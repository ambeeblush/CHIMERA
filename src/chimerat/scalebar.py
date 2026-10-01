"""Scale bar and channel labels, drawn directly into the frames.
 
add_scalebar() draws a bar of a given length in µm. Its length in pixels is
computed from the XY pixel size, which is why isotropic resampling only
changes Z: XY keeps the size written in the file metadata, and the bar stays
correct. If the bar doesn't fit in the image, an error is raised instead of
a bar that silently gets cut off.
 
add_channel_labels() writes the name of each channel, in the color that
channel is shown with (taken from colormaps.py). A gradient LUT has no single
color, so its label uses a representative shade of the LUT. 'grays' becomes
white, because mid-gray text would be hard to read on a dark background.
 
Both work on a single image or on a whole stack of frames:
 
    (H, W)          2D grayscale        (Z, H, W)       grayscale stack
    (H, W, 3/4)     2D RGB / RGBA       (Z, H, W, 3/4)  RGB / RGBA stack
 
Text is rendered with Pillow, using the first TrueType font found on the
system (DejaVu, Liberation or Arial, bold by default), so the output looks
the same on Linux, macOS and Windows as long as one of them is installed.
"""
 
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from skimage.draw import rectangle
 
from chimerat.colormaps import to_rgb
 
 
_FONT_CANDIDATES = [
    '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
    '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf',
    '/Library/Fonts/Arial.ttf',
    '/System/Library/Fonts/Supplemental/Arial.ttf',
    'C:\\Windows\\Fonts\\arial.ttf',
]
 
_FONT_CANDIDATES_BOLD = [
    '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
    '/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf',
    '/usr/share/fonts/truetype/freefont/FreeSansBold.ttf',
    '/Library/Fonts/Arial Bold.ttf',
    '/System/Library/Fonts/Supplemental/Arial Bold.ttf',
    'C:\\Windows\\Fonts\\arialbd.ttf',
]
 
 
def _load_font(font_path, label_size, bold=True):
    """Load a TrueType font, falling back to Pillow's default font."""
    if font_path:
        paths = [font_path]
    else:
        paths = _FONT_CANDIDATES_BOLD + _FONT_CANDIDATES if bold else _FONT_CANDIDATES
    for p in paths:
        try:
            return ImageFont.truetype(p, label_size)
        except Exception:
            continue
    try:
        # Pillow >= 10 also accepts a size for the default font
        return ImageFont.load_default(size=label_size)
    except TypeError:
        return ImageFont.load_default()
 
 
def classify_array(array, is_stack=None):
    """
    Tell what kind of array was passed in.
 
    Returns
    -------
    str : one of '2d_gray', '2d_rgb', 'stack_gray', 'stack_rgb'
 
    Notes
    -----
    The ambiguous case is ndim == 3: it can be an (H, W, 3) RGB image or a
    (Z, H, W) grayscale stack with 3 slices. By default it is read as RGB if
    the last dimension is 3 or 4; pass is_stack=True/False to force it.
    """
    ndim = array.ndim
    last_is_channel = array.shape[-1] in (3, 4)
 
    if ndim == 2:
        return '2d_gray'
 
    if ndim == 4:
        if not last_is_channel:
            raise ValueError(
                f"4D array with last dimension {array.shape[-1]}: "
                f"expected (Z, H, W, 3) or (Z, H, W, 4), got {array.shape}"
            )
        return 'stack_rgb'
 
    if ndim == 3:
        if is_stack is True:
            return 'stack_gray'
        if is_stack is False:
            return '2d_rgb'
        return '2d_rgb' if last_is_channel else 'stack_gray'
 
    raise ValueError(f"Unsupported number of dimensions: ndim={ndim}, shape={array.shape}")
 
 
def _bar_origin(H, W, bar_len_px, thickness_px, position, margin_px):
    """(row, column) of the top-left corner of the bar."""
    if position == 'bottom-right':
        return H - thickness_px - margin_px, W - bar_len_px - margin_px
    if position == 'bottom-left':
        return H - thickness_px - margin_px, margin_px
    if position == 'top-right':
        return margin_px, W - bar_len_px - margin_px
    if position == 'top-left':
        return margin_px, margin_px
    raise ValueError(
        f"position '{position}' not recognized. Use: bottom-right, "
        f"bottom-left, top-right, top-left"
    )
 
 
def _draw_on_frame(frame, bar_len_px, label_text, position, thickness_px,
                   color, label_color, font, margin_px, show_label,
                   stroke_width=0, antialias=True, alpha_threshold=0.5,
                   label_gap_px=6):
    """
    Draw bar + label on a single 2D frame, (H, W) or (H, W, C).
    Works on a copy and returns it.
    """
    frame = frame.copy()
    H, W = frame.shape[:2]
    is_color = frame.ndim == 3
 
    if bar_len_px > W - 2 * margin_px:
        raise ValueError(
            f"The scale bar ({bar_len_px} px) doesn't fit in an image {W} px wide. "
            f"Reduce scalebar_length_um or the margin."
        )
 
    y0, x0 = _bar_origin(H, W, bar_len_px, thickness_px, position, margin_px)
 
    # --- bar ---
    rr, cc = rectangle(start=(y0, x0), extent=(thickness_px, bar_len_px), shape=(H, W))
    rr, cc = rr.astype(int), cc.astype(int)
 
    if is_color:
        n_ch = frame.shape[2]
        frame[rr, cc] = np.array(color[:n_ch] if n_ch <= 3 else tuple(color) + (255,),
                                 dtype=frame.dtype)
    else:
        frame[rr, cc] = int(np.mean(color))
 
    if not show_label:
        return frame
 
    # --- label: Pillow text mask, blended onto the frame ---
    mask_img = Image.new('L', (W, H), 0)
    mdraw = ImageDraw.Draw(mask_img)
 
    bbox = mdraw.textbbox((0, 0), label_text, font=font, stroke_width=stroke_width)
    text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
 
    text_x = x0 + (bar_len_px - text_w) // 2 - bbox[0]
    text_y = y0 - text_h - label_gap_px - bbox[1]
    if text_y + bbox[1] < 0:                 # no room above the bar: put it below
        text_y = y0 + thickness_px + label_gap_px - bbox[1]
    text_x = int(np.clip(text_x, 0, max(0, W - text_w)))
    text_y = int(text_y)
 
    mdraw.text((text_x, text_y), label_text, fill=255, font=font,
               stroke_width=stroke_width, stroke_fill=255)
 
    alpha = np.asarray(mask_img).astype(np.float32) / 255.0
    if not antialias:
        # no intermediate levels: every glyph pixel becomes fully opaque
        alpha = (alpha >= alpha_threshold).astype(np.float32)
 
    return composite_mask(frame, alpha, label_color)
 
 
def resolve_channel_color(colormap):
    """RGB color for the text label of a channel (see colormaps.to_rgb)."""
    return to_rgb(colormap)
 
 
def composite_mask(frame, alpha, color):
    """
    Blend a solid color onto the frame through an (H, W) alpha mask in 0-1.
    Works on (H, W), (H, W, 3) and (H, W, 4) frames.
    """
    if frame.ndim == 3:
        n_ch = frame.shape[2]
        fill = np.array(list(color[:3]) + [255] * max(0, n_ch - 3),
                        dtype=np.float32)[:n_ch]
        blended = frame.astype(np.float32) * (1 - alpha[..., None]) + fill * alpha[..., None]
    else:
        blended = frame.astype(np.float32) * (1 - alpha) + float(np.mean(color)) * alpha
    return np.clip(blended, 0, 255).astype(frame.dtype)
 
 
def _channel_label_masks(W, H, labels, font, position, margin_px, line_gap_px,
                         stroke_width, antialias, alpha_threshold):
    """
    Build one mask per channel label, stacked vertically. The masks are the
    same on every frame of a stack, so they are computed only once.
 
    Returns
    -------
    list of ndarray : one float (H, W) mask in 0-1 per label
    """
    probe = Image.new('L', (1, 1))
    pdraw = ImageDraw.Draw(probe)
 
    boxes = [pdraw.textbbox((0, 0), t, font=font, stroke_width=stroke_width)
             for t in labels]
    widths = [b[2] - b[0] for b in boxes]
    heights = [b[3] - b[1] for b in boxes]
    line_h = max(heights) + line_gap_px
 
    on_top = position.startswith('top')
    on_left = position.endswith('left')
 
    masks = []
    for i, (text, box, w, h) in enumerate(zip(labels, boxes, widths, heights)):
        if on_top:
            y = margin_px + i * line_h
        else:
            # from the bottom up, keeping the order of the labels
            y = H - margin_px - (len(labels) - i) * line_h + line_gap_px
        x = margin_px if on_left else W - margin_px - w
 
        mask_img = Image.new('L', (W, H), 0)
        mdraw = ImageDraw.Draw(mask_img)
        # box[0]/box[1] compensate for the glyph bearing
        mdraw.text((int(x - box[0]), int(y - box[1])), text, fill=255, font=font,
                   stroke_width=stroke_width, stroke_fill=255)
 
        alpha = np.asarray(mask_img).astype(np.float32) / 255.0
        if not antialias:
            alpha = (alpha >= alpha_threshold).astype(np.float32)
        masks.append(alpha)
 
    return masks
 
 
def add_channel_labels(
    image_array,
    labels,
    colormaps,
    position='top-left',
    label_size=None,
    margin_px=10,
    line_gap_px=4,
    bold=True,
    stroke_width=0,
    antialias=True,
    alpha_threshold=0.5,
    font_path=None,
    is_stack=None,
    verbose=True,
):
    """
    Write the channel names on the image, each in the color its channel is
    shown with.
 
    Parameters
    ----------
    image_array : ndarray
        (H, W), (H, W, 3/4), (Z, H, W) or (Z, H, W, 3/4).
    labels : list of str
        Channel names, e.g. ['DAPI', 'TOM20', 'Phalloidin'].
    colormaps : list
        Colormap of each label, in the same order as the channels of the
        tile: e.g. ['blue', 'green', 'yellow']. RGB tuples and gradient LUTs
        are accepted too.
    position : str
        'top-left' (default), 'top-right', 'bottom-left', 'bottom-right'.
        Labels are stacked vertically in the order of the list. Pick a
        different corner from the scale bar.
    label_size : int or None
        Font size in pixels; None scales it with the frame height.
    line_gap_px : int
        Vertical space between labels.
    bold, stroke_width, antialias, alpha_threshold, font_path
        As in add_scalebar().
 
    Returns
    -------
    ndarray : copy of the input with the labels, same shape and dtype.
 
    Examples
    --------
    >>> out = add_channel_labels(
    ...     merged_frames,
    ...     labels=['DAPI', 'TOM20', 'Phalloidin'],
    ...     colormaps=['blue', 'green', 'yellow'],   # same as in the merge
    ...     position='top-left',
    ... )
    """
    if len(labels) != len(colormaps):
        raise ValueError(
            f"labels has {len(labels)} items and colormaps {len(colormaps)}: "
            f"they must match one to one."
        )
    if len(labels) == 0:
        return image_array.copy()
 
    kind = classify_array(image_array, is_stack=is_stack)
    colors = [resolve_channel_color(c) for c in colormaps]
 
    frame_h = image_array.shape[0] if kind in ('2d_gray', '2d_rgb') else image_array.shape[1]
    frame_w = image_array.shape[1] if kind in ('2d_gray', '2d_rgb') else image_array.shape[2]
    if label_size is None:
        label_size = max(11, int(frame_h / 18))
 
    if position not in ('top-left', 'top-right', 'bottom-left', 'bottom-right'):
        raise ValueError(
            f"position '{position}' not recognized. Use: top-left, top-right, "
            f"bottom-left, bottom-right"
        )
 
    font = _load_font(font_path, label_size, bold=bold)
 
    masks = _channel_label_masks(
        frame_w, frame_h, list(labels), font, position, margin_px, line_gap_px,
        stroke_width, antialias, alpha_threshold,
    )
 
    if verbose:
        pairs = ', '.join(f"{t}={c}" for t, c in zip(labels, colormaps))
        print(f"Channel labels ({position}, font {label_size} px): {pairs}")
 
    if kind in ('2d_gray', '2d_rgb'):
        out = image_array.copy()
        for alpha, col in zip(masks, colors):
            out = composite_mask(out, alpha, col)
        return out
 
    result = image_array.copy()
    for z in range(result.shape[0]):
        frame = result[z]
        for alpha, col in zip(masks, colors):
            frame = composite_mask(frame, alpha, col)
        result[z] = frame
    return result
 
 
def add_scalebar(
    image_array,
    spacing_um,
    scalebar_length_um=10,
    position='bottom-right',
    thickness_px=3,
    color=(255, 255, 255),
    label_color=(255, 255, 255),
    label_size=None,
    margin_px=10,
    font_path=None,
    show_label=True,
    label_text=None,
    bold=True,
    stroke_width=0,
    antialias=True,
    alpha_threshold=0.5,
    label_gap_px=6,
    is_stack=None,
    verbose=True,
):
    """
    Add a scale bar to a 2D image or to a stack.
 
    Parameters
    ----------
    image_array : ndarray
        (H, W), (H, W, 3/4), (Z, H, W) or (Z, H, W, 3/4).
    spacing_um : float or tuple
        Pixel size in µm. If a tuple, the last value (spacing_x) is used,
        consistent with the (spacing_z, spacing_y, spacing_x) format.
    scalebar_length_um : float
        Length of the bar in µm.
    position : str
        'bottom-right', 'bottom-left', 'top-right', 'top-left'.
    thickness_px : int
        Thickness of the bar in pixels.
    color, label_color : tuple (R, G, B)
        Color of the bar and of the text. On grayscale images the mean of
        the three values is used.
    label_size : int or None
        Font size in pixels. If None, it scales with the image height
        (about H/18, minimum 11).
    bold : bool
        Use a bold system font (default True): strokes stay solid even at
        small sizes.
    stroke_width : int
        Outline added to the glyphs, in the same color as the text. 1 or 2
        make the letters thicker.
    antialias : bool
        If False, the text mask is binarized: no gray edge pixels, fully
        solid letters (useful on dark backgrounds or when the image will be
        enlarged).
    alpha_threshold : float
        Threshold (0-1) used when antialias=False.
    label_gap_px : int
        Space in pixels between the bar and the text.
    margin_px : int
        Distance from the image border in pixels.
    font_path : str or None
        Path to a .ttf file; if None, the most common system fonts are tried.
    show_label : bool
        If False, only the bar is drawn, without text.
    label_text : str or None
        Custom text. If None, "<length> µm" is used.
    is_stack : bool or None
        Disambiguates 3D arrays: True = (Z, H, W), False = (H, W, 3).
        None = guess from the shape.
    verbose : bool
        Print the µm -> pixel conversion.
 
    Returns
    -------
    ndarray : copy of the input with the scale bar, same shape and dtype.
    """
    spacing_xy = spacing_um[-1] if isinstance(spacing_um, (list, tuple)) else spacing_um
    bar_len_px = int(round(scalebar_length_um / spacing_xy))
 
    if bar_len_px < 1:
        raise ValueError(
            f"The scale bar would be {bar_len_px} px long: increase scalebar_length_um "
            f"(spacing = {spacing_xy} um/px)."
        )
 
    kind = classify_array(image_array, is_stack=is_stack)
 
    if label_text is None:
        length_str = (f"{scalebar_length_um:g}")
        label_text = f"{length_str} \u00b5m"
 
    if verbose:
        print(f"Shape {image_array.shape} -> {kind}")
        print(f"Scale bar: {scalebar_length_um} um = {bar_len_px} px "
              f"(spacing {spacing_xy} um/px)")
 
    if label_size is None:
        frame_h = image_array.shape[-3] if kind.startswith('stack') or image_array.ndim >= 3 \
                  else image_array.shape[0]
        frame_h = image_array.shape[1] if kind == 'stack_gray' else frame_h
        label_size = max(11, int(frame_h / 18))
 
    font = _load_font(font_path, label_size, bold=bold) if show_label else None
 
    common = dict(
        bar_len_px=bar_len_px,
        label_text=label_text,
        position=position,
        thickness_px=thickness_px,
        color=color,
        label_color=label_color,
        font=font,
        margin_px=margin_px,
        show_label=show_label,
        stroke_width=stroke_width,
        antialias=antialias,
        alpha_threshold=alpha_threshold,
        label_gap_px=label_gap_px,
    )
 
    if kind in ('2d_gray', '2d_rgb'):
        return _draw_on_frame(image_array, **common)
 
    # stack: frame by frame
    result = image_array.copy()
    for z in range(result.shape[0]):
        result[z] = _draw_on_frame(result[z], **common)
 
    if verbose:
        print(f"Scale bar added to {result.shape[0]} frames")
 
    return result