"""Render panels: rotating 3D views of several channels, side by side.

A panel is a row of tiles defined in the config. Each tile shows one channel,
or the merge of several channels, rotating around one axis.

Every channel goes through the same steps:

    prepare_channel_frames()   clip and scale to 8 bit (clip range from the config)
                               -> isotropic resampling (Z interpolated to the XY pixel size)
                               -> rotating maximum intensity projection
    compose_tile()             colormaps, merge, channel labels

render_panel() puts the tiles side by side and adds the scale bar and the
file name. render_and_save() renders a panel once and saves it as .mov
(save_movie) and/or .gif (save_gif).
"""

import logging

import numpy as np
from pathlib import Path
from scipy.ndimage import rotate as ndi_rotate, zoom as ndi_zoom

from chimerat.clipping import clip_plane
from chimerat.colormaps import apply_colormap
from chimerat.scalebar import add_channel_labels, add_scalebar

logger = logging.getLogger(__name__)

# L'assenza di supporto alle cl.Image e' una proprieta' del device, non della
# singola immagine: l'avviso ha senso una volta per sessione, non a ogni canale.
_LINEAR_UNSUPPORTED_WARNED = False



def make_isotropic(image, anisotropy_factor, dtype=np.uint8,
                   linear_interpolation=False, on_unsupported='scipy'):
    """
    Porta lo stack a voxel isotropici interpolando Z, con clesperanto.

    Al contrario di resample_to_isotropic() (che porta tutto alla risoluzione
    piu' grossolana, buttando via il dettaglio laterale), qui Z viene portato
    alla risoluzione di XY: e' quello che serve per una figura.

    Parameters
    ----------
    image : ndarray (Z, Y, X)
    anisotropy_factor : float
        spacing_z / spacing_xy. Con (0.4, 0.1, 0.1) vale 4.
    dtype : numpy dtype
        Tipo del buffer di uscita. I piani arrivano qui gia' uint8 da
        clip_plane(), quindi uint8 va bene sia per le maschere sia per i
        canali di intensita'; passa uint16 solo se ricampioni dati grezzi
        prima del clip.
    linear_interpolation : bool
        False (nearest, default) OBBLIGATORIO per label e maschere:
        l'interpolazione lineare creerebbe valori intermedi tra due etichette
        adiacenti, cioe' oggetti che non esistono.
        True per canali continui: evita lo scalino in Z che si vede ruotando.

        Attenzione: cle.scale con linear_interpolation=True richiede il
        supporto alle cl.Image, che parecchi device (tipicamente i backend
        CPU sui nodi di calcolo) non hanno. In quel caso clesperanto solleva
        "No supported ImageFormat found": vedi on_unsupported.
    on_unsupported : {'scipy', 'nearest', 'raise'}
        Cosa fare se il device non supporta l'interpolazione lineare:
        'scipy'   rifa' l'interpolazione lineare su CPU con ndimage.zoom
                  (stesso risultato, piu' lento) e logga un avviso
        'nearest' ripiega su nearest e logga un avviso
        'raise'   propaga l'errore
    """
    import pyclesperanto_prototype as cle
    cle.select_device("cpu")

    dimension_x = image.shape[-1]
    dimension_y = image.shape[-2]
    dimension_z = int(round(image.shape[-3] * anisotropy_factor))
    factor_z = dimension_z / image.shape[-3]

    def _run(linear):
        # Il buffer va ricreato a ogni tentativo: uno scale fallito puo'
        # averlo lasciato scritto a meta'.
        resized = cle.create([dimension_z, dimension_y, dimension_x], dtype=dtype)
        # I fattori restano float: int(dimension_z / shape[-3]) troncava, e il
        # buffer veniva riempito solo in parte lasciando slice nere in cima.
        # Il caso insidioso e' un rapporto apparentemente intero: 0.3/0.1 in
        # virgola mobile e' 2.9999999999999996, quindi int() dava 2.
        cle.scale(image, resized,
                  factor_z=factor_z, factor_y=1.0, factor_x=1.0,
                  centered=False, linear_interpolation=linear)
        return np.array(resized, dtype=dtype)

    if not linear_interpolation:
        return _run(False)

    try:
        return _run(True)
    except Exception as err:
        # cle solleva ValueError da _get_image_format quando il device non ha
        # formati immagine; il suo fallback interno intercetta solo
        # pyopencl.RuntimeError, quindi qui l'errore arriva grezzo.
        if on_unsupported == 'raise':
            raise
        global _LINEAR_UNSUPPORTED_WARNED
        if not _LINEAR_UNSUPPORTED_WARNED:
            logger.warning(
                "Interpolazione lineare non disponibile su questo device OpenCL "
                "(%s): il device non espone formati immagine. Ripiego su '%s' "
                "per tutta la sessione.", type(err).__name__, on_unsupported,
            )
            _LINEAR_UNSUPPORTED_WARNED = True
        if on_unsupported == 'nearest':
            return _run(False)
        if on_unsupported != 'scipy':
            raise ValueError(
                f"on_unsupported={on_unsupported!r} non valido: "
                f"usa 'scipy', 'nearest' o 'raise'"
            ) from err

    # fallback CPU: interpolazione lineare vera, solo lungo Z
    out = ndi_zoom(image.astype(np.float32), (factor_z, 1.0, 1.0), order=1)
    if out.shape[0] != dimension_z:                 # arrotondamenti di zoom
        fixed = np.zeros((dimension_z, dimension_y, dimension_x), np.float32)
        z = min(out.shape[0], dimension_z)
        fixed[:z] = out[:z]
        out = fixed
    if np.issubdtype(np.dtype(dtype), np.integer):
        info = np.iinfo(dtype)
        out = np.clip(np.rint(out), info.min, info.max)
    return out.astype(dtype)


def cle_resampler(dtype=np.uint8, linear_interpolation=False, on_unsupported='scipy'):
    """
    Adatta make_isotropic all'interfaccia che si aspetta render_tile,
    cioe' una callable (plane, spacing_um) -> plane.

    >>> render_tile(stack, cm, 'skels', spacing_um,
    ...             resampler=cle_resampler(linear_interpolation=True))

    Il default e' nearest, l'unico corretto per label e maschere e l'unico
    disponibile su device OpenCL senza supporto alle cl.Image. Passa
    linear_interpolation=True per i canali continui.
    """
    def _resample(plane, spacing_um):
        spacing_z, spacing_y, spacing_x = spacing_um
        if not np.isclose(spacing_y, spacing_x):
            raise ValueError(
                f"make_isotropic assume spacing_y == spacing_x, "
                f"trovati y={spacing_y}, x={spacing_x}"
            )
        return make_isotropic(
            plane,
            anisotropy_factor=spacing_z / spacing_y,
            dtype=dtype,
            linear_interpolation=linear_interpolation,
            on_unsupported=on_unsupported,
        )
    return _resample


def _default_resampler(plane, spacing_um):
    """Isotropic voxels with scipy: interpolate Z only, keep XY untouched.

    Same behavior as cle_resampler (nearest neighbor), so the scalebar,
    computed with the original XY spacing, stays correct.
    """
    spacing_z, spacing_y, spacing_x = spacing_um
    return ndi_zoom(plane, (spacing_z / spacing_y, 1.0, 1.0), order=0)


# Plane in which the volume rotates, for each rotation axis (Z=0, Y=1, X=2).
_ROTATION_AXES = {"x": (0, 1), "y": (0, 2), "z": (1, 2)}


def _pad_for_rotation(plane, axes):
    """Pad the rotation plane to a square as large as its diagonal.

    ndi_rotate(reshape=False) keeps the original shape, so everything that
    rotates outside the box is lost: at 90 degrees only the central part of
    the crop would be visible. With the padding, every angle fits.
    """
    diagonal = int(np.ceil(np.hypot(plane.shape[axes[0]], plane.shape[axes[1]])))
    pad = [(0, 0)] * plane.ndim
    for axis in axes:
        extra = diagonal - plane.shape[axis]
        pad[axis] = (extra // 2, extra - extra // 2)
    return np.pad(plane, pad)


def _rotated_mip(plane, angle, axes):
    rotated = ndi_rotate(plane, angle, axes=axes, order=1, reshape=False)
    return rotated.max(axis=0)


def _rotate_one(args):
    """Picklable version for ProcessPoolExecutor."""
    return _rotated_mip(*args)


def rotate_mip_gray(plane, n_frames=36, axis="y", workers=None):
    """Rotate a grayscale (Z, Y, X) volume and return the MIPs, (n_frames, H, W) uint8.

    The volume is padded first (see _pad_for_rotation), so the frames are
    larger than the crop, and nothing is cut at any angle.
    """
    axes = _ROTATION_AXES[axis.lower()]
    plane = _pad_for_rotation(plane, axes)
    angles = np.linspace(0, 360, n_frames, endpoint=False)

    if workers and workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(workers) as pool:
            frames = list(pool.map(_rotate_one, [(plane, a, axes) for a in angles]))
    else:
        frames = [_rotated_mip(plane, a, axes) for a in angles]

    return np.asarray(frames, dtype=np.uint8)


def prepare_channel_frames(stack, channel_map, role, spacing_um, n_frames=36,
                           axis="y", resampler=None, workers=None,
                           debug_viewer=None, verbose=False):
    """
    Da uno stack multicanale ai frame ruotati di UN canale, ancora in grigio.

    clip (dal config) -> resampling isotropico -> rotazione MIP

    E' la parte cara e, soprattutto, e' indipendente dal riquadro in cui il
    canale finira': lo stesso canale che compare in un riquadro singolo e nel
    MERGE si calcola una volta sola (vedi la cache in render_panel).
    """
    if resampler is None:
        resampler = _default_resampler

    idx = channel_map[role]
    clip_range = channel_map.clip_for(role)
    plane = clip_plane(stack[idx], clip_range, out_dtype="uint8")

    if debug_viewer is not None:
        debug_viewer.add_image(stack[idx], name=f"{role}_raw",
                               blending="additive", visible=False)
        debug_viewer.add_image(plane, name=f"{role}_clipped",
                               blending="additive", visible=False)
    if verbose:
        print(f"  {role}: clip={clip_range} -> uint8 [{plane.min()}, {plane.max()}]")

    plane = resampler(plane, spacing_um)
    return rotate_mip_gray(plane, n_frames=n_frames, axis=axis, workers=workers)


def compose_tile(gray_frames, colormaps, labels, title=None,
                 title_color=(255, 255, 255), blend_mode="add",
                 normalize_to=None, show_labels=True, label_size=30,
                 label_position="top-left", stroke_width=0, verbose=False):
    """
    Da frame in grigio al riquadro RGB finito: colormap, merge, etichette.

    Lavora su immagini 2D, quindi costa pochissimo rispetto alla rotazione.
    E' la parte che cambia da riquadro a riquadro.
    """
    gray_frames = list(gray_frames)
    n_frames = gray_frames[0].shape[0]

    if normalize_to is None:
        scales = [255] * len(gray_frames)
    elif isinstance(normalize_to, (int, float)):
        scales = [normalize_to] * len(gray_frames)
    else:
        if len(normalize_to) != len(gray_frames):
            raise ValueError(
                f"normalize_to ha {len(normalize_to)} elementi per "
                f"{len(gray_frames)} canali"
            )
        scales = list(normalize_to)

    out = np.zeros((n_frames, *gray_frames[0].shape[1:], 3), dtype=np.float32)
    for frames, cmap, scale in zip(gray_frames, colormaps, scales):
        if scale != 255:
            frames = (frames.astype(np.float32) * (scale / 255.0)).astype(np.uint8)
        colored = np.stack([apply_colormap(f, cmap) for f in frames]).astype(np.float32)
        if blend_mode == "add":
            out += colored
        elif blend_mode == "max":
            out = np.maximum(out, colored)
        else:
            raise ValueError(f"blend_mode '{blend_mode}' non riconosciuto")

    frames = np.clip(out, 0, 255).astype(np.uint8)

    if show_labels:
        if title is not None:
            texts, text_colors = [title], [title_color]
        else:
            texts, text_colors = labels, colormaps
        frames = add_channel_labels(
            frames, texts, text_colors, position=label_position,
            label_size=label_size, stroke_width=stroke_width, verbose=verbose,
        )
    return frames


def render_panel(
    stacks,
    cfg,
    panel_name,
    spacing_um,
    filename=None,
    n_frames=36,
    axis='y',
    resampler=None,
    label_size=30,
    stroke_width=0,
    scalebar_length_um=10,
    scalebar_kwargs=None,
    tile_kwargs=None,
    channel_axis=0,
    workers=None,
    debug_viewer=None,
    verbose=False,
):
    """
    Compone un pannello descritto nel config.

    La funzione non sa quali pannelli esistono ne' cosa contengono: legge
    cfg.panel(panel_name) e itera. Un pannello nuovo e' un blocco di YAML.

    Parameters
    ----------
    stacks : dict
        Sezione -> stack multicanale (C, Z, Y, X). I riquadri pescano da
        sezioni diverse, quindi non basta un array solo:

            stacks = {'acquisition': raw,
                      'segmentation_output': seg,
                      'mito_output': mito}

        Devono esserci solo le sezioni citate dal pannello.
    cfg : Config
    panel_name : str
        Chiave sotto 'panels:' nel YAML.
    spacing_um : tuple
        (spacing_z, spacing_y, spacing_x) in micron.
    filename : str or None
        Testo scritto in basso a sinistra sul riquadro indicato da
        filename_on. Non sta nel config perche' cambia a ogni immagine.
    scalebar_length_um : float
        Lunghezza della barra, sul riquadro indicato da scalebar_on.
    tile_kwargs : dict or None
        Argomenti extra inoltrati a render_tile (blend_mode, normalize_to...).

    Returns
    -------
    ndarray : (n_frames, H, W_totale, 3) uint8
    """
    panel = cfg.panel(panel_name)
    tile_kwargs = dict(tile_kwargs or {})
    scalebar_kwargs = dict(scalebar_kwargs or {})

    needed = {ref.section for tile in panel.tiles for ref in tile.channels}
    missing = needed - set(stacks)
    if missing:
        raise ValueError(
            f"Il pannello '{panel_name}' usa le sezioni {sorted(needed)} ma in "
            f"'stacks' mancano {sorted(missing)}."
        )

    spacing = spacing_um
    scalebar_tiles = set(panel.tile_indices(panel.scalebar_on))
    filename_tiles = set(panel.tile_indices(panel.filename_on))

    # Un canale che compare in piu' riquadri (la falloidina sta nel riquadro
    # singolo E nel MERGE) va ricampionato e ruotato una volta sola: e' la
    # parte cara, e non dipende dal riquadro in cui finisce.
    gray_cache: dict = {}

    def _gray_for(ref):
        key = (ref.section, ref.role.lower())
        if key not in gray_cache:
            channel_map = getattr(cfg, ref.section).channels
            stack = np.moveaxis(stacks[ref.section], channel_axis, 0)
            gray_cache[key] = prepare_channel_frames(
                stack, channel_map, ref.role, spacing,
                n_frames=n_frames, axis=axis, resampler=resampler,
                workers=workers, debug_viewer=debug_viewer, verbose=verbose,
            )
        elif verbose:
            print(f"    {ref.section}/{ref.role}: riuso dalla cache")
        return gray_cache[key]

    rendered = []
    for i, tile in enumerate(panel.tiles):
        if verbose:
            refs = ", ".join(f"{r.section}/{r.role}" for r in tile.channels)
            print(f"[{panel_name}] riquadro {i}: {refs} title={tile.title!r}")

        gray = [_gray_for(ref) for ref in tile.channels]
        colormaps, labels = [], []
        for ref in tile.channels:
            channel_map = getattr(cfg, ref.section).channels
            colormaps.append(channel_map.colormap_for(ref.role))
            labels.append(channel_map.label(ref.role))

        shapes = {f.shape for f in gray}
        if len(shapes) > 1:
            detail = ", ".join(f"{r.section}/{r.role}: {f.shape}"
                               for r, f in zip(tile.channels, gray))
            raise ValueError(
                f"[{panel_name}.tiles[{i}]] i canali non hanno la stessa shape "
                f"({detail})."
            )

        frames = compose_tile(
            gray, colormaps, labels, title=tile.title,
            label_size=label_size, stroke_width=stroke_width,
            verbose=verbose, **tile_kwargs,
        )

        # -- annotazioni di bordo, sui riquadri indicati dal config ---------- #
        if i in filename_tiles and filename:
            frames = add_channel_labels(
                frames, [filename], [(255, 255, 255)],
                position='bottom-left', label_size=label_size,
                stroke_width=stroke_width, verbose=False,
            )
        if i in scalebar_tiles:
            frames = add_scalebar(
                frames,
                spacing_um=spacing_um,     # XY invariato dal resampling
                scalebar_length_um=scalebar_length_um,
                position='bottom-right',
                label_size=label_size,
                stroke_width=stroke_width,
                verbose=False,
                **scalebar_kwargs,
            )

        rendered.append(frames)

    # -- assemblaggio orizzontale ------------------------------------------- #
    shapes = {f.shape[:2] + f.shape[2:3] for f in rendered}
    heights = {f.shape[1] for f in rendered}
    n_frames_out = {f.shape[0] for f in rendered}
    if len(heights) > 1 or len(n_frames_out) > 1:
        detail = ", ".join(f"riquadro {i}: {f.shape}" for i, f in enumerate(rendered))
        raise ValueError(
            f"I riquadri di '{panel_name}' non sono componibili ({detail}). "
            f"Gli stack delle diverse sezioni devono avere la stessa shape XY "
            f"e lo stesso numero di frame."
        )

    if panel.gutter_px:
        gutter = np.zeros(
            (rendered[0].shape[0], rendered[0].shape[1], panel.gutter_px, 3),
            dtype=np.uint8,
        )
        pieces = []
        for i, f in enumerate(rendered):
            if i:
                pieces.append(gutter)
            pieces.append(f)
    else:
        pieces = rendered

    panel_frames = np.concatenate(pieces, axis=2)

    if debug_viewer is not None:
        debug_viewer.add_image(panel_frames, name=f"panel_{panel_name}",
                               blending='opaque')

    return panel_frames


def save_movie(
    frames,
    path,
    fps=6,
    quality=6,
    codec='libx264',
    macro_block_size=16,
    pad_value=0,
    verbose=True,
):
    """
    Salva i frame di un pannello come filmato (.mov, .mp4...).

    Parameters
    ----------
    frames : ndarray (n_frames, H, W, 3) uint8
        L'uscita di render_panel().
    path : str or Path
        File di destinazione; l'estensione decide il contenitore.
    fps : int
        Frame al secondo. Con 36 frame di rotazione, fps=6 da' 6 secondi.
    quality : int
        0-10, 10 e' la migliore.
    macro_block_size : int
        H.264 lavora a blocchi: imageio, se le dimensioni non sono multiple
        di 16, RIDIMENSIONA l'immagine, e lo fa su ogni asse separatamente.
        Un pannello 300x1156 diventerebbe 304x1168, cioe' +1.33% in altezza
        e +1.04% in larghezza: le cellule si deformano e la scalebar non
        misura piu' l'asse verticale.

        Qui si fa invece un padding nero fino al multiplo successivo, che
        lascia i pixel esattamente dove sono. Metti 1 per non aggiungere
        nulla, a rischio di incompatibilita' con qualche player.
    pad_value : int
        Colore del bordo aggiunto (0 = nero, come il fondo dei pannelli).

    Returns
    -------
    Path : il file scritto.
    """
    import imageio.v3 as iio

    frames = np.asarray(frames)
    if frames.ndim != 4 or frames.shape[-1] not in (3, 4):
        raise ValueError(
            f"Attesi frame (n, H, W, 3), trovato {frames.shape}"
        )
    if frames.dtype != np.uint8:
        raise ValueError(
            f"I codec video vogliono uint8, trovato {frames.dtype}"
        )

    n, H, W = frames.shape[:3]

    if macro_block_size > 1:
        H_pad = int(np.ceil(H / macro_block_size) * macro_block_size)
        W_pad = int(np.ceil(W / macro_block_size) * macro_block_size)
        if (H_pad, W_pad) != (H, W):
            padded = np.full((n, H_pad, W_pad, frames.shape[3]), pad_value,
                             dtype=np.uint8)
            padded[:, :H, :W] = frames        # in alto a sinistra: il bordo
            frames = padded                    # cresce in basso e a destra
            if verbose:
                print(f"  padding {H}x{W} -> {H_pad}x{W_pad} "
                      f"(multiplo di {macro_block_size}, nessun ridimensionamento)")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    iio.imwrite(
        path,
        frames,
        fps=fps,
        codec=codec,
        quality=quality,
        macro_block_size=1,      # il padding l'abbiamo gia' fatto noi
    )

    if verbose:
        size_mb = path.stat().st_size / 1e6
        print(f"  {path.name}: {n} frame a {fps} fps "
              f"({n / fps:.1f} s), {size_mb:.1f} MB")

    return path


MOVIE_FORMATS = ("mov", "gif")


def normalize_formats(formats):
    """Turn the formats option into a tuple, e.g. ("mov", "gif").

    Accepts a single string ("gif"), "both", or a list/tuple of formats.
    A plain string must be handled apart: tuple("gif") would give
    ("g", "i", "f"), because Python iterates a string letter by letter.
    """
    if isinstance(formats, str):
        formats = MOVIE_FORMATS if formats.lower() == "both" else (formats,)
    formats = tuple(f.lower().lstrip(".") for f in formats)
    unknown = [f for f in formats if f not in MOVIE_FORMATS]
    if unknown or not formats:
        raise ValueError(
            f"formats={formats!r} not valid: use 'mov', 'gif', 'both' "
            f"or a list such as ['mov', 'gif']"
        )
    return formats


def save_gif(frames, path, fps=6, verbose=True):
    """Save the frames of a panel as an animated GIF (loops forever).

    GIF has no codec, so no padding is needed: pixels stay exactly where they are.
    GIF can only show 256 colors: the palette is computed once on all frames
    together, so colors don't flicker from one frame to the next. Dithering is
    off, because it would add noise to the black background.

    Parameters
    ----------
    frames : ndarray (n_frames, H, W, 3) uint8
        The output of render_panel().
    path : str or Path
    fps : int
        Frames per second. GIF timing has a 10 ms resolution, so the real
        speed can differ slightly from the .mov one.

    Returns
    -------
    Path : the written file.
    """
    from PIL import Image

    frames = np.asarray(frames)
    if frames.ndim != 4 or frames.shape[-1] != 3 or frames.dtype != np.uint8:
        raise ValueError(f"Expected uint8 frames (n, H, W, 3), found {frames.shape} {frames.dtype}")

    # One shared palette: quantize all frames stacked as a single tall image.
    all_frames = Image.fromarray(frames.reshape(-1, frames.shape[2], 3))
    palette = all_frames.quantize(colors=256, method=Image.Quantize.MEDIANCUT)
    images = [
        Image.fromarray(frame).quantize(palette=palette, dither=Image.Dither.NONE)
        for frame in frames
    ]

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        duration=round(1000 / fps),   # milliseconds per frame
        loop=0,                       # 0 = repeat forever
    )

    if verbose:
        size_mb = path.stat().st_size / 1e6
        print(f"  {path.name}: {len(frames)} frames at {fps} fps, {size_mb:.1f} MB")
    return path


def render_and_save(
    stacks,
    cfg,
    panel_name,
    spacing_um,
    output_dir,
    filename=None,
    fps=6,
    quality=6,
    suffix=None,
    save_tiff=False,
    formats=("mov",),
    **panel_kwargs,
):
    """
    Renderizza un pannello e lo salva come filmato.

    Il nome del file esce da panel_name e da `filename`, cosi' i pannelli di
    immagini diverse non si sovrascrivono a vicenda.

    formats : str or list of str
        Which files to write: "mov", "gif", "both", or a list like ["mov", "gif"].
        The panel is rendered once and saved in every format.

    Returns (frames, list of written paths).

    >>> render_and_save(stacks, cfg, 'segmentation', spacing_um,
    ...                 output_dir='out', filename=Path(tif).stem,
    ...                 resampler=cle_resampler())
    out/panel_segmentation_q6.mov
    """
    frames = render_panel(stacks, cfg, panel_name, spacing_um,
                          filename=filename, **panel_kwargs)

    output_dir = Path(output_dir)
    stem = f"{filename}_{panel_name}" if filename else panel_name
    stem += suffix if suffix is not None else f"_q{quality}"

    formats = normalize_formats(formats)

    written = []
    if "mov" in formats:
        written.append(save_movie(frames, output_dir / f"{stem}.mov", fps=fps, quality=quality))
    if "gif" in formats:
        written.append(save_gif(frames, output_dir / f"{stem}.gif", fps=fps))

    if save_tiff:
        import tifffile
        tifffile.imwrite(output_dir / f"{stem}.tif", frames)

    return frames, written
