"""Colormaps and LUTs: every color used by CHIMERA is defined here.
 
Two kinds of colormaps:
 
    single colors   ImageJ colors (green, magenta, cyan, blue, red, yellow,
                    grays) or a custom (R, G, B) triplet. Coloring multiplies
                    the gray value by the color. Use them for channels that
                    have to be told apart in a merge.
    gradient LUTs   a 256-row RGB table indexed by the pixel value, taken from
                    matplotlib ('inferno', 'viridis', ...). Use them when the
                    pixel value is a quantity to read, like a distance map.
 
The module has no internal imports, so every other module can use it:
config.py validates the colormap names written in the YAML, panels.py colors
the frames (apply_colormap), scalebar.py writes each label in the color of
its channel (to_rgb). Matplotlib is imported only the first time one of its
LUTs is requested, so loading a config stays fast.
"""
 
import numpy as np
 
# nome -> (R, G, B); None significa grayscale
IMAGEJ_COLORMAPS = {
    'green':   (0, 255, 0),
    'magenta': (255, 0, 255),
    'cyan':    (0, 255, 255),
    'blue':    (0, 0, 255),
    'red':     (255, 0, 0),
    'yellow':  (255, 255, 0),
    'grays':   None,
}
 
# LUT prese da matplotlib. Costruite alla prima richiesta e messe in cache:
# colormaps.py resta un modulo foglia, e config.py puo' validare i nomi senza
# tirarsi dietro matplotlib all'import.
MPL_LUT_NAMES = (
    "inferno", "magma", "plasma", "viridis", "cividis", "turbo", "hot",
)
 
_MPL_CACHE: dict = {}
 
 
def _mpl_lut(name):
    """Tabella (256, 3) uint8 da una colormap di matplotlib."""
    if name in _MPL_CACHE:
        return _MPL_CACHE[name]
    try:
        from matplotlib import colormaps as mpl_colormaps
    except ImportError as err:                       # matplotlib < 3.5
        try:
            from matplotlib import cm
            table = cm.get_cmap(name)(np.linspace(0, 1, 256))
        except Exception:
            raise ColormapError(
                f"colormap '{name}' richiede matplotlib (pip install matplotlib)"
            ) from err
    else:
        table = mpl_colormaps[name](np.linspace(0, 1, 256))
 
    lut = (np.asarray(table)[:, :3] * 255).round().clip(0, 255).astype(np.uint8)
    _MPL_CACHE[name] = lut
    return lut
 
 
# Colore rappresentativo di ogni LUT, usato dove serve una tinta sola:
# l'etichetta del canale. Prendo un valore alto ma non l'estremo, perche' il
# bianco di fondo scala non direbbe quale LUT usa quel canale.
_LUT_LABEL_INDEX = 224
 
 
LUT_NAMES = MPL_LUT_NAMES
 
COLORMAP_NAMES = tuple(IMAGEJ_COLORMAPS) + LUT_NAMES
 
 
def is_lut(colormap) -> bool:
    """True se il valore e' il nome di una LUT a gradiente."""
    return isinstance(colormap, str) and colormap.lower() in LUT_NAMES
 
 
def get_lut(colormap) -> np.ndarray:
    """Tabella (256, 3) uint8 della LUT richiesta."""
    key = str(colormap).lower()
    if key in MPL_LUT_NAMES:
        return _mpl_lut(key)
    raise ColormapError(
        f"'{colormap}' non e' una LUT. Disponibili: {sorted(LUT_NAMES)}"
    )
 
 
class ColormapError(ValueError):
    """Colormap non riconosciuta."""
 
 
def is_valid(colormap) -> bool:
    """True se il valore e' un nome noto o una tripletta RGB valida."""
    try:
        normalize(colormap)
    except ColormapError:
        return False
    return True
 
 
def normalize(colormap):
    """
    Porta il valore in forma canonica: nome minuscolo oppure tupla RGB di int.
 
    Accetta:
        'Green' / 'green'      -> 'green'
        (0, 255, 0)            -> (0, 255, 0)
        [0, 255, 0]            -> (0, 255, 0)
    """
    if isinstance(colormap, str):
        key = colormap.lower()
        if key in LUT_NAMES:
            return key
        if key not in IMAGEJ_COLORMAPS:
            raise ColormapError(
                f"colormap '{colormap}' non riconosciuta. "
                f"Nomi validi: {sorted(COLORMAP_NAMES)}, "
                f"oppure una tripletta [R, G, B] 0-255"
            )
        return key
 
    if isinstance(colormap, (tuple, list)) and len(colormap) == 3:
        rgb = []
        for v in colormap:
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise ColormapError(f"componente RGB non numerica: {v!r}")
            if not 0 <= v <= 255:
                raise ColormapError(f"componente RGB fuori range 0-255: {v}")
            rgb.append(int(v))
        return tuple(rgb)
 
    raise ColormapError(
        f"atteso un nome di colormap o una tripletta [R, G, B], "
        f"trovato {colormap!r}"
    )
 
 
def to_rgb(colormap):
    """
    Tupla (R, G, B) effettiva. 'grays' diventa bianco: e' il valore giusto
    sia per il canale grigio sia per un testo leggibile su fondo scuro.
    """
    norm = normalize(colormap)
    if isinstance(norm, tuple):
        return norm
    if norm in LUT_NAMES:
        return tuple(int(v) for v in get_lut(norm)[_LUT_LABEL_INDEX])
    rgb = IMAGEJ_COLORMAPS[norm]
    return (255, 255, 255) if rgb is None else tuple(rgb)
 
 
def apply_colormap(image, colormap='grays'):
    """Color a grayscale 8-bit image (H, W) and return it as RGB (H, W, 3).
 
    colormap can be an ImageJ color ('green', 'magenta', ...), 'grays',
    a gradient LUT ('inferno', ...) or a custom (R, G, B) tuple.
    """
    
    H, W = image.shape
    
    # Normalizza a [0, 1]
    img_norm = image.astype(float) / 255.0
    
    if is_lut(colormap):
        # LUT a gradiente: il valore del pixel INDICIZZA la tabella, non
        # moltiplica un colore unico. Serve uno stack gia' a 8 bit, che e'
        # esattamente quello che produce clip_plane(out_dtype='uint8').
        rgb = get_lut(colormap)[image.astype(np.uint8)]
 
    elif colormap == 'grays':
        # Grayscale: replica il canale
        rgb = np.stack([image, image, image], axis=-1)
    
    elif isinstance(colormap, str) and colormap in IMAGEJ_COLORMAPS:
        # Colormap ImageJ predefinita
        color = IMAGEJ_COLORMAPS[colormap]  # (R, G, B)
        # Moltiplica l'immagine grayscale per il colore
        rgb = np.zeros((H, W, 3), dtype=np.uint8)
        rgb[:, :, 0] = (img_norm * color[0]).astype(np.uint8)
        rgb[:, :, 1] = (img_norm * color[1]).astype(np.uint8)
        rgb[:, :, 2] = (img_norm * color[2]).astype(np.uint8)
    
    elif isinstance(colormap, (tuple, list)) and len(colormap) == 3:
        # Colormap personalizzata RGB
        rgb = np.zeros((H, W, 3), dtype=np.uint8)
        rgb[:, :, 0] = (img_norm * colormap[0]).astype(np.uint8)
        rgb[:, :, 1] = (img_norm * colormap[1]).astype(np.uint8)
        rgb[:, :, 2] = (img_norm * colormap[2]).astype(np.uint8)
    
    else:
        raise ValueError(
            f"Colormap '{colormap}' non riconosciuta. "
            f"Opzioni: {list(IMAGEJ_COLORMAPS) + list(LUT_NAMES)} o tupla RGB")
    
    return rgb