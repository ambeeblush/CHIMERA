"""Clip channel intensities and scale them to 8 bit, using the clip ranges in the config.

The same clip range [lo, hi] does two jobs, like Brightness/Contrast in ImageJ:
values are clipped to [lo, hi], and lo -> 0, hi -> 255 when scaling to 8 bit.
Scaling to each image's own min/max after clipping would give different
mappings to images of the same experiment, and side-by-side comparisons
would be misleading.

    clip_plane()     clip one channel and, optionally, scale it to a target dtype
    suggest_clip()   print a ready-to-paste `clip:` block from the percentiles
                     of a real image, to choose the clip ranges the first time

target_range() and resolve_bounds() are helpers of clip_plane().
"""

import numpy as np


def target_range(out_dtype, like=None):
    """
    Range di arrivo della normalizzazione, in funzione del dtype richiesto.

    out_dtype puo' essere:
        None        -> nessuna normalizzazione (solo clip)
        'same'      -> il dtype dell'array di partenza (`like`)
        'uint8'     -> 0-255
        'uint16'    -> 0-65535
        np.float32  -> 0.0-1.0
        qualsiasi dtype numpy intero o float

    Returns
    -------
    (dtype, vmax) oppure (None, None) se out_dtype e' None.
    """
    if out_dtype is None:
        return None, None

    if out_dtype == 'same':
        if like is None:
            raise ValueError("out_dtype='same' richiede l'array di riferimento")
        dtype = np.dtype(like.dtype)
    else:
        dtype = np.dtype(out_dtype)

    if np.issubdtype(dtype, np.integer):
        return dtype, float(np.iinfo(dtype).max)
    if np.issubdtype(dtype, np.floating):
        # per i float la convenzione e' 0-1: e' quella che si aspettano
        # matplotlib, skimage e i writer OME-TIFF float
        return dtype, 1.0
    raise ValueError(f"out_dtype {out_dtype!r} non e' un tipo numerico")


def resolve_bounds(plane, clip_range):
    """
    Traduce un ClipRange nei due valori di intensita' effettivi per questo piano.

    Con mode='absolute' restituisce (lo, hi) cosi' come sono; con
    mode='percentile' li calcola sui dati. In entrambi i casi il risultato e'
    una coppia di numeri utilizzabile sia per np.clip sia per la
    normalizzazione.
    """
    if clip_range is None:
        return None
    if clip_range.mode == "absolute":
        return float(clip_range.lo), float(clip_range.hi)
    lo, hi = np.percentile(plane, [clip_range.lo, clip_range.hi])
    if hi <= lo:                      # piano piatto: evita la divisione per zero
        hi = lo + 1.0
    return float(lo), float(hi)


def clip_plane(plane, clip_range, out_dtype=None, to_uint8=None):
    """
    Applica il clip a un canale (Z, Y, X) o (Y, X) e, se richiesto, normalizza.

    Parameters
    ----------
    plane : ndarray
        Il canale, di qualsiasi dtype.
    clip_range : ClipRange or None
        Estremi presi dal config. None = nessun clip.
    out_dtype : None | 'same' | 'uint8' | 'uint16' | dtype
        Dtype di arrivo della normalizzazione:
          None     solo clip, dtype e valori originali conservati
          'same'   normalizza sul range pieno del dtype di partenza
                   (uint16 -> 0-65535, uint8 -> 0-255, float -> 0-1)
          'uint8'  0-255, la forma che serve a merge_channels()
          'uint16' 0-65535, per salvare TIFF quantitativi
    to_uint8 : bool, deprecato
        Vecchio parametro booleano. True equivale a out_dtype='uint8'.

    Returns
    -------
    ndarray : canale clippato e, se out_dtype non e' None, normalizzato in
        modo che clip_range.lo -> 0 e clip_range.hi -> massimo del dtype.

    Note
    ----
    Se clip_range e' None ma chiedi una normalizzazione, gli estremi vengono
    presi da min/max dei dati: e' l'unico riferimento disponibile, ma il
    risultato NON e' confrontabile tra immagini diverse. Per confronti
    quantitativi dichiara sempre il clip nel config.
    """
    if to_uint8 is not None:
        out_dtype = 'uint8' if to_uint8 else None

    dtype, vmax = target_range(out_dtype, like=plane)
    bounds = resolve_bounds(plane, clip_range)

    if bounds is None:
        if dtype is None:
            return plane
        lo, hi = float(plane.min()), float(plane.max())
        if hi <= lo:
            hi = lo + 1.0
    else:
        lo, hi = bounds
        # np.clip con estremi float promuove uint16 -> float64: riporto il dtype
        clipped = np.clip(plane, lo, hi)
        if dtype is None:
            return clipped.astype(plane.dtype, copy=False)
        plane = clipped

    scaled = (plane.astype(np.float32) - lo) / (hi - lo) * vmax
    if np.issubdtype(dtype, np.integer):
        scaled = np.rint(scaled)
    return np.clip(scaled, 0, vmax).astype(dtype)



# --------------------------------------------------------------------------- #
# Punto di ingresso: dai dati grezzi ai piani pronti per il merge
# --------------------------------------------------------------------------- #

def suggest_clip(raw, channel_map, lo_pct=1.0, hi_pct=99.5, channel_axis=0):
    """
    Stampa un blocco `clip:` pronto da incollare nel YAML.

    Serve a scegliere brightness/contrast la prima volta: guardi i percentili
    dei dati reali, li incolli come valori ASSOLUTI e da quel momento la
    mappatura e' fissa e confrontabile tra immagini. Usare direttamente
    mode='percentile' e' piu' comodo ma ricalcola gli estremi su ogni file,
    quindi due immagini non sono piu' confrontabili.
    """
    channel_map.validate_stack(raw.shape[channel_axis])
    raw = np.moveaxis(raw, channel_axis, 0)

    print("  clip:")
    for role in channel_map.names:
        plane = raw[channel_map[role]]
        lo, hi = np.percentile(plane, [lo_pct, hi_pct])
        print(f"    {role}: [{lo:.0f}, {hi:.0f}]"
              f"    # {channel_map.label(role)}: dati in "
              f"[{plane.min()}, {plane.max()}]")
