"""Run the manifest: load, check, crop and render each row.

For every row, run_job():

    1. loads only the sections the requested panels need: the raw image
       with BioIO, at the right series, and each processed file as a
       (Z, C, Y, X) TIFF
    2. checks that every stack has the channels the config expects, and
       that all stacks share the same (Z, Y, X) shape
    3. applies the SAME crop to every stack
    4. reads the voxel size from the raw file metadata (or uses the value
       passed by the caller)
    5. renders and saves one movie per panel

The shape check happens BEFORE cropping, on purpose. The crop uses the same
coordinates on raw and processed data, so they must share the same reference
frame. A processed file made on an already cropped field would otherwise give
misaligned crops with no error: plausible-looking but wrong figures.

run_manifest() runs every row and, by default, carries on when one fails,
listing the failures at the end: in a long batch, 58 panels out of 60 plus
a list of what went wrong beat losing everything at row 47. It also saves
config_used.yaml next to the movies, so every run can be reproduced.
"""

import logging
from pathlib import Path

import numpy as np

from chimerat.manifest import ManifestError
from chimerat.panels import normalize_formats, render_and_save

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Lettura delle immagini
# --------------------------------------------------------------------------- #

def open_nd2_image(file_path, series_index0):
    """
    Apre un file raw con BioImage e restituisce (stack CZYX, spacing_um).

    spacing_um esce come (z, y, x) in micron, lo stesso ordine usato ovunque
    nel progetto.

    Nota: physical_pixel_sizes viene letto DOPO set_scene. In un file
    multi-serie le scene possono avere calibrazioni diverse, e leggerlo prima
    darebbe sempre quella della scena 0.
    """
    from bioio import BioImage

    img = BioImage(file_path)
    img.set_scene(series_index0)
    sizes = img.physical_pixel_sizes
    #stack = np.squeeze(img.get_image_data("CZYX"))
    img1 = img.get_image_data("CZYX")
    return img1, (sizes.Z, sizes.Y, sizes.X)


def read_spacing(file_path, series_index0):
    """Solo la calibrazione, senza caricare i pixel.

    Serve quando i pannelli richiesti non usano la sezione 'acquisition' ma
    lo spacing va comunque letto dal raw.
    """
    from bioio import BioImage

    img = BioImage(file_path)
    img.set_scene(series_index0)
    sizes = img.physical_pixel_sizes
    return (sizes.Z, sizes.Y, sizes.X)


def validate_spacing(spacing, source=""):
    """Controlla che la calibrazione sia utilizzabile.

    I metadati mancanti arrivano come None e, senza questo controllo,
    finirebbero in una divisione dentro make_isotropic con un TypeError
    lontano dalla causa.
    """
    where = f" ({source})" if source else ""
    if spacing is None:
        raise ManifestError(f"spacing non disponibile{where}")
    if len(spacing) != 3:
        raise ManifestError(
            f"spacing{where}: attesi 3 valori (z, y, x), trovato {spacing!r}"
        )
    if any(v is None for v in spacing):
        raise ManifestError(
            f"spacing{where}: il file non dichiara la calibrazione "
            f"{tuple(spacing)}.\n"
            f"    Passa spacing_um=(z, y, x) esplicitamente a run_manifest()."
        )
    if any(v <= 0 for v in spacing):
        raise ManifestError(f"spacing{where}: valori non positivi {tuple(spacing)}")
    return tuple(float(v) for v in spacing)


def default_raw_reader(path, series_index0):
    """Lettore raw predefinito: BioImage. Vedi open_nd2_image."""
    return open_nd2_image(path, series_index0)


def open_tif_image(file_path, expected_channels=None):
    """
    Carica un file elaborato salvato come (Z, C, Y, X) e lo porta a (C, Z, Y, X).

    Il transpose (1, 0, 2, 3) e' la sua stessa inversa: applicato a un file
    gia' CZYX lo scambia al contrario, e il risultato e' uno stack in cui i
    "canali" sono piani z. Con expected_channels il caso viene intercettato;
    se pero' C e Z hanno la stessa dimensione nessun controllo puo'
    accorgersene, quindi la convenzione ZCYX va rispettata a monte.
    """
    import tifffile as tiff

    file_path = Path(file_path)
    if not file_path.exists():
        raise ManifestError(f"file elaborato mancante: {file_path}")

    img = tiff.imread(file_path)
    if img.ndim != 4:
        raise ManifestError(
            f"{file_path.name}: attese 4 dimensioni (Z, C, Y, X), "
            f"trovato {img.shape}"
        )

    img = np.transpose(img, (1, 0, 2, 3))

    if expected_channels is not None and img.shape[0] != expected_channels:
        hint = ""
        if img.shape[1] == expected_channels:
            hint = (f"\n    Le dimensioni tornano senza il transpose: il file "
                    f"sembra gia' in ordine (C, Z, Y, X).")
        raise ManifestError(
            f"{file_path.name}: dopo il transpose i canali sono "
            f"{img.shape[0]}, il config ne prevede {expected_channels}.{hint}"
        )

    return img



# --------------------------------------------------------------------------- #
# Coerenza geometrica
# --------------------------------------------------------------------------- #

def check_same_geometry(stacks, channel_axis=0):
    """
    Verifica che tutti gli stack condividano (Z, Y, X).

    Le coordinate del manifest sono uniche per tutte le sezioni: se un file
    elaborato fosse stato prodotto su un campo gia' ritagliato, lo stesso
    crop cadrebbe su regioni diverse e otterresti un merge sfalsato, con
    l'aria di funzionare.
    """
    geometry = {}
    for section, stack in stacks.items():
        zyx = tuple(s for i, s in enumerate(stack.shape) if i != channel_axis)
        geometry.setdefault(zyx, []).append(section)

    if len(geometry) > 1:
        detail = "\n    ".join(
            f"{zyx}: {', '.join(sections)}" for zyx, sections in geometry.items()
        )
        raise ManifestError(
            "Gli stack non condividono la geometria (Z, Y, X), quindi lo "
            "stesso crop cadrebbe su regioni diverse:\n    " + detail
        )
    return next(iter(geometry))


def check_channels(stacks, cfg, channel_axis=0):
    """Confronta il numero di canali di ogni stack con la sua sezione."""
    for section, stack in stacks.items():
        channel_map = getattr(cfg, section).channels
        channel_map.validate_stack(stack.shape[channel_axis], filename=section)


# --------------------------------------------------------------------------- #
# Esecuzione
# --------------------------------------------------------------------------- #

def run_job(
    job,
    cfg,
    output_dir,
    spacing_um=None,
    data_root=None,
    panels=None,
    raw_reader=None,
    channel_axis=0,
    skip_existing=False,
    verbose=True,
    **render_kwargs,
):
    """
    Esegue una riga del manifest: carica, croppa, renderizza, salva.

    Parameters
    ----------
    job : CropJob
    cfg : Config
    spacing_um : tuple, callable or None
        None (default): la calibrazione viene letta dai metadati del file
        raw, che e' l'unico posto dove non puo' divergere dai dati. Passa una
        tupla (z, y, x) solo per sovrascriverla, o una callable
        (job, raw_path) -> tupla per casi particolari.
    output_dir : str or Path
    panels : list of str or None
        Sovrascrive quanto indicato nella riga. None usa job.panels, e se
        anche quello e' vuoto tutti i pannelli del config.
    raw_reader : callable or None
        (path, series_index0) -> (C, Z, Y, X). None usa default_raw_reader.
    skip_existing : bool
        Salta i pannelli il cui .mov esiste gia': serve a riprendere una
        batch interrotta senza rifare tutto.
    **render_kwargs
        Inoltrati a render_and_save (resampler, n_frames, fps, quality...).

    Returns
    -------
    list of Path : i filmati prodotti.
    """
    output_dir = Path(output_dir)
    raw_reader = raw_reader or default_raw_reader

    wanted = list(panels or job.panels or sorted(cfg.panels))
    unknown = [p for p in wanted if p not in cfg.panels]
    if unknown:
        raise ManifestError(
            f"[riga {job.row}] pannelli non definiti: {unknown}\n"
            f"  disponibili: {sorted(cfg.panels)}"
        )

    if skip_existing:
        # A panel is done only if ALL the requested formats exist.
        quality = render_kwargs.get("quality", 6)
        formats = normalize_formats(render_kwargs.get("formats", "mov"))
        todo = [p for p in wanted
                if not all((output_dir / f"{job.name}_{p}_q{quality}.{ext}").exists()
                           for ext in formats)]
        if not todo:
            if verbose:
                print(f"[riga {job.row}] {job.name}: gia' fatto, salto")
            return []
        wanted = todo

    # -- quali sezioni servono davvero a questi pannelli -------------------- #
    sections = {
        ref.section
        for panel in wanted
        for tile in cfg.panel(panel).tiles
        for ref in tile.channels
    }

    if verbose:
        print(f"[riga {job.row}] {job.name}: pannelli {wanted}, "
              f"sezioni {sorted(sections)}")

    # -- caricamento -------------------------------------------------------- #
    stacks = {}
    spacing_from_file = None
    if "acquisition" in sections:
        result = raw_reader(job.raw, job.series_index0)
        # Un lettore puo' restituire (stack, spacing) oppure il solo stack.
        if isinstance(result, tuple) and len(result) == 2:
            stacks["acquisition"], spacing_from_file = result
        else:
            stacks["acquisition"] = result
        sections.discard("acquisition")

    elab = job.elab_paths(cfg, sections=sorted(sections), data_root=data_root)
    for section, path in elab.items():
        expected = getattr(cfg, section).channels.n_channels
        stacks[section] = open_tif_image(path, expected_channels=expected)

    if verbose:
        for section, stack in stacks.items():
            print(f"    {section:20s} {stack.shape} {stack.dtype}")

    # -- verifiche PRIMA di tagliare ---------------------------------------- #
    check_channels(stacks, cfg, channel_axis=channel_axis)
    check_same_geometry(stacks, channel_axis=channel_axis)

    # -- crop identico su tutti gli stack ----------------------------------- #
    stacks = {s: job.crop(stack, channel_axis=channel_axis)
              for s, stack in stacks.items()}
    if verbose:
        shape = next(iter(stacks.values())).shape
        print(f"    crop -> {shape}")

    # -- calibrazione ------------------------------------------------------- #
    if spacing_um is None:
        if spacing_from_file is None:
            # nessun pannello usa il raw: leggo solo i metadati, non i pixel
            spacing_from_file = read_spacing(job.raw, job.series_index0)
        spacing = validate_spacing(spacing_from_file, f"{job.raw.name}, metadati")
        origin = "metadati"
    elif callable(spacing_um):
        spacing = validate_spacing(spacing_um(job, job.raw), "callable")
        origin = "callable"
    else:
        spacing = validate_spacing(tuple(spacing_um), "argomento")
        origin = "argomento"

    if verbose:
        print(f"    spacing (z, y, x) = {spacing} um  [{origin}]")

    # -- un filmato per pannello -------------------------------------------- #
    produced = []
    for panel in wanted:
        _, files = render_and_save(
            stacks, cfg, panel, spacing,
            output_dir=output_dir,
            filename=job.name,
            channel_axis=channel_axis,
            **render_kwargs,
        )
        produced.extend(files)

    return produced


def run_manifest(
    jobs,
    cfg,
    output_dir,
    spacing_um=None,
    data_root=None,
    on_error="continue",
    save_provenance=True,
    verbose=True,
    **job_kwargs,
):
    """
    Esegue tutte le righe del manifest.

    Parameters
    ----------
    spacing_um : tuple, callable or None
        None (default): letta dai metadati di ogni file raw, quindi immagini
        con calibrazioni diverse nella stessa batch vengono trattate bene.
        Una tupla la impone uguale per tutte le righe.
    on_error : {'continue', 'raise'}
        'continue' (default) tira avanti e riepiloga i fallimenti alla fine:
        in una batch lunga e' meglio avere 58 pannelli su 60 piu' l'elenco
        di cosa e' andato storto, che perdere tutto alla riga 47. La
        validazione del manifest e' gia' avvenuta in lettura, quindi qui
        restano solo errori di runtime (file corrotto, memoria, geometria).
    save_provenance : bool
        Scrive la config effettiva in output_dir/config_used.yaml. E' quel
        file, non il path al default.yaml, che rende la run ricostruibile.

    Returns
    -------
    dict : {'produced': [...], 'failed': [(row, messaggio), ...]}
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if save_provenance:
        cfg.to_yaml(output_dir / "config_used.yaml")

    produced, failed = [], []
    for job in jobs:
        try:
            produced.extend(
                run_job(job, cfg, output_dir, spacing_um=spacing_um,
                        data_root=data_root, verbose=verbose, **job_kwargs)
            )
        except Exception as err:
            if on_error == "raise":
                raise
            logger.exception("Riga %d (%s) fallita", job.row, job.name)
            failed.append((job.row, f"{type(err).__name__}: {err}"))
            if verbose:
                print(f"    FALLITA: {type(err).__name__}: {err}")

    if verbose:
        print(f"\n{len(produced)} filmati prodotti, {len(failed)} righe fallite")
        for row, message in failed:
            print(f"  riga {row}: {message.splitlines()[0]}")

    return {"produced": produced, "failed": failed}
