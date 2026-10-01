#!/usr/bin/env python3
"""Command line interface: render the panels described by a config and a manifest.

    chimera --config src/configs/default_config.yaml \\
            --manifest input_csv/csv_example.csv \\
            --output-dir out

(equivalent to `python -m chimera.cli ...`). Run `chimera --help` for all options.

Built to run on a laptop as well as inside a SLURM job array:

1. Sharding. With --num-shards N the manifest is split round-robin, and each
   task renders its own share. In a job array the values are read from
   SLURM_ARRAY_TASK_ID and SLURM_ARRAY_TASK_COUNT, so there is nothing to pass.
2. Exit codes. 0 if everything worked, 1 if some rows failed, 2 for a
   configuration error. Without them, SLURM would mark as COMPLETED a run
   that produced no movie at all.
3. Provenance. Every run writes config_used.yaml, manifest_used.csv and a
   JSON summary per shard next to the movies, so any output can be traced
   back to the exact settings that produced it.
"""

import argparse
import json
import logging
import os
import shutil
import sys
from pathlib import Path

logger = logging.getLogger("chimera.cli")

EXIT_OK = 0
EXIT_SOME_FAILED = 1
EXIT_FATAL = 2


# --------------------------------------------------------------------------- #
# Argomenti
# --------------------------------------------------------------------------- #

def build_parser():
    p = argparse.ArgumentParser(
        prog="chimera",
        description="Produce i pannelli di rotazione 3D da un manifest CSV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    required = p.add_argument_group("input")
    required.add_argument("--config", required=True, type=Path,
                          help="YAML di configurazione")
    required.add_argument("--manifest", required=True, type=Path,
                          help="CSV con una riga per crop")
    required.add_argument("--output-dir", required=True, type=Path,
                          help="cartella di destinazione")
    required.add_argument("--data-root", type=Path, default=None,
                          help="radice dei dati; sovrascrive paths.data_root "
                               "del config, utile per girare su una copia locale")

    selection = p.add_argument_group("selezione")
    selection.add_argument("--panels", default=None,
                           help="pannelli da produrre, separati da virgola; "
                                "sovrascrive la colonna del manifest")
    selection.add_argument("--rows", default=None,
                           help="righe da elaborare, es. '1,4' o '1-10'")
    selection.add_argument("--shard", type=int, default=None,
                           help="indice di questo task (0-based); "
                                "default da SLURM_ARRAY_TASK_ID")
    selection.add_argument("--num-shards", type=int, default=None,
                           help="numero di task; default da SLURM_ARRAY_TASK_COUNT")

    render = p.add_argument_group("rendering")
    render.add_argument("--spacing", default=None,
                        help="z,y,x in micron; default: dai metadati del raw")
    render.add_argument("--n-frames", type=int, default=36)
    render.add_argument("--axis", default="y", choices=["x", "y", "z"])
    render.add_argument("--label-size", type=int, default=30)
    render.add_argument("--scalebar-um", type=float, default=10.0)
    render.add_argument("--fps", type=int, default=6)
    render.add_argument("--format", default="mov", choices=["mov", "gif", "both"],
                        help="output file type: mov (default), gif, or both")
    render.add_argument("--quality", type=int, default=6,
                        help="0-10; su testo bianco 8-9 pulisce parecchio")
    render.add_argument("--linear-interpolation", action="store_true",
                        help="interpolazione lineare in Z; lasciala spenta "
                             "per label e maschere")
    render.add_argument("--cle-device", default=None,
                        help="nome del device OpenCL da usare")

    behaviour = p.add_argument_group("comportamento")
    behaviour.add_argument("--dry-run", action="store_true",
                           help="elenca cosa verrebbe prodotto e esce")
    behaviour.add_argument("--skip-existing", action="store_true",
                           help="salta i pannelli gia' prodotti")
    behaviour.add_argument("--on-error", default="continue",
                           choices=["continue", "raise"])
    behaviour.add_argument("--log-level", default="INFO",
                           choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    behaviour.add_argument("--no-provenance", action="store_true",
                           help="non copiare config e manifest negli output")
    return p


def parse_rows(spec):
    """'1,4' oppure '2-10' oppure '1,5-7' -> set di numeri di riga."""
    if not spec:
        return None
    rows = set()
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            start, _, end = chunk.partition("-")
            rows.update(range(int(start), int(end) + 1))
        else:
            rows.add(int(chunk))
    return rows


def parse_spacing(spec):
    if not spec:
        return None
    parts = [float(v) for v in spec.replace(" ", "").split(",")]
    if len(parts) != 3:
        raise ValueError(f"--spacing vuole tre valori z,y,x, ricevuto {spec!r}")
    return tuple(parts)


def resolve_sharding(args):
    """Indice e numero di shard, dagli argomenti o dall'ambiente SLURM."""
    shard = args.shard
    total = args.num_shards

    if shard is None and "SLURM_ARRAY_TASK_ID" in os.environ:
        # gli array SLURM sono di solito 1-based: porto a 0-based
        task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
        task_min = int(os.environ.get("SLURM_ARRAY_TASK_MIN", 0))
        shard = task_id - task_min
    if total is None and "SLURM_ARRAY_TASK_COUNT" in os.environ:
        total = int(os.environ["SLURM_ARRAY_TASK_COUNT"])

    if shard is None and total is None:
        return 0, 1
    if shard is None or total is None:
        raise ValueError(
            "--shard e --num-shards vanno indicati insieme "
            f"(shard={shard}, num_shards={total})"
        )
    if not 0 <= shard < total:
        raise ValueError(f"shard {shard} fuori intervallo per {total} task")
    return shard, total


def select_jobs(jobs, rows=None, shard=0, num_shards=1):
    """Filtra per numero di riga e poi prende la fetta di questo task.

    Lo sharding e' a giro (jobs[shard::num_shards]) e non a blocchi
    contigui: i crop hanno costi molto diversi fra loro, e a blocchi il task
    che becca le immagini grandi tiene occupato il nodo mentre gli altri
    hanno gia' finito.
    """
    if rows:
        jobs = [j for j in jobs if j.row in rows]
    return jobs[shard::num_shards]


# --------------------------------------------------------------------------- #
# Esecuzione
# --------------------------------------------------------------------------- #

def main(argv=None):
    args = build_parser().parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )

    # import qui dentro: un errore negli argomenti non deve aspettare il
    # caricamento di tutto lo stack scientifico
    from chimera.config import Config, ConfigError
    from chimera.manifest import read_manifest, dry_run, ManifestError
    from chimera.panels import cle_resampler
    from chimera.runner import run_manifest

    try:
        shard, num_shards = resolve_sharding(args)
        spacing = parse_spacing(args.spacing)
        rows = parse_rows(args.rows)
    except ValueError as err:
        logger.error("%s", err)
        return EXIT_FATAL

    tag = f"[shard {shard + 1}/{num_shards}] " if num_shards > 1 else ""

    try:
        cfg = Config.from_yaml(args.config)
        jobs_all = read_manifest(args.manifest, cfg=cfg, data_root=args.data_root)
    except (ConfigError, ManifestError) as err:
        logger.error("%s", err)
        return EXIT_FATAL

    jobs = select_jobs(jobs_all, rows=rows, shard=shard, num_shards=num_shards)
    logger.info("%s%d righe su %d da elaborare", tag, len(jobs), len(jobs_all))
    if not jobs:
        logger.warning("%snessuna riga selezionata", tag)
        return EXIT_OK

    if args.dry_run:
        ok = dry_run(jobs, cfg, data_root=args.data_root,
                     output_dir=args.output_dir)
        return EXIT_OK if ok else EXIT_SOME_FAILED

    if args.cle_device:
        import pyclesperanto_prototype as cle
        cle.select_device(args.cle_device)
    try:
        import pyclesperanto_prototype as cle
        logger.info("%sdevice OpenCL: %s", tag, cle.get_device())
    except Exception as err:       # noqa: BLE001 - solo informativo
        logger.warning("%sdevice OpenCL non interrogabile: %s", tag, err)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_provenance and shard == 0:
        # solo il primo task, altrimenti N copie della stessa cosa
        shutil.copy2(args.manifest, args.output_dir / "manifest_used.csv")

    result = run_manifest(
        jobs, cfg,
        output_dir=args.output_dir,
        spacing_um=spacing,
        data_root=args.data_root,
        panels=args.panels.split(",") if args.panels else None,
        resampler=cle_resampler(linear_interpolation=args.linear_interpolation),
        n_frames=args.n_frames,
        axis=args.axis,
        label_size=args.label_size,
        scalebar_length_um=args.scalebar_um,
        fps=args.fps,
        quality=args.quality,
        formats=("mov", "gif") if args.format == "both" else (args.format,),
        skip_existing=args.skip_existing,
        on_error=args.on_error,
        save_provenance=not args.no_provenance and shard == 0,
        verbose=True,
    )

    summary = {
        "shard": shard,
        "num_shards": num_shards,
        "rows": [job.row for job in jobs],
        "produced": [str(p) for p in result["produced"]],
        "failed": [{"row": r, "error": m} for r, m in result["failed"]],
    }
    # un file per shard: due task che scrivessero lo stesso nome si
    # sovrascriverebbero a vicenda
    summary_path = args.output_dir / f"summary_shard{shard:03d}.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    logger.info("%s%d filmati, %d righe fallite -> %s",
                tag, len(result["produced"]), len(result["failed"]), summary_path)

    return EXIT_SOME_FAILED if result["failed"] else EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
