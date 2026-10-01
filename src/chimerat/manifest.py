"""Read and validate the manifest CSV: one row = one crop to render.

The config says WHAT to render, the manifest says WHERE the data is.

Columns
-------
    raw_img_path     raw image file (required). Feeds the `acquisition`
                     section of the config; the voxel size is read from it.
    series_index     series inside the file, starting from 1 (required)
    <section name>   one column for each processed section of the config,
                     e.g. `mito_output`: path of that processed file.
                     Can be empty if no panel of that row needs it.
    z0,z1,y0,y1,x0,x1
                     crop, as numpy slicing (z1 excluded). Each cell can be
                     empty: empty start = from the first plane/pixel, empty
                     stop = up to the last one. Negative values are not
                     accepted (in numpy, :-1 would drop the last plane).
    panels           optional, panel names separated by ';'. Empty = all panels
    crop_id          optional, used in the output file names. Must be unique.
    notes            ignored by the code

Relative paths start from `data_root` (the --data-root option), or from the
folder you run the command in if data_root is not given.

Everything that can be checked without opening the images is checked when the
file is read: unknown columns, missing files, panels not in the config,
duplicate output names. All problems are reported together, so a typo in row
47 is found now and not after forty minutes of rendering. Crop bounds need
the image size, so they are checked when each image is opened.

The file is read with csv.DictReader instead of pandas on purpose: one empty
cell is enough for pandas to turn a whole column into floats (2.0 instead of 2).
"""

import csv
import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

RAW_SECTION = "acquisition"    # the section read from raw_img_path
REQUIRED_COLUMNS = {"raw_img_path", "series_index"}
CROP_COLUMNS = ("z0", "z1", "y0", "y1", "x0", "x1")
OPTIONAL_COLUMNS = set(CROP_COLUMNS) | {"panels", "crop_id", "notes"}


class ManifestError(ValueError):
    """Error in the manifest, with the row number."""


@dataclass(frozen=True)
class CropJob:
    """One row of the manifest, already converted and validated."""

    row: int                          # row number in the CSV, header excluded
    raw: Path                         # raw image file
    series_index: int                 # starts from 1, as written in the CSV
    processed: dict[str, Path] = field(default_factory=dict)   # section -> file
    # (start, stop); None inside the pair = "from the beginning" / "to the end"
    z: tuple[int | None, int | None] = (None, None)
    y: tuple[int | None, int | None] = (None, None)
    x: tuple[int | None, int | None] = (None, None)
    panels: tuple[str, ...] = ()      # empty = all panels of the config
    crop_id: str = ""
    notes: str = ""

    @property
    def series_index0(self) -> int:
        """0-based index, the one image readers expect."""
        return self.series_index - 1

    @property
    def name(self) -> str:
        """Name used for the output movies."""
        if self.crop_id:
            return self.crop_id
        return f"row{self.row:03d}_{self.raw.stem}_s{self.series_index}"

    def elab_paths(self, cfg=None, sections=None, data_root=None) -> dict[str, Path]:
        """Processed files needed for `sections` (all of them if None).

        cfg and data_root are accepted for compatibility with runner.py:
        paths are already complete when the manifest is read.
        """
        if sections is None:
            return dict(self.processed)
        missing = [s for s in sections if s not in self.processed]
        if missing:
            raise ManifestError(
                f"[row {self.row}] no file given for {missing}: fill the "
                f"column(s) {missing} in the manifest"
            )
        return {s: self.processed[s] for s in sections}

    def crop(self, stack, channel_axis=0):
        """Apply the crop to a (C, Z, Y, X) stack.

        Fails if the crop goes outside the image: numpy would silently cut it,
        or return an empty array.
        """
        spatial_axes = [axis for axis in range(stack.ndim) if axis != channel_axis]
        index = [slice(None)] * stack.ndim
        for axis, name, (start, stop) in zip(spatial_axes, "zyx", (self.z, self.y, self.x)):
            size = stack.shape[axis]
            start = 0 if start is None else start
            stop = size if stop is None else stop
            if start >= size or stop > size:
                raise ManifestError(
                    f"[row {self.row}] crop {name}={start}:{stop} goes outside "
                    f"the image ({name} size = {size})"
                )
            index[axis] = slice(start, stop)
        return stack[tuple(index)]


# --------------------------------------------------------------------------- #
# Reading
# --------------------------------------------------------------------------- #

def _int_or_none(value, row, column):
    value = (value or "").strip()
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        raise ManifestError(f"[row {row}] column '{column}': expected an integer, found {value!r}") from None


def _interval(cells, row, start_col, stop_col):
    """(start, stop) pair. An empty cell becomes None = no limit on that side."""
    start = _int_or_none(cells.get(start_col), row, start_col)
    stop = _int_or_none(cells.get(stop_col), row, stop_col)
    for column, value in ((start_col, start), (stop_col, stop)):
        if value is not None and value < 0:
            raise ManifestError(
                f"[row {row}] '{column}'={value}: negative values are not allowed. "
                f"To go up to the last plane, leave '{stop_col}' empty."
            )
    if start is not None and stop is not None and stop <= start:
        raise ManifestError(f"[row {row}] need {start_col} < {stop_col}, found {start}:{stop}")
    return (start, stop)


def _full_path(path_text, root):
    path = Path(path_text)
    return path if path.is_absolute() else root / path


def read_manifest(path, cfg, data_root=None, check_files=True):
    """Read the manifest and return a list of CropJob.

    Everything is validated before returning, so a typo in row 47 is found
    now and not after forty minutes of rendering.
    """
    path = Path(path)
    root = Path(data_root) if data_root else Path(".")
    processed_sections = [s for s in cfg.sections if s != RAW_SECTION]

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ManifestError(f"{path}: empty file")

        columns = {c.strip() for c in reader.fieldnames}
        valid = REQUIRED_COLUMNS | OPTIONAL_COLUMNS | set(processed_sections)
        missing = REQUIRED_COLUMNS - columns
        unknown = columns - valid
        if missing or unknown:
            raise ManifestError(
                f"{path}: missing columns {sorted(missing)}, unknown columns {sorted(unknown)}\n"
                f"  valid columns: {sorted(valid)}"
            )

        jobs, problems = [], []
        for row_number, cells in enumerate(reader, start=1):
            cells = {k.strip(): (v or "").strip() for k, v in cells.items() if k}
            try:
                jobs.append(_build_job(cells, row_number, root, cfg,
                                       processed_sections, check_files))
            except ManifestError as err:
                problems.append(str(err))

    if not jobs and not problems:
        raise ManifestError(f"{path}: no data rows")

    # Two rows with the same name would overwrite each other's movies.
    seen = {}
    for job in jobs:
        if job.name in seen:
            problems.append(f"[row {job.row}] name '{job.name}' already used by row {seen[job.name]}")
        seen[job.name] = job.row

    if problems:
        raise ManifestError(f"{path}: {len(problems)} problem(s)\n  " + "\n  ".join(problems))

    logger.info("Manifest %s: %d valid rows", path, len(jobs))
    return jobs


def _build_job(cells, row, root, cfg, processed_sections, check_files):
    if not cells.get("raw_img_path"):
        raise ManifestError(f"[row {row}] 'raw_img_path' is empty")
    raw = _full_path(cells["raw_img_path"], root)

    processed = {
        section: _full_path(cells[section], root)
        for section in processed_sections
        if cells.get(section)
    }

    if check_files:
        for label, file in [("raw_img_path", raw), *processed.items()]:
            if not file.is_file():
                raise ManifestError(f"[row {row}] {label}: file not found: {file}")

    series = _int_or_none(cells.get("series_index"), row, "series_index")
    if series is None or series < 1:
        raise ManifestError(f"[row {row}] 'series_index' must be 1 or more (first series = 1)")

    panels = tuple(p.strip() for p in cells.get("panels", "").split(";") if p.strip())
    unknown = [p for p in panels if p not in cfg.panels]
    if unknown:
        raise ManifestError(f"[row {row}] panels not in the config: {unknown}. Available: {sorted(cfg.panels)}")

    return CropJob(
        row=row,
        raw=raw,
        series_index=series,
        processed=processed,
        z=_interval(cells, row, "z0", "z1"),
        y=_interval(cells, row, "y0", "y1"),
        x=_interval(cells, row, "x0", "x1"),
        panels=panels,
        crop_id=cells.get("crop_id", ""),
        notes=cells.get("notes", ""),
    )


def dry_run(jobs, cfg, data_root=None, output_dir="out"):
    """Print what would be produced, without rendering anything."""
    output_dir = Path(output_dir)
    for job in jobs:
        panels = job.panels or tuple(sorted(cfg.panels))
        crop = ", ".join(
            f"{axis}={'' if start is None else start}:{'' if stop is None else stop}"
            for axis, (start, stop) in (("z", job.z), ("y", job.y), ("x", job.x))
        )
        print(f"[row {job.row}] {job.name}")
        print(f"    raw        {job.raw}  (series {job.series_index})")
        for section, file in job.processed.items():
            print(f"    {section:10s} {file}")
        print(f"    crop       {crop}")
        for panel in panels:
            print(f"    -> {output_dir / f'{job.name}_{panel}.mov'}")
        print()
    print(f"{len(jobs)} rows")
    return True
