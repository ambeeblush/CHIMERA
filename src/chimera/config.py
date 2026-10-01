"""Load and validate the CHIMERA configuration file (YAML).

Structure of the YAML file:

    config_version: 1
    acquisition:         # the raw image (manifest column raw_img_path)
      channels: ...      #   role -> channel index in the image file
      channel_labels: ...
      colormaps: ...
      clip: ...
    <section>:           # any other name, e.g. mito_output: one processed file,
      ...                #   same keys, path in the manifest column of the same name
    panels:
      <panel name>:
        tiles: ...       # each tile lists (section, role) pairs: one = single channel,
                         #   more = merge
        gutter_px, scalebar_on, filename_on   # layout, optional

Every top-level key other than config_version and panels is read as a
channel section, so adding a processed output only requires editing the YAML.
The config says WHAT to render, the manifest says WHERE the data is.

Validation is strict: unknown keys are an error (with a "did you mean"
suggestion), and every panel is checked against the sections and roles that
actually exist before anything is rendered.

Design choice: scientific parameters (channels, colormaps, clip ranges) have
no hidden defaults and must be written in the YAML. Only layout options and
labels have defaults, because their effect is visible on the movie itself.
"""

from __future__ import annotations

import difflib
import logging
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from chimera.colormaps import COLORMAP_NAMES, ColormapError
from chimera.colormaps import normalize as normalize_colormap

logger = logging.getLogger(__name__)

CONFIG_VERSION = 1

# Top-level keys that are NOT channel sections.
RESERVED_KEYS = {"config_version", "panels"}


class ConfigError(ValueError):
    """Error in the configuration file, with a readable message."""


def _check_keys(data: Any, allowed: set, required: set, where: str) -> None:
    """Check that `data` is a dict with only allowed keys and all required ones.

    Unknown keys are an error, not a warning: a typo like 'colormap' instead
    of 'colormaps' would otherwise be ignored silently.
    """
    if not isinstance(data, dict):
        raise ConfigError(f"[{where}] expected a block of keys, found {type(data).__name__}")

    unknown = set(data) - allowed
    if unknown:
        msg = f"[{where}] unknown keys: {sorted(unknown)}\n  valid keys: {sorted(allowed)}"
        for key in sorted(unknown):
            close = difflib.get_close_matches(key, allowed, n=1, cutoff=0.6)
            if close:
                msg += f"\n  '{key}' -> did you mean '{close[0]}'?"
        raise ConfigError(msg)

    missing = required - set(data)
    if missing:
        raise ConfigError(f"[{where}] missing keys: {sorted(missing)}")


# --------------------------------------------------------------------------- #
# Channels
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class ClipRange:
    """Intensity limits for one channel, like Brightness/Contrast in ImageJ.

    lo and hi are used both to clip and to rescale the channel.
    mode='absolute'   -> lo/hi are intensity values
    mode='percentile' -> lo/hi are percentiles (0-100) computed on the data
    """

    lo: float
    hi: float
    mode: str = "absolute"

    def __post_init__(self) -> None:
        if self.mode not in ("absolute", "percentile"):
            raise ConfigError(f"[clip] invalid mode '{self.mode}': use 'absolute' or 'percentile'")
        for name, value in (("lo", self.lo), ("hi", self.hi)):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ConfigError(f"[clip] '{name}' must be a number, found {value!r}")
        if self.lo >= self.hi:
            raise ConfigError(f"[clip] lo ({self.lo}) must be smaller than hi ({self.hi})")
        if self.mode == "percentile" and not (0 <= self.lo < self.hi <= 100):
            raise ConfigError(f"[clip] percentiles must be between 0 and 100: lo={self.lo}, hi={self.hi}")

    @classmethod
    def from_yaml_value(cls, value: Any, where: str) -> ClipRange | None:
        """Accept [lo, hi], {lo, hi, mode} or null (= no clipping)."""
        if value is None:
            return None
        if isinstance(value, (list, tuple)):
            if len(value) != 2:
                raise ConfigError(f"[{where}] clip list needs exactly [lo, hi]")
            return cls(lo=value[0], hi=value[1])
        _check_keys(value, allowed={"lo", "hi", "mode"}, required={"lo", "hi"}, where=where)
        return cls(**value)

    def to_yaml_value(self) -> Any:
        if self.mode == "absolute":
            return [self.lo, self.hi]
        return {"lo": self.lo, "hi": self.hi, "mode": self.mode}


@dataclass(frozen=True)
class ChannelMap(Mapping):
    """Role -> channel index, plus label, colormap and clip range of each role.

    Behaves like a read-only dict: channel_map["mito"] returns the index.
    Role names are case-insensitive.
    """

    mapping: dict[str, int]
    labels: dict[str, str] = field(default_factory=dict)
    colormaps: dict[str, Any] = field(default_factory=dict)
    clip: dict[str, ClipRange | None] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Frozen dataclass: object.__setattr__ is the standard way to
        # normalize a field inside __post_init__.
        for name in ("mapping", "labels", "colormaps", "clip"):
            lowered = {k.lower(): v for k, v in getattr(self, name).items()}
            object.__setattr__(self, name, lowered)

        if not self.mapping:
            raise ConfigError("[channels] no channels defined")

        for role, index in self.mapping.items():
            if isinstance(index, bool) or not isinstance(index, int) or index < 0:
                raise ConfigError(f"[channels] '{role}': index must be an integer >= 0, found {index!r}")
        if len(set(self.mapping.values())) != len(self.mapping):
            raise ConfigError(f"[channels] two roles share the same index: {self.mapping}")

        # Labels, colormaps and clip ranges may only refer to existing roles.
        for block in ("labels", "colormaps", "clip"):
            orphans = set(getattr(self, block)) - set(self.mapping)
            if orphans:
                raise ConfigError(
                    f"[{block}] roles not defined in 'channels': {sorted(orphans)}\n"
                    f"  defined roles: {sorted(self.mapping)}"
                )

        # Colormaps are required for every channel, because every channel
        # has to be drawn with some color ('grays' for grayscale).
        missing = set(self.mapping) - set(self.colormaps)
        if missing:
            raise ConfigError(f"[colormaps] missing for: {sorted(missing)}")
        normalized = {}
        for role, value in self.colormaps.items():
            try:
                normalized[role] = normalize_colormap(value)
            except (ColormapError, TypeError) as err:
                raise ConfigError(
                    f"[colormaps] '{role}': {err}\n  valid names: {sorted(COLORMAP_NAMES)}"
                ) from None
        object.__setattr__(self, "colormaps", normalized)

        # The clip block is optional, but if present it must list every
        # channel. Write `null` for a channel you don't want to clip.
        if self.clip:
            missing = set(self.mapping) - set(self.clip)
            if missing:
                raise ConfigError(
                    f"[clip] missing for: {sorted(missing)} (use null for no clipping)"
                )

    # -- dict-like access (required by Mapping) ---------------------------- #

    def __getitem__(self, role: str) -> int:
        try:
            return self.mapping[role.lower()]
        except KeyError:
            raise ConfigError(
                f"Channel '{role}' not defined. Available: {sorted(self.mapping)}"
            ) from None

    def __iter__(self) -> Iterator[str]:
        """Iterate roles in index order."""
        return iter(sorted(self.mapping, key=self.mapping.get))

    def __len__(self) -> int:
        return len(self.mapping)

    # -- per-role information --------------------------------------------- #

    @property
    def names(self) -> list[str]:
        return list(self)

    @property
    def n_channels(self) -> int:
        """Minimum number of channels the image file must have."""
        return max(self.mapping.values()) + 1

    def label(self, role: str) -> str:
        """Label shown on the movie (falls back to the capitalized role name)."""
        self[role]  # raises a clear error if the role doesn't exist
        return self.labels.get(role.lower(), role.capitalize())

    def colormap_for(self, role: str) -> Any:
        self[role]
        return self.colormaps[role.lower()]

    def clip_for(self, role: str) -> ClipRange | None:
        self[role]
        return self.clip.get(role.lower())

    def validate_stack(self, n_channels_found: int, filename: str = "") -> None:
        """Fail early if the image has fewer channels than the config expects."""
        if n_channels_found < self.n_channels:
            where = f" in '{filename}'" if filename else ""
            raise ConfigError(
                f"The config expects {self.n_channels} channels {self.names} "
                f"but {n_channels_found} were found{where}."
            )

    @classmethod
    def from_yaml_section(cls, data: Any, where: str) -> ChannelMap:
        _check_keys(
            data,
            allowed={"channels", "channel_labels", "colormaps", "clip"},
            required={"channels", "colormaps"},
            where=where,
        )
        clip = {
            role: ClipRange.from_yaml_value(value, f"{where}.clip.{role}")
            for role, value in (data.get("clip") or {}).items()
        }
        return cls(
            mapping=data["channels"],
            labels=data.get("channel_labels") or {},
            colormaps=data["colormaps"],
            clip=clip,
        )

    def to_yaml_section(self) -> dict:
        section = {
            "channels": dict(self.mapping),
            "channel_labels": dict(self.labels),
            "colormaps": {r: list(c) if isinstance(c, tuple) else c
                          for r, c in self.colormaps.items()},
        }
        if self.clip:
            section["clip"] = {r: c.to_yaml_value() if c else None
                               for r, c in self.clip.items()}
        return section


@dataclass(frozen=True)
class ChannelSection:
    """A channel section of the YAML (e.g. 'acquisition').

    Only a wrapper around ChannelMap, kept so that the rest of the package
    can keep writing `section.channels`.
    """

    channels: ChannelMap


# --------------------------------------------------------------------------- #
# Panels
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class ChannelRef:
    """Points to one channel: which section, which role."""

    section: str
    role: str


@dataclass(frozen=True)
class TileSpec:
    """One tile of a panel. One channel = single tile, more channels = merge."""

    channels: tuple[ChannelRef, ...]
    title: str | None = None

    @classmethod
    def from_yaml_value(cls, data: Any, where: str) -> TileSpec:
        _check_keys(data, allowed={"channels", "title"}, required={"channels"}, where=where)
        if not isinstance(data["channels"], list) or not data["channels"]:
            raise ConfigError(f"[{where}] 'channels' must be a non-empty list")

        refs = []
        for i, ref in enumerate(data["channels"]):
            _check_keys(ref, allowed={"section", "role"}, required={"section", "role"},
                        where=f"{where}.channels[{i}]")
            refs.append(ChannelRef(section=ref["section"], role=ref["role"]))

        if len(set(refs)) != len(refs):
            raise ConfigError(f"[{where}] the same channel appears twice in one tile")
        return cls(channels=tuple(refs), title=data.get("title"))


@dataclass(frozen=True)
class PanelSpec:
    """A panel: a row of tiles, plus where to draw scalebar and filename.

    scalebar_on / filename_on: 'first', 'last', 'all', null or a tile index.
    Layout options have defaults: unlike scientific parameters, you can see
    them by looking at the movie.
    """

    tiles: tuple[TileSpec, ...]
    gutter_px: int = 8
    scalebar_on: Any = "last"
    filename_on: Any = "first"

    def __post_init__(self) -> None:
        if isinstance(self.gutter_px, bool) or not isinstance(self.gutter_px, int) or self.gutter_px < 0:
            raise ConfigError(f"[panels] gutter_px must be an integer >= 0, found {self.gutter_px!r}")
        for name in ("scalebar_on", "filename_on"):
            self.tile_indices(getattr(self, name), check_name=name)

    def tile_indices(self, anchor: Any, check_name: str = "anchor") -> list[int]:
        """Indices of the tiles where an annotation goes."""
        n = len(self.tiles)
        if anchor is None:
            return []
        if anchor == "first":
            return [0]
        if anchor == "last":
            return [n - 1]
        if anchor == "all":
            return list(range(n))
        if isinstance(anchor, int) and not isinstance(anchor, bool) and 0 <= anchor < n:
            return [anchor]
        raise ConfigError(
            f"[panels] {check_name}={anchor!r} is not valid: use 'first', 'last', "
            f"'all', null or a tile index between 0 and {n - 1}"
        )

    @classmethod
    def from_yaml_value(cls, data: Any, where: str) -> PanelSpec:
        _check_keys(
            data,
            allowed={"tiles", "gutter_px", "scalebar_on", "filename_on"},
            required={"tiles"},
            where=where,
        )
        if not isinstance(data["tiles"], list) or not data["tiles"]:
            raise ConfigError(f"[{where}] 'tiles' must be a non-empty list")
        tiles = tuple(
            TileSpec.from_yaml_value(tile, f"{where}.tiles[{i}]")
            for i, tile in enumerate(data["tiles"])
        )
        options = {k: v for k, v in data.items() if k != "tiles"}
        return cls(tiles=tiles, **options)


# --------------------------------------------------------------------------- #
# Main config
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Config:
    """The whole configuration file.

    Channel sections are read from the YAML, so cfg.acquisition and
    cfg.section("acquisition") return the same ChannelSection.
    """

    sections: dict[str, ChannelSection]
    panels: dict[str, PanelSpec]
    config_version: int = CONFIG_VERSION
    source: Path | None = None   # file the config was loaded from

    def __post_init__(self) -> None:
        # Cross-check: every channel used in a panel must exist.
        for panel_name, panel in self.panels.items():
            for t, tile in enumerate(panel.tiles):
                for c, ref in enumerate(tile.channels):
                    where = f"panels.{panel_name}.tiles[{t}].channels[{c}]"
                    if ref.section not in self.sections:
                        raise ConfigError(
                            f"[{where}] section '{ref.section}' does not exist.\n"
                            f"  sections: {sorted(self.sections)}"
                        )
                    channel_map = self.sections[ref.section].channels
                    if ref.role.lower() not in channel_map.mapping:
                        raise ConfigError(
                            f"[{where}] role '{ref.role}' not defined in '{ref.section}'.\n"
                            f"  roles: {channel_map.names}"
                        )

    # -- access ------------------------------------------------------------ #

    def section(self, name: str) -> ChannelSection:
        if name not in self.sections:
            raise ConfigError(f"Section '{name}' not defined. Available: {sorted(self.sections)}")
        return self.sections[name]

    def __getattr__(self, name: str) -> ChannelSection:
        # Called only when normal attribute lookup fails: lets the rest of
        # the package keep writing cfg.acquisition or getattr(cfg, section).
        sections = self.__dict__.get("sections", {})
        if name in sections:
            return sections[name]
        raise AttributeError(name)

    def panel(self, name: str) -> PanelSpec:
        if name not in self.panels:
            raise ConfigError(f"Panel '{name}' not defined. Available: {sorted(self.panels)}")
        return self.panels[name]

    # -- loading ----------------------------------------------------------- #

    @classmethod
    def from_dict(cls, raw: dict, source: Path | None = None) -> Config:
        if not isinstance(raw, dict):
            raise ConfigError("The config file must contain a block of keys")

        section_names = [key for key in raw if key not in RESERVED_KEYS]
        if not section_names:
            raise ConfigError("No channel section found (e.g. 'acquisition:')")

        sections = {
            name: ChannelSection(ChannelMap.from_yaml_section(raw[name], where=name))
            for name in section_names
        }

        panels_raw = raw.get("panels") or {}
        if not isinstance(panels_raw, dict):
            raise ConfigError("'panels' must be a block: panel name -> panel")
        panels = {
            name: PanelSpec.from_yaml_value(spec, f"panels.{name}")
            for name, spec in panels_raw.items()
        }

        return cls(
            sections=sections,
            panels=panels,
            config_version=raw.get("config_version", CONFIG_VERSION),
            source=source,
        )

    @classmethod
    def from_yaml(cls, path: Path | str) -> Config:
        path = Path(path)
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        logger.info("Config loaded from %s (sections: %s)",
                    path, [k for k in raw if k not in RESERVED_KEYS])
        return cls.from_dict(raw, source=path)

    # -- saving (provenance) ----------------------------------------------- #

    def to_dict(self) -> dict:
        out: dict[str, Any] = {"config_version": self.config_version}
        for name, section in self.sections.items():
            out[name] = section.channels.to_yaml_section()

        out["panels"] = {}
        for name, panel in self.panels.items():
            out["panels"][name] = {
                "tiles": [
                    {
                        **({"title": tile.title} if tile.title is not None else {}),
                        "channels": [{"section": r.section, "role": r.role}
                                     for r in tile.channels],
                    }
                    for tile in panel.tiles
                ],
                "gutter_px": panel.gutter_px,
                "scalebar_on": panel.scalebar_on,
                "filename_on": panel.filename_on,
            }

        return out

    def to_yaml(self, path: Path | str) -> Path:
        """Write the config actually used, next to the output movies."""
        path = Path(path)
        path.write_text(
            yaml.safe_dump(self.to_dict(), sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        return path
