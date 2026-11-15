"""Empirical reference distributions for percentile-based render scores."""

from __future__ import annotations

import json
import math
from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class ReferenceDistribution:
    """A sorted empirical distribution of raw values from real videos."""

    probe: str
    higher_is_better: bool
    samples: tuple[float, ...]
    two_sided: bool = False

    def __post_init__(self) -> None:
        if not self.samples:
            raise ValueError("reference samples must not be empty")
        normalized = tuple(sorted(float(value) for value in self.samples))
        if any(not math.isfinite(value) for value in normalized):
            raise ValueError("reference samples must be finite")
        object.__setattr__(self, "samples", normalized)

    @classmethod
    def from_samples(
        cls,
        probe: str,
        higher_is_better: bool,
        samples: Iterable[float],
        two_sided: bool = False,
    ) -> "ReferenceDistribution":
        """Build a validated reference distribution from raw measurements."""
        return cls(probe, higher_is_better, tuple(samples), two_sided)

    def percentile(self, value: float) -> float:
        """Return the empirical midrank percentile of a raw value.

        A two-sided axis asks how far the value sits from the middle of real
        footage rather than how far above or below it is: a render that is
        crisper than every real clip is as unlike real footage as one that is
        blurrier.
        """
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError("percentile value must be finite")
        samples = self.samples
        if self.two_sided:
            center = samples[len(samples) // 2]
            numeric = abs(numeric - center)
            samples = tuple(sorted(abs(sample - center) for sample in samples))
        left = bisect_left(samples, numeric)
        right = bisect_right(samples, numeric)
        return 100.0 * (left + 0.5 * (right - left)) / len(samples)

    def to_dict(self) -> dict[str, Any]:
        """Serialize this distribution to its versioned JSON shape."""
        return {
            "schema_version": 1,
            "probe": self.probe,
            "higher_is_better": self.higher_is_better,
            "samples": list(self.samples),
            "two_sided": self.two_sided,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ReferenceDistribution":
        """Deserialize and validate a versioned distribution object."""
        if data.get("schema_version") != 1:
            raise ValueError("unsupported reference distribution schema version")
        if not isinstance(data.get("probe"), str) or not data["probe"]:
            raise ValueError("probe must be a non-empty string")
        if type(data.get("higher_is_better")) is not bool:
            raise ValueError("higher_is_better must be a boolean")
        samples = data.get("samples")
        if not isinstance(samples, list):
            raise ValueError("samples must be a list")
        two_sided = data.get("two_sided", False)
        if type(two_sided) is not bool:
            raise ValueError("two_sided must be a boolean")
        return cls.from_samples(
            data["probe"], data["higher_is_better"], samples, two_sided
        )

    @classmethod
    def load(cls, path: Path) -> "ReferenceDistribution":
        """Load a reference distribution from a JSON file."""
        with Path(path).open("r", encoding="utf-8") as stream:
            data = json.load(stream)
        if not isinstance(data, dict):
            raise ValueError("reference distribution JSON must contain one object")
        return cls.from_dict(data)

    def save(self, path: Path) -> None:
        """Save this reference distribution as deterministic JSON."""
        with Path(path).open("w", encoding="utf-8") as stream:
            json.dump(self.to_dict(), stream, indent=2, sort_keys=True)
            stream.write("\n")
