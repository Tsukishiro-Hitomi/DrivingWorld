"""Dataset manifests and validation for model evaluation."""

from .event_judge import DatasetRecord, load_manifest, validate_record

__all__ = ["DatasetRecord", "load_manifest", "validate_record"]
