"""Persistent, reusable rollout artifacts for checkpoint analysis.

The expensive operation in zero-shot analysis is model inference.  A single rollout
already contains everything needed for VRMSE, videos, prediction distributions and
flow metrics, so this module stores that rollout once and lets all downstream
analyses read it without loading the model again.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import re
import shutil
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Optional

import numpy as np
from the_well.data.datasets import WellMetadata

CACHE_SCHEMA_VERSION = 2
DEFAULT_CACHE_DIR = pathlib.Path("./_analysis_cache")


def _jsonable(value: Any) -> Any:
    """Convert common config/path/numpy values to deterministic JSON data."""
    if isinstance(value, pathlib.Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in sorted(value.items(), key=lambda x: str(x[0]))}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def _slug(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    return value[:100] or "unknown"


def cache_fingerprint(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(_jsonable(payload), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()[:16]


@dataclass(frozen=True)
class RolloutCacheKey:
    """Identity of one cached model/run/dataset/split evaluation.

    Directory layout is explicit so two wandb projects, two datasets, or two
    models can never share a folder:

        {root}/{entity}/{project}/{model}/{run_name}__{run_id}/{data}/{split}/epoch_{N}__{hash}
    """

    entity: str
    project: str
    model: str
    run_id: str
    run_name: str
    data: str
    split: str
    checkpoint_epoch: Optional[int]
    checkpoint_path: str
    settings: Mapping[str, Any]

    @property
    def fingerprint(self) -> str:
        return cache_fingerprint(
            {
                "schema": CACHE_SCHEMA_VERSION,
                "entity": self.entity,
                "project": self.project,
                "model": self.model,
                "run_id": self.run_id,
                "data": self.data,
                "split": self.split,
                "checkpoint_epoch": self.checkpoint_epoch,
                "checkpoint_path": self.checkpoint_path,
                "settings": self.settings,
            }
        )

    def directory(self, root: pathlib.Path | str = DEFAULT_CACHE_DIR) -> pathlib.Path:
        epoch = "unknown" if self.checkpoint_epoch is None else str(self.checkpoint_epoch)
        return (
            pathlib.Path(root)
            / _slug(self.entity)
            / _slug(self.project)
            / _slug(self.model)
            / f"{_slug(self.run_name)}__{_slug(self.run_id)}"
            / _slug(self.data)
            / _slug(self.split)
            / f"epoch_{epoch}__{self.fingerprint}"
        )

    def identity(self) -> dict[str, Any]:
        """Fields that uniquely name this evaluation, independent of settings hash."""
        return {
            "schema_version": CACHE_SCHEMA_VERSION,
            "entity": self.entity,
            "project": self.project,
            "model": self.model,
            "run_id": self.run_id,
            "run_name": self.run_name,
            "data": self.data,
            "split": self.split,
            "checkpoint_epoch": self.checkpoint_epoch,
            "checkpoint_path": self.checkpoint_path,
            "fingerprint": self.fingerprint,
        }

    def manifest(self) -> dict[str, Any]:
        return {
            **self.identity(),
            "settings": _jsonable(self.settings),
        }


@dataclass
class CachedRollout:
    """One cached embryo/sample on physical (unnormalized) field units."""

    path: pathlib.Path
    dataset: str
    file: Optional[str]
    batch: int
    sample: int
    field_names: tuple[str, ...]
    pred: np.ndarray
    ref: np.ndarray
    context: np.ndarray
    input_time: np.ndarray
    output_time: np.ndarray
    space_grid: np.ndarray
    metadata: WellMetadata

    @property
    def time(self) -> np.ndarray:
        return np.concatenate([self.input_time, self.output_time])[
            : self.context.shape[0] + self.ref.shape[0]
        ]


def metadata_to_dict(metadata: WellMetadata) -> dict[str, Any]:
    return _jsonable(asdict(metadata))


def metadata_from_dict(value: Mapping[str, Any]) -> WellMetadata:
    data = dict(value)
    data["field_names"] = {int(k): v for k, v in data["field_names"].items()}
    data["constant_field_names"] = {
        int(k): v for k, v in data["constant_field_names"].items()
    }
    data["spatial_resolution"] = tuple(data["spatial_resolution"])
    return WellMetadata(**data)


class RolloutCache:
    """Read/write one cache key, using an explicit completion marker."""

    def __init__(
        self,
        key: RolloutCacheKey,
        root: pathlib.Path | str = DEFAULT_CACHE_DIR,
    ):
        self.key = key
        self.root = pathlib.Path(root)
        self.path = key.directory(root)
        self.artifact_dir = self.path / "artifacts"
        self.video_dir = self.path / "videos"
        self.manifest_path = self.path / "manifest.json"
        self.summary_path = self.path / "summary.json"
        self.complete_path = self.path / "COMPLETE"

    def _stored_manifest(self) -> Optional[dict[str, Any]]:
        if not self.manifest_path.is_file():
            return None
        try:
            return json.loads(self.manifest_path.read_text())
        except json.JSONDecodeError:
            return None

    def matches_key(self) -> bool:
        """True if the on-disk manifest is this exact project/model/run/dataset/settings."""
        stored = self._stored_manifest()
        if stored is None:
            return False
        expected = self.key.identity()
        return all(stored.get(field) == value for field, value in expected.items())

    @property
    def complete(self) -> bool:
        return (
            self.complete_path.is_file()
            and self.manifest_path.is_file()
            and self.summary_path.is_file()
            and self.artifact_dir.is_dir()
            and self.matches_key()
        )

    def reset(self) -> None:
        if self.path.exists():
            shutil.rmtree(self.path)

    def begin(self, *, overwrite: bool = False) -> None:
        if (
            self.complete_path.is_file()
            and self.matches_key()
            and not overwrite
        ):
            raise FileExistsError(
                f"Refusing to overwrite complete cache at {self.path}. "
                "Pass refresh_cache=True to rebuild this project/model/dataset entry."
            )
        self.reset()
        self.path.mkdir(parents=True, exist_ok=True)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.video_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path.write_text(
            json.dumps(self.key.manifest(), indent=2, sort_keys=True) + "\n"
        )
        self.complete_path.unlink(missing_ok=True)

    def save_batch(
        self,
        *,
        dataset: str,
        batch_index: int,
        pred: np.ndarray,
        ref: np.ndarray,
        context: np.ndarray,
        input_time: np.ndarray,
        output_time: np.ndarray,
        space_grid: np.ndarray,
        field_names: list[str],
        metadata: WellMetadata,
        file_paths: list[str],
    ) -> None:
        """Save each batch item independently so embryo identities remain stable."""
        pred = np.asarray(pred)
        ref = np.asarray(ref)
        context = np.asarray(context)
        batch_size = pred.shape[0]
        metadata_json = json.dumps(metadata_to_dict(metadata), sort_keys=True)
        fields = np.asarray(field_names, dtype=np.str_)

        def sample(array: np.ndarray, i: int) -> np.ndarray:
            if array.ndim and array.shape[0] == batch_size:
                return array[i]
            return array

        for i in range(batch_size):
            flat_index = batch_index * batch_size + i
            file_path = file_paths[flat_index] if flat_index < len(file_paths) else ""
            file_name = pathlib.Path(file_path).name if file_path else ""
            stem = _slug(pathlib.Path(file_name).stem) if file_name else f"batch_{batch_index:04d}_{i}"
            out = self.artifact_dir / f"{flat_index:05d}_{stem}.npz"
            np.savez_compressed(
                out,
                dataset=np.asarray(dataset),
                source_path=np.asarray(file_path),
                batch=np.asarray(batch_index),
                sample=np.asarray(i),
                field_names=fields,
                metadata_json=np.asarray(metadata_json),
                pred=sample(pred, i).astype(np.float32, copy=False),
                ref=sample(ref, i).astype(np.float32, copy=False),
                context=sample(context, i).astype(np.float32, copy=False),
                input_time=sample(np.asarray(input_time), i).astype(np.float32, copy=False),
                output_time=sample(np.asarray(output_time), i).astype(np.float32, copy=False),
                space_grid=sample(np.asarray(space_grid), i).astype(np.float32, copy=False),
            )

    def finish(self, summary: Mapping[str, Any]) -> None:
        self.summary_path.write_text(
            json.dumps(_jsonable(summary), indent=2, sort_keys=True) + "\n"
        )
        self.complete_path.write_text("complete\n")

    def summary(self) -> dict[str, Any]:
        if not self.complete:
            raise FileNotFoundError(
                f"Incomplete or mismatched rollout cache: {self.path}"
            )
        return json.loads(self.summary_path.read_text())

    def rollouts(self) -> list[CachedRollout]:
        if not self.complete:
            raise FileNotFoundError(
                f"Incomplete or mismatched rollout cache: {self.path}"
            )
        results = []
        for path in sorted(self.artifact_dir.glob("*.npz")):
            with np.load(path, allow_pickle=False) as z:
                metadata = metadata_from_dict(json.loads(str(z["metadata_json"].item())))
                results.append(
                    CachedRollout(
                        path=path,
                        dataset=str(z["dataset"].item()),
                        file=str(z["source_path"].item()) or None,
                        batch=int(z["batch"].item()),
                        sample=int(z["sample"].item()),
                        field_names=tuple(str(x) for x in z["field_names"].tolist()),
                        pred=np.asarray(z["pred"]),
                        ref=np.asarray(z["ref"]),
                        context=np.asarray(z["context"]),
                        input_time=np.asarray(z["input_time"]),
                        output_time=np.asarray(z["output_time"]),
                        space_grid=np.asarray(z["space_grid"]),
                        metadata=metadata,
                    )
                )
        return results
