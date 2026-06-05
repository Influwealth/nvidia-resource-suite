"""Triton model repository management — config generation, ensemble pipelines, versioning."""
from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from loguru import logger


class ModelBackend(str, Enum):
    TENSORRT = "tensorrt_plan"
    ONNX = "onnxruntime_onnx"
    PYTORCH = "pytorch_libtorch"
    PYTHON = "python"
    ENSEMBLE = "ensemble"
    VLLM = "vllm"


@dataclass
class DynamicBatchConfig:
    max_queue_delay_microseconds: int = 100
    preferred_batch_size: list[int] = field(default_factory=lambda: [1, 4, 8])
    max_batch_size: int = 32


@dataclass
class ModelInput:
    name: str
    data_type: str  # e.g. "TYPE_FP32", "TYPE_INT32", "TYPE_STRING"
    dims: list[int]  # -1 for dynamic


@dataclass
class ModelOutput:
    name: str
    data_type: str
    dims: list[int]


@dataclass
class ModelConfig:
    name: str
    backend: ModelBackend
    max_batch_size: int = 8
    inputs: list[ModelInput] = field(default_factory=list)
    outputs: list[ModelOutput] = field(default_factory=list)
    dynamic_batching: DynamicBatchConfig | None = None
    instance_count: int = 1
    gpu_ids: list[int] = field(default_factory=lambda: [0])
    cpu_only: bool = False
    parameters: dict[str, str] = field(default_factory=dict)

    def to_pbtxt(self) -> str:
        """Render Triton protobuf text config."""
        lines = [
            f'name: "{self.name}"',
            f'backend: "{self.backend.value}"',
            f"max_batch_size: {self.max_batch_size}",
        ]
        for inp in self.inputs:
            lines += [
                "input {",
                f'  name: "{inp.name}"',
                f"  data_type: {inp.data_type}",
                f"  dims: [{', '.join(str(d) for d in inp.dims)}]",
                "}",
            ]
        for out in self.outputs:
            lines += [
                "output {",
                f'  name: "{out.name}"',
                f"  data_type: {out.data_type}",
                f"  dims: [{', '.join(str(d) for d in out.dims)}]",
                "}",
            ]
        if self.dynamic_batching:
            db = self.dynamic_batching
            lines += [
                "dynamic_batching {",
                f"  max_queue_delay_microseconds: {db.max_queue_delay_microseconds}",
            ]
            for ps in db.preferred_batch_size:
                lines.append(f"  preferred_batch_size: {ps}")
            lines.append("}")
        if not self.cpu_only:
            for gpu_id in self.gpu_ids:
                lines += [
                    "instance_group {",
                    f"  count: {self.instance_count}",
                    "  kind: KIND_GPU",
                    f"  gpus: [{gpu_id}]",
                    "}",
                ]
        else:
            lines += [
                "instance_group {",
                f"  count: {self.instance_count}",
                "  kind: KIND_CPU",
                "}",
            ]
        for key, val in self.parameters.items():
            lines += [
                "parameters {",
                f'  key: "{key}"',
                "  value {",
                f'    string_value: "{val}"',
                "  }",
                "}",
            ]
        return "\n".join(lines) + "\n"


@dataclass
class EnsemblePipeline:
    """Multi-model ensemble pipeline (e.g. preprocess → LLM → postprocess)."""
    name: str
    steps: list[dict]  # [{"model": str, "version": int, "input_map": {}, "output_map": {}}]

    def to_pbtxt(self) -> str:
        lines = [
            f'name: "{self.name}"',
            'backend: "ensemble"',
            "max_batch_size: 1",
        ]
        lines.append("ensemble_scheduling {")
        for step in self.steps:
            lines += [
                "  step {",
                f'    model_name: "{step["model"]}"',
                f'    model_version: {step.get("version", -1)}',
            ]
            for src, dst in step.get("input_map", {}).items():
                lines += [
                    "    input_map {",
                    f'      key: "{src}"',
                    f'      value: "{dst}"',
                    "    }",
                ]
            for src, dst in step.get("output_map", {}).items():
                lines += [
                    "    output_map {",
                    f'      key: "{src}"',
                    f'      value: "{dst}"',
                    "    }",
                ]
            lines.append("  }")
        lines.append("}")
        return "\n".join(lines) + "\n"


class ModelRepository:
    """Manages a local Triton model repository directory."""

    def __init__(self, root: str | Path = "./model_repository"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        logger.info(f"Model repository root: {self.root.resolve()}")

    def create_model(
        self,
        config: ModelConfig,
        model_file: str | Path | None = None,
        version: int = 1,
    ) -> Path:
        """Scaffold model directory and write config.pbtxt."""
        model_dir = self.root / config.name
        version_dir = model_dir / str(version)
        version_dir.mkdir(parents=True, exist_ok=True)

        config_path = model_dir / "config.pbtxt"
        config_path.write_text(config.to_pbtxt())
        logger.info(f"Wrote config: {config_path}")

        if model_file:
            src = Path(model_file)
            if src.exists():
                shutil.copy(src, version_dir / src.name)
                logger.info(f"Copied model file to {version_dir}")
            else:
                # Create placeholder so Triton can start in mock mode
                placeholder = version_dir / "model.placeholder"
                placeholder.write_text("# Replace with actual model file")
                logger.warning(f"Model file not found — placeholder written at {placeholder}")

        return model_dir

    def create_ensemble(self, pipeline: EnsemblePipeline) -> Path:
        pipeline_dir = self.root / pipeline.name
        version_dir = pipeline_dir / "1"
        version_dir.mkdir(parents=True, exist_ok=True)
        config_path = pipeline_dir / "config.pbtxt"
        config_path.write_text(pipeline.to_pbtxt())
        logger.info(f"Wrote ensemble config: {config_path}")
        return pipeline_dir

    def list_models(self) -> list[str]:
        return [d.name for d in self.root.iterdir() if d.is_dir()]

    def delete_model(self, model_name: str):
        model_dir = self.root / model_name
        if model_dir.exists():
            shutil.rmtree(model_dir)
            logger.info(f"Deleted model: {model_name}")

    def add_version(self, model_name: str, model_file: str | Path, version: int | None = None) -> Path:
        model_dir = self.root / model_name
        if not model_dir.exists():
            raise FileNotFoundError(f"Model {model_name} not found in repository")
        if version is None:
            existing = sorted(int(d.name) for d in model_dir.iterdir() if d.is_dir() and d.name.isdigit())
            version = (existing[-1] + 1) if existing else 1
        version_dir = model_dir / str(version)
        version_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(Path(model_file), version_dir)
        logger.info(f"Added version {version} to model {model_name}")
        return version_dir

    # ------------------------------------------------------------------
    # Preset configurations for common educational use cases
    # ------------------------------------------------------------------

    def create_llm_config(self, model_name: str = "llm-backend") -> ModelConfig:
        return ModelConfig(
            name=model_name,
            backend=ModelBackend.PYTHON,
            max_batch_size=1,
            inputs=[
                ModelInput("INPUT_IDS", "TYPE_INT32", [-1]),
                ModelInput("ATTENTION_MASK", "TYPE_INT32", [-1]),
            ],
            outputs=[ModelOutput("OUTPUT_IDS", "TYPE_INT32", [-1])],
            dynamic_batching=DynamicBatchConfig(preferred_batch_size=[1], max_batch_size=1),
        )

    def create_embedding_config(self, model_name: str = "embedding", dim: int = 4096) -> ModelConfig:
        return ModelConfig(
            name=model_name,
            backend=ModelBackend.ONNX,
            max_batch_size=32,
            inputs=[ModelInput("INPUT", "TYPE_STRING", [-1])],
            outputs=[ModelOutput("EMBEDDING", "TYPE_FP32", [dim])],
            dynamic_batching=DynamicBatchConfig(preferred_batch_size=[1, 8, 16, 32]),
        )

    def create_vision_config(self, model_name: str = "vision") -> ModelConfig:
        return ModelConfig(
            name=model_name,
            backend=ModelBackend.TENSORRT,
            max_batch_size=8,
            inputs=[ModelInput("IMAGE", "TYPE_FP32", [3, 224, 224])],
            outputs=[
                ModelOutput("FEATURES", "TYPE_FP32", [768]),
                ModelOutput("LABELS", "TYPE_STRING", [-1]),
            ],
            dynamic_batching=DynamicBatchConfig(preferred_batch_size=[1, 4, 8]),
        )

    def create_rag_ensemble(
        self,
        embed_model: str = "embedding",
        llm_model: str = "llm-backend",
    ) -> EnsemblePipeline:
        return EnsemblePipeline(
            name="rag-pipeline",
            steps=[
                {
                    "model": embed_model,
                    "version": 1,
                    "input_map": {"INPUT": "query_text"},
                    "output_map": {"EMBEDDING": "query_embedding"},
                },
                {
                    "model": llm_model,
                    "version": 1,
                    "input_map": {"INPUT_IDS": "context_ids"},
                    "output_map": {"OUTPUT_IDS": "response_ids"},
                },
            ],
        )
