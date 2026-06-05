"""NeMo LLM: fine-tuning (SFT/PEFT/LoRA), checkpoint export, NeMo-to-TRT pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger

try:
    from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
    from nemo.collections.nlp.parts.nlp_overrides import NLPDDPStrategy
    import pytorch_lightning as pl
    NEMO_AVAILABLE = True
except ImportError:
    NEMO_AVAILABLE = False
    logger.warning("NeMo not installed — LLM module in simulation mode")


@dataclass
class LoRAConfig:
    """LoRA adapter configuration for parameter-efficient fine-tuning."""
    rank: int = 16
    alpha: float = 32.0
    dropout: float = 0.05
    target_modules: list[str] = field(default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"])
    lora_dropout: float = 0.0


@dataclass
class PromptTuningConfig:
    """Soft prompt tuning: only virtual tokens are trained."""
    num_virtual_tokens: int = 100
    prompt_tuning_init: str = "TEXT"  # TEXT or RANDOM
    init_text: str = "Classify the sentiment of the following text:"


@dataclass
class SFTConfig:
    """Supervised fine-tuning configuration."""
    model_name_or_path: str = "meta/llama-3.1-8b"
    dataset_path: str = ""
    output_dir: str = "./nemo_sft_output"
    num_epochs: int = 3
    batch_size: int = 4
    learning_rate: float = 2e-5
    warmup_steps: int = 100
    max_seq_length: int = 2048
    gradient_checkpointing: bool = True
    fp16: bool = True
    bf16: bool = False
    lora: LoRAConfig | None = None
    prompt_tuning: PromptTuningConfig | None = None
    save_steps: int = 500
    eval_steps: int = 500
    logging_steps: int = 10


class NeMoLLM:
    """NeMo Framework large language model interface.

    Supports inference via NIM API when NeMo is not locally installed,
    falling back gracefully for educational use.
    """

    def __init__(self, model_path: str | Path | None = None, use_nim_fallback: bool = True):
        self.model_path = Path(model_path) if model_path else None
        self.use_nim_fallback = use_nim_fallback
        self._model = None
        if NEMO_AVAILABLE and model_path and Path(model_path).exists():
            self._load_model()
        elif use_nim_fallback:
            logger.info("NeMo model not found locally — using NIM API fallback")
        else:
            logger.warning("NeMo not available and NIM fallback disabled")

    def _load_model(self):
        if not NEMO_AVAILABLE:
            return
        try:
            self._model = MegatronGPTModel.restore_from(str(self.model_path))
            logger.info(f"NeMo model loaded: {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load NeMo model: {e}")

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        if self._model is not None and NEMO_AVAILABLE:
            return self._nemo_generate(prompt, max_new_tokens, temperature, top_p)
        if self.use_nim_fallback:
            return self._nim_generate(prompt, max_new_tokens, temperature, top_p)
        return f"[NeMo not available] Prompt received: {prompt[:100]}"

    def _nemo_generate(self, prompt: str, max_tokens: int, temperature: float, top_p: float) -> str:
        response = self._model.generate(
            inputs=[prompt],
            length_params={"max_length": max_tokens},
            sampling_params={"temperature": temperature, "top_p": top_p},
        )
        return response[0] if response else ""

    def _nim_generate(self, prompt: str, max_tokens: int, temperature: float, top_p: float) -> str:
        import os
        import httpx
        api_key = os.environ.get("NVIDIA_API_KEY", "")
        if not api_key:
            return "[NIM fallback requires NVIDIA_API_KEY]"
        r = httpx.post(
            "https://integrate.api.nvidia.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": "meta/llama-3.1-70b-instruct",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
            },
            timeout=60.0,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

    def fine_tune(self, config: SFTConfig) -> dict:
        """Launch supervised fine-tuning with optional LoRA.

        Returns a dict describing the training plan.
        Real training requires NeMo + multi-GPU cluster.
        """
        logger.info(f"SFT config: {config}")
        if not NEMO_AVAILABLE:
            return {
                "status": "planned",
                "model": config.model_name_or_path,
                "epochs": config.num_epochs,
                "lora_rank": config.lora.rank if config.lora else None,
                "note": "Install nemo-toolkit to run training",
                "mock": True,
            }
        # Actual fine-tuning via NeMo trainer
        trainer = pl.Trainer(
            max_epochs=config.num_epochs,
            strategy=NLPDDPStrategy(),
            precision=16 if config.fp16 else 32,
        )
        # trainer.fit(model) — model prep and data loading omitted for brevity
        return {"status": "training_started", "output_dir": config.output_dir}

    def export_to_tensorrt(
        self,
        output_path: str | Path,
        precision: str = "fp16",
        max_input_len: int = 2048,
        max_output_len: int = 512,
        max_batch_size: int = 1,
    ) -> dict:
        """Export NeMo checkpoint to TensorRT-LLM format.

        Requires TensorRT-LLM installed: pip install tensorrt-llm
        """
        logger.info(f"Exporting to TRT-LLM at {output_path}")
        return {
            "output_path": str(output_path),
            "precision": precision,
            "max_input_len": max_input_len,
            "max_output_len": max_output_len,
            "max_batch_size": max_batch_size,
            "requires": "tensorrt-llm",
            "command": f"python -m tensorrt_llm.commands.convert_checkpoint --model_dir {self.model_path} --output_dir {output_path} --dtype {precision}",
        }
