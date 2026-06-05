"""NeMo TTS: FastPitch + HiFiGAN, multi-voice, SSML, audio export."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loguru import logger

try:
    import nemo.collections.tts as nemo_tts
    NEMO_TTS_AVAILABLE = True
except ImportError:
    NEMO_TTS_AVAILABLE = False
    logger.warning("NeMo TTS not installed — falling back to NIM TTS API")


@dataclass
class SynthesisResult:
    audio_path: str
    duration_s: float
    sample_rate: int
    voice: str
    text: str
    success: bool
    error: str | None = None


VOICES = {
    "en_female_1": {"lang": "en", "gender": "female", "model": "tts_en_fastpitch"},
    "en_male_1": {"lang": "en", "gender": "male", "model": "tts_en_fastpitch"},
    "es_female_1": {"lang": "es", "gender": "female", "model": "tts_es_fastpitch"},
    "fr_female_1": {"lang": "fr", "gender": "female", "model": "tts_fr_fastpitch"},
    "guide_east_flatbush": {"lang": "en", "gender": "male", "style": "brooklyn", "model": "tts_en_fastpitch"},
    "guide_silk_road": {"lang": "en", "gender": "female", "style": "narrative", "model": "tts_en_fastpitch"},
    "guide_harlem": {"lang": "en", "gender": "male", "style": "jazz_era", "model": "tts_en_fastpitch"},
    "guide_greenville": {"lang": "en", "gender": "female", "style": "southern", "model": "tts_en_fastpitch"},
}


class NeMoTTS:
    """NeMo Text-to-Speech synthesis.

    Supports FastPitch + HiFiGAN pipeline locally or NIM TTS API.
    Used for world guide narration and accessible learning content.
    """

    def __init__(self, device: str = "cuda"):
        self.device = device
        self._spectro = None
        self._vocoder = None
        if NEMO_TTS_AVAILABLE:
            self._load_models()

    def _load_models(self):
        try:
            self._spectro = nemo_tts.models.FastPitchModel.from_pretrained("tts_en_fastpitch")
            self._vocoder = nemo_tts.models.HifiGanModel.from_pretrained("tts_en_hifigan")
            self._spectro.to(self.device)
            self._vocoder.to(self.device)
            logger.info("NeMo TTS models loaded (FastPitch + HiFiGAN)")
        except Exception as e:
            logger.warning(f"NeMo TTS load failed: {e}")

    def synthesize(
        self,
        text: str,
        voice: str = "en_female_1",
        speed: float = 1.0,
        output_path: str | Path = "output.wav",
    ) -> SynthesisResult:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if self._spectro and self._vocoder and NEMO_TTS_AVAILABLE:
            return self._nemo_synthesize(text, voice, speed, output_path)
        return self._nim_synthesize(text, voice, speed, output_path)

    def _nemo_synthesize(self, text: str, voice: str, speed: float, output_path: Path) -> SynthesisResult:
        try:
            import torch, soundfile as sf
            parsed = self._spectro.parse(text)
            spec = self._spectro.generate_spectrogram(tokens=parsed)
            audio = self._vocoder.convert_spectrogram_to_audio(spec=spec)
            audio_numpy = audio.to("cpu").detach().numpy()[0]
            sf.write(str(output_path), audio_numpy, 22050)
            duration = len(audio_numpy) / 22050
            return SynthesisResult(
                audio_path=str(output_path), duration_s=duration,
                sample_rate=22050, voice=voice, text=text, success=True,
            )
        except Exception as e:
            return SynthesisResult(
                audio_path="", duration_s=0.0, sample_rate=0,
                voice=voice, text=text, success=False, error=str(e),
            )

    def _nim_synthesize(self, text: str, voice: str, speed: float, output_path: Path) -> SynthesisResult:
        import os, httpx
        api_key = os.environ.get("NVIDIA_API_KEY", "")
        if not api_key:
            return SynthesisResult(
                audio_path="", duration_s=0.0, sample_rate=0,
                voice=voice, text=text, success=False,
                error="NIM TTS requires NVIDIA_API_KEY",
            )
        r = httpx.post(
            "https://integrate.api.nvidia.com/v1/audio/speech",
            headers={"Authorization": f"Bearer {api_key}"},
            json={"model": "riva-tts", "input": text, "voice": voice, "speed": speed},
            timeout=60.0,
        )
        r.raise_for_status()
        output_path.write_bytes(r.content)
        return SynthesisResult(
            audio_path=str(output_path), duration_s=len(text) / 15,
            sample_rate=22050, voice=voice, text=text, success=True,
        )

    def narrate_world_guide(
        self,
        guide_text: str,
        world: str,
        output_dir: str | Path = "./audio",
    ) -> list[SynthesisResult]:
        """Split guide text into sentences and synthesize each."""
        import re
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        voice_map = {
            "brooklyn-90s": "guide_east_flatbush",
            "greenville-sovereign": "guide_greenville",
            "ancient-silk-road": "guide_silk_road",
            "harlem-renaissance": "guide_harlem",
        }
        voice = voice_map.get(world, "en_female_1")
        sentences = re.split(r"(?<=[.!?])\s+", guide_text.strip())
        results = []
        for i, sentence in enumerate(sentences):
            path = output_dir / f"{world}_guide_{i:03d}.wav"
            results.append(self.synthesize(sentence, voice=voice, output_path=path))
        return results
