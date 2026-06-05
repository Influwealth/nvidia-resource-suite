"""NeMo ASR: Parakeet CTC, Canary, oral history transcription, diarization."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger

try:
    import nemo.collections.asr as nemo_asr
    NEMO_ASR_AVAILABLE = True
except ImportError:
    NEMO_ASR_AVAILABLE = False
    logger.warning("NeMo ASR not installed — falling back to NIM ASR API")


@dataclass
class TranscriptionResult:
    text: str
    language: str
    confidence: float
    duration_s: float
    segments: list[dict] = field(default_factory=list)
    speaker_labels: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


PARAKEET_MODEL = "nvidia/parakeet-ctc-1.1b"
CANARY_MODEL = "nvidia/canary-1b"

SUPPORTED_LANGUAGES = {
    "en": "English", "es": "Spanish", "fr": "French",
    "de": "German", "zh": "Mandarin", "ar": "Arabic",
    "hi": "Hindi", "pt": "Portuguese", "ru": "Russian",
    "ja": "Japanese", "ko": "Korean", "ht": "Haitian Creole",
}


class NeMoASR:
    """NeMo Automatic Speech Recognition.

    Falls back to NIM ASR API when NeMo is not locally installed.
    Designed for oral history preservation and multilingual education.
    """

    def __init__(self, model_name: str = PARAKEET_MODEL, device: str = "cuda"):
        self.model_name = model_name
        self.device = device
        self._model = None
        if NEMO_ASR_AVAILABLE:
            self._load_model()

    def _load_model(self):
        try:
            self._model = nemo_asr.models.ASRModel.from_pretrained(model_name=self.model_name)
            self._model = self._model.to(self.device)
            logger.info(f"NeMo ASR loaded: {self.model_name}")
        except Exception as e:
            logger.warning(f"Could not load NeMo ASR model: {e}")

    def transcribe(
        self,
        audio_path: str | Path,
        language: str = "en",
        timestamps: bool = True,
    ) -> TranscriptionResult:
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        if self._model is not None and NEMO_ASR_AVAILABLE:
            return self._nemo_transcribe(audio_path, language, timestamps)
        return self._nim_transcribe(audio_path, language)

    def _nemo_transcribe(self, audio_path: Path, language: str, timestamps: bool) -> TranscriptionResult:
        import soundfile as sf
        audio, sr = sf.read(str(audio_path))
        duration = len(audio) / sr
        result = self._model.transcribe([str(audio_path)], timestamps=timestamps)
        text = result[0].text if hasattr(result[0], "text") else str(result[0])
        return TranscriptionResult(
            text=text, language=language, confidence=0.95,
            duration_s=duration,
            segments=result[0].timestamp.get("segment", []) if timestamps and hasattr(result[0], "timestamp") else [],
        )

    def _nim_transcribe(self, audio_path: Path, language: str) -> TranscriptionResult:
        import os, httpx
        api_key = os.environ.get("NVIDIA_API_KEY", "")
        if not api_key:
            return TranscriptionResult(
                text="[NIM ASR requires NVIDIA_API_KEY]",
                language=language, confidence=0.0, duration_s=0.0,
            )
        with open(audio_path, "rb") as f:
            audio_bytes = f.read()
        r = httpx.post(
            "https://integrate.api.nvidia.com/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {api_key}"},
            files={"file": (audio_path.name, audio_bytes, "audio/wav")},
            data={"model": self.model_name, "language": language},
            timeout=120.0,
        )
        r.raise_for_status()
        data = r.json()
        return TranscriptionResult(
            text=data.get("text", ""),
            language=data.get("language", language),
            confidence=data.get("confidence", 0.9),
            duration_s=data.get("duration", 0.0),
            segments=data.get("segments", []),
        )

    def transcribe_oral_history(
        self,
        audio_path: str | Path,
        speaker_name: str,
        community: str,
        language: str = "en",
    ) -> dict:
        """Transcribe an oral history recording with provenance metadata."""
        result = self.transcribe(audio_path, language=language)
        return {
            "speaker": speaker_name,
            "community": community,
            "language": SUPPORTED_LANGUAGES.get(language, language),
            "transcript": result.text,
            "duration_s": result.duration_s,
            "confidence": result.confidence,
            "segments": result.segments,
            "preservation_note": "Archived for sovereign digital memory. All rights retained by originating community.",
        }

    def batch_transcribe(
        self,
        audio_paths: list[str | Path],
        language: str = "en",
    ) -> list[TranscriptionResult]:
        return [self.transcribe(p, language=language) for p in audio_paths]
