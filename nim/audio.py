"""
NVIDIA NIM — Audio Client

Speech-to-text and text-to-speech for:
  - Oral history recording transcription
  - Multilingual accessibility
  - AI guide voiceover generation
  - Student voice input for world interaction
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import requests

log = logging.getLogger(__name__)


class AudioClient:
    """NIM audio client for ASR and TTS tasks."""

    PARAKEET_MODEL = "nvidia/parakeet-ctc-1.1b"
    CANARY_MODEL = "nvidia/canary-1b"

    def __init__(self, api_key: str | None = None, base_url: str | None = None):
        self.api_key = api_key or os.environ.get("NVIDIA_API_KEY", "")
        self.base_url = (base_url or os.environ.get("NIM_BASE_URL", "https://integrate.api.nvidia.com/v1")).rstrip("/")

    def transcribe(
        self,
        audio_path: str,
        model: str = PARAKEET_MODEL,
        language: str = "en",
    ) -> str:
        """
        Transcribe audio file to text using NVIDIA Parakeet or Canary NIM.
        Supports: .wav, .mp3, .flac, .ogg
        """
        if not self.api_key:
            return "[Audio NIM unavailable — set NVIDIA_API_KEY]"

        path = Path(audio_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        with open(audio_path, "rb") as f:
            files = {"audio": (path.name, f, "audio/wav")}
            headers = {"Authorization": f"Bearer {self.api_key}"}
            data = {"model": model, "language": language}
            resp = requests.post(
                f"{self.base_url}/audio/transcriptions",
                headers=headers,
                files=files,
                data=data,
                timeout=120,
            )
            resp.raise_for_status()
            return resp.json().get("text", "")

    def transcribe_oral_history(
        self,
        audio_path: str,
        speaker_name: str = "",
        community: str = "",
    ) -> dict[str, str]:
        """
        Transcribe an oral history recording with metadata.
        Used for community content ingestion into the world-building pipeline.
        """
        transcript = self.transcribe(audio_path)
        return {
            "transcript": transcript,
            "speaker": speaker_name,
            "community": community,
            "source_file": str(audio_path),
            "word_count": len(transcript.split()),
            "ready_for_indexing": True,
        }

    def synthesize(
        self,
        text: str,
        voice: str = "default",
        speed: float = 1.0,
        output_path: str = "output.wav",
    ) -> str:
        """
        Synthesize text to speech for AI guide voiceover.
        Returns path to output audio file.
        """
        if not self.api_key:
            log.warning("Audio NIM unavailable — TTS skipped")
            return ""

        payload = {"model": "nvidia/tts-v1", "input": text, "voice": voice, "speed": speed}
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        resp = requests.post(
            f"{self.base_url}/audio/speech",
            headers=headers,
            json=payload,
            timeout=60,
        )
        resp.raise_for_status()
        Path(output_path).write_bytes(resp.content)
        return output_path
