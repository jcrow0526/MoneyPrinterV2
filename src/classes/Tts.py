import os
import sys
import site
from typing import Dict

from config import ROOT_DIR
from TTS.utils.manage import ModelManager
from TTS.utils.synthesizer import Synthesizer


class TTS:
    """Class for Text-to-Speech using Coqui TTS with multi-language support."""

    def __init__(self) -> None:
        """Initializes the TTS class and prepares language specific synthesizers."""
        # Detect virtual environment site packages
        if hasattr(sys, "real_prefix") or (
            hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix
        ):
            # We're in a virtual environment
            site_packages = site.getsitepackages()[0]
        else:
            # We're not in a virtual environment, use the user's site packages
            site_packages = site.getusersitepackages()

        # Path to the .models.json file
        models_json_path = os.path.join(site_packages, "TTS", ".models.json")

        # Create directory if it doesn't exist
        tts_dir = os.path.dirname(models_json_path)
        if not os.path.exists(tts_dir):
            os.makedirs(tts_dir)

        # Initialize the ModelManager
        self._model_manager = ModelManager(models_json_path)

        # Cache synthesizers to avoid re-downloading models
        self._synthesizers: Dict[str, Synthesizer] = {}

        # Map supported languages to their Coqui model identifiers
        self._language_models = {
            "english": {
                "tts": "tts_models/en/ljspeech/tacotron2-DDC_ph",
                "vocoder": "vocoder_models/en/ljspeech/univnet",
            },
            "spanish": {
                "tts": "tts_models/es/mai/tacotron2-DDC",
                "vocoder": "vocoder_models/universal/libri-tts/fullband-multi-speaker",
            },
        }

    def _normalize_language(self, language: str) -> str:
        """Normalizes and validates the requested language."""
        if not language:
            return "english"

        normalized = language.strip().lower()

        if normalized in {"es", "español", "espanol", "spanish"}:
            return "spanish"

        return "english"

    def _get_synthesizer(self, language: str) -> Synthesizer:
        """Returns a synthesizer for the requested language, downloading models on demand."""
        language_key = self._normalize_language(language)

        if language_key in self._synthesizers:
            return self._synthesizers[language_key]

        model_config = self._language_models.get(language_key, self._language_models["english"])

        tts_model_id = model_config["tts"]
        vocoder_model_id = model_config["vocoder"]

        tts_path, tts_config_path, _ = self._model_manager.download_model(tts_model_id)
        voc_path, voc_config_path, _ = self._model_manager.download_model(vocoder_model_id)

        synthesizer = Synthesizer(
            tts_checkpoint=tts_path,
            tts_config_path=tts_config_path,
            vocoder_checkpoint=voc_path,
            vocoder_config=voc_config_path,
        )

        self._synthesizers[language_key] = synthesizer
        return synthesizer

    def synthesize(
        self,
        text: str,
        output_file: str = os.path.join(ROOT_DIR, ".mp", "audio.wav"),
        language: str = "english",
    ) -> str:
        """Synthesizes the given text into speech in the requested language."""

        synthesizer = self._get_synthesizer(language)

        # Synthesize the text
        outputs = synthesizer.tts(text)

        # Save the synthesized speech to the output file
        synthesizer.save_wav(outputs, output_file)

        return output_file

