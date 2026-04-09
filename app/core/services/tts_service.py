import subprocess
import tempfile
from pathlib import Path

from gtts import gTTS  # нужен будет пакет gTTS
from loguru import logger


class TextToSpeechService:
    """Сервис TTS: English text -> OGG/Opus для отправки как voice в Telegram."""

    def __init__(self, language: str = "en"):
        self.language = language

    async def synthesize_to_ogg_bytes(self, text: str) -> bytes:
        """
        Синтезировать голос для текста и вернуть байты .ogg (opus).

        NB: gTTS использует внешнее API Google, нужен интернет.
        """
        logger.info("TTS synthesize text: %r", text)

        with tempfile.TemporaryDirectory() as tmpdir_str:
            tmpdir = Path(tmpdir_str)
            mp3_path = tmpdir / "tts.mp3"
            ogg_path = tmpdir / "tts.ogg"

            # 1) Синтез в mp3
            tts = gTTS(text=text, lang=self.language)
            tts.save(str(mp3_path))

            # 2) Конвертация mp3 -> ogg/opus через ffmpeg
            cmd = [
                "ffmpeg",
                "-y",
                "-i",
                str(mp3_path),
                "-c:a",
                "libopus",
                "-b:a",
                "48k",
                str(ogg_path),
            ]
            try:
                subprocess.run(
                    cmd,
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except subprocess.CalledProcessError as exc:
                logger.error("ffmpeg failed to convert TTS audio: {}", exc)
                raise

            data = ogg_path.read_bytes()
            logger.debug("TTS synthesized {} bytes", len(data))
            return data
