from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass

from loguru import logger

try:
    import joblib  # type: ignore
    import librosa  # type: ignore
    import numpy as np  # type: ignore
except ImportError:  # pragma: no cover
    joblib = None       # type: ignore
    librosa = None      # type: ignore
    np = None           # type: ignore


@dataclass
class SpeechEmotionResult:
    label: str      # 'joy', 'sadness', 'anger', 'neutral', ...
    score: float    # уверенность модели [0, 1]
    raw_probs: dict[str, float]


class SpeechEmotionAnalyzer:
    """
    Классификатор эмоций по аудио (по голосу, а не по тексту).

    Ожидается, что оффлайн обученная модель лежит по пути
    SPEECH_EMOTION_MODEL_PATH (по умолчанию: data/speech_emotion_model.joblib)
    и представляет собой sklearn-пайплайн, принимающий
    1D-вектор фич и выдающий predict/predict_proba.
    """

    def __init__(self) -> None:
        self.model_path = os.getenv(
            "SPEECH_EMOTION_MODEL_PATH",
            "data/speech_emotion_model.joblib",
        )
        self._model = None  # type: ignore[assignment]

        if any(m is None for m in (joblib, librosa, np)):
            logger.warning(
                "joblib/librosa/numpy не установлены. "
                "SpeechEmotionAnalyzer будет работать как заглушка."
            )

    # ---------------------- Публичный API ----------------------

    async def analyze_audio(self, audio_bytes: bytes) -> SpeechEmotionResult | None:
        """
        Основной метод, который будет вызываться из хендлера.

        :param audio_bytes: байты голосового сообщения (ogg/opus).
        :return: SpeechEmotionResult или None, если модель недоступна.
        """
        if joblib is None or librosa is None or np is None:
            logger.debug("SpeechEmotionAnalyzer: нет зависимостей → возвращаем None")
            return None

        model = self._get_model()
        if model is None:
            logger.debug("SpeechEmotionAnalyzer: модель не загружена → None")
            return None

        tmp_path = None
        try:
            # 1. складываем байты во временный файл (как в PronunciationEvaluator)
            with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp:
                tmp.write(audio_bytes)
                tmp_path = tmp.name

            # 2. загружаем аудио через librosa (оно само дернёт ffmpeg, если нужно)
            y, sr = librosa.load(tmp_path, sr=16000)  # моно, 16kHz

            if y.size == 0:
                logger.warning("SpeechEmotionAnalyzer: пустой аудиосигнал")
                return None

            features = self._extract_features(y, sr)  # 1D-вектор

            # 3. предсказание
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba([features])[0]
                labels = model.classes_
                best_idx = int(proba.argmax())
                label = str(labels[best_idx])
                score = float(proba[best_idx])
                probs_dict = {
                    str(lbl): float(p)
                    for lbl, p in zip(labels, proba)
                }
            else:
                label = str(model.predict([features])[0])
                score = 1.0
                probs_dict = {label: 1.0}

            logger.debug(
                "SpeechEmotionAnalyzer: label='{}', score={:.3f}, probs={}",
                label,
                score,
                probs_dict,
            )

            return SpeechEmotionResult(
                label=label,
                score=score,
                raw_probs=probs_dict,
            )

        except Exception as exc:  # pragma: no cover
            logger.error("SpeechEmotionAnalyzer: ошибка при анализе аудио: {}", exc)
            return None
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

    # ---------------------- Внутреннее ----------------------

    def _get_model(self):
        """
        Ленивая загрузка sklearn-пайплайна из joblib.
        """
        if self._model is not None:
            return self._model

        if joblib is None:
            return None

        if not os.path.exists(self.model_path):
            logger.warning(
                "Файл модели эмоций по аудио не найден: {}",
                self.model_path,
            )
            return None

        try:
            logger.info(
                "Загрузка модели эмоций по аудио из {}",
                self.model_path,
            )
            self._model = joblib.load(self.model_path)
        except Exception as exc:  # pragma: no cover
            logger.error("Не удалось загрузить speech_emotion_model.joblib: {}", exc)
            self._model = None

        return self._model

    def _extract_features(self, y, sr):
        """
        ДОЛЖНО 1-в-1 совпадать с тем, что ты будешь использовать в train-скрипте.
        y: np.ndarray, sr: int
        Возвращает np.ndarray shape=(D,)
        """
        import numpy as np  # type: ignore

        # нормализуем
        y = librosa.util.normalize(y)

        # MFCC (13 коэффициентов)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = mfcc.mean(axis=1)
        mfcc_std = mfcc.std(axis=1)

        # Хрома
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_mean = chroma.mean(axis=1)
        chroma_std = chroma.std(axis=1)

        # Энергия (RMS)
        rms = librosa.feature.rms(y=y)
        rms_mean = rms.mean(axis=1)
        rms_std = rms.std(axis=1)

        # Zero-crossing rate
        zcr = librosa.feature.zero_crossing_rate(y=y)
        zcr_mean = zcr.mean(axis=1)
        zcr_std = zcr.std(axis=1)

        # Можно добавить ещё spectral_centroid / bandwidth при желании
        feat_vec = np.concatenate(
            [
                mfcc_mean,
                mfcc_std,
                chroma_mean,
                chroma_std,
                rms_mean,
                rms_std,
                zcr_mean,
                zcr_std,
            ]
        )

        return feat_vec
