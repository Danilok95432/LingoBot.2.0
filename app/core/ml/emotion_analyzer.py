import os
import tempfile
from typing import Optional, Dict, Any

import numpy as np
from loguru import logger

try:
    import joblib  # type: ignore
except ImportError:  # pragma: no cover
    joblib = None  # type: ignore

try:
    import torch  # type: ignore
except ImportError:  # pragma: no cover
    torch = None  # type: ignore

try:
    import librosa  # type: ignore
except ImportError:  # pragma: no cover
    librosa = None  # type: ignore


class EmotionAnalyzer:
    """
    Анализатор эмоций по голосу.

    Ожидает, что в models/speech_emotion_rf.pt лежит:
    - либо sklearn-классификатор (RandomForest / Pipeline) с .predict_proba и .classes_
    - либо dict {"model": ..., "label_encoder": ...}
    """

    DEFAULT_MODEL_PATH = "app/core/ml/models/speech_emotion_rf.pkl"

    def __init__(self, model_path: Optional[str] = None) -> None:
        self.model_path = model_path or os.getenv(
            "SPEECH_EMOTION_MODEL_PATH",
            self.DEFAULT_MODEL_PATH,
        )
        self.model = None
        self.labels: list[str] = []
        self._load_model()

    # ------------------------------------------------------------------ #
    # Загрузка модели
    # ------------------------------------------------------------------ #

    def _load_model(self) -> None:
        if not os.path.exists(self.model_path):
            logger.warning(
                "EmotionAnalyzer: файл модели не найден: {}",
                self.model_path,
            )
            return

        obj = None
        # 1) пробуем joblib (для sklearn)
        if joblib is not None:
            try:
                obj = joblib.load(self.model_path)  # type: ignore[arg-type]
                logger.info(
                    "EmotionAnalyzer: модель эмоций загружена через joblib из {}",
                    self.model_path,
                )
            except Exception as e:
                logger.warning(
                    "EmotionAnalyzer: не удалось загрузить модель через joblib: {}",
                    e,
                )

        # 2) если не получилось — пробуем torch.load
        if obj is None and torch is not None:
            try:
                obj = torch.load(self.model_path, map_location="cpu")  # type: ignore[arg-type]
                logger.info(
                    "EmotionAnalyzer: модель эмоций загружена через torch.load из {}",
                    self.model_path,
                )
            except Exception as e:
                logger.error(
                    "EmotionAnalyzer: не удалось загрузить модель ни через joblib, ни через torch: {}",
                    e,
                )
                return

        if obj is None:
            logger.error("EmotionAnalyzer: объект модели None после загрузки")
            return

        model = None
        labels: list[str] = []

        # Вариант 1: мы сохранили dict {"model": ..., "label_encoder": ...}
        if isinstance(obj, dict):
            model = obj.get("model") or obj.get("rf") or obj.get("classifier")
            le = obj.get("label_encoder") or obj.get("le")
            if le is not None and hasattr(le, "classes_"):
                labels = [str(c) for c in le.classes_]
        else:
            model = obj

        if model is None:
            logger.error("EmotionAnalyzer: в файле не найден объект модели")
            return

        # Попробуем вытащить labels из самой модели
        if not labels and hasattr(model, "classes_"):
            labels = [str(c) for c in model.classes_]  # type: ignore[attr-defined]

        self.model = model
        self.labels = labels
        logger.info(
            "EmotionAnalyzer: модель загружена. Метки классов: {}",
            self.labels,
        )

    # ------------------------------------------------------------------ #
    # Извлечение аудио-признаков
    # ------------------------------------------------------------------ #

    def _extract_features(self, audio_bytes: bytes) -> np.ndarray | None:
        """
        Извлечь признаки ровно в том же формате, что и при обучении:
        - 40 лог-мел каналов
        - для каждого mean и std
        -> вектор длиной 80.
        """
        if librosa is None or np is None:
            logger.error("EmotionAnalyzer: нет librosa/np, не могу извлечь признаки")
            return None

        tmp_path: str | None = None
        try:
            # пишем байты во временный файл
            with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp:
                tmp.write(audio_bytes)
                tmp_path = tmp.name

            # грузим так же, как в train_speech_emotion_model.py
            y, sr = librosa.load(tmp_path, sr=16000, mono=True)
            if y.size == 0:
                logger.warning("EmotionAnalyzer: пустой сигнал после загрузки")
                return None

            # выравниваем длину до 3 секунд
            target_len = 3 * sr
            if len(y) < target_len:
                y = np.pad(y, (0, target_len - len(y)))
            else:
                y = y[:target_len]

            # лог-мел спектр
            melspec = librosa.feature.melspectrogram(
                y=y,
                sr=sr,
                n_fft=1024,
                hop_length=512,
                n_mels=40,
                fmax=8000,
            )
            logmelspec = librosa.power_to_db(melspec, ref=np.max)

            feat_mean = logmelspec.mean(axis=1)
            feat_std = logmelspec.std(axis=1)
            feat = np.concatenate([feat_mean, feat_std], axis=0).astype(np.float32)

            if feat.shape[0] != 80:
                logger.error(
                    "EmotionAnalyzer: ожидали 80 признаков, получили %d", feat.shape[0]
                )
                return None

            return feat.astype("float32").reshape(1, -1)

        except Exception as exc:
            logger.error("EmotionAnalyzer: ошибка при извлечении признаков: {}", exc)
            return None
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

    # ------------------------------------------------------------------ #
    # Публичный метод для PronunciationService
    # ------------------------------------------------------------------ #

    async def analyze_audio(
            self,
            audio_bytes: bytes,
    ) -> Optional[Dict[str, Any]]:
        """
        Главный метод, который вызывает PronunciationService.

        Возвращает словарь с ключами:
            - label: итоговая метка эмоции с учётом порогов
                      ("happy", "sad", "neutral", "uncertain", ...)
            - raw_label: метка напрямую из модели (без постобработки)
            - prob: вероятность raw_label (0..1)
            - score: нормированный скор в [0..1], сейчас равен prob
            - probs: словарь {метка: вероятность} по всем классам

        Или None, если модель не готова или произошла ошибка.
        """
        if self.model is None:
            logger.warning(
                "EmotionAnalyzer: модель не загружена, emotion_info будет None"
            )
            return None

        # 1. Извлекаем признаки
        feats = self._extract_features(audio_bytes)
        if feats is None:
            logger.warning("EmotionAnalyzer: не удалось извлечь признаки из аудио")
            return None

        # _extract_features уже должен вернуть (1, n_features),
        # но на всякий случай нормализуем форму
        if isinstance(feats, np.ndarray) and feats.ndim == 1:
            feats = feats.reshape(1, -1)

        try:
            # 2. Предсказание вероятностей классов
            probs: np.ndarray = self.model.predict_proba(feats)[0]
            max_idx = int(np.argmax(probs))
            max_prob = float(probs[max_idx])

            # Словарь вероятностей по всем классам
            if self.labels and len(self.labels) == len(probs):
                probs_dict: Dict[str, float] = {
                    str(lbl): float(p) for lbl, p in zip(self.labels, probs)
                }
            else:
                probs_dict = {str(i): float(p) for i, p in enumerate(probs)}

            # Базовая метка от модели
            if self.labels and 0 <= max_idx < len(self.labels):
                raw_label = str(self.labels[max_idx])
            else:
                raw_label = str(max_idx)

            label = raw_label

            # Мягкий порог уверенности:
            # если модель совсем не уверена, говорим "uncertain",
            # не затирая уверенные предсказания типа "neutral".
            UNCERTAIN_THRESHOLD = 0.20
            if max_prob < UNCERTAIN_THRESHOLD:
                label = "uncertain"

            logger.debug(
                "EmotionAnalyzer: raw_label='{}', final_label='{}', prob={:.3f}, probs={}",
                raw_label,
                label,
                max_prob,
                probs_dict,
            )

            score = max_prob

            return {
                "label": label,
                "raw_label": raw_label,
                "prob": max_prob,
                "score": score,
                "probs": probs_dict,
            }

        except Exception as e:
            logger.error("EmotionAnalyzer: ошибка при предсказании эмоции: {}", e)
            return None

