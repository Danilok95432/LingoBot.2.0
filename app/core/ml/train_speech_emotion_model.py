import os
import re
import glob
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import librosa
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

BASE_DATA_DIR = r"C:\Users\danil\PycharmProjects\datasets"

DATASETS = {
    "ravdess": os.path.join(BASE_DATA_DIR, "Ravdess", "audio_speech_actors_01-24"),
    "crema": os.path.join(BASE_DATA_DIR, "Crema"),
    "savee": os.path.join(BASE_DATA_DIR, "Savee"),
    "tess": os.path.join(BASE_DATA_DIR, "Tess"),
    "telegram": os.path.join(BASE_DATA_DIR, "Telegram")
}

MODEL_OUTPUT_PATH = os.path.join("models", "speech_emotion_rf.pkl")


EMOTION_LABELS = {
    "neutral": "neutral",
    "happy": "happy",
    "sad": "sad",
    "angry": "angry",
    "fear": "fear",
    "disgust": "disgust",
    "surprise": "surprise",
}


def parse_ravdess_emotion(path: str) -> str | None:
    name = os.path.basename(path)
    parts = name.split(".")[0].split("-")
    if len(parts) < 3:
        return None
    code = parts[2]
    mapping = {
        "01": "neutral",
        "02": "neutral",
        "03": "happy",
        "04": "sad",
        "05": "angry",
        "06": "fear",
        "07": "disgust",
        "08": "surprise",
    }
    return mapping.get(code)


def parse_crema_emotion(path: str) -> str | None:
    name = os.path.basename(path)
    parts = name.split(".")[0].split("_")
    if len(parts) < 3:
        return None
    code = parts[2]
    mapping = {
        "ANG": "angry",
        "DIS": "disgust",
        "FEA": "fear",
        "HAP": "happy",
        "NEU": "neutral",
        "SAD": "sad",
    }
    return mapping.get(code)


def parse_savee_emotion(path: str) -> str | None:
    name = os.path.basename(path)
    m = re.search(r"_([a-z]+)\d+\.wav$", name)
    if not m:
        return None
    code = m.group(1)
    mapping = {
        "a": "angry",
        "d": "disgust",
        "f": "fear",
        "h": "happy",
        "n": "neutral",
        "sa": "sad",
        "su": "surprise",
    }
    return mapping.get(code)


def parse_tess_emotion(path: str) -> str | None:
    p = Path(path)
    parent = p.parent.name
    if "_" in parent:
        emotion = parent.split("_")[-1].lower()
    else:
        emotion = parent.lower()

    mapping = {
        "neutral": "neutral",
        "happy": "happy",
        "sad": "sad",
        "angry": "angry",
        "fear": "fear",
        "disgust": "disgust",
        "surprise": "surprise",
        "ps": "surprise",
        "calm": "neutral",
    }
    return mapping.get(emotion)


def collect_files_with_labels() -> List[Tuple[str, str]]:
    items: list[tuple[str, str]] = []

    # RAVDESS
    ravdess_root = DATASETS.get("ravdess")
    if ravdess_root and os.path.isdir(ravdess_root):
        for wav_path in glob.glob(os.path.join(ravdess_root, "Actor_*", "*.wav")):
            label = parse_ravdess_emotion(wav_path)
            if label:
                items.append((wav_path, label))

    # CREMA-D
    crema_root = DATASETS.get("crema")
    if crema_root and os.path.isdir(crema_root):
        for wav_path in glob.glob(os.path.join(crema_root, "*.wav")):
            label = parse_crema_emotion(wav_path)
            if label:
                items.append((wav_path, label))

    # SAVEE
    savee_root = DATASETS.get("savee")
    if savee_root and os.path.isdir(savee_root):
        for wav_path in glob.glob(os.path.join(savee_root, "*.wav")):
            label = parse_savee_emotion(wav_path)
            if label:
                items.append((wav_path, label))

    # TESS
    tess_root = DATASETS.get("tess")
    if tess_root and os.path.isdir(tess_root):
        for wav_path in glob.glob(os.path.join(tess_root, "*", "*.wav")):
            label = parse_tess_emotion(wav_path)
            if label:
                items.append((wav_path, label))

    # TELEGRAM
    telegram_root = DATASETS.get("telegram")
    if telegram_root and os.path.isdir(telegram_root):
        telegram_count = 0
        for folder_name in os.listdir(telegram_root):
            subdir = os.path.join(telegram_root, folder_name)
            if not os.path.isdir(subdir):
                continue

            norm_label = EMOTION_LABELS.get(folder_name.lower())
            if norm_label is None:
                print(f"[TELEGRAM] Пропускаю подпапку {folder_name!r}: неизвестная эмоция")
                continue

            for ext in ("*.ogg", "*.wav", "*.mp3", "*.m4a"):
                pattern = os.path.join(subdir, ext)
                for audio_path in glob.glob(pattern):
                    items.append((audio_path, norm_label))
                    items.append((audio_path, norm_label))
                    items.append((audio_path, norm_label))
                    items.append((audio_path, norm_label))
                    items.append((audio_path, norm_label))
                    items.append((audio_path, norm_label))
                    items.append((audio_path, norm_label))
                    items.append((audio_path, norm_label))
                    telegram_count += 8

        print(f"[TELEGRAM] Добавлено {telegram_count} файлов из {telegram_root}")
    else:
        print("[TELEGRAM] Папка с датасетом не найдена или пуста:", telegram_root)


    print(f"Всего файлов с метками: {len(items)}")
    return items

def extract_features(path: str, sr: int = 16000) -> np.ndarray:
    y, sr = librosa.load(path, sr=sr, mono=True)
    target_len = 3 * sr
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]

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
    feat = np.concatenate([feat_mean, feat_std], axis=0)
    return feat.astype(np.float32)


def main():
    os.makedirs(os.path.dirname(MODEL_OUTPUT_PATH), exist_ok=True)

    items = collect_files_with_labels()
    if not items:
        print("Не нашёл ни одного аудиофайла. Проверь пути в DATASETS.")
        return

    X_list: list[np.ndarray] = []
    y_list: list[str] = []

    print("Извлекаю признаки...")
    for path, label in tqdm(items):
        try:
            feat = extract_features(path)
            X_list.append(feat)
            y_list.append(label)
        except Exception as e:
            print(f"Ошибка при обработке {path}: {e}")

    X = np.stack(X_list, axis=0)
    y = np.array(y_list)

    print(f"Финальный датасет: X.shape={X.shape}, y.shape={y.shape}")

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=300,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    print("Обучаю модель...")
    model.fit(X_train, y_train)

    print("Оценка на валидации:")
    y_pred = model.predict(X_val)

    report_str = classification_report(y_val, y_pred)
    print(report_str)

    labels = sorted(np.unique(y_val))
    cm = confusion_matrix(y_val, y_pred, labels=labels)
    print("Матрица ошибок (строки — истинные классы, столбцы — предсказанные):")
    print("labels:", labels)
    print(cm)

    report_dict = classification_report(y_val, y_pred, output_dict=True)

    metrics = {
        "labels": labels,
        "confusion_matrix": cm.tolist(),
        "classification_report": report_dict,
    }

    metrics_path = MODEL_OUTPUT_PATH.replace(".pkl", "_metrics.json")
    print(f"Сохраняю метрики в {metrics_path} ...")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"Сохраняю модель в {MODEL_OUTPUT_PATH} ...")
    joblib.dump(model, MODEL_OUTPUT_PATH)
    print("Готово.")


if __name__ == "__main__":
    main()
