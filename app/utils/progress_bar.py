def render_progress_bar(current: int, total: int, length: int = 10) -> str:
    current = max(0, min(current, total))
    filled = int(length * current / total) if total > 0 else 0
    bar = "█" * filled + "░" * (length - filled)
    return f"Прогресс: [{bar}] {current}/{total}"
