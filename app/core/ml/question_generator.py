from __future__ import annotations

import json
import os
import random
import re
import textwrap
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Sequence

import requests
from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Question

GRAMMAR_TOPICS_BY_LEVEL: dict[str, list[str]] = {
    "A1": ["present_simple", "basic_questions", "possessive_adjectives", "articles", "prepositions_place"],
    "A2": ["past_simple", "comparatives", "future_intentions", "adverbs_frequency", "countable_uncountable"],
    "B1": ["present_perfect", "conditionals", "passive_voice", "modal_verbs", "relative_clauses"],
    "B2": ["past_perfect", "reported_speech", "modals_deduction", "conditional_perfect", "inversion"],
    "C1": ["inversion", "mixed_conditionals", "participles", "subjunctive", "ellipsis"],
    "C2": ["subjunctive", "discourse_markers", "rhetorical_devices", "nominalisation", "advanced_passives"],
}

VOCAB_TOPICS_BY_LEVEL: dict[str, list[str]] = {
    "A1": ["family", "daily_routines", "food", "animals", "school"],
    "A2": ["travel", "health", "shopping", "hobbies", "jobs"],
    "B1": ["technology", "environment", "education", "relationships", "culture"],
    "B2": ["business", "science", "media", "global_issues", "art"],
    "C1": ["philosophy", "psychology", "economics", "politics", "innovation"],
    "C2": ["academic_language", "law", "literature", "linguistics", "ethics"],
}


@dataclass
class GeneratedQuestion:
    type: str
    level: str
    topic: str
    text: str
    options: list[str]
    correct_index: int
    audio_file_path: str | None = None
    pronunciation_phrase: str | None = None
    difficulty: float = 0.5


class LLMQuestionBackend:
    """
    Обёртка над Ollama Mistral.

    Настройки через env:
    - OLLAMA_BASE_URL (по умолчанию http://ollama:11434)
    - OLLAMA_MODEL (по умолчанию mistral)
    """

    def __init__(self) -> None:
        self.base_url = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434").rstrip("/")
        self.model_name = os.getenv("OLLAMA_MODEL", "mistral")
        self.available = True  # считаем доступной, пока не упали с ошибкой

        logger.info("Используем Ollama LLM для вопросов: base_url={}, model={}", self.base_url, self.model_name)

    # ----------- низкоуровневый вызов Ollama -----------

    def _generate_raw(self, prompt: str, max_tokens: int = 512) -> str:
        """
        Вызов /api/generate у Ollama без стриминга.
        Возвращаем просто text (data["response"]).
        """
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": 0.7,
                "top_p": 0.9,
            },
        }

        try:
            resp = requests.post(url, json=payload, timeout=120)
            resp.raise_for_status()
            data = resp.json()
            text = data.get("response", "") or ""
            return text.strip()
        except Exception as exc:
            logger.error("Ошибка при обращении к Ollama: {}", exc)
            self.available = False
            return ""

    # ----------- публичные методы генерации -----------

    def generate_questions(
            self,
            level: str,
            qtype: str,
            n: int = 1,
    ) -> list[dict[str, Any]]:
        """
        Сгенерировать n вопросов указанного типа и уровня через Ollama.

        Возвращает список словарей с ключами:
        * text
        * options
        * correct_index
        * topic
        """
        # Если бэкенд не доступен — сразу выходим
        if not self.available:
            return []

        skill_map = {
            "grammar": "grammar (choose the correct form in a sentence)",
            "vocabulary": "vocabulary (choose the correct word or translation)",
        }
        skill = skill_map.get(qtype, "general English")

        prompt = textwrap.dedent(
            f"""
            You are an experienced ESL teacher.
            Create {n} multiple-choice questions for a student with CEFR level {level}.
            Focus on {skill}.

            For each question you MUST provide:

            - "text": question text in Russian with English examples where needed
            - "options": 4 answer options as a JSON array of strings
            - "correct_index": index (0-3) of the correct option
            - "topic": short topic identifier in English (like "articles", "present_simple")

            Return ONLY valid JSON: a list of objects, e.g.

            [
              {{
                "text": "[{level}][articles] Выберите правильный вариант.",
                "options": ["a cat", "an cat", "the cat", "cat"],
                "correct_index": 0,
                "topic": "articles"
              }}
            ]

            No explanations, no comments, no Markdown.
            Only pure JSON list.
            """
        ).strip()

        # ---- 1. Дёргаем Ollama ----
        raw = self._generate_raw(prompt)
        if not raw:
            logger.error("LLM вернула пустой ответ при generate_questions (qtype=%s, level=%s)", qtype, level)
            return []

        # ---- 2. Вырезаем JSON массив из ответа ----
        # иногда модель добавляет лишний текст до/после
        start = raw.find("[")
        end = raw.rfind("]")
        if start == -1 or end == -1 or end <= start:
            logger.error("LLM ответ без JSON-массива (qtype=%s, level=%s): %r", qtype, level, raw[:2000])
            return []

        json_text = raw[start: end + 1]

        # ---- 3. Подчищаем типичные артефакты ----

        # убираем ```json ... ``` если модель их добавила
        json_text = re.sub(r"```(?:json)?", "", json_text, flags=re.IGNORECASE)
        json_text = json_text.replace("```", "")

        # убираем возможные trailing commas перед ] или }
        # [...,]  или {...,}
        json_text = re.sub(r",(\s*[}\]])", r"\1", json_text)

        # иногда модель ставит комментарии // ... в JSON — вырезаем строки с ними
        lines = []
        for line in json_text.splitlines():
            # обрежем всё после //, если оно не внутри строки — это уже сложно детектить,
            # поэтому делаем грубо: просто убираем // и дальше до конца строки.
            if "//" in line:
                line = line.split("//", 1)[0]
            lines.append(line)
        json_text = "\n".join(lines).strip()

        # ---- 4. Парсим JSON ----
        try:
            data = json.loads(json_text)
        except json.JSONDecodeError as e:
            logger.error(
                "Не удалось распарсить JSON от LLM: %s\nСырой JSON (обрезано): %r",
                e,
                json_text[:2000],
            )
            return []

        if isinstance(data, dict):
            # На всякий случай, если модель вернула один объект вместо списка
            data = [data]

        if not isinstance(data, list):
            logger.error("LLM JSON не является списком объектов: %r", type(data))
            return []

        # ---- 5. Нормализуем структуру вопросов ----
        result: list[dict[str, Any]] = []
        for item in data:
            if not isinstance(item, dict):
                continue

            text_val = str(item.get("text", "")).strip()
            options_val = [str(o) for o in item.get("options", [])][:4]

            if len(options_val) < 2:
                continue

            try:
                correct_index_val = int(item.get("correct_index", 0))
            except (TypeError, ValueError):
                correct_index_val = 0

            topic_val = str(item.get("topic", "general")).strip() or "general"

            result.append(
                {
                    "text": text_val,
                    "options": options_val,
                    "correct_index": max(0, min(correct_index_val, len(options_val) - 1)),
                    "topic": topic_val,
                }
            )

        return result[:n]

    def generate_pronunciation_phrase(self, level: str) -> str | None:
        """
        Сгенерировать одно слово или короткую фразу для произношения.
        """
        if not self.available:
            return None

        prompt = textwrap.dedent(
            f"""
            You are an English teacher.

            Generate ONE short natural English phrase or ONE word suitable
            for a student with CEFR level {level}.

            Requirements:
            - 1–8 words
            - everyday, common language
            - NO numbering, NO explanations
            - output ONLY the phrase itself, without quotes or comments
            """
        ).strip()

        raw = self._generate_raw(prompt, max_tokens=32)
        if not raw:
            return None

        # берём последнюю непустую строку как фразу
        lines = [l.strip() for l in raw.splitlines() if l.strip()]
        if not lines:
            return None

        phrase = lines[-1].strip().strip('"\' ')
        if not phrase or len(phrase.split()) > 10:
            return None

        return phrase

    def generate_listening_mcq(self, level: str) -> tuple[str, list[str], int] | None:
        """
        Генерирует одно предложение (6–12 слов) и 4 варианта,
        где ровно один вариант — точная копия предложения.

        Формат:
        Sentence: <sentence>
        A) ...
        B) ...
        C) ...
        D) ...
        Correct: <letter>
        """
        if not self.available:
            return None

        prompt = textwrap.dedent(
            f"""
            You are an English teacher.
            Create ONE listening comprehension question for a CEFR {level} student.

            Requirements:
            - First line: "Sentence: <English sentence of 6-12 words>"
            - Next 4 lines: options labeled A), B), C), D)
            - Exactly ONE option must be EXACTLY the same as the Sentence line text.
            - Other 3 options must be similar but slightly different
              (word order, small changes in time, place, details, etc.).
            - Last line: "Correct: <letter of correct option>"

            Example:
            Sentence: I like to read books in the evening.
            A) I like reading books in the evening.
            B) I like to read books in the evening.
            C) I like to read books at night.
            D) I love to read books every evening.
            Correct: B

            Now create ONLY ONE such question.
            """
        ).strip()

        text = self._generate_raw(prompt, max_tokens=256)
        if not text:
            return None

        sentence_match = re.search(r"Sentence:\s*(.+)", text)
        options_matches = re.findall(r"^[ABCD]\)\s*(.+)", text, flags=re.MULTILINE)
        correct_match = re.search(r"Correct:\s*([ABCD])", text)

        if not sentence_match or len(options_matches) != 4 or not correct_match:
            logger.warning("LLM listening parsing failed: {}", text)
            return None

        sentence = sentence_match.group(1).strip()
        options = [o.strip() for o in options_matches]
        letter = correct_match.group(1).upper()
        letter_to_index = {"A": 0, "B": 1, "C": 2, "D": 3}
        correct_index = letter_to_index.get(letter, 0)

        if correct_index < 0 or correct_index > 3:
            logger.warning("LLM listening invalid correct index: {}", text)
            return None

        return sentence, options, correct_index


class QuestionGenerator:
    """
    Генератор вопросов.

    - Для grammar/vocabulary умеет работать батчами через Ollama.
    - Для listening/speaking — отдельные генераторы.
    """

    def __init__(self, use_llm: bool | None = None) -> None:
        if use_llm is None:
            # по умолчанию LLM включена, если явно не выключили
            use_llm = os.getenv("USE_LLM_QUESTIONS", "1") == "1"
        self.use_llm = use_llm
        self._llm_backend: LLMQuestionBackend | None = None

    # ------------------------------------------------------------------
    # Публичный метод: батч-генерация и сохранение в БД
    # ------------------------------------------------------------------

    async def generate_and_store_bulk(
        self,
        session: AsyncSession,
        qtypes: Sequence[str],
        level: str,
        n: int,
    ) -> None:
        """
        Сгенерировать и сохранить n вопросов заданных типов.

        ВАЖНО:
        - Для grammar/vocabulary делаем ОДИН запрос в LLM на пачку вопросов по уровню.
        - Если LLM не справилась или отключена — докидываем fallback'ами.
        - Для listening/speaking пока генерим поштучно (их обычно меньше).
        """
        qtypes_list = list(qtypes) if qtypes else ["grammar"]

        # Распределяем n по типам более-менее равномерно
        base = n // len(qtypes_list)
        remainder = n % len(qtypes_list)
        counts: dict[str, int] = {}
        for i, qt in enumerate(qtypes_list):
            counts[qt] = base + (1 if i < remainder else 0)

        generated: list[GeneratedQuestion] = []

        backend = self._get_llm_backend()

        for qtype, count in counts.items():
            if count <= 0:
                continue

            if qtype in {"grammar", "vocabulary"} and backend is not None:
                topics_source = GRAMMAR_TOPICS_BY_LEVEL if qtype == "grammar" else VOCAB_TOPICS_BY_LEVEL
                allowed_topics = topics_source.get(level, topics_source["A1"])

                llm_items = backend.generate_questions(
                    level=level,
                    qtype=qtype,
                    n=count,
                )

                logger.debug(
                    "LLM сгенерировала {} вопросов типа {} для уровня {}: {}",
                    len(llm_items),
                    qtype,
                    level,
                    llm_items,
                )

                for item in llm_items:
                    gen = self._from_llm_item(qtype=qtype, level=level, item=item)
                    generated.append(gen)

                # Если LLM вернула меньше, чем запросили — добиваем шаблонными
                missing = count - len(llm_items)
                for _ in range(missing):
                    if qtype == "grammar":
                        generated.append(self._grammar_question(level, random.choice(allowed_topics)))
                    else:
                        generated.append(self._vocabulary_question(level, random.choice(allowed_topics)))

            elif qtype == "listening":
                for _ in range(count):
                    topic_source = GRAMMAR_TOPICS_BY_LEVEL.get(level, GRAMMAR_TOPICS_BY_LEVEL["A1"])
                    topic = random.choice(topic_source)
                    generated.append(self._listening_question(level, topic))

            elif qtype == "speaking":
                for _ in range(count):
                    topic_source = GRAMMAR_TOPICS_BY_LEVEL.get(level, GRAMMAR_TOPICS_BY_LEVEL["A1"])
                    topic = random.choice(topic_source)
                    generated.append(self._speaking_question(level, topic))

            else:
                # На всякий случай, если прилетел неизвестный тип
                topic_source = GRAMMAR_TOPICS_BY_LEVEL.get(level, GRAMMAR_TOPICS_BY_LEVEL["A1"])
                topic = random.choice(topic_source)
                generated.append(self._grammar_question(level, topic))

        # конвертируем GeneratedQuestion → ORM-модель и сохраняем
        questions: list[Question] = []
        for gen in generated:
            q = Question(
                type=gen.type,
                level=gen.level,
                topic=gen.topic,
                payload={
                    "text": gen.text,
                    "options": gen.options,
                },
                correct_option_index=gen.correct_index,
                audio_file_path=gen.audio_file_path,
                pronunciation_phrase=gen.pronunciation_phrase,
                difficulty=gen.difficulty,
                created_at=datetime.utcnow(),
            )
            questions.append(q)

        session.add_all(questions)
        await session.commit()

    # ------------------------------------------------------------------
    # Внутренние методы
    # ------------------------------------------------------------------

    def _get_llm_backend(self) -> LLMQuestionBackend | None:
        if not self.use_llm:
            return None
        if self._llm_backend is None:
            self._llm_backend = LLMQuestionBackend()
            if not self._llm_backend.available:
                logger.warning("LLMQuestionBackend недоступен, будут использованы шаблонные вопросы")
        return self._llm_backend if self._llm_backend and self._llm_backend.available else None

    def _from_llm_item(self, qtype: str, level: str, item: dict[str, Any]) -> GeneratedQuestion:
        """
        Приведение сырого LLM-ответа к нашему GeneratedQuestion.
        Здесь же жёстко прошиваем уровень в текст.
        """
        raw_text = str(item.get("text", "")).strip()
        topic_from_llm = str(item.get("topic", "general")).strip() or "general"

        # убираем старые [A1]/[B2] в начале
        raw_text = re.sub(r"^\[[A-C][12]\]\s*(\[[^\]]+\])?\s*", "", raw_text)

        text = f"[{level.upper()}][{topic_from_llm}] {raw_text}"

        options = [str(o) for o in item.get("options", [])][:4]
        if len(options) < 2:
            # fallback, не должно особо происходить
            options = ["Option 1", "Option 2"]
        try:
            correct_index = int(item.get("correct_index", 0))
        except (TypeError, ValueError):
            correct_index = 0

        correct_index = max(0, min(correct_index, len(options) - 1))

        return GeneratedQuestion(
            type=qtype,
            level=level,
            topic=topic_from_llm,
            text=text,
            options=options,
            correct_index=correct_index,
            difficulty=self._base_difficulty(level),
        )

    def _generate_single(self, qtype: str, level: str) -> GeneratedQuestion:
        """
        Старый путь (по одному вопросу). Используется там, где ещё вызывается.
        Для уроков основная оптимизация теперь в generate_and_store_bulk.
        """
        topic_source = GRAMMAR_TOPICS_BY_LEVEL if qtype == "grammar" else VOCAB_TOPICS_BY_LEVEL
        topic_pool = topic_source.get(level, topic_source["A1"])
        topic = random.choice(topic_pool)

        if qtype in {"grammar", "vocabulary"}:
            backend = self._get_llm_backend()
            if backend is not None:
                llm_questions = backend.generate_questions(
                    level=level,
                    qtype=qtype,
                    n=1,
                )
                if llm_questions:
                    item = llm_questions[0]
                    logger.debug("LLM сгенерировала вопрос: {}", item)
                    return self._from_llm_item(qtype=qtype, level=level, item=item)

        if qtype == "grammar":
            return self._grammar_question(level, topic)
        if qtype == "vocabulary":
            return self._vocabulary_question(level, topic)
        if qtype == "listening":
            return self._listening_question(level, topic)
        if qtype == "speaking":
            return self._speaking_question(level, topic)

        return self._grammar_question(level, topic)

    # ----------------- Шаблонные вопросы (fallback) --------------------

    def _grammar_question(self, level: str, topic: str) -> GeneratedQuestion:
        sentence = "She __ to the gym every morning."
        options = ["go", "goes", "is going", "going"]
        correct_index = 1
        text = f"[{level.upper()}][{topic}] Выберите правильную форму глагола."
        return GeneratedQuestion(
            type="grammar",
            level=level,
            topic=topic,
            text=text,
            options=options,
            correct_index=correct_index,
            difficulty=self._base_difficulty(level),
        )

    def _vocabulary_question(self, level: str, topic: str) -> GeneratedQuestion:
        # маленький словарик для fallback, чтобы задания отличались
        vocab = [
            ("horse", "лошадь"),
            ("table", "стол"),
            ("house", "дом"),
            ("cat", "кошка"),
            ("book", "книга"),
            ("car", "машина"),
        ]

        eng, ru_correct = random.choice(vocab)

        wrong_options = [ru for e, ru in vocab if ru != ru_correct]
        wrong_sample = random.sample(wrong_options, k=3)

        options = [ru_correct] + wrong_sample
        random.shuffle(options)
        correct_index = options.index(ru_correct)

        text = f"[{level.upper()}][{topic}] Выберите правильный перевод слова '{eng}'"

        return GeneratedQuestion(
            type="vocabulary",
            level=level,
            topic=topic,
            text=text,
            options=options,
            correct_index=correct_index,
            difficulty=self._base_difficulty(level),
        )

    def _listening_question(self, level: str, topic: str) -> GeneratedQuestion:
        backend = self._get_llm_backend()
        sentence: str | None = None
        options: list[str] | None = None
        correct_index: int = 0

        if backend is not None:
            res = backend.generate_listening_mcq(level)
            if res is not None:
                sentence, options, correct_index = res
                logger.debug(
                    "Listening question from LLM: sentence=%r, correct_index=%s",
                    sentence,
                    correct_index,
                )
            else:
                logger.debug("Listening question: LLM returned None, fallback.")
        else:
            logger.debug("Listening question: LLM backend not available, fallback.")

        if sentence is None or options is None:
            sentence = "I like to read books in the evening."
            options = [
                "I like to read books in the evening.",  # правильный
                "I like reading books in the evening.",
                "I like to read books at night.",
                "I love to read books every evening.",
            ]
            correct_index = 0
            logger.debug("Listening question: using fallback sentence/options.")

        text = "👂 Прослушайте аудио и выберите предложение, которое вы услышали."

        return GeneratedQuestion(
            type="listening",
            level=level,
            topic=topic,
            text=text,
            options=options,
            correct_index=correct_index,
            difficulty=self._base_difficulty(level) + 0.05,
        )

    def _speaking_question(self, level: str, topic: str) -> GeneratedQuestion:
        backend = self._get_llm_backend()
        phrase: str | None = None

        if backend is not None:
            phrase = backend.generate_pronunciation_phrase(level)
            logger.debug("Speaking question phrase from LLM: {!r}", phrase)
        else:
            logger.debug("Speaking question: LLM backend is not available, using fallback.")

        if not phrase:
            fallback_phrases_by_level = {
                "A1": [
                    "I like cats and dogs.",
                    "My name is Anna.",
                    "The sky is blue.",
                    "I have a red book.",
                    "She is very happy.",
                    "We eat green apples.",
                    "He reads every day.",
                    "They play football together."
                ],
                "A2": [
                    "I want to be a programmer.",
                    "The weather is nice today.",
                    "She reads books every day.",
                    "I love learning new languages.",
                    "I go to school by bus.",
                    "We watch movies on weekends.",
                    "My family lives in London.",
                    "He works in a big office."
                ],
                "B1": [
                    "Artificial intelligence is transforming our world.",
                    "Programming requires creativity and logical thinking.",
                    "Global warming affects ecosystems worldwide.",
                    "Modern technology connects people globally.",
                    "Learning languages opens cultural opportunities.",
                    "Regular exercise improves mental health.",
                    "Effective communication is professionally essential.",
                    "Sustainable development balances economic growth."
                ],
                "B2": [
                    "Technology significantly alters human interaction patterns.",
                    "Climate change requires international mitigation strategies.",
                    "Digital literacy is fundamental in education.",
                    "Globalization presents opportunities and challenges.",
                    "Cultural diversity fosters creativity and innovation.",
                    "Resilience develops through mindfulness practices."
                ],
                "C1": [
                    "Quantum computing represents a paradigm shift.",
                    "Automation necessitates comprehensive policy reforms.",
                    "Neuroplasticity revolutionizes cognitive development understanding.",
                    "Epistemology underlies qualitative research methodologies.",
                    "Integration policies address structural interpersonal dynamics.",
                    "Technological singularity raises ethical questions."
                ],
                "C2": [
                    "Postmodern thought challenges ontological assumptions.",
                    "Geopolitics features multipolar transnational interdependencies.",
                    "Consciousness remains elusive in neuroscience.",
                    "Hermeneutics interprets textual historical meaning.",
                    "Anthropocene reevaluates anthropocentric environmental ethics.",
                    "Sociolinguistics reveals language social stratification."
                ]
            }
            phrases = fallback_phrases_by_level.get(level.upper(), fallback_phrases_by_level["A2"])

            # Выбираем случайную фразу для данного уровня
            phrase = random.choice(phrases)
            logger.debug("Speaking question: using fallback phrase for level {}: {!r}", level, phrase)

        text = f"[{level.upper()}][{topic}] Произнесите фразу: \"{phrase}\""
        options = ["Готово! Отправьте голосовое сообщение с произношением."] * 4

        return GeneratedQuestion(
            type="speaking",
            level=level,
            topic=topic,
            text=text,
            options=options,
            correct_index=0,
            pronunciation_phrase=phrase,
            difficulty=self._base_difficulty(level) + 0.15,
        )

    # ------------------------------------------------------------------

    def _base_difficulty(self, level: str) -> float:
        mapping = {
            "A1": 0.2,
            "A2": 0.3,
            "B1": 0.4,
            "B2": 0.6,
            "C1": 0.75,
            "C2": 0.9,
        }
        return mapping.get(level, 0.5)
