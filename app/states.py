from aiogram.fsm.state import State, StatesGroup


class LessonStates(StatesGroup):
    IN_LESSON = State()
    WAITING_NEXT = State()


class QuickQuestionStates(StatesGroup):
    CHOOSING_TYPE = State()
    WAITING_ANSWER = State()
    AFTER_ANSWER = State()


class LevelTestStates(StatesGroup):
    QUIZ_IN_PROGRESS = State()
    TEXT_INPUT = State()
