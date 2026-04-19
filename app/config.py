from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    bot_token: str = Field(alias="BOT_TOKEN")

    # DB
    database_async_url: str | None = Field(default=None, alias="DATABASE_URL")
    database_sync_url: str | None = Field(default=None, alias="DATABASE_SYNC_URL")

    db_user: str = Field(default="english_bot", alias="DB_USER")
    db_password: str = Field(default="english_bot", alias="DB_PASSWORD")
    db_name: str = Field(default="english_bot", alias="DB_NAME")
    db_host: str = Field(default="db", alias="DB_HOST")
    db_port: int = Field(default=5432, alias="DB_PORT")

    # LLM
    use_llm_questions: bool = Field(default=True, alias="USE_LLM_QUESTIONS")
    llm_provider: str = Field(default="deepseek", alias="LLM_PROVIDER")

    # DeepSeek
    deepseek_api_key: str | None = Field(default=None, alias="DEEPSEEK_API_KEY")
    deepseek_base_url: str = Field(default="https://api.deepseek.com", alias="DEEPSEEK_BASE_URL")
    deepseek_model: str = Field(default="deepseek-chat", alias="DEEPSEEK_MODEL")

    # Legacy / fallback
    ollama_base_url: str = Field(default="http://ollama:11434", alias="OLLAMA_BASE_URL")
    ollama_model: str = Field(default="mistral", alias="OLLAMA_MODEL")

    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    @property
    def database_url(self) -> str:
        if self.database_async_url:
            return self.database_async_url

        return (
            f"postgresql+asyncpg://{self.db_user}:{self.db_password}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}"
        )

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


@lru_cache
def get_settings() -> Settings:
    return Settings()