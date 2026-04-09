from pydantic import Field
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    bot_token: str = Field(alias="BOT_TOKEN")

    db_user: str = Field(default="english_bot", alias="DB_USER")
    db_password: str = Field(default="english_bot", alias="DB_PASSWORD")
    db_name: str = Field(default="english_bot", alias="DB_NAME")
    db_host: str = Field(default="db", alias="DB_HOST")
    db_port: int = Field(default=5432, alias="DB_PORT")

    database_sync_url: str | None = Field(default=None, alias="DATABASE_SYNC_URL")

    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    @property
    def database_url(self) -> str:
        return f"postgresql+asyncpg://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


@lru_cache
def get_settings() -> Settings:
    return Settings()
