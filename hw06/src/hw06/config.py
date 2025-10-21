from importlib.resources import files

from pydantic import BaseModel
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)


class TrainingSettings(BaseModel):
    """Settings for model training."""

    split: float = 0.1
    n_embed: int = 32
    vocab_size: int = 100
    n_blocks: int = 5
    context_length: int = 8
    n_heads: int = 4
    batch_size: int = 16
    train_steps: int = 10
    learning_rate: float = 0.02
    momentum: float = 0.9


class AppSettings(BaseSettings):
    """Main application settings."""

    debug: bool = False
    random_seed: int = 31451
    training: TrainingSettings = TrainingSettings()

    model_config = SettingsConfigDict(
        toml_file=files("hw06").joinpath("config.toml"),
        env_nested_delimiter="__",
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """
        Set the priority of settings sources.

        We use a TOML file for configuration.
        """
        return (
            init_settings,
            TomlConfigSettingsSource(settings_cls),
            env_settings,
            dotenv_settings,
            file_secret_settings,
        )


def load_settings() -> AppSettings:
    """Load application settings."""
    return AppSettings()
