from importlib.resources import files

from pydantic import BaseModel
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)


class DataSettings(BaseModel):
    """Settings for data generation."""

    val_split: float = 0.2
    dataset: str = "cifar100"


class TrainingSettings(BaseModel):
    """Settings for model training."""

    fresh_start: bool = True
    input_depth: int = 3
    num_classes: int = 100
    num_block: tuple = (5, 5, 5)
    layer_depths: tuple[int] = (64, 128, 256)
    layer_kernel_sizes: tuple[int] = (3, 3, 3)
    batch_size: int = 256
    train_steps: int = 2000
    learning_rate: float = 0.02
    momentum: float = 0.9


class AppSettings(BaseSettings):
    """Main application settings."""

    debug: bool = False
    random_seed: int = 31451
    data: DataSettings = DataSettings()
    training: TrainingSettings = TrainingSettings()

    model_config = SettingsConfigDict(
        toml_file=files("hw04").joinpath("config.toml"),
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
