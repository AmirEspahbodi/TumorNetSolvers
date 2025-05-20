from pydantic_settings import BaseSettings, SettingsConfigDict


class _AppSettings(BaseSettings): 
    PAMOUNT_DIR: str
    DATASET_ID: int
    BASE_DIR: str
    model_config = SettingsConfigDict(env_file='.env')

AppConfig = _AppSettings()