
import os


def get_env_variable(var_name: str) -> str:
    value = os.getenv(var_name)
    if value is None:
        raise EnvironmentError(f"env var '{var_name}' not found.")
    return value
