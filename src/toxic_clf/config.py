"""Central params.yaml loader -- single source of truth for tunable values."""

import logging
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

DEFAULT_PARAMS_PATH = "params.yaml"


def load_params(path: str = DEFAULT_PARAMS_PATH) -> dict:
    """Load the project's params.yaml as a dict."""
    params_path = Path(path)
    if not params_path.exists():
        raise FileNotFoundError(f"params.yaml not found at: {params_path}")
    with open(params_path, "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)
    logger.info("Loaded params from %s", params_path)
    return params
