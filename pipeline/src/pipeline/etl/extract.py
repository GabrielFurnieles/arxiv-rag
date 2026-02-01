from rich.logging import RichHandler
from pathlib import Path
import logging
import os

logging.basicConfig(
    level=logging.INFO, handlers=[RichHandler(rich_tracebacks=True, markup=True)]
)
logger = logging.getLogger(__name__)


class KaggleDatasetExtractor:
    def __init__(self, kaggle_username: str, kaggle_key: str):
        self._set_auth(kaggle_username, kaggle_key)

        # Import after setting credentials to avoid error
        from kaggle.api.kaggle_api_extended import KaggleApi

        self.api = KaggleApi()

    def _set_auth(self, kaggle_username: str, kaggle_key: str) -> None:
        logger.info(
            f"[yellow]Setting Kaggle auth for user: '{kaggle_username}' with api key '...{kaggle_key[-4:]}'[/yellow]"
        )
        os.environ["KAGGLE_USERNAME"] = kaggle_username
        os.environ["KAGGLE_KEY"] = kaggle_key

    def get_dataset(self, dataset: str, path: str = "./data") -> Path:
        """Pulls the last version of the Cornell-University/arxiv dataset"""
        download_path = Path(path)
        download_path.mkdir(parents=True, exist_ok=True)

        self.api.authenticate()
        self.api.dataset_download_files(
            dataset, path=str(download_path), unzip=True, force=True, quiet=False
        )

        return download_path
