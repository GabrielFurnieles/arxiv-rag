from qdrant_client import QdrantClient, models
from decouple import config
from pathlib import Path
import polars as pl
import numpy as np
import os

from rich.logging import RichHandler
import logging
import tqdm

from ..embeddings import VectorEmbeddings
from ..db.models import JobStatus

logging.basicConfig(
    level=logging.INFO, handlers=[RichHandler(rich_tracebacks=True, markup=True)]
)
logger = logging.getLogger(__name__)


class QdrantLoader:
    def __init__(self):
        self.client = QdrantClient(
            url=config("QDRANT_URL"), api_key=config("QDRANT_API_KEY")
        )
        self._embeddings = None
        self._metadata = None

    def create_collection(self, name: str, emb_dim: int, recreate: bool = False):
        if recreate:
            self.client.delete_collection(collection_name=name)
            logger.info(f"🧹 Removed '{name}' collection")

        self.client.create_collection(
            collection_name=name,
            vectors_config=models.VectorParams(
                size=emb_dim, distance=models.Distance.COSINE, on_disk=True
            ),
            quantization_config=models.BinaryQuantization(
                binary=models.BinaryQuantizationConfig(always_ram=True)
            ),
            optimizers_config=models.OptimizersConfigDiff(
                max_segment_size=60_000  # ~1/8 of the total number of vectors
            ),
            hnsw_config=models.HnswConfigDiff(
                m=0,  # Disable indexing for for upload. NOTE. Make sure to activate indices after upload
                on_disk=False,  # Keep indices always on RAM
            ),
        )

    def load_vectors(
        self,
        job_id: int,
        collection: str,
        chunk_size: int = 10_000,
        recreate: bool = False,
    ):
        if not self._check_job_status(job_id):
            return

        if recreate:
            current_collection = self.client.get_collection(collection)
            emb_dim = current_collection.config.params.vectors.size
            self.create_collection(collection, emb_dim, recreate=True)

        total_points = self._init_resources(job_id)
        logger.info(
            f"Loading {total_points} vectors into '{collection}' Qdrant collection..."
        )

        for offset in range(0, total_points, chunk_size):
            end = min(offset + chunk_size, total_points)

            vectors, metadata = self._download_chunk(job_id, offset, end)

            self.client.upload_collection(
                collection_name=collection,
                vectors=vectors.tolist(),
                payload=metadata.collect().to_dicts(),
                ids=tqdm.tqdm(range(offset, end)),
                batch_size=256,
                parallel=os.cpu_count(),
            )

        # Update indices
        self.client.update_collection(
            collection,
            hnsw_config=models.HnswConfigDiff(m=16, on_disk=False),
        )

    def _check_job_status(self, job_id: int) -> bool:
        Embeddings = VectorEmbeddings()
        status = Embeddings.check_status(job_id, display=False)

        if status != JobStatus.COMPLETED:
            logger.warning(
                f"Job {job_id} is not {JobStatus.COMPLETED}. Current status is {status.value if status else None}"
            )
            return False

        return True

    def _get_emb_file(self, job_id: int) -> Path:
        parent = Path(f"./data/embeddings/{str(job_id).zfill(4)}")
        files = [f for f in os.listdir(parent) if f.endswith(".npy")]

        if len(files) > 1:
            raise ValueError(
                f"There are too many NPY files in '{parent}' directory. Files = '{files}'"
            )
        if not files:
            raise FileNotFoundError(
                f"No NPY file was found in the job's '{parent}' directory"
            )

        return parent / files[0]

    def _init_resources(self, job_id: int) -> int:
        emb_file = self._get_emb_file(job_id)
        md_file = Path("./data/interim/clean-arxiv-metadata-oai.parquet")

        self._embeddings = np.load(emb_file, mmap_mode="r")
        self._metadata = pl.scan_parquet(md_file)

        # Check
        total_points = self._embeddings.shape[0]
        total_records = self._metadata.select(pl.len()).collect().item()

        if total_points != total_records:
            raise ValueError(
                f"The number of Vectors: {total_points} do not match the number of Records in metadata: {total_records}"
            )

        return total_points

    def _download_chunk(
        self, job_id: int, start: int, end: int
    ) -> tuple[np.array, pl.LazyFrame]:
        if self._embeddings is None or self._metadata is None:
            self._init_resources(job_id)

        return (
            self._embeddings[start:end],
            self._metadata.slice(start, end - start),
        )
