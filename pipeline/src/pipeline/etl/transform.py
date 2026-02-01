from rich.logging import RichHandler
from rich.console import Console
from typing import Optional
from pathlib import Path
import polars as pl
import logging
import time

logging.basicConfig(
    level=logging.INFO, handlers=[RichHandler(rich_tracebacks=True, markup=True)]
)
logger = logging.getLogger(__name__)

console = Console()  # Rich prints

JSON_SCHEMA = {
    "id": pl.String,
    "submitter": pl.String,
    "authors": pl.String,
    "title": pl.String,
    "comments": pl.String,
    "journal-ref": pl.String,
    "doi": pl.String,
    "report-no": pl.String,
    "categories": pl.String,
    "license": pl.String,
    "abstract": pl.String,
    "versions": pl.List(
        pl.Struct([pl.Field("version", pl.String), pl.Field("created", pl.String)])
    ),
    "update_date": pl.String,
    "authors_parsed": pl.List(pl.List(pl.String)),
}

CATEGORIES = [
    "cs.AI",  # "Artificial Intelligence",
    "cs.CG",  # "Computational Geometry",
    "cs.CL",  # "Computational Language",
    "cs.CV",  # "Computer Vision and Pattern Recognition",
    "cs.IR",  # "Information Retrieval",
    "cs.LG",  # "Machine Learning",
    "cs.MA",  # "Multiagent Systems",
    "cs.NE",  # "Neural and Evolutionary Computing",
    "cs.RO",  # "Robotics"
]


class ArxivDatasetTransformer:
    def __init__(self, jsonfile: str):
        self.file = Path(jsonfile)

        assert self.file.exists(), f"JSON file not found: {self.file}"
        assert (
            self.file.suffix == ".json"
        ), "argument jsonfile must be a valid JSON file"

    def clean(self, outputfile: Optional[str | Path] = None):
        if outputfile is None:
            outputfile = f"{self.file.parent}/{self.file.stem}-clean.parquet"

        outputfile = Path(outputfile)
        assert (
            outputfile.suffix == ".parquet"
        ), f"Output file must be parquet, got {outputfile}"
        outputfile.parent.mkdir(parents=True, exist_ok=True)

        console.print(
            f"\n[bold cyan]Starting cleaning process[/bold cyan] 🫧\n"
            f"[dim]File: {self.file}[/dim]\n"
            f"[dim]Categories: {CATEGORIES}[/dim]\n"
        )

        start_time = time.perf_counter()

        with console.status(
            "[bold yellow]Processing data. This may take a few seconds.", spinner="dots"
        ):

            lf = pl.scan_ndjson(source=self.file, schema=JSON_SCHEMA)

            lf = (
                lf.with_columns(
                    pl.col("categories").str.split(by=" ").alias("categories_parsed")
                )
                .filter(
                    pl.col("categories_parsed")
                    .list.set_intersection(CATEGORIES)
                    .list.len()
                    > 0
                )
                .unique(subset="id", keep="first", maintain_order=False)
                .with_columns(
                    pl.col("title")
                    .str.replace_all("\n", " ")
                    .str.replace_all(r"\s+", " ")
                    .str.strip_chars(),
                    pl.col("abstract")
                    .str.replace_all("\n", " ")
                    .str.replace_all(r"\s+", " ")
                    .str.strip_chars(),
                    pl.col("authors_parsed").list.eval(
                        pl.element().list.reverse().list.join(" ").str.strip_chars()
                    ),
                    pl.col("versions")
                    .list.last()
                    .struct.field("created")
                    .str.to_datetime(format="%a, %d %b %Y %H:%M:%S GMT")
                    .alias("last_submitted"),
                )
                .select(
                    "id",
                    "doi",
                    "title",
                    "abstract",
                    "authors_parsed",
                    "categories_parsed",
                    "journal-ref",
                    "license",
                    "last_submitted",
                )
            )

            lf.sink_parquet(outputfile, compression="zstd")

        logger.info(
            f"[green]✅ Cleaning complete in {(time.perf_counter() - start_time):.2f}s![/green]"
        )

        return outputfile
