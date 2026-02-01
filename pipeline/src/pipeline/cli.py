import typer
from decouple import config
from .etl import KaggleDatasetExtractor, ArxivDatasetTransformer
from .embeddings import VectorEmbeddings

app = typer.Typer()


@app.command()
def hello_pipeline():
    typer.echo("Hello pipeline!")


@app.command()
def bye_pipeline():
    typer.echo("Bye pipeline!")


# TEST
@app.command()
def extract_dataset():
    extractor = KaggleDatasetExtractor(config("KAGGLE_USERNAME"), config("KAGGLE_KEY"))
    path = extractor.get_dataset(dataset="Cornell-University/arxiv", path="./data")
    typer.echo(f"Files downloaded at: /{path}")


# TEST
@app.command()
def clean_dataset():
    transformer = ArxivDatasetTransformer(
        jsonfile="./data/arxiv-metadata-oai-snapshot.json"
    )
    clean_path = transformer.clean(outputfile="./data/clean-arxiv-metadata-oai.parquet")
    typer.echo(f"Files created at: /{clean_path}")


# TEST
@app.command()
def compute_embeddings():
    embeddings = VectorEmbeddings(model="text-embedding-3-small")
    df = embeddings.get_embeddings(
        file="./data/clean-arxiv-metadata-oai.parquet",
        text_column=["title", "abstract"],
        max_concurrent_requests=10,
    )
    df.write_parquet("./data/embs-arxiv-metadata-oai.parquet", compression="zstd")
    typer.echo(f"\nComputed embeddings for {len(df)} records!")


# TEST
@app.command()
def post_embeddings_batch():
    embeddings = VectorEmbeddings(model="Qwen/Qwen3-Embedding-8B")
    embeddings.post_embeddings_batch(
        file="./data/clean-arxiv-metadata-oai.parquet",
        text_column=["title", "abstract"],
    )
    typer.echo("post-embeddings-batch finished!")


if __name__ == "__main__":
    app()
