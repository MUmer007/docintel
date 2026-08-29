"""
Single CLI entrypoint for the whole project.

Usage (after `uv sync`):
    uv run docintel info
    uv run docintel ingest run --source data/raw
    uv run docintel serve
"""

import typer
from rich.console import Console

from docintel.core.config import get_settings

app = typer.Typer(help="DocIntel: enterprise document Q&A platform.")
console = Console()

ingest_app = typer.Typer(help="Ingestion pipeline commands.")
eval_app = typer.Typer(help="Eval harness commands.")
app.add_typer(ingest_app, name="ingest")
app.add_typer(eval_app, name="eval")


@app.command()
def info() -> None:
    """Print resolved configuration (secrets redacted) -- useful for debugging env setup."""
    settings = get_settings()
    console.print(f"[bold]Environment:[/bold] {settings.environment}")
    console.print(f"[bold]Groq model:[/bold] {settings.llm.groq_model}")
    console.print(f"[bold]Judge model:[/bold] {settings.llm.judge_model}")
    console.print(f"[bold]Chroma dir:[/bold] {settings.retrieval.chroma_persist_dir}")
    console.print(f"[bold]Collection:[/bold] {settings.retrieval.collection_name}")
    console.print(f"[bold]Hybrid search:[/bold] {settings.retrieval.use_hybrid_search}")
    has_groq = bool(settings.llm.groq_api_key)
    has_anthropic = bool(settings.llm.anthropic_api_key)
    console.print(f"[bold]GROQ_API_KEY set:[/bold] {'yes' if has_groq else 'NO -- set it in .env'}")
    console.print(
        f"[bold]ANTHROPIC_API_KEY set:[/bold] {'yes' if has_anthropic else 'NO -- set it in .env'}"
    )


@app.command()
def serve() -> None:
    """Run the FastAPI server."""
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "docintel.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.environment == "development",
    )


@ingest_app.command("run")
def ingest_run(
    source: str = typer.Option("data/raw", help="Directory of source documents."),
) -> None:
    """Run the ingestion pipeline: parse -> chunk -> embed -> upsert into Chroma."""
    console.print(f"[yellow]Ingestion pipeline not yet implemented. Source: {source}[/yellow]")


@eval_app.command("run")
def eval_run(
    suite: str = typer.Option("regression", help="Which eval suite to run."),
) -> None:
    """Run the eval harness against the gold dataset."""
    console.print(f"[yellow]Eval suite '{suite}' not yet implemented.[/yellow]")


if __name__ == "__main__":
    app()
    