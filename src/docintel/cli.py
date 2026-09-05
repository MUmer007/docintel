"""
Single CLI entrypoint for the whole project.

Usage (after `uv sync`):
    uv run docintel info
    uv run docintel ask "question"
    uv run docintel ingest run --source data/raw
    uv run docintel ingest index
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
def ask(question: str) -> None:
    """Ask a question against the indexed filings and get a cited answer."""
    from docintel.generation.rag import answer_question

    response = answer_question(question)
    console.print(f"\n[bold]Answer:[/bold] {response.answer}\n")
    if response.citations:
        console.print(f"[dim]Citations: {', '.join(response.citations)}[/dim]")
    if response.insufficient_context:
        console.print("[yellow]Note: model flagged this as insufficient context.[/yellow]")
@app.command()
def agent(question: str) -> None:
    """Ask a question using the full agent (search + SQL + calculator tools)."""
    from docintel.agents.orchestrator import run_agent

    response = run_agent(question)
    console.print(f"\n[bold]Answer:[/bold] {response.answer}\n")
    if response.steps:
        console.print(f"[dim]Tool calls made: {len(response.steps)}[/dim]")
        for i, step in enumerate(response.steps, 1):
            console.print(f"[dim]  {i}. {step.tool_name}({step.tool_args})[/dim]")
    if response.hit_max_steps:
        console.print("[yellow]Note: hit max tool-use steps without a final answer.[/yellow]")


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
    """Parse and chunk all filings in `source`, saving results to data/processed/."""
    from pathlib import Path

    from docintel.ingestion.pipeline import run_pipeline

    output_paths = run_pipeline(raw_dir=Path(source))
    console.print(f"[green]Processed {len(output_paths)} filing(s).[/green]")


@ingest_app.command("index")
def ingest_index() -> None:
    """Embed chunks (dense), build BM25 (sparse), and load tables into DuckDB (structured)."""
    from docintel.agents.tools.sql_store import build_sql_store
    from docintel.retrieval.indexer import run_indexing
    from docintel.retrieval.sparse import build_bm25_index, save_bm25_index

    count = run_indexing()
    console.print(f"[green]Indexed {count} chunks into Chroma (dense).[/green]")

    bm25_index = build_bm25_index()
    save_bm25_index(bm25_index)
    console.print(
        f"[green]Built BM25 index with {len(bm25_index.chunk_ids)} chunks (sparse).[/green]"
    )

    sql_summary = build_sql_store()
    console.print(
        f"[green]Loaded {sql_summary['loaded']} tables into DuckDB "
        f"({sql_summary['skipped']} skipped as not SQL-worthy).[/green]"
    )


@eval_app.command("run")
def eval_run(
    suite: str = typer.Option("regression", help="Which eval suite to run."),
) -> None:
    """Run the eval harness against the gold dataset."""
    from datetime import datetime
    from pathlib import Path

    from docintel.evals.runner import run_eval_suite, save_results, summarize_results

    console.print(f"[bold]Running eval suite: {suite}[/bold]\n")
    results = run_eval_suite()
    summary = summarize_results(results)

    for r in results:
        status = "[green]OK[/green]"
        if r.contains_expected_terms is False or r.correctly_refused is False:
            status = "[red]FAIL[/red]"
        console.print(f"{status} [{r.question_id}] {r.question}")

    console.print("\n[bold]Summary:[/bold]")
    for k, v in summary.items():
        console.print(f"  {k}: {v}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path("data/eval_datasets") / f"results_{timestamp}.json"
    save_results(results, summary, output_path)
    console.print(f"\n[dim]Results saved to {output_path}[/dim]")




if __name__ == "__main__":
    app()
