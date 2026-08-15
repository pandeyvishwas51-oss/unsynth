"""UnSynth CLI — detect, rewrite, clean, strip, eval, batch."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from unsynth import __version__
from unsynth.config import Settings, load_settings
from unsynth.eval.report import render_markdown_report
from unsynth.exceptions import UnSynthError
from unsynth.logging import configure_logging
from unsynth.metadata.strip import strip_path
from unsynth.pipeline.document import iter_files
from unsynth.pipeline.orchestrator import UnSynthPipeline
from unsynth.types import PipelineMode

app = typer.Typer(
    name="unsynth",
    help=(
        "Detect AI-text / statistical watermark signals and rewrite existing "
        "prose so those signals weaken. Local-first. Not a cryptographic eraser."
    ),
    no_args_is_help=True,
    add_completion=False,
)
console = Console(stderr=True)
out = Console()


def _as_float(value: object, default: float = 0.0) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    return default


def _settings(config: Path | None) -> Settings:
    return load_settings(config)


def _read_input(path: Path | None, stdin_ok: bool = True) -> str:
    if path is not None:
        try:
            return path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            raise typer.BadParameter(f"cannot read {path}: {exc}") from exc
    if stdin_ok and not sys.stdin.isatty():
        return sys.stdin.read()
    raise typer.BadParameter("provide a file or pipe text on stdin")


def _write_output(text: str, dest: Path | None) -> None:
    if dest is None:
        # Always use stdout for the payload so piping works.
        sys.stdout.write(text)
        if not text.endswith("\n"):
            sys.stdout.write("\n")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(text, encoding="utf-8")
    console.print(f"[green]wrote[/green] {dest}")


def _print_score_table(title: str, result_dict: dict[str, object]) -> None:
    table = Table(title=title, show_header=True, header_style="bold")
    table.add_column("detector")
    table.add_column("family")
    table.add_column("score", justify="right")
    table.add_column("label")
    table.add_column("conf", justify="right")
    details = result_dict.get("details")
    members = None
    if isinstance(details, dict):
        members = details.get("members")
    if members is None:
        members = result_dict.get("members")
    if isinstance(members, list):
        for m in members:
            if not isinstance(m, dict):
                continue
            table.add_row(
                str(m.get("name")),
                str(m.get("family")),
                f"{_as_float(m.get('score')):.3f}",
                str(m.get("label")),
                f"{_as_float(m.get('confidence')):.2f}",
            )
    table.add_row(
        str(result_dict.get("name", "ensemble")),
        str(result_dict.get("family", "ensemble")),
        f"{_as_float(result_dict.get('score')):.3f}",
        str(result_dict.get("label", "")),
        f"{_as_float(result_dict.get('confidence')):.2f}",
        style="bold",
    )
    console.print(table)


@app.callback()
def _root(
    ctx: typer.Context,
    config: Path | None = typer.Option(None, "--config", "-c", help="YAML/TOML config file."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    configure_logging("DEBUG" if verbose else "INFO")
    ctx.obj = {"config": config}


@app.command()
def version() -> None:
    """Print the package version."""

    out.print(__version__)


@app.command()
def doctor(
    ctx: typer.Context,
    config: Path | None = typer.Option(None, "--config", "-c"),
) -> None:
    """Check local backends, plugins, and extras."""

    settings = _settings(config or (ctx.obj or {}).get("config"))
    from unsynth.backends import is_available
    from unsynth.detectors.registry import DetectorRegistry
    from unsynth.rewriters.registry import RewriterRegistry

    table = Table(title="UnSynth doctor", show_header=True)
    table.add_column("check")
    table.add_column("status")
    table.add_column("detail")
    table.add_row("version", "[green]ok[/green]", __version__)
    table.add_row("backend.kind", "info", settings.backend.kind.value)
    table.add_row(
        "backend reachable",
        "[green]yes[/green]" if is_available(settings) else "[yellow]no[/yellow]",
        settings.backend.host
        if settings.backend.kind.value != "none"
        else "disabled (heuristic only)",
    )
    dets = DetectorRegistry(settings).available()
    rws = RewriterRegistry(settings).available()
    table.add_row("detectors", "[green]ok[/green]", ", ".join(dets))
    table.add_row("rewriters", "[green]ok[/green]", ", ".join(rws))
    try:
        import sentence_transformers  # noqa: F401

        table.add_row("embeddings", "[green]installed[/green]", "sentence-transformers")
    except Exception:
        table.add_row(
            "embeddings",
            "[yellow]fallback[/yellow]",
            "tfidf (install unsynth[embeddings])",
        )
    try:
        import PIL  # noqa: F401

        table.add_row("metadata extras", "[green]installed[/green]", "Pillow")
    except Exception:
        table.add_row("metadata extras", "[yellow]partial[/yellow]", "install unsynth[metadata]")
    console.print(table)
    console.print(
        Panel(
            "UnSynth never claims 100% undetectable output. Private-key "
            "watermarks can only be disrupted by changing tokens and context.",
            title="honesty",
            border_style="yellow",
        )
    )


@app.command()
def config_show(
    ctx: typer.Context,
    config: Path | None = typer.Option(None, "--config", "-c"),
) -> None:
    """Print the resolved configuration as JSON."""

    settings = _settings(config or (ctx.obj or {}).get("config"))
    out.print_json(settings.model_dump_json(indent=2))


@app.command()
def detect(
    ctx: typer.Context,
    path: Path | None = typer.Argument(None, help="Text/markdown file. Defaults to stdin."),
    config: Path | None = typer.Option(None, "--config", "-c"),
    json_out: bool = typer.Option(False, "--json", help="Machine-readable result."),
) -> None:
    """Score existing text with the detector ensemble."""

    settings = _settings(config or (ctx.obj or {}).get("config"))
    text = _read_input(path)
    pipeline = UnSynthPipeline(settings)
    result = pipeline.run(text, mode=PipelineMode.DETECT)
    if json_out:
        out.print_json(json.dumps(result.before.as_dict(), indent=2))
        return
    _print_score_table("UnSynth detect", result.before.as_dict())
    details = result.before.details
    if isinstance(details, dict) and "disclaimer" in str(details):
        pass
    console.print(
        f"[bold]ensemble[/bold] {result.before.score:.3f} "
        f"({result.before.label}, conf={result.before.confidence:.2f})"
    )


@app.command()
def rewrite(
    ctx: typer.Context,
    path: Path | None = typer.Argument(None),
    output: Path | None = typer.Option(None, "--output", "-o"),
    strength: float | None = typer.Option(None, "--strength", "-s", min=0.0, max=1.0),
    config: Path | None = typer.Option(None, "--config", "-c"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Score the planned rewrite only."),
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    """Rewrite once at the given strength. No adaptive loop."""

    settings = _settings(config or (ctx.obj or {}).get("config"))
    if strength is not None:
        settings.rewrite.initial_strength = strength
    text = _read_input(path)
    pipeline = UnSynthPipeline(settings)
    if dry_run:
        before = pipeline.detect(text)
        console.print(f"would rewrite {len(text)} chars; current score {before.score:.3f}")
        raise typer.Exit(0)
    result = pipeline.run(text, mode=PipelineMode.REWRITE)
    if json_out:
        payload = {"result": result.as_dict(), "text": result.output}
        out.print_json(json.dumps(payload, indent=2))
    elif result.after:
        _print_score_table("after one pass", result.after.as_dict())
    if output is not None or not json_out:
        _write_output(result.output, output)


@app.command()
def clean(
    ctx: typer.Context,
    path: Path | None = typer.Argument(None, help="Published article, blog post, notes…"),
    output: Path | None = typer.Option(None, "--output", "-o"),
    report: Path | None = typer.Option(None, "--report", "-r", help="Write a markdown report."),
    config: Path | None = typer.Option(None, "--config", "-c"),
    max_passes: int | None = typer.Option(None, "--max-passes", min=1, max=12),
    min_similarity: float | None = typer.Option(None, "--min-similarity", min=0.0, max=1.0),
    json_out: bool = typer.Option(False, "--json"),
    quiet: bool = typer.Option(False, "--quiet", "-q"),
) -> None:
    """Detect → rewrite → re-detect until targets are met or passes run out."""

    settings = _settings(config or (ctx.obj or {}).get("config"))
    if max_passes is not None:
        settings.rewrite.max_passes = max_passes
    if min_similarity is not None:
        settings.rewrite.min_similarity = min_similarity
    text = _read_input(path)
    pipeline = UnSynthPipeline(settings)

    def _progress(msg: str) -> None:
        if not quiet:
            console.print(f"[dim]{msg}[/dim]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
        disable=quiet,
    ) as progress:
        progress.add_task("cleaning", total=None)
        result = pipeline.run(text, mode=PipelineMode.CLEAN, progress=_progress)

    if json_out:
        payload = {"result": result.as_dict(), "text": result.output}
        out.print_json(json.dumps(payload, indent=2))
    else:
        if result.after:
            _print_score_table("after clean", result.after.as_dict())
        delta = result.before.score - (result.after.score if result.after else result.before.score)
        console.print(
            f"score {result.before.score:.3f} → "
            f"{(result.after.score if result.after else 0):.3f} "
            f"(Δ {delta:+.3f})  target_met={result.target_met}"
        )
    if output is not None or not json_out:
        _write_output(result.output, output)
    if report is not None:
        report.write_text(render_markdown_report(result), encoding="utf-8")
        console.print(f"[green]report[/green] {report}")


@app.command()
def strip(
    path: Path = typer.Argument(..., exists=True, readable=True),
    output: Path | None = typer.Option(None, "--output", "-o"),
    inplace: bool = typer.Option(False, "--inplace"),
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    """Strip C2PA / XMP / EXIF / HTML provenance from a file."""

    try:
        report = strip_path(path, output, inplace=inplace)
    except UnSynthError as exc:
        console.print(f"[red]strip failed:[/red] {exc}")
        raise typer.Exit(1) from exc
    if json_out:
        out.print_json(json.dumps(report.as_dict(), indent=2))
        return
    console.print(f"wrote {report.output}")
    if report.removed:
        console.print("removed: " + ", ".join(report.removed))
    if report.skipped:
        console.print("skipped: " + ", ".join(report.skipped))
    if report.warnings:
        for w in report.warnings:
            console.print(f"[yellow]{w}[/yellow]")


@app.command(name="eval")
def eval_cmd(
    ctx: typer.Context,
    original: Path = typer.Argument(..., exists=True, readable=True),
    rewritten: Path = typer.Argument(..., exists=True, readable=True),
    config: Path | None = typer.Option(None, "--config", "-c"),
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    """Compare two files with the full evaluation harness."""

    from unsynth.eval.report import compare_texts

    settings = _settings(config or (ctx.obj or {}).get("config"))
    report = compare_texts(
        original.read_text(encoding="utf-8", errors="replace"),
        rewritten.read_text(encoding="utf-8", errors="replace"),
        settings=settings,
    )
    if json_out:
        out.print_json(json.dumps(report.as_dict(), indent=2))
        return
    _print_score_table("before", report.before.as_dict())
    if report.after:
        _print_score_table("after", report.after.as_dict())
    if report.quality:
        console.print(
            f"similarity={report.quality.similarity:.3f} "
            f"change={report.quality.token_change_rate:.3f} "
            f"passed={report.quality.passed}"
        )


@app.command()
def batch(
    ctx: typer.Context,
    source: Path = typer.Argument(..., exists=True),
    dest: Path = typer.Option(..., "--output", "-o", help="Directory for cleaned files."),
    config: Path | None = typer.Option(None, "--config", "-c"),
    mode: PipelineMode = typer.Option(PipelineMode.CLEAN, "--mode"),
    suffix: str = typer.Option(".clean.md", "--suffix"),
) -> None:
    """Run detect/clean over a file or a directory tree."""

    settings = _settings(config or (ctx.obj or {}).get("config"))
    pipeline = UnSynthPipeline(settings)
    dest.mkdir(parents=True, exist_ok=True)
    files = list(iter_files([source]))
    if not files:
        console.print("[yellow]no files found[/yellow]")
        raise typer.Exit(1)
    for file in files:
        console.print(f"[bold]{file}[/bold]")
        text = file.read_text(encoding="utf-8", errors="replace")
        result = pipeline.run(text, mode=mode)
        rel = file.name if source.is_file() else str(file.relative_to(source))
        target = dest / (rel + suffix if source.is_file() else Path(rel).with_suffix(suffix))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(result.output, encoding="utf-8")
        after = result.after.score if result.after else result.before.score
        console.print(
            f"  {result.before.score:.3f} → {after:.3f}  target_met={result.target_met}  → {target}"
        )


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8765, "--port"),
    config: Path | None = typer.Option(None, "--config", "-c"),
) -> None:
    """Optional local web UI (requires unsynth[web])."""

    try:
        from unsynth.web.app import run_server
    except ImportError as exc:
        console.print("[red]web extra missing[/red]: pip install 'unsynth[web]'")
        raise typer.Exit(1) from exc
    settings = _settings(config)
    run_server(settings, host=host, port=port)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
