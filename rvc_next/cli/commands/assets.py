"""assets list | download | verify."""

from typing import Annotated

import typer

from rvc_next.cli.output import human_size, open_services, print_json, print_table
from rvc_next.cli.progress import finish, job_progress

Group = typer.Option(
    help="inference (HuBERT and the pitch models), embedder (Applio's content-feature models), training, separation, community (base models) or all (all but embedder and community)"
)


def assets_list(json_output: Annotated[bool, typer.Option("--json", help="Print JSON")] = False) -> None:
    """Show each asset: installed, missing or corrupt, and its size."""
    with open_services() as services:
        items = services.assets.list_assets()
        if json_output:
            print_json(items)
            return
        print_table(
            f"Assets in {services.assets.root}",
            ["Id", "Title", "Group", "Size", "State"],
            [(a.id, a.title, a.group, human_size(a.size), a.state + (" ✓" if a.verified else "")) for a in items],
        )
        repo = next(r for r in services.assets.repositories() if r.selected)
        d = services.settings.settings.downloads
        where = f"{repo.modelscope} on ModelScope" if d.source == "modelscope" and repo.modelscope else f"{repo.repo} ({d.source})"
        typer.echo(f"Downloads come from {where}; change with --repository and --source, or downloads.repository and downloads.source")


def assets_download(
    ids: Annotated[list[str] | None, typer.Argument(help="Asset ids; see 'rvc-next assets list'")] = None,
    group: Annotated[str | None, Group] = None,
    source: Annotated[str | None, typer.Option(help="Source: huggingface, hf-mirror, modelscope or custom (saved to settings)")] = None,
    repository: Annotated[str | None, typer.Option(help="rvc-model or official (saved to settings)")] = None,
) -> None:
    """Download assets, resuming partial files and checking their checksums."""
    with open_services() as services:
        if source:
            services.settings.update({"downloads": {"source": source}})
        if repository:
            services.settings.update({"downloads": {"repository": repository}})
        if not ids and not group:
            group = "inference"
        with job_progress(services):
            job = services.assets.download(ids or [], group, foreground=True)
        finish(job)
        typer.echo(f"Installed in {services.assets.root}")


def assets_verify(
    ids: Annotated[list[str] | None, typer.Argument(help="Asset ids")] = None,
    group: Annotated[str | None, Group] = None,
) -> None:
    """Check installed assets against their checksums."""
    with open_services() as services:
        chosen = ids or [a.id for a in services.assets.list_assets() if a.state in ("installed", "corrupt")]
        if not chosen and not group:
            typer.echo("Nothing installed to verify")
            return
        with job_progress(services):
            job = services.assets.download(chosen, group, verify_only=True, foreground=True)
        finish(job)
        typer.echo("All checksums match")
