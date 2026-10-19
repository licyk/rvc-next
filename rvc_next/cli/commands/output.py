"""output check."""

from typing import Annotated

import typer

from rvc_next.cli.output import console, open_services, print_json, print_table


def output_check(
    fix: Annotated[bool, typer.Option("--fix", help="Rewrite the records of files found elsewhere, and old absolute ones inside the data directory")] = False,
    prune_missing: Annotated[bool, typer.Option("--prune-missing", help="Delete the records of outputs and audio files found nowhere")] = False,
    rebase: Annotated[tuple[str, str] | None, typer.Option("--rebase", metavar="OLD NEW", help="Records under folder OLD point under NEW instead")] = None,
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Check that every output, its input and every uploaded or server audio file is where its record says (they are
    found again after the data directory moved); list the missing ones. Exits 1 when some are missing."""
    from rvc_next.core.storage import PathCheckRequest

    with open_services() as services:
        report = services.storage.check(PathCheckRequest(fix=fix, prune_missing=prune_missing, rebase_from=rebase[0] if rebase else None, rebase_to=rebase[1] if rebase else None))
        if json_output:
            print_json(report)
        else:
            found = "rewritten" if report.fixed else "--fix rewrites their records"
            console.print(f"Data directory: {report.data_dir}", highlight=False, markup=False)
            console.print(f"{report.relative} inside it, {report.external} elsewhere, {report.relocated} found away from their records ({found})", highlight=False)
            if report.rebased:
                console.print(f"{report.rebased} rebased", highlight=False)
            if report.missing:
                print_table(f"Missing ({len(report.missing)})", ["Kind", "Id", "Name", "Recorded at"], [(m.kind, m.id, m.name, m.recorded) for m in report.missing])
            if report.pruned:
                console.print(f"Deleted {report.pruned} records of missing files", highlight=False)
        if any(m.kind != "source" for m in report.missing) and not report.pruned:
            raise typer.Exit(1)
