"""``rvc-next effects``: the effects ``--effect`` takes."""

from __future__ import annotations

from typing import Annotated

import typer

from rvc_next.cli.output import console, err_console, print_json


def effects(json_output: Annotated[bool, typer.Option("--json", help="Print the catalog as JSON")] = False) -> None:
    """List the effects with their parameters' defaults and ranges, and whether pedalboard is installed."""
    from rvc_next.core.conversion.service import ConversionService

    catalog = ConversionService.effects_catalog()
    if json_output:
        print_json(catalog)
        return
    for spec in catalog.effects:
        params = ", ".join(f"{p.name}={p.default:g} ({p.min:g}..{p.max:g}{' ' + p.unit if p.unit else ''})" for p in spec.params)
        console.print(f"[bold]{spec.kind}[/bold]  {params}", highlight=False)
    if not catalog.available:
        err_console.print("[yellow]pedalboard is not installed: pip install rvc-next[effects][/yellow]")
