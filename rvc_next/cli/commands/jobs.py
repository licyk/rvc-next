"""jobs list | show."""

from typing import Annotated

import typer

from rvc_next.cli.output import console, open_services, print_json, print_table


def jobs_list(
    state: Annotated[str | None, typer.Option(help="A state, 'active' or 'finished'")] = None,
    limit: Annotated[int, typer.Option(min=1, max=200)] = 30,
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """List jobs, newest first, including those of a running server."""
    with open_services() as services:
        page = services.jobs.list_jobs(state=state, limit=limit)
        if json_output:
            print_json(page)
            return
        print_table(
            None,
            ["Id", "Kind", "State", "Progress", "Title", "Created"],
            [(j.id, j.kind, j.state, "" if j.progress is None else f"{j.progress:.0%}", j.title, j.created_at[:19]) for j in page.items],
        )


def jobs_show(
    job_id: Annotated[str, typer.Argument(help="Job id")],
    log: Annotated[bool, typer.Option("--log", help="Print the job's log")] = False,
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Show one job, and optionally its log."""
    with open_services() as services:
        job = services.jobs.get(job_id)
        if json_output:
            print_json(job)
        else:
            print_table(
                job.title,
                ["Field", "Value"],
                [
                    ("Id", job.id),
                    ("Kind", job.kind),
                    ("State", job.state),
                    ("Created", job.created_at),
                    ("Finished", job.finished_at or ""),
                    ("Error", job.error.message if job.error else ""),
                    *((f"Step {s.title}", s.state) for s in job.steps),
                ],
            )
        if log:
            offset = 0
            while True:
                chunk = services.jobs.read_log(job_id, offset)
                for line in chunk.lines:
                    console.print(line, markup=False, highlight=False)
                if chunk.next_offset == offset:
                    break
                offset = chunk.next_offset
