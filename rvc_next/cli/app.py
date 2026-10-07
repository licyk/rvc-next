"""Entry point: ``get_app()`` registers every command; ``main()`` runs it with uniform error handling."""

import sys
import traceback

import typer
from typer import Abort, Exit

from rvc_next.cli.commands.analyse import analyse
from rvc_next.cli.commands.assets import assets_download, assets_list, assets_verify
from rvc_next.cli.commands.config import config_get, config_path, config_set, config_show
from rvc_next.cli.commands.convert import convert
from rvc_next.cli.commands.effects import effects
from rvc_next.cli.commands.jobs import jobs_list, jobs_show
from rvc_next.cli.commands.live import live_check, live_devices, live_latency, live_run, live_test_tone
from rvc_next.cli.commands.model import (
    index_assign,
    index_attach,
    index_build,
    model_catalog,
    model_download,
    model_edit,
    model_export,
    model_extract,
    model_import,
    model_info,
    model_list,
    model_merge,
    model_remove,
)
from rvc_next.cli.commands.preset import preset_list, preset_remove, preset_save
from rvc_next.cli.commands.separate import separate
from rvc_next.cli.commands.system import doctor, env, version
from rvc_next.cli.commands.train import train_dataset_scan, train_dataset_speakers, train_export, train_import, train_list, train_new, train_run, train_status
from rvc_next.cli.commands.webui import webui
from rvc_next.cli.factory import ClickException, typer_factory
from rvc_next.logger import setup_logging

logger = setup_logging()


def get_app() -> typer.Typer:
    """Build the rvc-next command line. Every command is registered here, and nowhere else."""
    app = typer_factory("Retrieval-based voice conversion: convert, live, separate and train")

    app.command(help="Start the server and open the web UI", name="webui")(webui)
    app.command(help="Show the version of rvc-next and its main components", name="version")(version)
    app.command(help="List the environment variables rvc-next reads", name="env")(env)
    app.command(help="Check torch and its GPU backend, audio devices and the assets", name="doctor")(doctor)
    app.command(help="Convert audio files or folders with a voice", name="convert")(convert)
    app.command(help="List the effects convert and live run can add", name="effects")(effects)
    app.command(help="Separate vocals, accompaniment and reverb", name="separate")(separate)
    app.command(help="Show an audio file's pitch and format; write its pitch curve", name="analyse")(analyse)

    config_cli = typer_factory(help="Show and change settings")
    config_cli.command(help="Show the effective settings", name="show")(config_show)
    config_cli.command(help="Print one setting", name="get")(config_get)
    config_cli.command(help="Change one setting and save it", name="set")(config_set)
    config_cli.command(help="Print the path of the settings file", name="path")(config_path)
    app.add_typer(config_cli, name="config")

    assets_cli = typer_factory(help="HuBERT, RMVPE, base models and separation weights")
    assets_cli.command(help="Show what is installed", name="list")(assets_list)
    assets_cli.command(help="Download assets", name="download")(assets_download)
    assets_cli.command(help="Check installed assets against their checksums", name="verify")(assets_verify)
    app.add_typer(assets_cli, name="assets")

    model_cli = typer_factory(help="The voice library")
    model_cli.command(help="List voices", name="list")(model_list)
    model_cli.command(help="Show a voice", name="info")(model_info)
    model_cli.command(help="Import .pth, .zip or an RVC install", name="import")(model_import)
    model_cli.command(help="List the ready-made voices that can be downloaded", name="catalog")(model_catalog)
    model_cli.command(help="Download ready-made voices into the library", name="download")(model_download)
    model_cli.command(help="Rename a voice or name its speakers", name="edit")(model_edit)
    model_cli.command(help="Remove a voice", name="remove")(model_remove)
    model_cli.command(help="Write a voice and its indexes to a zip", name="export")(model_export)
    model_cli.command(help="Merge two voices", name="merge")(model_merge)
    model_cli.command(help="Extract a voice from a training checkpoint", name="extract")(model_extract)
    index_cli = typer_factory(help="Retrieval indexes")
    index_cli.command(help="Attach an index file", name="attach")(index_attach)
    index_cli.command(help="Attach an unassigned index to a voice", name="assign")(index_assign)
    index_cli.command(help="Build an index from an experiment", name="build")(index_build)
    model_cli.add_typer(index_cli, name="index")
    app.add_typer(model_cli, name="model")

    preset_cli = typer_factory(help="Saved voice parameters")
    preset_cli.command(help="List presets", name="list")(preset_list)
    preset_cli.command(help="Save a preset", name="save")(preset_save)
    preset_cli.command(help="Remove a preset", name="remove")(preset_remove)
    app.add_typer(preset_cli, name="preset")

    train_cli = typer_factory(help="Training experiments")
    train_cli.command(help="List experiments", name="list")(train_list)
    train_cli.command(help="Create an experiment", name="new")(train_new)
    dataset_cli = typer_factory(help="An experiment's dataset")
    dataset_cli.command(help="Check clips, duration and problems", name="scan")(train_dataset_scan)
    dataset_cli.command(help="Show or set the speaker table", name="speakers")(train_dataset_speakers)
    train_cli.add_typer(dataset_cli, name="dataset")
    train_cli.command(help="Run stages, or all of them", name="run")(train_run)
    train_cli.command(help="Show stages, checkpoints and the last metrics", name="status")(train_status)
    train_cli.command(help="Export a checkpoint as a library voice", name="export")(train_export)
    train_cli.command(help="Import an original RVC experiment folder", name="import")(train_import)
    app.add_typer(train_cli, name="train")

    live_cli = typer_factory(help="Live conversion with the server's audio devices")
    live_cli.command(help="List audio devices", name="devices")(live_devices)
    live_cli.command(help="Check the saved devices and format", name="check")(live_check)
    live_cli.command(help="Play a test sound on an output", name="test-tone")(live_test_tone)
    live_cli.command(help="Measure the real latency with a loopback", name="latency")(live_latency)
    live_cli.command(help="Convert live until Ctrl+C", name="run")(live_run)
    app.add_typer(live_cli, name="live")

    jobs_cli = typer_factory(help="Jobs of this machine, including a running server's")
    jobs_cli.command(help="List jobs", name="list")(jobs_list)
    jobs_cli.command(help="Show a job", name="show")(jobs_show)
    app.add_typer(jobs_cli, name="jobs")

    return app


def main() -> None:
    """Run the command line."""
    from rvc_next.core.errors import RvcNextError

    try:
        get_app()()
    except Exit as e:
        sys.exit(e.exit_code)
    except Abort:
        logger.error("Cancelled")
        sys.exit(1)
    except ClickException as e:
        e.show()
        sys.exit(e.exit_code)
    except RvcNextError as e:
        logger.error("%s", e.message)
        if e.detail.get("assets"):
            logger.error("Install with: rvc-next assets download %s", " ".join(e.detail["assets"]))
        sys.exit(e.exit_code)
    except KeyboardInterrupt:
        logger.error("Interrupted")
        sys.exit(130)
    except Exception as e:
        traceback.print_exc()
        logger.error("Command failed: %s", e)
        sys.exit(1)
