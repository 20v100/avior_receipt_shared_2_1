#
# ────────────────────────────────────────────────────────────────────────────────── I ──────────
#   :::::: M A I N   A V I O R   N L P   M O D U L E : :  :   :    :     :        :          :
# ────────────────────────────────────────────────────────────────────────────────────────────
#

import logging
from typing import Optional
import logging
import typer

import debugpy

# from invoice_train.predict import test_ner
from settings import settings
from modeling.train import train_modeling


# ─── CONFIG ─────────────────────────────────────────────────────────────────────

log = logging.getLogger("spleeter")


# ─── TYPER STRUCTURE ────────────────────────────────────────────────────────────

__version__ = "0.1.1"


def version_callback(value: bool):
    if value:
        log.info(f"Avior Computer Vision Trainer version: {__version__}")
        raise typer.Exit()


app_typer = typer.Typer()


@app_typer.command()
def system(
    version: Optional[bool] = typer.Option(
        None, "--version", callback=version_callback
    ),
):
    """
    Assert that Typer app is working properly.
    """
    log.info("Avior visual inspection app is working.")


@app_typer.command()
def nlp(
    vocab: Optional[str] = typer.Option(
        None,
        help="Create a Vocabulary file for the tokenizer",
    ),
    dataset: Optional[bool] = typer.Option(
        None,
        help="Process process into a dataset.",
    ),
    modeling: Optional[bool] = typer.Option(
        False,
        help="Save train_modeling",
    ),
    save: Optional[bool] = typer.Option(
        False,
        help="Save ...",
    ),
    detail: Optional[bool] = typer.Option(
        False,
        help="Produce a more detailled log. Note: Warlock logs are in blue color.",
    ),
    debug: Optional[bool] = typer.Option(
        None,
        help="Wait for VS Code debug to attach to port 5768. See debugging with VS code for more information: https://code.visualstudio.com/docs/editor/debugging",
    ),
):
    """
    Educational deep learning NLP training for Avior invoice project.
    ex.: docker-compose exec app python /app/main.py nlp --help
    """

    log.info(f"Starting Avior Invoice Training App.")

    if debug:
        log.info(
            "Debug mode activated, please launch a VS Code debugger that will attach to port 5678."
        )
        debugpy.listen(("0.0.0.0", 5678))
        debugpy.wait_for_client()

    log.info("Starting Avoir NLP invoice module.")

    if vocab:
        log.info("Completed the --vocab task.")

    if dataset:

        log.info("Completed the --dataset task.")

    if modeling:
        # train_modeling()
        log.info(
            f"Completed --train task. Saved results in: ${settings.invoice_transformer_ner_model_path}"
        )
    if save:

        log.info(f"Saved model to {settings.invoice_model_path} under ")


if __name__ == "__main__":
    app_typer()
