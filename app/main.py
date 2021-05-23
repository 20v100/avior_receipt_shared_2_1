#
# ────────────────────────────────────────────────────────────────────────────────── I ──────────
#   :::::: M A I N   A V I O R   N L P   M O D U L E : :  :   :    :     :        :          :
# ────────────────────────────────────────────────────────────────────────────────────────────
#

import logging
from typing import Optional
import logging
import typer
import admin.logger  # Do not delete as it instance the logging utility

import debugpy

from invoice_train.et import create_ner_dataset
from invoice_train.train import train_ner, save_model, post_to_server

# from invoice_train.predict import test_ner
from settings import settings


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
def invoice(
    dataset: Optional[bool] = typer.Option(
        None,
        help="Process Tesseract TSV files, Labels from the Avior ERP in a dataset ready for NER fine-tuning.",
    ),
    train: Optional[str] = typer.Option(
        None,
        help="Load ner_dataset_01.tfrecord dataset and perform deep learning NER training. Required a name for the trained model. Ex.: python /app/main.py invoice --train beta_1_model --detail",
    ),
    save: Optional[str] = typer.Option(
        None,
        help="Save latest trained model for TensorFlow serving. Model version is required.",
    ),
    post: Optional[bool] = typer.Option(
        False,
        help="Post a invoice request to server.",
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
    ex.: docker-compose exec app python /app/main.py invoice --help
    """

    log.info(f"Starting Avior Invoice Training App.")

    if debug:
        log.info(
            "Debug mode activated, please launch a VS Code debugger that will attach to port 5678."
        )
        debugpy.listen(("0.0.0.0", 5678))
        debugpy.wait_for_client()

    log.info("Starting Avoir NLP invoice module.")

    if dataset:
        create_ner_dataset(detail)
        log.info("Completed the --dataset task.")

    if train:
        train_ner(train, detail)
        log.info(
            f"Completed --train task. Saved results in: ${settings.invoice_transformer_ner_model_path}"
        )

    if save:
        save_model(save_model, detail)
        log.info(f"Saved model to {settings.invoice_model_path} under {save_model}")

    if post:
        post_to_server(detail)
        log.info(f"Posted to server")


if __name__ == "__main__":
    app_typer()
