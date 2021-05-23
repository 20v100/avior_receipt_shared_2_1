""" Main module of Nova Bus DL Training """

import logging
from typing import Optional
import logging
import typer
import debugpy
from app.conversion import (
    convert_transformer_pytorch_to_tf,
    pytorch_max_pooling,
)


# ─── LOGGING SETUP ──────────────────────────────────────────────────────────────


class TyperLoggerHandler(logging.Handler):
    """ A custom logger handler that use Typer echo. """

    def emit(self, record: logging.LogRecord) -> None:
        typer.echo(
            typer.style(
                self.format(record),
                fg=typer.colors.BRIGHT_CYAN,
                bold=False,
            )
        )


formatter = logging.Formatter(
    "%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s"
)
handler = TyperLoggerHandler()
handler.setFormatter(formatter)
log: logging.Logger = logging.getLogger("spleeter")
log.addHandler(handler)
log.setLevel(logging.INFO)


# ─── TYPER STRUCTURE ────────────────────────────────────────────────────────────


__version__ = "0.1.1"


def version_callback(value: bool):
    if value:
        log.info(f"Nova Bus Computer Vision Trainer version: {__version__}")
        raise typer.Exit()


app_typer = typer.Typer()


@app_typer.command()
def admin(
    version: Optional[bool] = typer.Option(
        None, "--version", callback=version_callback
    ),
):
    """
    Assert that Typer app is working properly.
    """
    log.info(f"Nova Bus visual inspection app is workding")


# Concurency wrapper for asynch functions
# def coro(f):
#     @wraps(f)
#     def wrapper(*args, **kwargs):
#         return asyncio.run(f(*args, **kwargs))

#     return wrapper


@app_typer.command()
def convert(
    transformer_model_name: str = typer.Argument(
        ...,
        help="Transformer model name",
    ),
    saving_pathname: str = typer.Argument(
        ...,
        help="Path to save the TF model starting with /tf",
    ),
    model_version: str = typer.Argument(
        ...,
        help="Version of the model",
    ),
    debug: Optional[bool] = typer.Option(
        None,
        help="Wait for VS Code debug to attach to port 5768.",
    ),
):
    """Convert Transformer Pytorch models to Tensorflow
    ex.: docker-compose exec aux python /aux/main.py convert xlm-roberta-base /mutual/models/xml_roberta_base 000001
    """
    if debug:
        # debugpy.listen(("localhost", 5678))
        log.info(
            "Debug mode activated, plase launch a VS Code debugger that will attach to port 5678"
        )
        debugpy.listen(("0.0.0.0", 5678))
        debugpy.wait_for_user()

    log.info("Starting")
    convert_transformer_pytorch_to_tf(
        transformer_model_name, saving_pathname, model_version
    )


@app_typer.command()
def pytorch_max_pooling(
    debug: Optional[bool] = typer.Option(
        None,
        help="Wait for VS Code debug to attach to port 5768.",
    ),
):
    """Convert Transformer Pytorch models to Tensorflow
    ex.: docker-compose exec aux python aux/app/main.py pytorch_max_pooling --debug
    From: https://huggingface.co/sentence-transformers/msmarco-distilbert-base-v3
    """
    pytorch_max_pooling()


@app_typer.command()
def test(
    # transformer_model_name: str = typer.Argument(
    #     ...,
    #     help="Transformer model name",
    # ),
    # saving_path: str = typer.Argument(
    #     ...,
    #     help="Path to save the TF model starting with /tf",
    # ),
    debug: Optional[bool] = typer.Option(
        None,
        help="Wait for VS Code debug to attach to port 5768.",
    ),
):
    """Convert Transformer Pytorch models to Tensorflow
    ex.: docker-compose exec tf_training python tf_training/app/main.py convert --help msmarco-distilbert-base-v3
    """
    if debug:
        # debugpy.listen(("localhost", 5678))
        log.info(
            "Debug mode activated, plase launch a VS Code debugger that will attach to port 5678"
        )
        debugpy.listen(("0.0.0.0", 5678))
        debugpy.wait_for_user()

    log.info("Starting")
    mscarco_bi_encoder()


if __name__ == "__main__":
    app_typer()
