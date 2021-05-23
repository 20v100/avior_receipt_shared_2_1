#
# ────────────────────────────────────────────────────────────────────────────────────────────────── I ──────────
#   :::::: P Y T H O N   L O G G I N G   C O N F I G U R A T I O N S : :  :   :    :     :        :          :
# ────────────────────────────────────────────────────────────────────────────────────────────────────────────
#

import logging
import typer


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
