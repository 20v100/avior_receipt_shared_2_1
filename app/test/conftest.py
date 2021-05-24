import pytest
import debugpy
import logging
from psutil import process_iter
from signal import SIGKILL

log = logging.getLogger("spleeter")


# Tying to VS Code debug
def pytest_addoption(parser):
    parser.addoption("--db", action="store_true", help="run all combinations")


# Tying to VS Code debug
def pytest_generate_tests(metafunc):
    if metafunc.config.getoption("db"):
        # try:
        log.info("Pls launch debugger no port 5678!")
        debugpy.listen(("0.0.0.0", 5678))
        debugpy.wait_for_client()
        # except:
        #     for proc in process_iter():
        #         for conns in proc.get_connections(kind="inet"):
        #             if conns.laddr[1] == 1300:
        #                 proc.send_signal(SIGKILL)
        #                 continue


# if debug:
#     log.info(
#         "Debug mode activated, please launch a VS Code debugger that will attach to port 5678."
#     )
#     debugpy.listen(("0.0.0.0", 5678))
#     debugpy.wait_for_client()
