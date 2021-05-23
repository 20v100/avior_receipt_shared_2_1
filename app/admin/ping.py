import sys
import tensorflow as tf

from fastapi import APIRouter, Depends


router = APIRouter()


@router.get("/ping", summary="Facility to test connection")
async def pong():
    return {
        "Python version": f"{sys.version_info[0]}.{sys.version_info[1]}.{sys.version_info[2]}",
        "Tensorflow version": tf.__version__,
    }