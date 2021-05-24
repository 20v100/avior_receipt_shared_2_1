import io
import torch
from torchtext.datasets import WikiText2
from torchtext.data.utils import get_tokenizer
from collections import Counter
from torchtext.vocab import Vocab
from torch.utils.data import DataLoader

train_iter = WikiText2(split="train")
tokenizer = get_tokenizer("basic_english")
counter = Counter()
for line in train_iter:
    counter.update(tokenizer(line))
vocab = Vocab(counter)


import logging
import pathlib
import json

import pathlib

# import os
import random
from typing import List

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

import torch
from torch.utils.data.dataset import Dataset, IterableD
from app.reception.tokenizer import TechnicalTokenizer

log = logging.getLogger("spleeter")


def train_test_split():
    # todo: create train and test set
    pass


class TechnicalNLPDataset(Dataset):
    """Create Pytorch dataset for AV&R detection project."""

    def __init__(
        self,
        doc_folder_paths: List[str],
        filename_glob_pattern: str = "*.txt",
        detail: bool = False,
    ):
        self.doc_folder_paths = doc_folder_paths
        self.filename_glob_pattern = filename_glob_pattern
        self.detail = detail
        self.doc_filepaths = []
        self.tok = TechnicalTokenizer()

    def _get_doc_filepaths(self):
        for doc_folder_path in self.doc_folder_paths:
            doc_folder_path_pl = pathlib.Path(doc_folder_path)
            doc_filepath_ls = [
                str(path.resolve())
                for path in doc_folder_path_pl.glob(self.filename_glob_pattern)
            ]
            for filepath in doc_filepath_ls:
                self.doc_filepaths.append(filepath)

    def transform(self):
        # todo: associate each tokens to a label
        # todo: create a tensor for absolute positionning
        return ["we", "are"]

    def __getitem__(self, index):
        # Select y at random
        # todo: Need to handle padding and spliting documents that are beyond the limit.
        with open(self.doc_filepaths[index], "r") as f:
            doc = f.read()
        token_ls = self.tok.tokenize(doc)
        token_ts = torch.tensor(token_ls)
        return token_ts

    def __len__(self):
        return len(self.doc_filepaths)

        train_dataloader = DataLoader(
            avr_image_dataset,
            batch_size=settings.batch,
            shuffle=True,
            num_workers=0,
        )


train_data = TechnicalNLPDataset()
test_data = TechnicalNLPDataset()


def batching(data_batch):
    pass


train_dataloader = DataLoader(
    train_data, batch_size=64, shuffle=True, collate_fn=batching
)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)
