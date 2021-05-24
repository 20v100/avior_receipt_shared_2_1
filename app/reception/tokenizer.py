from curses.ascii import isupper
from enum import Enum
import re
import os
import pdb
import json

# import pickle
import glob

from collections import defaultdict
from typing import Dict, List
from torchtext.data.utils import get_tokenizer
from pydantic import BaseModel
import logging

log = logging.getLogger("spleeter")


class Token(BaseModel):
    key: str
    id: int
    # string: str
    start: int
    stop: int


class VocabKindEnum(str, Enum):
    NUMERIC = "NB"
    SEQUENCE = "SQ"
    ALPHA = "AP"
    MARKER = "MK"


class MarkerKindEnum(str, Enum):
    NUMERIC_START = "<NU->"
    NUMERIC_STOP = "<-NU>"
    NUMERIC_NONE = "<NUx>"
    SEQUENCE_START = "<SQ->"
    SEQUENCE_STOP = "<-SQ>"
    SEQUENCE_NONE = "<SQx>"
    SEQUENCE_UPPERCASE_START = "<SU->"
    SEQUENCE_UPPERCASE_STOP = "<-SU>"
    ALPHA_NONE = "<ALx>"
    ALPHA_UPPERCASE_START = "<AU->"
    ALPHA_UPPERCASE_STOP = "<-AU>"
    TITLE_CASE_START = "<TC->"
    TITLE_CASE_STOP = "<-TC>"


# class VocabItem(BaseModel):
#     token_key: str
#     id: int
#     string: str
#     kind: VocabKindEnum


# class Vocab():
#     def __init__():


class TechnicalTokenizer:
    """
    Customer tokenizer adapted implementing latest numeracy technics.
    """

    def __init__(self, vocab_filepath="/mutual/data/tokenizer/vocab.pkl"):
        self.tokens = []
        # self.doc = str(sentence)
        self.vocab_filepath: str = vocab_filepath
        self._split_doc_re = re.compile(
            r"\.\s|\,\s|\:\s|\-\s|\_\s|\@\s|\.$|\,$|\:$|\-$|\_$|\@$|[\s\$\;\?\!\|\%\&\#\*\(\)\[\]\{\}\"\'\`\/\\]"
        )
        self._is_numeric_re = re.compile(r"^[1-9]+$|^[1-9]+[\.\,][1-9]+$")
        self._is_sequencial_re = re.compile(r"[1-9\.\,\-\_\@\:]")
        self._load_vocab()

    def _load_vocab(self):
        if os.path.isfile(self.vocab_filepath):
            with open(self.vocab_filepath, "r") as f:
                self.vocab = json.loads(f.read())
        else:
            self.vocab = {}
            log.info("Could not find a vocab file!")

    def _tokenize_numeric(self, word, start_pos, stop_pos) -> List[Token]:
        token_ls = []
        key_start = f"{VocabKindEnum.MARKER}_{MarkerKindEnum.NUMERIC_START}"
        numeric_marker_start_id = self.vocab.get(key_start)
        token_ls.append(
            Token(
                key=key_start,
                id=numeric_marker_start_id,
                start=start_pos,
                stop=(start_pos + 1),
            )
        )

        for idx, digit in enumerate(word):
            if digit == " ":
                continue
            key = f"{VocabKindEnum.NUMERIC}_{str(digit).lower()}"
            vocab_id = self.vocab.get(key, None)
            if not vocab_id:
                key = f"{VocabKindEnum.MARKER}_{MarkerKindEnum.NUMERIC_NONE}"
                vocab_id = self.vocab.get(key)

            token_ls.append(
                Token(
                    key=key,
                    id=vocab_id,
                    start=(start_pos + idx),
                    stop=(start_pos + idx + 1),
                )
            )

        key_end = f"{VocabKindEnum.MARKER}_{MarkerKindEnum.NUMERIC_STOP}"
        numeric_marker_stop_id = self.vocab.get(key_end)
        token_ls.append(
            Token(
                key=key_end,
                id=numeric_marker_stop_id,
                start=stop_pos - 1,
                stop=stop_pos,
            )
        )
        return token_ls

    def _tokenize_sequence(self, sequence, start_pos, stop_pos) -> List[Token]:
        token_ls = []
        key_start = f"{VocabKindEnum.MARKER}_{MarkerKindEnum.SEQUENCE_START}"
        numeric_marker_start_id = self.vocab.get(key_start)
        token_ls.append(
            Token(
                key=key_start,
                id=numeric_marker_start_id,
                start=start_pos,
                stop=(start_pos + 1),
            )
        )

        for idx, seq in enumerate(sequence):
            is_uppercase = False
            if seq.isupper():
                is_uppercase = True
                key_upper = (
                    f"{VocabKindEnum.MARKER}_{MarkerKindEnum.SEQUENCE_UPPERCASE_START}"
                )
                vocab_id = self.vocab.get(key_upper)
                token_ls.append(
                    Token(
                        key=key_upper,
                        id=vocab_id,
                        start=(start_pos + idx),
                        stop=(start_pos + idx + 1),
                    )
                )
            if seq == " ":
                continue
            key = f"{VocabKindEnum.SEQUENCE}_{str(seq).lower()}"
            vocab_id = self.vocab.get(key, None)
            if not vocab_id:
                key = f"{VocabKindEnum.MARKER}_{MarkerKindEnum.SEQUENCE_NONE}"
                vocab_id = self.vocab.get(key)
            token_ls.append(
                Token(
                    key=key,
                    id=vocab_id,
                    start=(start_pos + idx),
                    stop=(start_pos + idx + 1),
                )
            )
            if is_uppercase:
                key_upper = (
                    f"{VocabKindEnum.MARKER}_{MarkerKindEnum.SEQUENCE_UPPERCASE_STOP}"
                )
                vocab_id = self.vocab.get(key_upper)
                token_ls.append(
                    Token(
                        key=key_upper,
                        id=vocab_id,
                        start=(start_pos + idx),
                        stop=(start_pos + idx + 1),
                    )
                )

        key_end = f"{VocabKindEnum.MARKER}_{MarkerKindEnum.SEQUENCE_STOP}"
        numeric_marker_stop_id = self.vocab.get(key_end)
        token_ls.append(
            Token(
                key=key_end,
                id=numeric_marker_stop_id,
                start=stop_pos - 1,
                stop=(stop_pos),
            )
        )
        return token_ls

    def _tokenize_alpha(self, word, start_pos, stop_pos) -> List[Token]:
        token_ls = []
        is_uppercase = False
        is_titlecase = False
        if word.isupper():
            is_uppercase = True
            key_upper = f"{VocabKindEnum.MARKER}_{MarkerKindEnum.ALPHA_UPPERCASE_START}"
            vocab_id = self.vocab.get(key_upper)
            token_ls.append(
                Token(
                    key=key_upper,
                    id=vocab_id,
                    start=(start_pos),
                    stop=(start_pos + 1),
                )
            )
        elif word.istitle():
            is_titlecase = True
            key_upper = f"{VocabKindEnum.MARKER}_{MarkerKindEnum.TITLE_CASE_START}"
            vocab_id = self.vocab.get(key_upper)
            token_ls.append(
                Token(
                    key=key_upper,
                    id=vocab_id,
                    start=start_pos,
                    stop=(start_pos + 1),
                )
            )

        key = f"{VocabKindEnum.ALPHA}_{str(word).lower().strip()}"
        vocab_id = self.vocab.get(key, None)
        if not vocab_id:
            key = f"{VocabKindEnum.MARKER}_{MarkerKindEnum.ALPHA_NONE}"
            vocab_id = self.vocab.get(key)
        # pdb.set_trace()
        token_ls.append(
            Token(
                key=key,
                id=vocab_id,
                start=start_pos,
                stop=stop_pos,
            )
        )

        if is_titlecase:
            key_upper = f"{VocabKindEnum.MARKER}_{MarkerKindEnum.TITLE_CASE_STOP}"
            vocab_id = self.vocab.get(key_upper)
            token_ls.append(
                Token(
                    key=key_upper,
                    id=vocab_id,
                    start=(stop_pos - 1),
                    stop=stop_pos,
                )
            )
        elif is_uppercase:
            key_upper = f"{VocabKindEnum.MARKER}_{MarkerKindEnum.ALPHA_UPPERCASE_STOP}"
            vocab_id = self.vocab.get(key_upper)
            token_ls.append(
                Token(
                    key=key_upper,
                    id=vocab_id,
                    start=(stop_pos - 1),
                    stop=stop_pos,
                )
            )
        # key_end = f"{VocabKindEnum.MARKER}_{MarkerKindEnum.SEQUENCE_STOP}"
        # numeric_marker_stop_id = self.vocab.get(key_end)
        # token_ls.append(
        #     Token(
        #         key=key_end,
        #         id=numeric_marker_stop_id,
        #         start=stop_pos,
        #         stop=(stop_pos + 1),
        #     )
        # )
        return token_ls

    def _select_tokenizer(self, doc):
        self.doc = doc
        split_pos = []
        for m in self._split_doc_re.finditer(self.doc):
            split_pos.append(m.span()[0])
            split_pos.append(m.span()[1])
        if split_pos[0] != 0:
            split_pos = [0] + split_pos
        if split_pos[-1] != len(self.doc):
            split_pos.append(len(self.doc))

        for idx in range(len(split_pos) - 1):
            word = self.doc[split_pos[idx] : split_pos[idx + 1]]
            word_formated = word.strip().lower()
            if self._is_numeric_re.search(word_formated):
                tokens = self._tokenize_numeric(
                    word, split_pos[idx], split_pos[idx + 1]
                )
            elif (
                self._is_sequencial_re.search(word_formated) and len(word_formated) > 1
            ):
                tokens = self._tokenize_sequence(
                    word, split_pos[idx], split_pos[idx + 1]
                )
            else:
                tokens = self._tokenize_alpha(word, split_pos[idx], split_pos[idx + 1])
            for token in tokens:
                if word_formated == "":
                    continue
                yield token

    def tokenize(self, doc: str):
        for token in self._select_tokenizer(doc=doc):
            self.tokens.append(token)

    def token_to_char(self, token_idx):
        return self.tokens[token_idx].dict()

    def char_to_token(self, char_idx):
        for token in self.tokens:
            if token.key[:3] != "ML":
                if token.token_start >= char_idx and token.token_stop <= char_idx:
                    return token.dict()

    @property
    def token_ids(self):
        ls = []
        for token in self.tokens:
            ls.append(token.id)
        return ls

    def create_vocab(self, corpus_folder_path: str, min_occurence: int = 2):
        candidate_token_dc = defaultdict(int)
        max_key = max(self.vocab, default=0, key=self.vocab.get)
        counter = 0

        if os.path.isfile(self.vocab_filepath):
            with open(self.vocab_filepath, "r") as f:
                self.vocab = json.loads(f.read())
                log.info(
                    f"Saved {counter} new tokens into the vocab. The file has now a total of {max_key} tokens."
                )
        else:
            log.info("Creating a new vocab file!")

        for marker in MarkerKindEnum:
            if marker not in self.vocab:
                # candidate_token_dc[f"{VocabKindEnum.MARKER}:{marker}"] += 9000000
                max_key += 1
                self.vocab[f"{VocabKindEnum.MARKER}_{marker}"] = max_key

        for filepath in glob.glob(corpus_folder_path):
            with open(filepath, "r") as f:
                for line in f:
                    for token in self._select_tokenizer(line):
                        if re.search(
                            f"{VocabKindEnum.MARKER}_{MarkerKindEnum.NUMERIC_NONE}",
                            token.key,
                        ):
                            candidate_token_dc[
                                f"{VocabKindEnum.NUMERIC}_{self.doc[token.start:token.stop].strip().lower()}"
                            ] += 1
                        elif re.search(
                            f"{VocabKindEnum.MARKER}_{MarkerKindEnum.SEQUENCE_NONE}",
                            token.key,
                        ):
                            candidate_token_dc[
                                f"{VocabKindEnum.SEQUENCE}_{self.doc[token.start:token.stop].strip().lower()}"
                            ] += 1
                        elif re.search(
                            f"{VocabKindEnum.MARKER}_{MarkerKindEnum.ALPHA_NONE}",
                            token.key,
                        ):
                            # pdb.set_trace()
                            candidate_token_dc[
                                f"{VocabKindEnum.ALPHA}_{self.doc[token.start:token.stop].strip().lower()}"
                            ] += 1

        # for marker in MarkerKindEnum:
        #     if marker not in self.vocab:
        #         candidate_token_dc[f"{VocabKindEnum.MARKER}_{marker}"] += 1

        for key, val in sorted(
            candidate_token_dc.items(), key=lambda item: item[1], reverse=True
        ):
            if val >= min_occurence:
                max_key += 1
                counter += 1
                self.vocab[key] = max_key

        with open(self.vocab_filepath, "w") as f:
            # pdb.set_trace()
            json.dump(self.vocab, f, indent=4)
            log.info(
                f"Saved {counter} new tokens into the vocab. The file has now a total of {max_key} tokens."
            )

    # def __iter__(self):
    #     return self

    # def __next__(self):
    #     if self._index < len(self.tokens):
    #         result = self.tokens[self._index]
    #         self._index+=1
    #         return result
    #     raise StopIteration
