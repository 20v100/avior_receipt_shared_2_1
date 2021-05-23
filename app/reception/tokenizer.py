import enum
# import json
import pickle
# import operator
from collections import defaultdict
import token
from typing import Dict, List, 
from torchtext.data.utils import get_tokenizer
from collection
from pydantic import BaseModel
import logging

log = logging.getLogger("spleeter")

class Token(BaseModel):
    key: str
    id: int
    string: str
    char_start: int
    char_stop: int


class VocabKindEnum(str, enum):
    NUMERIC = "NB"
    SEQUENCE = "SQ"
    WORD = "WD"
    MARKER = "ML"


class MarkerKindEnum(str, enum):
    NUMERIC_START = "<nU;>" 
    NUMERIC_STOP = "<;nU>" 
    NUMERIC_NONE = "<nUX>"
    SEQUENCE_START = "<sQ;>" 
    SEQUENCE_STOP = "<;sQ>" 
    SEQUENCE_NONE = "<sQX>"
    UPPERCASE_START = "<uP;>"
    UPPERCASE_STOP = "<;uP>"
    TITLE_CASE_START = "<tC;>"
    TITLE_CASE_STOP = "<;tC>"
    ALPHA_NONE = "<aLX>"


# class VocabItem(BaseModel):
#     token_key: str
#     token_id: int
#     string: str
#     kind: VocabKindEnum


# class Vocab():
#     def __init__():


class TechnicalTokenizer(vocab_file_path="mutual/data/tokenizer"):
    """
    Customer tokenizer adapted implementing latest numeracy technics.
    """
    def __init__(self):
        self.tokens = []
        # self.doc = str(sentence)
        self.vocab_filepath: str = vocab_file_path
        self._split_doc_re = re.compile(r"(\.\s)|(\,\s)|(\:\s)|(\-\s)|(\_\s)|(\@\s)|[\s\$\:\;\-\?\!\|\%\&\#\*\(\)\[\]\{\}\"\'\`\/\\\*]")
        self._is_numeric_re = re.compile(r"(^[1-9]+$)|(^[1-9]+[\.\,][1-9]+$)")
        self._is_sequencial_re = re.compile(r"[1-9\.\,\-\_\@\:]")
        self._load_vocab()


    def _load_vocab(self):
        with open(corpus_path, 'r') as f:
            self.vocab = pickle.load(f)


    def _tokenize_numeric(self, word, start_pos, stop_pos) -> List[Token]:
        token_ls = []
        start_nbr_token = self.vocab.git(MarkerKindEnum.NUMERIC_START)
        token_ls.append(Token(token_id=start_nbr_token.token_id, string=start_nbr_token.string, start_pos=start_pos, stop_pos=(start_pos + 1)))

        for idx, char in enumerate(word):
            vocab_item = self.vocab.git(f"NB-{char}", default=MarkerKindEnum.NONE)
            token_ls.append(Token(token_id=vocab_item.token_id, string=vocab_item.string, start_pos=(start_pos + idx), stop_pos=(stop_pos + idx + 1)))

        stop_nbr_token = self._get_token(MarkerKindEnum.NUMERIC_END)
        token_ls.append(Token(token_id=stop_nbr_token.token_id, string=stop_nbr_token.string, start_pos=stop_pos, stop_pos=(stop_pos + 1)))
        return token_ls


    # def _tokenize_sequence(self, word) -> List[Token]:
    #     for char in word:
    #         vocab_item = self._get_token(f"SQ{char}")
    #         return Token(token_id=vocab_item.token_id, string=vocab_item.string, start_pos=start_pos, stop_pos=stop_pos)


    # def _tokenize_alpha(self, word, build_vocab=False) -> List[Token]:
    #     vocab_item = self._get_token(f"WD-{word}")
    #     if word.isupper():

    #     elif word.istile():
        
    #     return Token(token_id=vocab_item.token_id, string=vocab_item.string, start_pos=start_pos, stop_pos=stop_pos)


    def select_tokenizer(self, doc):
        space_pos = self._split_doc_re.search(self.doc)
        for idx in ennumerate(len(space_pos)-1):
            word = self.doc[space_pos[idx], space_pos[idx+1]]
            if self._is_numeric_re.search(word, space_pos[idx], space_pos[idx+1]):
                token = self._tokenize_numeric(word)
            elif self._is_sequential_re.search(word, space_pos[idx], space_pos[idx+1]):
                token = self._is_tokenize_sequence(word)
            else:
                token = self._is_tokenize_alpha(word)
            yield token


    def tokenize(self, doc: str):
        self.doc = doc
        select_tokenizer(doc = doc)


    def token_to_char(self):
        pass


    def char_to_token(self):
        pass


    @parameters
    def token_ids(self):
        pass


    def create_vocab(self, corpus_filepath: str, min_occurence: int = 2):
        candidate_token_dc = defaultdict(int)
        max_key = max(self.vocab, key=self.vocab.get)
        counter = 0
        
        for marker in reversed(MarkerKindEnum):
            if marker not in self.vocab:
                candidate_token_dc[f"{VocabKindEnum.MARKER}:{marker}"] += 9000000
        
        with open(corpus_filepath, 'r') as f:
            for line in f:
                for token_ls in self.select_tokenizer(line):
                    for token in token_ls:
                        if re.find(f"{VocabKindEnum.MARKER}:{MarkerKindEnum.NUMERIC_NONE}|{VocabKindEnum.MARKER}:{MarkerKindEnum.SEQUENCE_NONE}|{VocabKindEnum.MARKER}:{MarkerKindEnum.ALPHA_NONE}", token_item):
                            candidate_token_dc[token.key] += 1

        for marker in MarkerKindEnum:
            if marker not in self.vocab:
                candidate_token_dc[f"{VocabKindEnum.MARKER}:{marker}"] += 1

        for key, val in sorted(candidate_token_dc.items(), key=lambda item: item[1], reverse=True):
            if val >= min_occurence:
                max_key += 1
                counter += 1
                self.vocab[key] = max_key

        with open(self.vocabfilepath, 'w') as f:
            pickle.dump(self.vocab, f, protocol=pickle.HIGHEST_PROTOCOL)
            log.info(f"Saved {counter} new tokens into the vocab. The file has now a total of {max_key} tokens.")

                






    def __iter__(self):
        return self

    def __next__(self):
        if self._index < len(self.tokens):
            result = self.tokens[self._index]
            self._index+=1
            return result
        raise StopIteration