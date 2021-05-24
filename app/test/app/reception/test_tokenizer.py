import pytest
import os
from app.reception.tokenizer import TechnicalTokenizer
import pdb


def test_create_vocab():
    vocab_filepath = "/app/test/app/reception/test_vocab.json"
    if os.path.isfile(vocab_filepath):
        os.remove(vocab_filepath)
    tok = TechnicalTokenizer(vocab_filepath)
    tok.create_vocab("/app/test/app/reception/test_corpus/*.txt", min_occurence=0)
    assert tok.vocab["AP_digit"] == 19

    tok2 = TechnicalTokenizer(vocab_filepath)
    tok2.tokenize("digit 51")
    assert tok2.token_ids == [19, 1, 14, 3, 2]

    tok3 = TechnicalTokenizer(vocab_filepath)
    tok3.tokenize("hello 51,44")
    assert tok3.token_ids == [9, 1, 14, 3, 21, 16, 16, 2]

    tok4 = TechnicalTokenizer(vocab_filepath)
    tok4.tokenize("A69: 54")
    assert tok4.token_ids == [4, 7, 24, 8, 17, 6, 5, 20, 1, 14, 16, 2]

    tok5 = TechnicalTokenizer(vocab_filepath)
    tok5.tokenize("Digit DIGIT")
    # pdb.set_trace()
    assert tok5.token_ids == [12, 19, 13, 10, 19, 11]


# def test_tokenize():
#     if os.path.isfile(vocab_filepath):
#         os.remove(vocab_filepath)
#     tok = TechnicalTokenizer(vocab_filepath)
#     tok.create_vocab("/app/test/app/reception/test_corpus/*.txt")

#     vocab_filepath = "/app/test/app/reception/test_vocab.json"

#     assert tok2.tokens[0].key == "dd"
#     assert tokens[15] == ""


def test_tokens():
    vocab_filepath = "/app/test/app/reception/test_vocab.json"
    tok = TechnicalTokenizer(vocab_filepath)
    tok.tokenize("/app/test/app/reception/test_corpus/*.txt")

    assert tok.tokens[0].key == "dd"
