import pytest
from gpt.tokenizer.tokenizer import Tokenizer

@pytest.fixture
def tokenizer():
    return Tokenizer("test tokenizer")

def test_encode_decode(tokenizer: Tokenizer):
    text = "test tokenizer"
    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)
    assert decoded == text

def test_vocab_size(tokenizer: Tokenizer):
    expected_vocab_size = len(set("test tokenizer"))
    assert tokenizer.vocab_size == expected_vocab_size

def test_encode_known_chars(tokenizer: Tokenizer):
    encoded = tokenizer.encode("t k")
    assert all(isinstance(i, int) for i in encoded)
    assert len(encoded) == 3

def test_decode_known_ids(tokenizer: Tokenizer):
    encoded = tokenizer.encode("so")
    decoded = tokenizer.decode(encoded)
    assert decoded == "so"