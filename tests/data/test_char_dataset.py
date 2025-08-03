import pytest
from gpt.data.char_dataset import CharDataset
from gpt.tokenizer.tokenizer import Tokenizer

@pytest.fixture
def tokenizer():
    return Tokenizer("test tokenizer")

@pytest.fixture
def dataset(tokenizer: Tokenizer):
    return CharDataset(text="test tokenizer", tokenizer=tokenizer, block_size=4)

def test_dataset_len(dataset: CharDataset):
    assert len(dataset) == len(dataset.data) - dataset.block_size

def test_dataset_item_shapes(dataset: CharDataset):
    x, y = dataset[0]
    assert len(x) == dataset.block_size
    assert len(y) == dataset.block_size

def test_dataset_pair_alignment(dataset: CharDataset):
    x, y = dataset[0]
    for i in range(len(x)):
        assert y[i] == dataset.data[i+1]