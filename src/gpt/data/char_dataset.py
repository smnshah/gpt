from gpt.tokenizer.tokenizer import Tokenizer

class CharDataset:
    def __init__(self, text: str, tokenizer: Tokenizer, block_size: int):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.data = self.tokenizer.encode(text)

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        chunk = self.data[idx:idx + self.block_size + 1]
        x = chunk[:-1]
        y = chunk[1:]
        return x, y