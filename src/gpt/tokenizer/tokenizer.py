class Tokenizer:
    def __init__(self, text: str):
        unique_chars = sorted(set(text))
        self.stoi = {ch: i for i, ch in enumerate(unique_chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}
        self.vocab_size = len(self.stoi)

    def encode(self, text: str) -> list[int]:
        return [self.stoi[ch] for ch in text]

    def decode(self, ids: list[int]) -> str:
        return ''.join(self.itos[i] for i in ids)