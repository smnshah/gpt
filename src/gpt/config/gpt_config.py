class GPTConfig:
    def __init__(
        self, 
        vocab_size: int, 
        block_size: int, 
        n_layers: int = 4, 
        n_heads: int = 4, 
        n_embed: int = 128, 
        dropout: float = 0.1
    ):
        self.vocab_size = vocab_size    # size of character vocabulary
        self.block_size = block_size    # max context length
        self.n_layers = n_layers        # number of transformer blocks
        self.n_heads = n_heads          # number of attention heads
        self.n_embed = n_embed          # embedding dimension
        self.dropout = dropout          # dropout rate
        