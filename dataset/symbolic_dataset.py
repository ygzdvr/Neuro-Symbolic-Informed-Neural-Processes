"""
Symbolic Equation Dataset for NS-INP

Provides tokenized symbolic equations for sinusoid data.
Equations are of the form: y = a*x + sin(b*x) + c

The tokenizer embeds numeric values directly into the equation
so the Transformer can learn numeric relationships.
"""

import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

ROOT = "data/"


class SymbolicEquationTokenizer:
    """
    Tokenizer for symbolic equations with embedded numeric values.

    Vocabulary (64 tokens):
    - PAD=0, SOS=1, EOS=2, UNK=3
    - OPERATORS: =, +, -, *, /, ^, (, )  [8 tokens]
    - FUNCTIONS: sin, cos, exp, log      [4 tokens]
    - VARIABLES: x, y, t                 [3 tokens]
    - DIGITS: 0-9                        [10 tokens]
    - SPECIAL: . (decimal), E (scientific) [2 tokens]
    - MASK: ? (for partial knowledge)    [1 token]
    - Reserved                           [~33 tokens]
    """

    # Special tokens
    PAD = 0
    SOS = 1
    EOS = 2
    UNK = 3

    # Vocabulary mapping
    VOCAB = {
        '<PAD>': 0, '<SOS>': 1, '<EOS>': 2, '<UNK>': 3,
        # Operators (4-11)
        '=': 4, '+': 5, '-': 6, '*': 7, '/': 8, '^': 9, '(': 10, ')': 11,
        # Functions (12-15)
        'sin': 12, 'cos': 13, 'exp': 14, 'log': 15,
        # Variables (16-18)
        'x': 16, 'y': 17, 't': 18,
        # Digits (19-28)
        '0': 19, '1': 20, '2': 21, '3': 22, '4': 23,
        '5': 24, '6': 25, '7': 26, '8': 27, '9': 28,
        # Special (29-30)
        '.': 29, 'E': 30,
        # Mask for partial knowledge (31)
        '?': 31,
    }

    # Reverse mapping for decoding
    IDX_TO_TOKEN = {v: k for k, v in VOCAB.items()}

    def __init__(self, max_len=50, vocab_size=64):
        self.max_len = max_len
        self.vocab_size = vocab_size

    def tokenize_number(self, num, precision=2):
        """
        Tokenize a number digit by digit.

        Args:
            num: float number to tokenize
            precision: decimal places to round to

        Returns:
            list of token strings
        """
        # Round to specified precision
        num = round(num, precision)

        # Convert to string
        num_str = f"{num:.{precision}f}"

        # Tokenize character by character
        tokens = []
        for char in num_str:
            if char in self.VOCAB:
                tokens.append(char)
            else:
                tokens.append('<UNK>')

        return tokens

    def tokenize_equation(self, a, b, c, knowledge_type="symbolic_full"):
        """
        Tokenize the sinusoid equation y = a*x + sin(b*x) + c

        Args:
            a, b, c: equation parameters
            knowledge_type: what parameters to reveal
                - "symbolic_full": all parameters
                - "symbolic_abc2": 1-2 parameters, others masked
                - "symbolic_abc": 1 parameter, others masked

        Returns:
            list of token strings
        """
        # Determine which parameters to reveal
        if knowledge_type == "symbolic_full":
            reveal_mask = [True, True, True]
        elif knowledge_type == "symbolic_abc2":
            num_revealed = np.random.choice([1, 2])
            revealed = np.random.choice([0, 1, 2], num_revealed, replace=False)
            reveal_mask = [i in revealed for i in range(3)]
        elif knowledge_type == "symbolic_abc":
            revealed = np.random.choice([0, 1, 2])
            reveal_mask = [i == revealed for i in range(3)]
        elif knowledge_type == "symbolic_a":
            reveal_mask = [True, False, False]
        elif knowledge_type == "symbolic_b":
            reveal_mask = [False, True, False]
        elif knowledge_type == "symbolic_c":
            reveal_mask = [False, False, True]
        elif knowledge_type == "symbolic_none":
            reveal_mask = [False, False, False]
        else:
            reveal_mask = [True, True, True]

        tokens = ['<SOS>', 'y', '=']

        # a*x term
        if reveal_mask[0]:
            tokens.extend(self.tokenize_number(a))
        else:
            tokens.append('?')
        tokens.extend(['*', 'x', '+'])

        # sin(b*x) term
        tokens.extend(['sin', '('])
        if reveal_mask[1]:
            tokens.extend(self.tokenize_number(b))
        else:
            tokens.append('?')
        tokens.extend(['*', 'x', ')'])

        # + c term
        tokens.append('+')
        if reveal_mask[2]:
            tokens.extend(self.tokenize_number(c))
        else:
            tokens.append('?')

        tokens.append('<EOS>')

        return tokens

    def encode(self, tokens):
        """Convert token strings to token IDs."""
        return [self.VOCAB.get(t, self.UNK) for t in tokens]

    def decode(self, token_ids):
        """Convert token IDs back to token strings."""
        return [self.IDX_TO_TOKEN.get(int(t), '<UNK>') for t in token_ids]

    def pad(self, token_ids):
        """Pad sequence to max_len."""
        if len(token_ids) >= self.max_len:
            return token_ids[:self.max_len]
        return token_ids + [self.PAD] * (self.max_len - len(token_ids))

    def __call__(self, a, b, c, knowledge_type="symbolic_full"):
        """
        Full tokenization pipeline.

        Args:
            a, b, c: equation parameters
            knowledge_type: what parameters to reveal

        Returns:
            torch.Tensor of shape [max_len] with token IDs
        """
        tokens = self.tokenize_equation(a, b, c, knowledge_type)
        token_ids = self.encode(tokens)
        token_ids = self.pad(token_ids)
        return torch.tensor(token_ids, dtype=torch.long)


class SymbolicSinusoidDataset(Dataset):
    """
    Symbolic sinusoid dataset wrapping the existing trending-sinusoids data.

    Returns (x, y, equation_tokens) where equation_tokens is tokenized
    symbolic representation of the underlying function.
    """

    def __init__(
        self,
        split="train",
        root=f"{ROOT}/trending-sinusoids",
        knowledge_type="symbolic_full",
        split_file="splits",
        max_len=50,
        vocab_size=64,
    ):
        self.data = pd.read_csv(f"{root}/data.csv")
        self.knowledge = pd.read_csv(f"{root}/knowledge.csv")
        self.value_cols = [c for c in self.data.columns if c.isnumeric()]
        self.dim_x = 1
        self.dim_y = 1

        if split_file is None:
            split_file = "splits"
        self.train_test_val_split = pd.read_csv(f"{root}/{split_file}.csv")
        self.split = split
        self.knowledge_type = knowledge_type

        # Tokenizer
        self.tokenizer = SymbolicEquationTokenizer(max_len=max_len, vocab_size=vocab_size)

        # For compatibility with existing code
        # knowledge_input_dim is set to max_len since that's the sequence length
        self.knowledge_input_dim = max_len
        self.vocab_size = vocab_size
        self.max_len = max_len

        self._split_data()

    def _split_data(self):
        if self.split == "train":
            split_ids = self.train_test_val_split[
                self.train_test_val_split["split"] == "train"
            ].curve_id
        elif self.split == "val" or self.split == "valid":
            split_ids = self.train_test_val_split[
                self.train_test_val_split["split"] == "val"
            ].curve_id
        elif self.split == "test":
            split_ids = self.train_test_val_split[
                self.train_test_val_split["split"] == "test"
            ].curve_id
        else:
            split_ids = self.data.curve_id.unique()

        self.data = self.data[self.data.curve_id.isin(split_ids)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        y = self.data.iloc[idx, :][self.value_cols].values
        x = np.linspace(-2, 2, len(y))
        curve_id = self.data.iloc[idx]["curve_id"]

        # Get parameters (a, b, c)
        knowledge_row = self.knowledge[self.knowledge.curve_id == curve_id]
        a = knowledge_row['a'].values[0]
        b = knowledge_row['b'].values[0]
        c = knowledge_row['c'].values[0]

        # Tokenize equation
        equation_tokens = self.tokenizer(a, b, c, self.knowledge_type)

        x = torch.tensor(x, dtype=torch.float32).unsqueeze(-1)
        y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)

        # Also return true parameters for auxiliary loss (normalized)
        # Normalize to roughly [-1, 1] range based on known parameter ranges
        # a in [-1, 1], b in [0, 6] -> normalized to [-1, 1], c in [-1, 1]
        true_params = torch.tensor([a, (b - 3) / 3, c], dtype=torch.float32)

        return x, y, equation_tokens, true_params

    def get_raw_knowledge(self, curve_id):
        """Get raw (a, b, c) parameters for a curve."""
        knowledge_row = self.knowledge[self.knowledge.curve_id == curve_id]
        return {
            'a': knowledge_row['a'].values[0],
            'b': knowledge_row['b'].values[0],
            'c': knowledge_row['c'].values[0],
        }


class SymbolicSinusoidDatasetDistShift(SymbolicSinusoidDataset):
    """Distribution shift variant of symbolic sinusoid dataset."""

    def __init__(
        self,
        split="train",
        root="./data/trending-sinusoids-dist-shift",
        knowledge_type="symbolic_full",
        split_file="splits",
        max_len=50,
        vocab_size=64,
    ):
        super().__init__(
            split=split,
            root=root,
            knowledge_type=knowledge_type,
            split_file=split_file,
            max_len=max_len,
            vocab_size=vocab_size,
        )


if __name__ == "__main__":
    # Test tokenizer
    tokenizer = SymbolicEquationTokenizer()

    # Test with sample parameters
    a, b, c = 0.5, 2.3, -0.8
    tokens = tokenizer.tokenize_equation(a, b, c, "symbolic_full")
    print("Tokens:", tokens)

    token_ids = tokenizer(a, b, c, "symbolic_full")
    print("Token IDs shape:", token_ids.shape)
    print("Token IDs:", token_ids[:30])

    # Decode back
    decoded = tokenizer.decode(token_ids[:30])
    print("Decoded:", decoded)

    # Test with partial knowledge
    print("\nPartial knowledge (abc2):")
    for _ in range(3):
        token_ids = tokenizer(a, b, c, "symbolic_abc2")
        decoded = tokenizer.decode(token_ids[:25])
        print(" ".join([t for t in decoded if t != '<PAD>']))

    # Test dataset
    print("\n\nTesting dataset...")
    try:
        dataset = SymbolicSinusoidDataset(split="train", knowledge_type="symbolic_full")
        print(f"Dataset length: {len(dataset)}")
        print(f"knowledge_input_dim: {dataset.knowledge_input_dim}")

        x, y, knowledge = dataset[0]
        print(f"x shape: {x.shape}")
        print(f"y shape: {y.shape}")
        print(f"knowledge shape: {knowledge.shape}")

        # Decode the equation
        decoded = tokenizer.decode(knowledge)
        equation_str = " ".join([t for t in decoded if t != '<PAD>'])
        print(f"Equation: {equation_str}")
    except Exception as e:
        print(f"Dataset test skipped: {e}")
