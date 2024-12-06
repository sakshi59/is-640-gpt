import torch

class Data:
    def __init__(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            self.text = f.read()
        self.chars = sorted(list(set(self.text)))
        self.vocab_size = len(self.chars)

        # Create mappings
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}

        # Encode text to integers
        self.data = torch.tensor([self.stoi[c] for c in self.text], dtype=torch.long)

    def encode(self, text):
        """Convert string to list of integers"""
        return [self.stoi[ch] for ch in text]

    def decode(self, indices):
        """Convert list of integers to string"""
        return ''.join([self.itos[i] for i in indices])

    def get_splits(self, split_ratio=0.9):
        """Split data into training and validation sets"""
        n = int(len(self.data) * split_ratio)
        return self.data[:n], self.data[n:]

    def get_batch(self, split_data, batch_size, block_size):
        """Generate a batch of input-output sequences"""
        ix = torch.randint(len(split_data) - block_size, (batch_size,))
        x = torch.stack([split_data[i:i+block_size] for i in ix])
        y = torch.stack([split_data[i+1:i+block_size+1] for i in ix])
        return x, y
