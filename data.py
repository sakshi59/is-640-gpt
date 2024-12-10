import torch

class Data:
    def __init__(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            self.text = f.read()
        self.chars = sorted(list(set(self.text)))
        self.vocab_size = len(self.chars)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}
        self.encode = lambda s: [self.stoi[c] for c in s]
        self.decode = lambda l: ''.join([self.itos[i] for i in l])

        data = torch.tensor(self.encode(self.text), dtype=torch.long)
        n = int(0.9 * len(data))
        self.train_data = data[:n]
        self.val_data = data[n:]

    
