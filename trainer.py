import torch

class Trainer:
    def __init__(self, data, model, optimizer):
        self.data = data
        self.model = model
        self.optimizer = optimizer
    def train(self, iterations, batch_size, block_size):
        for iter in range(iterations):
            xb, yb = self.data.get_batch('train', batch_size, block_size)
            logits, loss = self.model(xb, yb)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if iter % 100 == 0:
                print(f"Step {iter}, Loss: {loss.item()}")
