import torch

class Trainer:
    def __init__(self, data, model, optimizer):
        self.data = data
        self.model = model
        self.optimizer = optimizer
