import torch
from trainer import Trainer
from data import Data
from model import GPTLanguageModel

# Hyperparameters
random_seed = 1337
train_iterations = 100
word_count = 100
data_file = "input.txt"
block_size = 8
batch_size = 16
learning_rate = 0.001

def main():
    """Main function"""
    torch.manual_seed(random_seed)

    # Load data
    data = Data(data_file)

    # Initialize model
    model = GPTLanguageModel()
    model.to("cuda" if torch.cuda.is_available() else "cpu")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Initialize trainer
    trainer = Trainer(data, model, optimizer)

    # Train model
    trainer.train(iterations=train_iterations, batch_size=batch_size, block_size=block_size)

    # Generate text
    context = torch.zeros((1, 1), dtype=torch.long)
    generated = model.generate(context, max_new_tokens=word_count)
    print(data.decode(generated[0].tolist()))

if __name__ == "__main__":
    main()
