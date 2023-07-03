import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from tokenizer import Tokenizer


class TextDataset(Dataset):
    def __init__(self, file_path, sequence_length):
        self.tokenizer = Tokenizer()
        self.sequence_length = sequence_length

        # Read the text file and tokenize the text
        with open(file_path, "r") as file:
            text = file.read()
        self.tokens = self.tokenizer.tokenize(text)

    def __len__(self):
        return len(self.tokens) - self.sequence_length

    def __getitem__(self, idx):
        # Get a sequence of tokens and the next token as the target
        input_tokens = self.tokens[idx : idx + self.sequence_length]
        target_token = self.tokens[idx + self.sequence_length]
        return torch.tensor(input_tokens), torch.tensor(target_token)


def get_dataloaders(file_path, sequence_length, batch_size, validation_split=0.1):
    train_file = "data/train_dataset.pt"
    val_file = "data/val_dataset.pt"

    if os.path.exists(train_file) and os.path.exists(val_file):
        # Load the datasets from files
        train_dataset = torch.load(train_file)
        val_dataset = torch.load(val_file)
    else:
        # Create the dataset
        dataset = TextDataset(file_path, sequence_length)

        # Split the dataset into training and validation sets
        train_size = int((1.0 - validation_split) * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        # Save the datasets to files
        torch.save(train_dataset, train_file)
        torch.save(val_dataset, val_file)

    # Create data loaders for the training and validation sets
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

    return train_dataloader, val_dataloader
