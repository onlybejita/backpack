import torch
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm

class Trainer:
    def __init__(self, model, loss_fn, optimizer, scheduler, device, save_dir="models"):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.save_dir = save_dir

    def train_one_epoch(self, dataloader):
        self.model.train()
        total_loss = 0
        dataloader_len = len(dataloader)

        for batch, (input, target) in enumerate(tqdm(dataloader)):
            input = input.to(self.device)
            target = target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(input)
            loss = self.loss_fn(output, target)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / dataloader_len
        return avg_loss

    def validate(self, dataloader):
        self.model.eval()
        total_loss = 0
        dataloader_len = len(dataloader)

        with torch.no_grad():
            for input, target in dataloader:
                input = input.to(self.device)
                target = target.to(self.device)

                output = self.model(input)
                loss = self.loss_fn(output, target)

                total_loss += loss.item()

        avg_loss = total_loss / dataloader_len
        return avg_loss

    def save_model(self, epoch):
        torch.save(self.model.state_dict(), f"{self.save_dir}/model_{epoch}.pt")

