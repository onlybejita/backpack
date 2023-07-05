import torch
from transformers import get_linear_schedule_with_warmup
from torch.optim.lr_scheduler import StepLR
from src.dataset import get_dataloaders
from src.model import Backpack
from src.loss import get_loss
from src.optimizer import get_optimizer
from src.train import train
from src.validate import validate
from src.evaluate import evaluate

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the data
train_dataloader, val_dataloader = get_dataloaders(
    "/data/openwebtext/some_file.txt", sequence_length=100, batch_size=32
)

# Create the model
model = Backpack(
    vocab_size=50257, sense_size=10, embedding_dim=768, nhead=12, nhid=3072, nlayers=12
).to(device)

# Create the loss function
loss_fn = get_loss(model)

# Create the optimizer
optimizer = get_optimizer(model, learning_rate=0.001)

# Create the learning rate scheduler
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_dataloader)*10)

# Initialize the best validation loss
best_val_loss = float("inf")

# Run the training loop
for epoch in range(10):
    train_loss = train(model, train_dataloader, loss_fn, optimizer, device)
    print(f"Epoch {epoch}, Train Loss {train_loss}")

    val_loss = validate(model, val_dataloader, loss_fn, device)
    print(f"Epoch {epoch}, Val Loss {val_loss}")

    # Save the best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "models/best_model.pt")

    # Save the model every 5 epochs
    if epoch % 5 == 0:
        torch.save(model.state_dict(), f"models/model_{epoch}.pt")

    # Step the learning rate scheduler
    scheduler.step()

# Load the best model
model.load_state_dict(torch.load("models/best_model.pt"))


# Define a simple text generation function
def generate_text(model, device, max_length=100):
    model.eval()
    input = torch.randint(50257, (1, 1), dtype=torch.long).to(device)
    output = []

    with torch.no_grad():
        for _ in range(max_length):
            logits = model(input)
            word = torch.argmax(logits, dim=2)[-1]
            output.append(word.item())
            input = torch.cat([input, word.unsqueeze(0).unsqueeze(0)], dim=1)

    return output


# Define a set of reference texts
references = [[i for i in range(100)] for _ in range(100)]

# Evaluate the model
test_loss, perplexity, bleu_score = evaluate(
    model, val_dataloader, loss_fn, device, generate_text, references
)
print(f"Test Loss {test_loss}, Perplexity {perplexity}, BLEU Score {bleu_score}")
