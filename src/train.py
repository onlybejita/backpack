import torch
from transformers import AdamW, get_linear_schedule_with_warmup

def train(model, dataloader, loss_fn, device, save_interval, save_dir, num_epochs):
    model.train()

    # Create the AdamW optimizer
    optimizer = AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)

    # Create the learning rate scheduler
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(dataloader)*num_epochs)

    total_loss = 0

    for batch, (input, target) in enumerate(dataloader):
        input = input.to(device)
        target = target.to(device)

        optimizer.zero_grad()

        output = model(input)
        loss = loss_fn(output.view(-1, model.vocab_size), target.view(-1))

        loss.backward()
        optimizer.step()

        # Step the learning rate scheduler
        scheduler.step()

        total_loss += loss.item()

        # Save the model's parameters every save_interval batches
        if batch % save_interval == 0:
            torch.save(model.state_dict(), f"{save_dir}/model_{batch}.pt")

        # Print out the training progress
        if batch % 100 == 0:
            print(f"Batch {batch}, Loss {loss.item()}")

    return total_loss / len(dataloader)
