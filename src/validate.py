import torch


def validate(model, dataloader, loss_fn, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch, (input, target) in enumerate(dataloader):
            input = input.to(device)
            target = target.to(device)

            output = model(input)
            loss = loss_fn(output.view(-1, model.vocab_size), target.view(-1))

            total_loss += loss.item()

    return total_loss / len(dataloader)
