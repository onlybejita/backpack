import torch.optim as optim

def get_optimizer(model, learning_rate):
    return optim.Adam(model.parameters(), lr=learning_rate)
