import torch
import torch.nn as nn

def get_loss(model):
    def loss_fn(output, target):
        # Cross-entropy loss with a regularization term that penalizes the model when the sense vectors are too similar to each other
        ce_loss = nn.CrossEntropyLoss()(output, target)
        reg_loss = torch.sum(model.sense_vectors.pow(2)) / model.sense_vectors.shape[0]
        return ce_loss + 0.01 * reg_loss
    return loss_fn
