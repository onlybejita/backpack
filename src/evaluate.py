import math
from nltk.translate.bleu_score import sentence_bleu
from .validate import validate


def evaluate(model, dataloader, loss_fn, device, generate_fn, references):
    # Compute the loss and perplexity
    loss = validate(model, dataloader, loss_fn, device)
    perplexity = math.exp(loss)

    # Generate text from the model
    generated_texts = [generate_fn(model, device) for _ in range(len(references))]

    # Compute the BLEU score
    bleu_score = sum(
        sentence_bleu([ref], gen) for ref, gen in zip(references, generated_texts)
    ) / len(references)

    return loss, perplexity, bleu_score
