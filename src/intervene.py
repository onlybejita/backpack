from .model import Backpack

def intervene(model, word_index, sense_index=None, new_vector=None, suppress=False):
    if suppress:
        if sense_index is None:
            raise ValueError("sense_index must be provided when suppress is True")
        model.sense_vectors[word_index, sense_index] = 0
    elif sense_index is None:
        model.sense_vectors[word_index] = new_vector
    else:
        model.sense_vectors[word_index, sense_index] = new_vector
