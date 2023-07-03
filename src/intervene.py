from .model import Backpack

def intervene(model, word_index, sense_index=None, new_vector=None, suppress=False):
    # Check if word_index is valid
    if word_index < 0 or word_index >= model.sense_vectors.size(0):
        raise ValueError(f"word_index must be between 0 and {model.sense_vectors.size(0)-1}")

    # Check if sense_index is valid
    if sense_index is not None and (sense_index < 0 or sense_index >= model.sense_vectors.size(1)):
        raise ValueError(f"sense_index must be between 0 and {model.sense_vectors.size(1)-1}")

    # Check if new_vector is valid
    if new_vector is not None and new_vector.size() != model.sense_vectors[0, 0].size():
        raise ValueError(f"new_vector must have the same size as the sense vectors")

    if suppress:
        if sense_index is None:
            raise ValueError("sense_index must be provided when suppress is True")
        model.sense_vectors[word_index, sense_index] = 0
    elif sense_index is None:
        model.sense_vectors[word_index] = new_vector
    else:
        model.sense_vectors[word_index, sense_index] = new_vector
