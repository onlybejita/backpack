# 🎒 Backpack Language Model Implementation 🎒

I'm attempting to teach myself more about new and novel model architectures by trying to implement them in Python. This repository contains an attempt at a basic Python implementation of the Backpack Language Model as described in the paper [Backpack: A New Neural Architecture for NLP](https://arxiv.org/pdf/2305.16765.pdf).

## Overview

The Backpack Language Model is a new neural architecture that combines strong modeling performance with an interface for interpretability and control. It learns multiple non-contextual sense vectors for each word in a vocabulary, and represents a word in a sequence as a context-dependent, non-negative linear combination of sense vectors in this sequence.

This implementation includes the following components:

- A tokenizer for converting text into tokens
- A dataset for loading and preprocessing the data
- The Backpack model itself, including the sense vectors and a Transformer encoder
- A loss function for training the model
- An optimizer for updating the model's parameters
- A training loop for training the model
- A validation loop for evaluating the model on a validation set
- An evaluation script for computing various evaluation metrics
- An intervention script for performing interventions on the sense vectors

## Installation

To install the necessary dependencies, run the following command:

```
pip install -r requirements.txt
```

## Usage

To train the model, run the following command:

```
python main.py
```

## Acknowledgements
This implementation is based on the paper [Backpack: A New Neural Architecture for NLP](https://arxiv.org/pdf/2305.16765.pdf) by the authors of the paper.

## Citation

If you find this work useful, please cite the original paper:

```
@misc{hewitt2023backpack,
      title={Backpack Language Models}, 
      author={John Hewitt and John Thickstun and Christopher D. Manning and Percy Liang},
      year={2023},
      eprint={2305.16765},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```