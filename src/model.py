import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class Backpack(nn.Module):
    def __init__(self, vocab_size, sense_size, embedding_dim, nhead, nhid, nlayers):
        super(Backpack, self).__init__()

        # The sense vectors are initialized as a 3D tensor
        self.sense_vectors = nn.Parameter(torch.rand(vocab_size, sense_size, embedding_dim))
        self.sense_vectors = nn.ReLU()(self.sense_vectors)  # Apply ReLU to ensure non-negativity

        # The rest of the model is a standard Transformer encoder
        self.pos_encoder = PositionalEncoding(embedding_dim)
        encoder_layers = TransformerEncoderLayer(embedding_dim, nhead, nhid)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.fc = nn.Linear(embedding_dim, vocab_size)

    def forward(self, src):
        # The input is a sequence of token indices
        # For each token, wecompute a context-dependent, non-negative linear combination of its sense vectors
        word_embeddings = self.get_word_embeddings(src)

        # The rest of the forward pass is the same as a standard Transformer encoder
        output = self.transformer_encoder(word_embeddings)
        output = self.fc(output)
        return output

    def get_word_embeddings(self, src):
        # This method computes a context-dependent, non-negative linear combination of the sense vectors for each token
        # The context-dependent weights are computed by a dot product between the sense vectors and a context vector
        word_senses = self.sense_vectors[src]
        context_vector = torch.mean(word_senses, dim=1, keepdim=True)
        weights = torch.bmm(word_senses, context_vector.transpose(1, 2))
        weights = nn.Softmax(dim=1)(weights)
        return torch.bmm(weights, word_senses)

    def analyze_sense_vectors(self, word_index):
        # This method prints out the sense vectors for a given word
        print(self.sense_vectors[word_index])

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout) if dropout else None

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x) if self.dropout else x
