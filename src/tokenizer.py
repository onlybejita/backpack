from transformers import GPT2Tokenizer


class Tokenizer:
    def __init__(self):
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    def tokenize(self, text):
        return self.tokenizer.encode(text)

    def detokenize(self, tokens):
        return self.tokenizer.decode(tokens)
