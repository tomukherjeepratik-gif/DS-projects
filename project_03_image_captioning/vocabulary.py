"""
Vocabulary Builder for Image Captioning
Handles mapping between words and numerical indices.
"""

from collections import Counter
import re

class Vocabulary:
    def __init__(self, freq_threshold=1):
        self.freq_threshold = freq_threshold
        self.itos = {0: "<PAD>", 1: "<START>", 2: "<END>", 3: "<UNK>"}
        self.stoi = {"<PAD>": 0, "<START>": 1, "<END>": 2, "<UNK>": 3}
        self.idx = 4

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenizer_eng(text):
        text = text.lower()
        text = re.sub(r"[^\w\s]", "", text)
        return text.split()

    def build_vocabulary(self, sentence_list):
        frequencies = Counter()
        for sentence in sentence_list:
            for word in self.tokenizer_eng(sentence):
                frequencies[word] += 1
                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = self.idx
                    self.itos[self.idx] = word
                    self.idx += 1

    def numericalize(self, text):
        tokenized_text = self.tokenizer_eng(text)
        return (
            [self.stoi["<START>"]]
            + [self.stoi.get(token, self.stoi["<UNK>"]) for token in tokenized_text]
            + [self.stoi["<END>"]]
        )

    def decode(self, indices):
        words = []
        for idx in indices:
            if isinstance(idx, float) or hasattr(idx, 'item'):
                idx = int(idx)
            word = self.itos.get(idx, "<UNK>")
            if word in ["<PAD>", "<START>"]:
                continue
            if word == "<END>":
                break
            words.append(word)
        return " ".join(words)
