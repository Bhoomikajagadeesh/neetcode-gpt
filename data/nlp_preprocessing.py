import torch
import torch.nn as nn
from torchtyping import TensorType
from typing import List

class Solution:
    def get_dataset(self, positive: List[str], negative: List[str]) -> TensorType[float]:
        # 1. Build vocabulary: collect all unique words, sort them, assign integer IDs starting at 1
        # 2. Encode each sentence by replacing words with their IDs
        # 3. Combine positive + negative into one list of tensors
        # 4. Pad shorter sequences with 0s using nn.utils.rnn.pad_sequence(tensors, batch_first=True)
        # 1. Tokenize sentences
        pos_tokens = [s.split() for s in positive]
        neg_tokens = [s.split() for s in negative]
        vocab = sorted(set(word for sent in pos_tokens + neg_tokens for word in sent))
        word2idx = {word: i + 1 for i, word in enumerate(vocab)}
        all_sentences = pos_tokens + neg_tokens

        encoded = [torch.tensor([word2idx[word] for word in sent], dtype=torch.float)
            for sent in all_sentences]
        padded = nn.utils.rnn.pad_sequence(encoded, batch_first=True, padding_value=0)
        return padded



