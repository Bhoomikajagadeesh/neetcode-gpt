from typing import List, Dict

class Solution:
    def _greedy_tokenize(self, text: str, vocab: Dict[str, int]) -> List[str]:
        tokens = []
        i = 0
        while i < len(text):
            longest = text[i]
            for length in range(len(text) - i, 0, -1):
                if text[i:i + length] in vocab:
                    longest = text[i:i + length]
                    break
            tokens.append(longest)
            i += len(longest)
        return tokens

    def tokenize_numbers(self, numbers: List[int], vocab: Dict[str, int]) -> List[List[str]]:
        return [self._greedy_tokenize(str(n), vocab) for n in numbers]

    def count_tokens(self, text: str, vocab: Dict[str, int]) -> int:
        return len(self._greedy_tokenize(text, vocab))

    def fertility_score(self, text: str, vocab: Dict[str, int]) -> float:
        return round(self.count_tokens(text, vocab) / len(text.split()), 4)
