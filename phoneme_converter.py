# phoneme_converter.py
import nltk
from nltk.corpus import cmudict
import functools


class PhonemeConverter:
    def __init__(self):
        # 1. Load a pre-existing pronunciation dictionary
        try:
            self.cmu_dict = cmudict.dict()
        except LookupError:
            nltk.download('cmudict')
            self.cmu_dict = cmudict.dict()

        # 2. Initialize with your existing dictionary for fast access
        self.phoneme_dict = {
            "hello": ["HH", "AH", "L", "OW"],
            "world": ["W", "ER", "L", "D"],
            # ... your existing entries
        }

        # 3. Add caching with LRU to speed up repeated lookups
        self.get_phonemes = functools.lru_cache(maxsize=10000)(self._get_phonemes)

    def _get_phonemes(self, word):
        """Internal method to find phonemes for a word"""
        word = word.lower()

        # First check our custom dictionary (fastest)
        if word in self.phoneme_dict:
            return self.phoneme_dict[word]

        # Then check CMU dictionary (still fast)
        if word in self.cmu_dict:
            # Format CMU pronunciation to match your style
            pronunciation = self.cmu_dict[word][0]
            # Strip digits from phonemes like 'AH0' to get 'AH'
            return [p.rstrip('0123456789') for p in pronunciation]

        # If still not found, use NLP model for prediction
        return self._predict_phonemes(word)

    def _predict_phonemes(self, word):
        """Use a phoneme prediction model for unknown words"""
        # Implement your NLP model here - this is a placeholder
        # Options include:
        # 1. Use g2p (Grapheme-to-Phoneme) libraries
        # 2. Use a pre-trained transformer model
        # 3. Use rule-based fallback

        # Example with a hypothetical g2p model:
        try:
            # Try to use g2p_en if available
            from g2p_en import G2p
            if not hasattr(self, 'g2p_model'):
                self.g2p_model = G2p()
            return self.g2p_model(word)
        except ImportError:
            # Simple fallback if no model is available
            # This is just a placeholder - actual implementation would be more sophisticated
            return list(word.upper())  # Very basic fallback

    def add_to_dict(self, word, phonemes):
        """Add new word to custom dictionary for future fast lookups"""
        word = word.lower()
        self.phoneme_dict[word] = phonemes
        # Clear the specific cache entry if it exists
        if hasattr(self.get_phonemes, 'cache_clear'):
            self.get_phonemes.cache_clear()

converter = PhonemeConverter()

print(converter.get_phonemes("ashish"))  # From custom dict

# CMU dictionary lookup
print(converter.get_phonemes("computer"))  # From CMU
print(converter.get_phonemes("huh"))  # From CMU
# NLP model prediction for unknown words
print(converter.get_phonemes("supercalifragilisticexpialidocious"))

# Add new words to dictionary
converter.add_to_dict("custom", ["K", "AH", "S", "T", "AH", "M"])
