# viseme_mapper.py
class VisemeMapper:
    def __init__(self):
        """
        Initialize the mapper with phoneme-to-viseme mappings
        Viseme parameters are given as (jaw_open, lip_round, lip_width)
        """
        # Basic viseme parameters: (jaw_open, lip_round, lip_width)
        self.viseme_params = {
            "REST": (0.0, 0.0, 0.5),  # Neutral closed mouth
            "A": (0.7, 0.0, 0.7),  # Open mouth as in "car", "hat"
            "E": (0.5, 0.2, 0.7),  # Wide mouth as in "bed", "yes"
            "I": (0.3, 0.0, 0.8),  # Slight open mouth as in "sit", "bit"
            "O": (0.5, 0.8, 0.6),  # Rounded mouth as in "go", "boat"
            "U": (0.3, 0.8, 0.4),  # Pursed lips as in "blue", "tube"
            "F": (0.1, 0.5, 0.7),  # Lower lip touching upper teeth as in "far", "van"
            "P": (0.0, 0.0, 0.5),  # Closed lips as in "put", "but"
            "L": (0.3, 0.0, 0.6),  # Tongue tip up as in "lot", "doll"
            "S": (0.2, 0.4, 0.7),  # Teeth closed, slight open as in "sit", "this"
        }

        # Mapping from phonemes to visemes
        self.phoneme_to_viseme = {
            # Vowels
            "AA": "A",  # "father"
            "AE": "A",  # "cat"
            "AH": "A",  # "hut"
            "AO": "O",  # "dog"
            "AW": "A",  # "cow"
            "AY": "A",  # "hide"
            "EH": "E",  # "pet"
            "ER": "E",  # "fur"
            "EY": "E",  # "ate"
            "IH": "I",  # "sit"
            "IY": "I",  # "eat"
            "OW": "O",  # "boat"
            "OY": "O",  # "toy"
            "UH": "U",  # "book"
            "UW": "U",  # "boot"

            # Consonants
            "B": "P",  # "buy"
            "CH": "S",  # "church"
            "D": "L",  # "day"
            "DH": "L",  # "this"
            "F": "F",  # "for"
            "G": "P",  # "go"
            "HH": "REST",  # "help"
            "JH": "S",  # "judge"
            "K": "P",  # "key"
            "L": "L",  # "lay"
            "M": "P",  # "me"
            "N": "L",  # "no"
            "NG": "L",  # "sing"
            "P": "P",  # "put"
            "R": "L",  # "run"
            "S": "S",  # "see"
            "SH": "S",  # "she"
            "T": "L",  # "take"
            "TH": "F",  # "thin"
            "V": "F",  # "very"
            "W": "U",  # "way"
            "Y": "I",  # "yes"
            "Z": "S",  # "zoo"
            "ZH": "S",  # "measure"
        }

        # Default transition time (in seconds) between visemes
        self.transition_time = 0.05

    def get_viseme_for_phoneme(self, phoneme):
        """Convert phoneme to viseme name and parameters"""
        if phoneme in self.phoneme_to_viseme:
            viseme_name = self.phoneme_to_viseme[phoneme]
            return viseme_name, self.viseme_params[viseme_name]
        else:
            # Return REST viseme for unknown phonemes
            return "REST", self.viseme_params["REST"]

    def map_to_viseme_sequence(self, phoneme_timings):
        """
        Map phoneme timings to viseme sequence

        Parameters:
        - phoneme_timings: List of (phoneme, start_time, end_time) tuples

        Returns:
        - List of (viseme_name, viseme_params, start_time, end_time) tuples
        """
        viseme_sequence = []

        for phoneme, start_time, end_time in phoneme_timings:
            viseme_name, viseme_params = self.get_viseme_for_phoneme(phoneme)
            viseme_sequence.append((viseme_name, viseme_params, start_time, end_time))

        # Add REST visemes at start and end if needed
        if viseme_sequence and viseme_sequence[0][2] > 0:
            # Add REST at start
            viseme_sequence.insert(0, ("REST", self.viseme_params["REST"], 0, viseme_sequence[0][2]))

        # Add final REST if the sequence isn't empty
        if viseme_sequence:
            last_end = viseme_sequence[-1][3]
            viseme_sequence.append(("REST", self.viseme_params["REST"], last_end, last_end + 0.5))

        return viseme_sequence


if __name__ == "__main__":
    mapper = VisemeMapper()

    # Example phoneme timings (from the PhonemeAnalyzer)
    phoneme_timings = [
        ("HH", 0.0, 0.12),
        ("AH", 0.12, 0.22),
        ("L", 0.22, 0.30),
        ("OW", 0.30, 0.42),
        ("W", 0.42, 0.49),
        ("ER", 0.49, 0.64),
        ("L", 0.64, 0.72),
        ("D", 0.72, 0.78)
    ]

    # Convert to viseme timings
    viseme_timings = mapper.convert_phoneme_timings(phoneme_timings)

    # Optimize the sequence
    optimized_visemes = mapper.optimize_viseme_sequence(viseme_timings)

    print("Viseme sequence:")
    for viseme, params, start, end in optimized_visemes:
        jaw, round_val, width = params
        print(f"{viseme}: {start:.2f}s - {end:.2f}s, Params: jaw={jaw:.1f}, round={round_val:.1f}, width={width:.1f}")
