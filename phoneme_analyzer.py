# phoneme_analyzer.py
import numpy as np
from scipy.io import wavfile
import os
import re

from phoneme_converter import PhonemeConverter


class PhonemeAnalyzer:
    def __init__(self):
        # Sample phoneme dictionary (can be expanded using CMU Pronouncing Dictionary)
        self.phoneme_dict = {
            "hello": ["HH", "AH", "L", "OW"],
            "world": ["W", "ER", "L", "D"],
            "and": ["AE", "N", "D"],
            "the": ["DH", "AH"],
            "is": ["IH", "Z"],
            "it": ["IH", "T"],
            "for": ["F", "AO", "R"],
            "this": ["DH", "IH", "S"],
            "from": ["F", "R", "AH", "M"],
            "to": ["T", "UW"],
            "a": ["AH"],
            "with": ["W", "IH", "TH"],
            # Add more words as needed
        }

        # Typical durations (in seconds) for different phoneme types
        self.phoneme_durations = {
            # Vowels generally last longer
            "AA": 0.12, "AE": 0.12, "AH": 0.10, "AO": 0.12, "AW": 0.15,
            "AY": 0.15, "EH": 0.10, "ER": 0.15, "EY": 0.15, "IH": 0.08,
            "IY": 0.10, "OW": 0.12, "OY": 0.15, "UH": 0.08, "UW": 0.10,

            # Consonants are shorter
            "B": 0.06, "CH": 0.08, "D": 0.06, "DH": 0.07, "F": 0.08,
            "G": 0.06, "HH": 0.07, "JH": 0.08, "K": 0.06, "L": 0.08,
            "M": 0.07, "N": 0.07, "NG": 0.09, "P": 0.05, "R": 0.07,
            "S": 0.09, "SH": 0.09, "T": 0.05, "TH": 0.08, "V": 0.07,
            "W": 0.07, "Y": 0.07, "Z": 0.09, "ZH": 0.09
        }

        # Default duration for unknown phonemes
        self.default_duration = 0.08
        self.converter = PhonemeConverter()

    def analyze_text(self, text, estimate_duration=True, duration=None):
        """
        Analyze text and generate phoneme timings without audio

        Parameters:
        - text: Text to analyze
        - estimate_duration: Whether to estimate duration based on phoneme types
        - duration: Override the estimated duration with a specific value (in seconds)

        Returns:
        - List of (phoneme, start_time, end_time) tuples
        """
        # Get phoneme sequence from transcript
        phonemes = self.get_phoneme_sequence(text)

        if not phonemes:
            print("Error: No phonemes generated from text")
            return []

        # Estimate total duration if not provided
        if duration is None:
            if estimate_duration:
                # Estimate based on phoneme durations (average speaking rate)
                total_duration = sum(self.phoneme_durations.get(p, self.default_duration) for p in phonemes)
                # Apply a speaking rate factor (adjust as needed)
                speaking_rate_factor = 1.2
                audio_duration = total_duration * speaking_rate_factor
            else:
                # Default duration if not estimating
                audio_duration = len(phonemes) * 0.1  # 100ms per phoneme as fallback
        else:
            audio_duration = duration

        # Generate weighted timings
        phoneme_timings = self.assign_timings_weighted(phonemes, audio_duration)

        return phoneme_timings

    def get_phoneme_sequence(self, text):
        """Convert text into a sequence of phonemes"""
        words = re.findall(r'\b\w+\b', text.lower())
        phonemes = []

        for word in words:
            if word in self.phoneme_dict:
                phonemes.extend(self.phoneme_dict[word])
            else:
                # For unknown words, use a simple approximation
                # In a real system, you might use a more comprehensive dictionary
                # or a grapheme-to-phoneme converter
                print(f"Warning: Word '{word}' not in phoneme dictionary")
                phonemes.extend(self.converter.get_phonemes(word))
                print(phonemes)
                for char in word:
                    if char in 'aeiou':
                        phonemes.append("AH")  # Default vowel sound
                    else:
                        phonemes.append(char.upper())  # Use character as phoneme

        return phonemes

    def assign_timings_uniform(self, phonemes, audio_duration):
        """Assign uniform timing to each phoneme"""
        duration_per_phoneme = audio_duration / len(phonemes)
        timings = []
        current_time = 0

        for phoneme in phonemes:
            timings.append((phoneme, current_time, current_time + duration_per_phoneme))
            current_time += duration_per_phoneme

        return timings

    def assign_timings_weighted(self, phonemes, audio_duration):
        """Assign timings based on typical phoneme durations"""
        # Calculate total relative duration
        total_relative = sum(self.phoneme_durations.get(p, self.default_duration) for p in phonemes)

        # Scale factor to fit into audio duration
        scale = audio_duration / total_relative

        timings = []
        current_time = 0

        for phoneme in phonemes:
            duration = self.phoneme_durations.get(phoneme, self.default_duration) * scale
            timings.append((phoneme, current_time, current_time + duration))
            current_time += duration

        return timings

    def detect_speech_regions(self, audio_file, window_size=1024, threshold=0.01):
        """
        Basic Voice Activity Detection to find speech regions in audio
        Returns list of (start_time, end_time) tuples
        """
        sample_rate, audio_data = wavfile.read(audio_file)

        # Convert to mono if stereo
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)

        # Normalize audio
        audio_data = audio_data / np.max(np.abs(audio_data))

        # Calculate energy in windows
        num_windows = len(audio_data) // window_size
        energies = []

        for i in range(num_windows):
            start = i * window_size
            end = start + window_size
            window = audio_data[start:end]
            energy = np.sum(window ** 2) / window_size
            energies.append(energy)

        # Find speech segments
        is_speech = np.array(energies) > threshold

        # Group consecutive speech frames
        speech_regions = []
        in_speech = False
        start_frame = 0

        for i, speech in enumerate(is_speech):
            if speech and not in_speech:
                in_speech = True
                start_frame = i
            elif not speech and in_speech:
                in_speech = False
                # Convert frames to time
                start_time = start_frame * window_size / sample_rate
                end_time = i * window_size / sample_rate
                speech_regions.append((start_time, end_time))

        # Handle if audio ends during speech
        if in_speech:
            start_time = start_frame * window_size / sample_rate
            end_time = len(audio_data) / sample_rate
            speech_regions.append((start_time, end_time))

        return speech_regions

    def align_phonemes_to_speech(self, phoneme_timings, speech_regions):
        """
        Align phonemes to detected speech regions
        This is a simplified alignment strategy
        """
        if not speech_regions:
            return phoneme_timings

        # Total speech duration
        total_speech_duration = sum(end - start for start, end in speech_regions)

        # Original phoneme sequence duration
        original_duration = phoneme_timings[-1][2] - phoneme_timings[0][1]

        # Scale factor
        scale = total_speech_duration / original_duration

        # Combine all speech regions
        combined_regions = []
        current_pos = 0

        for start, end in speech_regions:
            duration = end - start
            region_end = current_pos + duration
            combined_regions.append((current_pos, region_end))
            current_pos = region_end

        # Assign phonemes to combined regions
        aligned_timings = []
        region_idx = 0
        region_start, region_end = combined_regions[region_idx]

        for phoneme, start, end in phoneme_timings:
            # Scale timings
            scaled_start = start * scale
            scaled_end = end * scale

            # Find appropriate region
            while scaled_start >= region_end and region_idx < len(combined_regions) - 1:
                region_idx += 1
                region_start, region_end = combined_regions[region_idx]

            # Clip to region boundaries
            actual_start = max(scaled_start, region_start)
            actual_end = min(scaled_end, region_end)

            if actual_end > actual_start:
                aligned_timings.append((phoneme, actual_start, actual_end))

        return aligned_timings

    def analyze_audio(self, audio_file, transcript, use_vad=True):
        """
        Main method to analyze audio and generate phoneme timings

        Parameters:
        - audio_file: Path to WAV audio file
        - transcript: Text transcript of the audio
        - use_vad: Whether to use voice activity detection

        Returns:
        - List of (phoneme, start_time, end_time) tuples
        """
        # Get audio duration
        sample_rate, audio_data = wavfile.read(audio_file)
        audio_duration = len(audio_data) / sample_rate

        # Get phoneme sequence from transcript
        phonemes = self.get_phoneme_sequence(transcript)

        if not phonemes:
            print("Error: No phonemes generated from transcript")
            return []

        # Generate weighted timings
        phoneme_timings = self.assign_timings_weighted(phonemes, audio_duration)

        if use_vad:
            # Detect speech regions
            speech_regions = self.detect_speech_regions(audio_file)

            if speech_regions:
                # Align phonemes to speech regions
                phoneme_timings = self.align_phonemes_to_speech(phoneme_timings, speech_regions)

        return phoneme_timings


# Example usage
if __name__ == "__main__":
    analyzer = PhonemeAnalyzer()

    # Example with a dummy audio file (replace with actual file)
    # audio_file = "audio.wav"
    # transcript = "hello world"

    # If you have the audio file:
    # phoneme_timings = analyzer.analyze_audio(audio_file, transcript)

    # For testing without an audio file:
    phonemes = analyzer.get_phoneme_sequence("hello world")
    phoneme_timings = analyzer.assign_timings_weighted(phonemes, 2.0)  # 2 second duration

    print("Phoneme timings:")
    for phoneme, start, end in phoneme_timings:
        print(f"{phoneme}: {start:.2f}s - {end:.2f}s")
