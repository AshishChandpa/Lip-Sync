# Sample phoneme dictionary (simplified)
phoneme_dict = {
    "hello": ["HH", "AH", "L", "OW"],
    "world": ["W", "ER", "L", "D"]
}

def get_phoneme_sequence(text):
    words = text.lower().split()
    phonemes = []
    for word in words:
        if word in phoneme_dict:
            phonemes.extend(phoneme_dict[word])
    return phonemes

def assign_timings(phonemes, audio_duration):
    # Assume uniform duration per phoneme
    duration_per_phoneme = audio_duration / len(phonemes)
    timings = []
    current_time = 0
    for phoneme in phonemes:
        timings.append((phoneme, current_time, current_time + duration_per_phoneme))
        current_time += duration_per_phoneme
    return timings

# Example usage
text = "hello world"
audio_duration = 2.0  # seconds
phonemes = get_phoneme_sequence(text)
phoneme_timings = assign_timings(phonemes, audio_duration)
# Output: [('HH', 0.0, 0.25), ('AH', 0.25, 0.5), ...]
print(phoneme_timings)