import re
from collections import namedtuple
import urllib.request
import bz2

import cv2
import dlib
import pygame
import numpy as np
import time
import os
from scipy.ndimage import gaussian_filter1d

# Define viseme shapes more precisely
VisemeShape = namedtuple('VisemeShape', ['jaw_open', 'lip_round', 'lip_width', 'tongue_visible', 'teeth_visible'])
# At the beginning of your animator.py file, add:
import dlib
import os
import urllib.request
import bz2


# Download shape predictor if missing
def download_shape_predictor(predictor_path):
    """Download and decompress the dlib shape predictor file."""
    url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
    compressed_file = predictor_path + ".bz2"
    try:
        print("Downloading shape_predictor_68_face_landmarks.dat (this may take a while)...")
        urllib.request.urlretrieve(url, compressed_file)
        print("Download complete. Decompressing...")
        with bz2.BZ2File(compressed_file, 'rb') as f_in:
            with open(predictor_path, 'wb') as f_out:
                f_out.write(f_in.read())
        os.remove(compressed_file)
        print("Decompression complete. Predictor ready.")
    except Exception as e:
        print("Error downloading shape predictor:", e)


# Path to the predictor file
predictor_path = "shape_predictor_68_face_landmarks.dat"
if not os.path.exists(predictor_path):
    download_shape_predictor(predictor_path)


# --------------- Utility: Download Shape Predictor if Missing ---------------
def download_shape_predictor(predictor_path):
    """
    Download and decompress the dlib shape predictor file.
    """
    url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
    compressed_file = predictor_path + ".bz2"
    try:
        print("Downloading shape_predictor_68_face_landmarks.dat (this may take a while)...")
        urllib.request.urlretrieve(url, compressed_file)
        print("Download complete. Decompressing...")
        with bz2.BZ2File(compressed_file, 'rb') as f_in:
            with open(predictor_path, 'wb') as f_out:
                f_out.write(f_in.read())
        os.remove(compressed_file)
        print("Decompression complete. Predictor ready.")
    except Exception as e:
        print("Error downloading shape predictor:", e)


# --------------- Utility: Compute Mouth Aspect Ratio (MAR) ---------------
def mouth_aspect_ratio(mouth_points):
    """
    Compute the Mouth Aspect Ratio (MAR) using inner mouth landmarks.
    """
    try:
        A = np.linalg.norm(np.array(mouth_points[14]) - np.array(mouth_points[18]))  # p62-p66
        B = np.linalg.norm(np.array(mouth_points[15]) - np.array(mouth_points[17]))  # p63-p65
        C = np.linalg.norm(np.array(mouth_points[12]) - np.array(mouth_points[16]))  # p60-p64
        mar = (A + B) / (2.0 * C) if C > 0 else 0.2
        return mar
    except (IndexError, ZeroDivisionError) as e:
        print(f"Error calculating MAR: {e}")
        return 0.2


# --------------- PhonemeAnalyzer Class ---------------
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

    def analyze_text(self, text, estimate_duration=True, duration=None):
        """
        Analyze text and generate phoneme timings without audio
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
                print(f"Warning: Word '{word}' not in phoneme dictionary")
                for char in word:
                    if char in 'aeiou':
                        phonemes.append("AH")  # Default vowel sound
                    else:
                        phonemes.append(char.upper())  # Use character as phoneme

        return phonemes

    def assign_timings_weighted(self, phonemes, audio_duration):
        """Assign timings based on typical phoneme durations"""
        # Calculate total relative duration
        total_relative = sum(self.phoneme_durations.get(p, self.default_duration) for p in phonemes)

        # Scale factor to fit into audio duration
        scale = audio_duration / total_relative if total_relative > 0 else 1.0

        timings = []
        current_time = 0

        for phoneme in phonemes:
            duration = self.phoneme_durations.get(phoneme, self.default_duration) * scale
            timings.append((phoneme, current_time, current_time + duration))
            current_time += duration

        return timings


# --------------- VisemeMapper Class ---------------
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


# Add this class before the RealisticLipSyncAnimator class
class VideoMouthExtractor:
    def __init__(self):
        # Initialize dlib's detector and predictor
        self.detector = dlib.get_frontal_face_detector()

        # Path to the predictor file
        predictor_path = "shape_predictor_68_face_landmarks.dat"
        if not os.path.exists(predictor_path):
            print(f"Error: {predictor_path} not found. Please download it.")
            raise FileNotFoundError(f"Missing {predictor_path}")

        self.predictor = dlib.shape_predictor(predictor_path)

        # Storage for processed frames
        self.processed_frames = {}
        self.speaking_video_frames = []
        self.silence_video_frames = []

    def extract_mouth_data_from_video(self, video_path, is_silence=False):
        """
        Extract mouth data from video frames.
        Returns a list of (frame_idx, mar_value, mouth_rect, viseme_label, warp_factor) tuples.
        Also stores the actual video frames for later use.
        """
        print(f"Processing video: {video_path}")

        # Check if we've already processed this video
        cache_key = f"{video_path}_{is_silence}"
        if cache_key in self.processed_frames:
            print(f"Using cached data for {video_path}")
            return self.processed_frames[cache_key]

        # Check if video file exists
        if not os.path.exists(video_path):
            print(f"Error: Video file {video_path} not found")
            return []

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return []

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        mouth_data = []
        video_frames = []
        frame_idx = 0

        print(f"Starting to process {frame_count} frames from {video_path}")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Store the frame
            video_frames.append(frame.copy())

            try:
                # Process the frame to extract mouth data
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.detector(gray)

                if len(faces) > 0:
                    face = faces[0]
                    shape = self.predictor(gray, face)

                    # Extract mouth landmarks (points 48-67)
                    mouth_points = []
                    for i in range(48, 68):
                        x = shape.part(i).x
                        y = shape.part(i).y
                        mouth_points.append((x, y))

                    # Calculate mouth aspect ratio
                    mar_value = self.mouth_aspect_ratio(mouth_points)

                    # Determine viseme and warp factor based on MAR
                    viseme_label, warp_factor = self.map_mar_to_viseme(mar_value)

                    # Get mouth bounding rectangle
                    xs = [p[0] for p in mouth_points]
                    ys = [p[1] for p in mouth_points]
                    left, right = min(xs), max(xs)
                    top, bottom = min(ys), max(ys)
                    padding = int((bottom - top) * 0.2)
                    top = max(0, top - padding)
                    bottom = min(frame.shape[0], bottom + padding)
                    mouth_rect = (left, top, right - left, bottom - top)

                    # Store the data
                    mouth_data.append((frame_idx, mar_value, mouth_rect, viseme_label, warp_factor))
                else:
                    # No face detected, use default values
                    mouth_data.append((frame_idx, 0.2, None, "REST", 1.0))
            except Exception as e:
                print(f"Error processing frame {frame_idx}: {e}")
                # Add default data for this frame
                mouth_data.append((frame_idx, 0.2, None, "REST", 1.0))

            frame_idx += 1

            # Print progress
            if frame_idx % 100 == 0 or frame_idx == frame_count:
                print(f"Processed {frame_idx}/{frame_count} frames from {video_path}")

        cap.release()

        # Store the frames
        if is_silence:
            self.silence_video_frames = video_frames
        else:
            self.speaking_video_frames = video_frames

        # Cache the results
        self.processed_frames[cache_key] = mouth_data

        print(f"Completed processing {video_path}: {len(mouth_data)} frames")
        return mouth_data

    def mouth_aspect_ratio(self, mouth_points):
        """
        Compute the Mouth Aspect Ratio (MAR) using inner mouth landmarks.
        """
        # Use points 61, 67, 63, 65 for vertical distance (inner mouth height)
        A = np.linalg.norm(np.array(mouth_points[13]) - np.array(mouth_points[19]))  # p61-p67
        # Use points 62, 66 for horizontal distance (inner mouth width)
        B = np.linalg.norm(np.array(mouth_points[14]) - np.array(mouth_points[18]))  # p62-p66
        # Use points 60, 64 for horizontal distance (inner mouth width)
        C = np.linalg.norm(np.array(mouth_points[12]) - np.array(mouth_points[16]))  # p60-p64

        # Calculate MAR
        mar = (A + B) / (2.0 * C) if C > 0 else 0.2
        return mar

    def map_mar_to_viseme(self, mar):
        """
        Map the MAR value to a discrete viseme label and a corresponding warp factor.
        """
        if mar < 0.25:
            return "REST", 1.0
        elif mar < 0.32:
            return "I", 1.2
        elif mar < 0.40:
            return "A", 1.5
        elif mar < 0.48:
            return "O", 1.8
        else:
            return "A", 2.2

    def get_frame(self, frame_idx, is_silence=False):
        """
        Get a specific frame from the video.
        """
        frames = self.silence_video_frames if is_silence else self.speaking_video_frames
        if not frames or frame_idx >= len(frames):
            return None
        return frames[frame_idx]


# --------------- RealisticLipSyncAnimator Class ---------------
class RealisticLipSyncAnimator:
    # def __init__(self, width=800, height=600, transition_time=0.05):
    #     # Initialize pygame
    #     pygame.init()
    #     pygame.mixer.init()
    #
    #     # Setup display
    #     self.width = width
    #     self.height = height
    #     self.screen = pygame.display.set_mode((width, height))
    #     pygame.display.set_caption("Realistic Lip Sync Animation")
    #
    #     # Setup clock
    #     self.clock = pygame.time.Clock()
    #     self.fps = 60
    #
    #     # Animation parameters
    #     self.bg_color = (240, 240, 240)
    #     self.transition_time = transition_time
    #
    #     # Initialize the phoneme analyzer and viseme mapper from the original codebase
    #     self.phoneme_analyzer = PhonemeAnalyzer()
    #     self.viseme_mapper = VisemeMapper()
    #
    #     # Initialize video extractor
    #     self.video_extractor = VideoMouthExtractor()
    #
    #     # Silence threshold for audio analysis
    #     self.silence_threshold = 0.15  # Reduced from default 0.21
    #
    #     # Storage for mouth data
    #     self.speaking_mouth_data = None
    #     self.silence_mouth_data = None
    #
    #     # Viseme sequence from script
    #     self.viseme_sequence = None

    def __init__(self, width=800, height=600):
        # Initialize pygame
        pygame.init()
        pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=4096)

        # Setup display
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Video-Based Lip Sync Animation")

        # Setup clock
        self.clock = pygame.time.Clock()
        self.fps = 30  # Match typical video FPS

        # Animation parameters
        self.bg_color = (0, 0, 0)  # Black background
        self.silence_threshold = 0.21  # Threshold for detecting silence in audio

        # Video processing
        self.video_extractor = VideoMouthExtractor()
        self.speaking_mouth_data = []
        self.silence_mouth_data = []

        # Phoneme and viseme analysis
        from phoneme_analyzer import PhonemeAnalyzer
        from viseme_mapper import VisemeMapper
        self.phoneme_analyzer = PhonemeAnalyzer()
        self.viseme_mapper = VisemeMapper()
        self.viseme_sequence = None
        self.transition_time = 0.05  # Transition time between visemes

    def process_frame_with_viseme(self, frame, viseme_name, viseme_params):
        """
        Process a frame to apply the viseme parameters to the mouth region.
        Uses the viseme parameters from the VisemeMapper class.
        """
        frame_copy = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.video_extractor.detector(gray)

        if len(faces) > 0:
            face = faces[0]
            shape = self.video_extractor.predictor(gray, face)
            mouth_points = []
            for i in range(48, 68):  # Mouth landmarks
                x = shape.part(i).x
                y = shape.part(i).y
                mouth_points.append((x, y))
                cv2.circle(frame_copy, (x, y), 2, (0, 0, 255), -1)

            # Extract mouth region parameters
            xs = [p[0] for p in mouth_points]
            ys = [p[1] for p in mouth_points]
            left, right = min(xs), max(xs)
            top, bottom = min(ys), max(ys)
            padding = int((bottom - top) * 0.2)
            top = max(0, top - padding)
            bottom = min(frame.shape[0], bottom + padding)
            mouth_rect = (left, top, right - left, bottom - top)

            # Extract jaw_open, lip_round, lip_width from viseme_params
            jaw_open, lip_round, lip_width = viseme_params

            # Extract the mouth region
            mouth_roi = frame[top:bottom, left:right]
            if mouth_roi.size > 0:
                # Apply different transformations based on viseme type
                if viseme_name == "REST":
                    # For REST, minimal jaw opening
                    jaw_factor = 0.9  # Slightly closed
                    width_factor = 1.0
                else:
                    # For other visemes, apply parameters more aggressively
                    # Map jaw_open (0.0-1.0) to a range that creates visible difference (0.8-2.0)
                    jaw_factor = 0.8 + jaw_open * 1.2

                    # Map lip_width (0.4-0.8) to a range that creates visible difference (0.8-1.6)
                    width_factor = 0.8 + lip_width * 0.8

                # Calculate new dimensions
                new_height = int((bottom - top) * jaw_factor)
                new_width = int((right - left) * width_factor)

                try:
                    # Resize the mouth region
                    warped_mouth = cv2.resize(mouth_roi, (new_width, new_height))

                    # Calculate placement coordinates
                    center_x = left + (right - left) // 2
                    center_y = top + (bottom - top) // 2

                    # Apply lip rounding effect
                    if lip_round > 0.5:
                        # For rounded visemes (O, U), make mouth more oval and move up slightly
                        # Adjust height to make more oval for rounded sounds
                        oval_factor = 1.0 + lip_round * 0.3
                        warped_mouth = cv2.resize(warped_mouth,
                                                  (new_width, int(new_height * oval_factor)))
                        # Move up slightly for rounded sounds
                        center_y -= int(padding * lip_round * 0.5)

                    # Calculate final placement
                    new_left = max(0, center_x - warped_mouth.shape[1] // 2)
                    new_top = max(0, center_y - warped_mouth.shape[0] // 2)
                    new_right = min(frame.shape[1], new_left + warped_mouth.shape[1])
                    new_bottom = min(frame.shape[0], new_top + warped_mouth.shape[0])

                    # Adjust warped_mouth if needed
                    if warped_mouth.shape[1] != new_right - new_left or warped_mouth.shape[0] != new_bottom - new_top:
                        warped_mouth = cv2.resize(warped_mouth, (new_right - new_left, new_bottom - new_top))

                    # Place the warped mouth back into the frame
                    frame_copy[new_top:new_bottom, new_left:new_right] = warped_mouth

                    # Draw a colored outline for debugging
                    color = (0, 255, 0) if viseme_name == "REST" else (0, 255, 255)
                    cv2.rectangle(frame_copy, (new_left, new_top), (new_right, new_bottom), color, 1)

                except Exception as e:
                    print(f"Error warping mouth: {e}")
                    cv2.rectangle(frame_copy, (left, top), (right, bottom), (255, 0, 0), 2)

        # Add viseme info to the frame
        info_text = f"Viseme: {viseme_name} | Jaw: {jaw_open:.1f}, Round: {lip_round:.1f}, Width: {lip_width:.1f}"
        cv2.putText(frame_copy, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return frame_copy

    def analyze_audio(self, audio_path, fps):
        """
        Analyze audio file to extract features for lip sync.
        Returns an array of normalized mouth openness values aligned with video frames.
        When there is silence the value is near 0.2.
        """
        print(f"Analyzing audio file: {audio_path}")
        try:
            # Load audio with librosa
            import librosa
            y, sr = librosa.load(audio_path, sr=None)
            hop_length = int(sr / fps)  # Align with video frames

            # Compute RMS energy per frame using librosa
            rms = librosa.feature.rms(y=y, frame_length=hop_length, hop_length=hop_length)[0]

            # Normalize RMS values to a range suitable for mouth aspect ratio (0.2 - 0.6)
            max_rms = np.max(rms) if np.max(rms) > 0 else 1
            normalized_rms = 0.2 + (rms / max_rms) * 0.4

            # Smooth the envelope using a Gaussian filter (adjust sigma if needed)
            smoothed_rms = gaussian_filter1d(normalized_rms, sigma=1)

            print(f"Audio analysis complete: {len(smoothed_rms)} frames processed")
            return smoothed_rms
        except Exception as e:
            print(f"Error analyzing audio: {e}")
            return None

    def get_current_viseme(self, viseme_sequence, current_time):
        """
        Get the current viseme based on time from the viseme sequence.
        Handles transitions between visemes.
        """
        if not viseme_sequence:
            return "REST", self.viseme_mapper.viseme_params["REST"]

        # Find the current viseme segment
        current_viseme = None
        for viseme_name, viseme_params, start_time, end_time in viseme_sequence:
            if start_time <= current_time < end_time:
                current_viseme = (viseme_name, viseme_params)
                break

        # If no current viseme found, use the first or last one depending on time
        if current_viseme is None:
            if current_time < viseme_sequence[0][2]:
                current_viseme = (viseme_sequence[0][0], viseme_sequence[0][1])
            else:
                current_viseme = (viseme_sequence[-1][0], viseme_sequence[-1][1])

        return current_viseme

    def get_mouth_data_for_frame(self, frame_idx, current_time, audio_mar=None):
        """
        Get mouth data for a specific frame index.
        Uses speaking video data when audio is active, silence video data otherwise.
        If viseme_sequence is provided, it overrides the viseme selection.
        """
        # IMPORTANT FIX: Lower the silence threshold to ensure more frames are classified as speech
        self.silence_threshold = 0.15  # Reduced from 0.21

        # Default to speaking unless explicitly determined to be silence
        use_silence = False

        # Check if we have a viseme sequence from the script
        if self.viseme_sequence:
            # Find the current viseme based on time
            current_viseme = None
            for viseme_name, viseme_params, start_time, end_time in self.viseme_sequence:
                if start_time <= current_time < end_time:
                    current_viseme = (viseme_name, viseme_params)
                    break

            # If we found a current viseme, use it to determine if we should show speaking
            if current_viseme:
                viseme_name, _ = current_viseme
                # Only use silence for REST viseme, otherwise use speaking
                use_silence = (viseme_name == "REST")
        # If no viseme sequence or no current viseme found, fall back to audio analysis
        elif audio_mar is not None:
            use_silence = (audio_mar < self.silence_threshold)

        # IMPORTANT: Force speaking frames for the first few seconds to debug
        if current_time < 5.0:
            use_silence = False

        # Find the closest frame in the appropriate dataset
        if use_silence and self.silence_mouth_data:
            # Use silence data
            closest_idx = min(range(len(self.silence_mouth_data)),
                              key=lambda i: abs(
                                  self.silence_mouth_data[i][0] - frame_idx % len(self.silence_mouth_data)))
            return self.silence_mouth_data[closest_idx], True
        elif self.speaking_mouth_data:
            # Use speaking data
            closest_idx = min(range(len(self.speaking_mouth_data)),
                              key=lambda i: abs(
                                  self.speaking_mouth_data[i][0] - frame_idx % len(self.speaking_mouth_data)))
            return self.speaking_mouth_data[closest_idx], False
        else:
            # Fallback if no data is available
            return (frame_idx, 0.2, None, "REST", 1.0), False

    def process_videos(self, speaking_video_path, silence_video_path):
        """
        Process both speaking and silence videos to extract mouth data.
        """
        print("Processing speaking video...")
        speaking_data = self.video_extractor.extract_mouth_data_from_video(speaking_video_path, False)
        if not speaking_data:
            print("Warning: No mouth data extracted from speaking video")
        self.speaking_mouth_data = speaking_data

        print("Processing silence video...")
        silence_data = self.video_extractor.extract_mouth_data_from_video(silence_video_path, True)
        if not silence_data:
            print("Warning: No mouth data extracted from silence video")
        self.silence_mouth_data = silence_data

        print("Video processing complete.")

        # Make sure we have at least some data to work with
        has_speaking_data = len(self.speaking_mouth_data) > 0 if self.speaking_mouth_data else False
        has_silence_data = len(self.silence_mouth_data) > 0 if self.silence_mouth_data else False

        if not has_speaking_data:
            print("Error: No mouth data available from speaking video")
        if not has_silence_data:
            print("Error: No mouth data available from silence video")

        return has_speaking_data and has_silence_data

    def generate_viseme_sequence_from_text(self, text, duration=None):
        """
        Generate a viseme sequence from text.
        If duration is provided, it will be used as the total duration of the sequence.
        """
        print(f"Generating viseme sequence from text: '{text}'")
        # Use the PhonemeAnalyzer to convert text to phoneme timings
        phoneme_timings = self.phoneme_analyzer.analyze_text(text, estimate_duration=True, duration=duration)

        # Debug: print phoneme timings
        print("Phoneme timings:")
        for phoneme, start, end in phoneme_timings:
            print(f"{phoneme}: {start:.2f}s - {end:.2f}s")

        # Use the VisemeMapper to convert phoneme timings to viseme sequence
        viseme_sequence = self.viseme_mapper.map_to_viseme_sequence(phoneme_timings)

        # Debug: print viseme sequence
        print("Viseme sequence:")
        for viseme_name, viseme_params, start, end in viseme_sequence:
            print(f"{viseme_name}: {start:.2f}s - {end:.2f}s, Params: {viseme_params}")

        return viseme_sequence

        # # IMPORTANT: Make sure we have non-REST visemes in our sequence
        # has_speech = False
        # for viseme_name, _, _, _ in viseme_sequence:
        #     if viseme_name != "REST":
        #         has_speech = True
        #         break
        #
        # if not has_speech:
        #     print("WARNING: No speech visemes found in the sequence! Forcing some speech visemes.")
        #     # Force some speech visemes if none were generated
        #     if duration and duration > 1.0:
        #         # Add a simple A-E-I-O-U sequence
        #         viseme_sequence = [
        #             ("REST", self.viseme_mapper.viseme_params["REST"], 0.0, 0.5),
        #             ("A", self.viseme_mapper.viseme_params["A"], 0.5, 1.0),
        #             ("E", self.viseme_mapper.viseme_params["E"], 1.0, 1.5),
        #             ("I", self.viseme_mapper.viseme_params["I"], 1.5, 2.0),
        #             ("O", self.viseme_mapper.viseme_params["O"], 2.0, 2.5),
        #             ("U", self.viseme_mapper.viseme_params["U"], 2.5, 3.0),
        #             ("REST", self.viseme_mapper.viseme_params["REST"], 3.0, duration)
        #         ]
        #
        # print(f"Generated {len(viseme_sequence)} viseme segments")
        # return viseme_sequence

    def animate_with_videos(self, audio_file, speaking_video_path, silence_video_path, script_text=None):
        """
        Animate lip sync using data extracted from videos and synchronized with audio.
        If script_text is provided, it will be used to generate a viseme sequence.
        """
        # Process videos first
        if not self.process_videos(speaking_video_path, silence_video_path):
            print("Error: Failed to process videos.")
            return

        # Load audio file
        try:
            # Check if audio file exists and is readable
            if not os.path.exists(audio_file):
                print(f"Error: Audio file {audio_file} not found.")
                return

            # Try to load the audio file
            pygame.mixer.music.load(audio_file)
            print(f"Successfully loaded audio file: {audio_file}")
        except Exception as e:
            print(f"Error loading audio {audio_file}: {e}")
            return

        # Get video properties from the speaking video
        cap = cv2.VideoCapture(speaking_video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {speaking_video_path}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        print(f"Video properties: {width}x{height}, {fps} FPS, {frame_count} frames")

        # Resize pygame window to match video dimensions
        if width > 0 and height > 0:
            self.width = width
            self.height = height
            self.screen = pygame.display.set_mode((width, height))

        # Analyze audio to get MAR values
        audio_mars = self.analyze_audio(audio_file, fps)
        if audio_mars is None:
            print("Warning: Failed to analyze audio. Using video-based lip sync only.")
        else:
            print(f"Audio analysis produced {len(audio_mars)} MAR values")

        # Generate viseme sequence from script text if provided
        if script_text:
            # Get audio duration
            audio_duration = None
            if audio_mars is not None:
                audio_duration = len(audio_mars) / fps
            self.viseme_sequence = self.generate_viseme_sequence_from_text(script_text, audio_duration)
            print(f"Generated viseme sequence with {len(self.viseme_sequence)} segments")
        else:
            self.viseme_sequence = None
            print("No script text provided, using audio analysis only")

        # Animation loop setup
        running = True
        paused = True
        start_time = None
        current_time = 0
        frame_idx = 0

        font = pygame.font.SysFont('Arial', 18)
        control_text = font.render('Space: Play/Pause, Esc: Quit, R: Restart', True, (255, 255, 255))
        control_rect = control_text.get_rect(topleft=(10, 10))

        print("Ready to animate. Press SPACE to start playback.")

        # Main animation loop
        while running:
            # Process events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        if paused:
                            if start_time is None:
                                # First play
                                pygame.mixer.music.play()
                                start_time = time.time()
                                print("Starting playback")
                            else:
                                # Resume from pause
                                pygame.mixer.music.unpause()
                                print("Resuming playback")
                            paused = False
                        else:
                            # Pause playback
                            pygame.mixer.music.pause()
                            paused = True
                            print("Paused playback")
                    elif event.key == pygame.K_r:
                        # Restart playback
                        pygame.mixer.music.stop()
                        pygame.mixer.music.play()
                        start_time = time.time()
                        frame_idx = 0
                        paused = False
                        print("Restarting playback")

            # Update animation if playing
            if start_time is not None and not paused:
                # Calculate current time and frame index
                current_time = time.time() - start_time
                target_frame = int(current_time * fps)

                # Handle frame skipping if necessary
                if target_frame > frame_idx + 1:
                    skip_frames = target_frame - frame_idx
                    print(f"Audio/video sync: Skipping {skip_frames} frames to catch up")
                    frame_idx = target_frame

                # Get current viseme from sequence if available
                current_viseme_name = "REST"
                current_viseme_params = self.viseme_mapper.viseme_params["REST"]

                if self.viseme_sequence:
                    for viseme_name, viseme_params, start_time, end_time in self.viseme_sequence:
                        if start_time <= current_time < end_time:
                            current_viseme_name = viseme_name
                            current_viseme_params = viseme_params
                            break

                # Get the appropriate video frame
                frame_to_use = frame_idx

                # IMPORTANT FIX: Always use speaking frames during active playback,
                # regardless of viseme type. Only use silence frames when paused or not started.
                if self.video_extractor.speaking_video_frames:
                    frame_to_use = frame_idx % len(self.video_extractor.speaking_video_frames)
                    video_frame = self.video_extractor.get_frame(frame_to_use, False)
                    use_silence = False
                elif self.video_extractor.silence_video_frames:
                    frame_to_use = frame_idx % len(self.video_extractor.silence_video_frames)
                    video_frame = self.video_extractor.get_frame(frame_to_use, True)
                    use_silence = True
                else:
                    video_frame = None
                    use_silence = True

                if video_frame is not None:
                    # Process the frame with the current viseme
                    processed_frame = self.process_frame_with_viseme(video_frame, current_viseme_name,
                                                                     current_viseme_params)

                    # Convert OpenCV BGR to RGB for pygame
                    video_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)

                    # Create pygame surface from numpy array
                    video_surface = pygame.surfarray.make_surface(video_frame_rgb.swapaxes(0, 1))

                    # Scale to fit screen if needed
                    if video_surface.get_width() != self.width or video_surface.get_height() != self.height:
                        video_surface = pygame.transform.scale(video_surface, (self.width, self.height))

                    # Display the frame
                    self.screen.blit(video_surface, (0, 0))

                    # Display current information
                    time_text = font.render(f'Time: {current_time:.2f}s | Frame: {frame_idx}', True, (255, 255, 255))
                    self.screen.blit(time_text, (10, self.height - 80))

                    viseme_text = font.render(f'Viseme: {current_viseme_name}', True, (255, 255, 255))
                    self.screen.blit(viseme_text, (10, self.height - 50))

                    status = 'Playing' if not paused else 'Paused'
                    status_text = font.render(
                        f'Status: {status} | Using: {"Silence" if use_silence else "Speaking"} frame', True,
                        (255, 255, 255))
                    self.screen.blit(status_text, (10, self.height - 20))
                else:
                    # If no frame is available, just fill with background color
                    self.screen.fill(self.bg_color)

                # Increment frame index
                frame_idx += 1

                # Check if we've reached the end of the audio
                if not pygame.mixer.music.get_busy() and not paused:
                    print("End of audio reached")
                    paused = True
            else:
                # When paused or not started, show the first frame of the silence video
                silence_frame = self.video_extractor.get_frame(0, True)
                if silence_frame is not None:
                    silence_frame_rgb = cv2.cvtColor(silence_frame, cv2.COLOR_BGR2RGB)
                    silence_surface = pygame.surfarray.make_surface(silence_frame_rgb.swapaxes(0, 1))
                    if silence_surface.get_width() != self.width or silence_surface.get_height() != self.height:
                        silence_surface = pygame.transform.scale(silence_surface, (self.width, self.height))
                    self.screen.blit(silence_surface, (0, 0))
                else:
                    self.screen.fill(self.bg_color)

                # Display paused/ready message
                if start_time is None:
                    msg = "Press SPACE to start"
                else:
                    msg = "PAUSED - Press SPACE to resume"

                msg_text = font.render(msg, True, (255, 255, 255))
                msg_rect = msg_text.get_rect(center=(self.width // 2, self.height // 2))
                self.screen.blit(msg_text, msg_rect)

            # Always display control instructions
            self.screen.blit(control_text, control_rect)
            pygame.display.flip()
            self.clock.tick(self.fps)

        pygame.mixer.music.stop()
        print("Animation complete")

    def run_video_based_demo(self, audio_file, speaking_video, silence_video, script_text=None):
        """
        Run a demo using video-based mouth extraction.

        Parameters:
        - audio_file: Path to the audio file to play
        - speaking_video: Path to the video with speaking expressions
        - silence_video: Path to the video with silence/neutral expressions
        - script_text: Optional text script for phoneme-to-viseme mapping
        """
        # Validate input files
        if not os.path.exists(audio_file):
            print(f"Error: Audio file {audio_file} not found. Please check the path.")
            return

        if not os.path.exists(speaking_video):
            print(f"Error: Speaking video file {speaking_video} not found.")
            return

        if not os.path.exists(silence_video):
            print(f"Error: Silence video file {silence_video} not found.")
            return

        # Run the animation
        self.animate_with_videos(audio_file, speaking_video, silence_video, script_text)


def main():
    # Default window size, will be adjusted to match video dimensions
    animator = RealisticLipSyncAnimator(width=800, height=600)

    # Ask user for input files
    audio_file = "sample_audio.wav"  # input("Enter path to audio file: ")
    speaking_video = "./speaking.mp4"  # input("Enter path to speaking video file: ")
    silence_video = "./silence.mp4"  # input("Enter path to silence video file: ")

    # Ask for optional script text
    script_text = input("Enter script text for phoneme-viseme mapping (or press ENTER to skip): ")
    if not script_text.strip():
        script_text = "Hello World, This is a Lip Sync test."

    # Run the demo
    animator.run_video_based_demo(audio_file, speaking_video, silence_video, script_text)


if __name__ == "__main__":
    main()
