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
import math
from scipy.ndimage import gaussian_filter1d

# Define viseme shapes more precisely using the same structure as in animator.py
VisemeShape = namedtuple('VisemeShape', ['jaw_open', 'lip_round', 'lip_width', 'tongue_visible', 'teeth_visible'])


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
        Using improved mapping from animator.py
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
            return "Wide", 2.2

    def get_frame(self, frame_idx, is_silence=False):
        """
        Get a specific frame from the video.
        """
        frames = self.silence_video_frames if is_silence else self.speaking_video_frames
        if not frames or frame_idx >= len(frames):
            return None
        return frames[frame_idx]


class RealisticLipSyncAnimator:
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

        # Improved transition time for smoother viseme changes (from animator.py)
        self.transition_time = 0.08  # Faster transitions for realism

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

        # Enhanced viseme map with more detailed parameters (from animator.py)
        self.viseme_map = {
            "REST": VisemeShape(0.0, 0.0, 0.5, False, False),  # Neutral closed mouth
            "A": VisemeShape(0.7, 0.0, 0.7, True, True),  # Wide open "ah" sound
            "E": VisemeShape(0.5, 0.0, 0.8, False, True),  # "eh" as in "bed"
            "I": VisemeShape(0.3, 0.2, 0.7, False, True),  # "ee" as in "see"
            "O": VisemeShape(0.5, 0.8, 0.6, False, True),  # Round "oh" sound
            "U": VisemeShape(0.3, 0.9, 0.4, False, False),  # Tight "oo" sound
            "F": VisemeShape(0.1, 0.5, 0.8, False, True),  # "f" and "v" sounds
            "M": VisemeShape(0.0, 0.5, 0.5, False, False),  # Closed lips for "m", "b", "p"
            "L": VisemeShape(0.3, 0.0, 0.6, True, True),  # Tongue visible for "l"
            "S": VisemeShape(0.2, 0.3, 0.7, False, True),  # "s", "z" sounds
            "T": VisemeShape(0.2, 0.0, 0.7, True, True),  # "t", "d", "n" sounds
            "SH": VisemeShape(0.2, 0.7, 0.5, False, True),  # "sh", "ch", "j" sounds
            "Wide": VisemeShape(0.7, 0.0, 0.8, True, True),  # Extra wide open mouth
            "Slight": VisemeShape(0.3, 0.0, 0.6, False, True),  # Slightly open mouth
        }

        # Blinking parameters (from animator.py)
        self.blink_timer = 0
        self.next_blink = np.random.uniform(2.0, 5.0)  # Random blink interval
        self.is_blinking = False
        self.blink_duration = 0.15  # Blink lasts 0.15 seconds

        # Idle animation parameters (from animator.py)
        self.idle_offset_x = 0
        self.idle_offset_y = 0
        self.idle_timer = 0

    def interpolate_viseme_params(self, shape1, shape2, blend):
        """
        Interpolate between two viseme shapes for smoother transitions.
        Taken from animator.py
        """
        return VisemeShape(
            shape1.jaw_open * (1 - blend) + shape2.jaw_open * blend,
            shape1.lip_round * (1 - blend) + shape2.lip_round * blend,
            shape1.lip_width * (1 - blend) + shape2.lip_width * blend,
            shape2.tongue_visible if blend > 0.5 else shape1.tongue_visible,
            shape2.teeth_visible if blend > 0.5 else shape1.teeth_visible
        )

    def update_blink(self, delta_time):
        """
        Update the blinking animation by checking if it's time to blink.
        Taken from animator.py
        """
        self.blink_timer += delta_time
        if self.is_blinking:
            # If currently blinking, check if the blink duration has passed
            if self.blink_timer >= self.blink_duration:
                self.is_blinking = False
                self.blink_timer = 0
                # Set a new random time until the next blink
                self.next_blink = np.random.uniform(2.0, 5.0)
        else:
            # If not blinking, check if it's time to start a blink
            if self.blink_timer >= self.next_blink:
                self.is_blinking = True
                self.blink_timer = 0

    def update_idle_animation(self, delta_time):
        """
        Update idle animations (like subtle head sway or breathing).
        Taken from animator.py
        """
        self.idle_timer += delta_time
        # Example: a gentle left-right sway and up-down breathing effect
        idle_offset_x = math.sin(self.idle_timer * 0.5) * 3  # Sway horizontally
        idle_offset_y = math.sin(self.idle_timer * 0.3) * 2  # Sway vertically (breathing)
        return idle_offset_x, idle_offset_y

    def process_frame_with_viseme(self, frame, viseme_name, viseme_params):
        """
        Process a frame to apply the viseme parameters to the mouth region.
        Enhanced with more detailed viseme parameters from animator.py
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

            # Extract parameters from viseme_params
            jaw_open, lip_round, lip_width = viseme_params

            # Get tongue and teeth visibility from viseme map if available
            tongue_visible = False
            teeth_visible = False
            if viseme_name in self.viseme_map:
                tongue_visible = self.viseme_map[viseme_name].tongue_visible
                teeth_visible = self.viseme_map[viseme_name].teeth_visible

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

    def get_current_viseme_params(self, viseme_sequence, current_time):
        """
        Get the current viseme parameters based on time.
        Handles transitions between visemes for smoother animation.
        Based on the implementation in animator.py
        """
        if not viseme_sequence:
            return self.viseme_map["REST"]

        current_viseme_idx = None
        for i, (viseme_name, viseme_params, start, end) in enumerate(viseme_sequence):
            if start <= current_time < end:
                current_viseme_idx = i
                break

        if current_viseme_idx is None:
            if current_time < viseme_sequence[0][2]:
                return self.viseme_map[viseme_sequence[0][0]]
            elif current_time >= viseme_sequence[-1][3]:
                return self.viseme_map[viseme_sequence[-1][0]]
            else:
                return self.viseme_map["REST"]

        current_viseme, current_params, start, end = viseme_sequence[current_viseme_idx]

        # Get the viseme shape from our map
        if current_viseme in self.viseme_map:
            current_shape = self.viseme_map[current_viseme]
        else:
            # Fallback to the provided parameters
            current_shape = VisemeShape(current_params[0], current_params[1], current_params[2], False, False)

        # Transition to next viseme if near the end of the segment
        if (current_viseme_idx < len(viseme_sequence) - 1 and
                current_time >= end - self.transition_time):
            next_viseme = viseme_sequence[current_viseme_idx + 1][0]
            next_shape = self.viseme_map.get(next_viseme,
                                             VisemeShape(viseme_sequence[current_viseme_idx + 1][1][0],
                                                         viseme_sequence[current_viseme_idx + 1][1][1],
                                                         viseme_sequence[current_viseme_idx + 1][1][2],
                                                         False, False))

            blend = (current_time - (end - self.transition_time)) / self.transition_time
            blend = max(0, min(1, blend))
            return self.interpolate_viseme_params(current_shape, next_shape, blend)

        # Transition from previous viseme if near the start of the segment
        elif current_time <= start + self.transition_time and current_viseme_idx > 0:
            prev_viseme = viseme_sequence[current_viseme_idx - 1][0]
            prev_shape = self.viseme_map.get(prev_viseme,
                                             VisemeShape(viseme_sequence[current_viseme_idx - 1][1][0],
                                                         viseme_sequence[current_viseme_idx - 1][1][1],
                                                         viseme_sequence[current_viseme_idx - 1][1][2],
                                                         False, False))

            blend = 1 - (current_time - start) / self.transition_time
            blend = max(0, min(1, blend))
            return self.interpolate_viseme_params(current_shape, prev_shape, blend)

        return current_shape

    def map_phonemes_to_visemes(self, phoneme_sequence):
        """
        Map phoneme sequence to viseme sequence.
        Using the improved mapping from animator.py
        """
        phoneme_to_viseme = {
            'AA': 'A', 'AE': 'A', 'AH': 'A',
            'AO': 'O', 'AW': 'A', 'AY': 'A',
            'EH': 'E', 'ER': 'E', 'EY': 'E',
            'IH': 'I', 'IY': 'I',
            'OW': 'O', 'OY': 'O',
            'UH': 'U', 'UW': 'U',
            'B': 'M', 'CH': 'SH', 'D': 'T', 'DH': 'T',
            'F': 'F', 'G': 'T', 'HH': 'REST', 'JH': 'SH',
            'K': 'T', 'L': 'L', 'M': 'M', 'N': 'T',
            'NG': 'T', 'P': 'M', 'R': 'L', 'S': 'S',
            'SH': 'SH', 'T': 'T', 'TH': 'T', 'V': 'F',
            'W': 'U', 'Y': 'I', 'Z': 'S', 'ZH': 'SH',
            'SIL': 'REST', 'SP': 'REST', 'SPX': 'REST', '': 'REST'
        }

        viseme_sequence = []
        for phoneme, start, end in phoneme_sequence:
            viseme = phoneme_to_viseme.get(phoneme, 'REST')
            viseme_params = self.viseme_map.get(viseme, self.viseme_map['REST'])
            viseme_sequence.append(
                (viseme, (viseme_params.jaw_open, viseme_params.lip_round, viseme_params.lip_width), start, end))
        return viseme_sequence

    def generate_viseme_sequence_from_text(self, text, duration=None):
        """
        Generate a viseme sequence from text.
        If duration is provided, it will be used as the total duration of the sequence.
        Enhanced with better phoneme-to-viseme mapping.
        """
        print(f"Generating viseme sequence from text: '{text}'")

        # Use the PhonemeAnalyzer to convert text to phoneme timings
        phoneme_timings = self.phoneme_analyzer.analyze_text(text, estimate_duration=True, duration=duration)

        # Check if we got valid phoneme timings
        if not phoneme_timings:
            print("Warning: No phoneme timings generated. Creating a default sequence.")
            # Create a default sequence if no phonemes were generated
            if duration is None:
                duration = 5.0  # Default duration if none provided

            # Create a simple default sequence
            return [
                ("REST", (
                self.viseme_map["REST"].jaw_open, self.viseme_map["REST"].lip_round, self.viseme_map["REST"].lip_width),
                 0.0, 0.5),
                ("A", (self.viseme_map["A"].jaw_open, self.viseme_map["A"].lip_round, self.viseme_map["A"].lip_width),
                 0.5, 1.0),
                ("E", (self.viseme_map["E"].jaw_open, self.viseme_map["E"].lip_round, self.viseme_map["E"].lip_width),
                 1.0, 1.5),
                ("I", (self.viseme_map["I"].jaw_open, self.viseme_map["I"].lip_round, self.viseme_map["I"].lip_width),
                 1.5, 2.0),
                ("O", (self.viseme_map["O"].jaw_open, self.viseme_map["O"].lip_round, self.viseme_map["O"].lip_width),
                 2.0, 2.5),
                ("U", (self.viseme_map["U"].jaw_open, self.viseme_map["U"].lip_round, self.viseme_map["U"].lip_width),
                 2.5, 3.0),
                ("REST", (
                self.viseme_map["REST"].jaw_open, self.viseme_map["REST"].lip_round, self.viseme_map["REST"].lip_width),
                 3.0, duration)
            ]

        # Debug: print phoneme timings
        print("Phoneme timings:")
        for phoneme, start, end in phoneme_timings:
            print(f"{phoneme}: {start:.2f}s - {end:.2f}s")

        # Use the improved mapping to convert phoneme timings to viseme sequence
        viseme_sequence = self.map_phonemes_to_visemes(phoneme_timings)

        # Debug: print viseme sequence
        print("Viseme sequence:")
        for viseme_name, viseme_params, start, end in viseme_sequence:
            print(f"{viseme_name}: {start:.2f}s - {end:.2f}s, Params: {viseme_params}")

        # Ensure we have non-REST visemes in our sequence
        has_speech = False
        for viseme_name, _, _, _ in viseme_sequence:
            if viseme_name != "REST":
                has_speech = True
                break

        if not has_speech and duration and duration > 1.0:
            print("WARNING: No speech visemes found in the sequence! Forcing some speech visemes.")
            # Add a simple A-E-I-O-U sequence
            viseme_sequence = [
                ("REST", (
                self.viseme_map["REST"].jaw_open, self.viseme_map["REST"].lip_round, self.viseme_map["REST"].lip_width),
                 0.0, 0.5),
                ("A", (self.viseme_map["A"].jaw_open, self.viseme_map["A"].lip_round, self.viseme_map["A"].lip_width),
                 0.5, 1.0),
                ("E", (self.viseme_map["E"].jaw_open, self.viseme_map["E"].lip_round, self.viseme_map["E"].lip_width),
                 1.0, 1.5),
                ("I", (self.viseme_map["I"].jaw_open, self.viseme_map["I"].lip_round, self.viseme_map["I"].lip_width),
                 1.5, 2.0),
                ("O", (self.viseme_map["O"].jaw_open, self.viseme_map["O"].lip_round, self.viseme_map["O"].lip_width),
                 2.0, 2.5),
                ("U", (self.viseme_map["U"].jaw_open, self.viseme_map["U"].lip_round, self.viseme_map["U"].lip_width),
                 2.5, 3.0),
                ("REST", (
                self.viseme_map["REST"].jaw_open, self.viseme_map["REST"].lip_round, self.viseme_map["REST"].lip_width),
                 3.0, duration)
            ]

        print(f"Generated {len(viseme_sequence)} viseme segments")
        return viseme_sequence

    def animate_with_videos(self, audio_file, speaking_video_path, silence_video_path, script_text=None):
        """
        Animate lip sync using data extracted from videos and synchronized with audio.
        If script_text is provided, it will be used to generate a viseme sequence.
        Enhanced with smoother transitions and better viseme mapping.
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

            # Add this check
            if self.viseme_sequence is None:
                print("Error: Failed to generate viseme sequence. Using audio analysis only.")
                self.viseme_sequence = []
            else:
                print(f"Generated viseme sequence with {len(self.viseme_sequence)} segments")
        else:
            self.viseme_sequence = []
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
            delta_time = 1.0 / self.fps  # For animation updates

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
                current_viseme_params = (
                    self.viseme_map["REST"].jaw_open,
                    self.viseme_map["REST"].lip_round,
                    self.viseme_map["REST"].lip_width
                )

                if self.viseme_sequence:
                    # Find the current viseme based on time
                    for viseme_name, viseme_params, start, end in self.viseme_sequence:
                        if start <= current_time < end:
                            # Check for transitions
                            if current_time >= end - self.transition_time and end < self.viseme_sequence[-1][3]:
                                # Transition to next viseme
                                next_idx = next((i for i, v in enumerate(self.viseme_sequence)
                                                 if v[2] >= end), None)
                                if next_idx is not None:
                                    next_viseme = self.viseme_sequence[next_idx][0]
                                    next_params = self.viseme_sequence[next_idx][1]
                                    blend = (current_time - (end - self.transition_time)) / self.transition_time
                                    blend = max(0, min(1, blend))

                                    # Interpolate between current and next viseme
                                    jaw_open = viseme_params[0] * (1 - blend) + next_params[0] * blend
                                    lip_round = viseme_params[1] * (1 - blend) + next_params[1] * blend
                                    lip_width = viseme_params[2] * (1 - blend) + next_params[2] * blend

                                    current_viseme_name = f"{viseme_name}->{next_viseme}"
                                    current_viseme_params = (jaw_open, lip_round, lip_width)
                            elif current_time <= start + self.transition_time and start > self.viseme_sequence[0][
                                2]:
                                # Transition from previous viseme
                                prev_idx = next((i for i, v in enumerate(reversed(self.viseme_sequence))
                                                 if v[3] <= start), None)
                                if prev_idx is not None:
                                    prev_idx = len(self.viseme_sequence) - 1 - prev_idx
                                    prev_viseme = self.viseme_sequence[prev_idx][0]
                                    prev_params = self.viseme_sequence[prev_idx][1]
                                    blend = (current_time - start) / self.transition_time
                                    blend = max(0, min(1, blend))

                                    # Interpolate between previous and current viseme
                                    jaw_open = prev_params[0] * (1 - blend) + viseme_params[0] * blend
                                    lip_round = prev_params[1] * (1 - blend) + viseme_params[1] * blend
                                    lip_width = prev_params[2] * (1 - blend) + viseme_params[2] * blend

                                    current_viseme_name = f"{prev_viseme}->{viseme_name}"
                                    current_viseme_params = (jaw_open, lip_round, lip_width)
                            else:
                                current_viseme_name = viseme_name
                                current_viseme_params = viseme_params
                            break
                elif audio_mars is not None and frame_idx < len(audio_mars):
                    # Use audio analysis to determine mouth openness
                    mar_value = audio_mars[frame_idx]

                    # Map MAR to viseme
                    if mar_value < 0.25:
                        current_viseme_name = "REST"
                        current_viseme_params = (
                            self.viseme_map["REST"].jaw_open,
                            self.viseme_map["REST"].lip_round,
                            self.viseme_map["REST"].lip_width
                        )
                    elif mar_value < 0.32:
                        current_viseme_name = "I"
                        current_viseme_params = (
                            self.viseme_map["I"].jaw_open,
                            self.viseme_map["I"].lip_round,
                            self.viseme_map["I"].lip_width
                        )
                    elif mar_value < 0.40:
                        current_viseme_name = "A"
                        current_viseme_params = (
                            self.viseme_map["A"].jaw_open,
                            self.viseme_map["A"].lip_round,
                            self.viseme_map["A"].lip_width
                        )
                    elif mar_value < 0.48:
                        current_viseme_name = "O"
                        current_viseme_params = (
                            self.viseme_map["O"].jaw_open,
                            self.viseme_map["O"].lip_round,
                            self.viseme_map["O"].lip_width
                        )
                    else:
                        current_viseme_name = "Wide"
                        current_viseme_params = (
                            self.viseme_map["Wide"].jaw_open,
                            self.viseme_map["Wide"].lip_round,
                            self.viseme_map["Wide"].lip_width
                        )

                # Get the appropriate video frame
                frame_to_use = frame_idx

                # Determine whether to use speaking or silence frame
                use_silence = False
                if self.viseme_sequence:
                    # Use silence frames only for REST visemes
                    use_silence = (current_viseme_name == "REST")
                elif audio_mars is not None and frame_idx < len(audio_mars):
                    # Use silence frames when audio is below threshold
                    use_silence = (audio_mars[frame_idx] < self.silence_threshold)

                # Update blinking and idle animations
                self.update_blink(delta_time)
                idle_x, idle_y = self.update_idle_animation(delta_time)

                # Get the frame from the appropriate video
                if self.video_extractor.speaking_video_frames and not use_silence:
                    frame_to_use = frame_idx % len(self.video_extractor.speaking_video_frames)
                    video_frame = self.video_extractor.get_frame(frame_to_use, False)
                elif self.video_extractor.silence_video_frames:
                    frame_to_use = frame_idx % len(self.video_extractor.silence_video_frames)
                    video_frame = self.video_extractor.get_frame(frame_to_use, True)
                else:
                    video_frame = None

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
                    time_text = font.render(f'Time: {current_time:.2f}s | Frame: {frame_idx}', True,
                                            (255, 255, 255))
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
                    # Process with REST viseme
                    rest_params = (
                        self.viseme_map["REST"].jaw_open,
                        self.viseme_map["REST"].lip_round,
                        self.viseme_map["REST"].lip_width
                    )

                    processed_silence = self.process_frame_with_viseme(silence_frame, "REST", rest_params)
                    silence_frame_rgb = cv2.cvtColor(processed_silence, cv2.COLOR_BGR2RGB)
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
    audio_file = input("Enter path to audio file (default: sample_audio.wav): ") or "sample_audio.wav"
    speaking_video = input("Enter path to speaking video file (default: ./speaking.mp4): ") or "./speaking.mp4"
    silence_video = input("Enter path to silence video file (default: ./silence.mp4): ") or "./silence.mp4"

    # Ask for optional script text
    script_text = input("Enter script text for phoneme-viseme mapping (default: Hello World, This is a Lip Sync test.): ") or "Hello World, This is a Lip Sync test."

    # Run the demo
    animator.run_video_based_demo(audio_file, speaking_video, silence_video, script_text)


if __name__ == "__main__":
    main()
