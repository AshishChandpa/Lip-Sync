import cv2
import dlib
import pygame
import numpy as np
import time
import os
import urllib.request
import bz2
import librosa
import soundfile as sf
from scipy.ndimage import gaussian_filter1d
import re


# pip install librosa soundfile scipy

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


# Path to the predictor file
predictor_path = "shape_predictor_68_face_landmarks.dat"
if not os.path.exists(predictor_path):
    download_shape_predictor(predictor_path)

# Initialize dlib's detector and predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)


# --------------- Audio Analysis Functions ---------------
def analyze_audio(audio_path, fps):
    """
    Analyze audio file to extract features for lip sync.
    This implementation uses RMS energy with Gaussian smoothing.
    Returns an array of normalized mouth openness values aligned with video frames.
    When there is silence the value is near 0.2.
    """
    print(f"Analyzing audio file: {audio_path}")
    try:
        # Load audio with librosa
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


# --------------- Utility: Compute Mouth Aspect Ratio (MAR) ---------------
def mouth_aspect_ratio(mouth_points):
    """
    Compute the Mouth Aspect Ratio (MAR) using inner mouth landmarks.
    """
    A = np.linalg.norm(np.array(mouth_points[14]) - np.array(mouth_points[18]))  # p62-p66
    B = np.linalg.norm(np.array(mouth_points[15]) - np.array(mouth_points[17]))  # p63-p65
    C = np.linalg.norm(np.array(mouth_points[12]) - np.array(mouth_points[16]))  # p60-p64
    mar = (A + B) / (2.0 * C) if C > 0 else 0.2
    return mar


# --------------- Utility: Map MAR to Viseme and Warp Factor (Fallback) ---------------
def map_mar_to_viseme(mar):
    """
    Map the MAR value to a discrete viseme label and a corresponding warp factor.
    This function is used as a fallback if no override viseme is provided.
    """
    if mar < 0.25:
        return "REST", 1.0
    elif mar < 0.32:
        return "Slight", 1.2
    elif mar < 0.40:
        return "A", 1.5
    elif mar < 0.48:
        return "O", 1.8
    else:
        return "Wide", 2.2


# --------------- Process a Single Frame ---------------
def process_frame(frame, audio_mar=None, detection_only=False, blend_factor=0.7,
                  silence_threshold=0.21, override_viseme=None, override_warp_factor=None):
    """
    Process a single frame to detect face and mouth.
    If audio_mar is provided, blend it with the video MAR.
    When the audio-derived value is below the silence_threshold,
    force a closed mouth (MAR set to 0.2) and skip mouth warping.

    If override_viseme and override_warp_factor are provided (from text analysis),
    they will override the viseme mapping. For example, if override_viseme is "REST",
    the function forces detection-only mode so that the mouth remains closed.
    """
    frame_copy = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    viseme_label = "None"
    warp_factor = 1.0
    mar_value = 0.0
    mouth_rect = None

    if len(faces) > 0:
        face = faces[0]
        shape = predictor(gray, face)
        mouth_points = []
        for i in range(48, 68):  # Mouth landmarks
            x = shape.part(i).x
            y = shape.part(i).y
            mouth_points.append((x, y))
            cv2.circle(frame_copy, (x, y), 2, (0, 0, 255), -1)

        # Compute MAR from the detected face landmarks (video-based)
        video_mar = mouth_aspect_ratio(mouth_points)

        # Use audio-derived MAR if available and above silence threshold,
        # or force closed mouth (MAR 0.2) when silence is detected.
        if audio_mar is not None and audio_mar < silence_threshold:
            mar_value = 0.2  # Closed mouth when silence detected
            detection_only = True
        else:
            if audio_mar is not None:
                mar_value = (audio_mar * blend_factor) + (video_mar * (1 - blend_factor))
            else:
                mar_value = video_mar

        # If no override is provided, determine the viseme from the computed MAR
        if override_viseme is None or override_warp_factor is None:
            viseme_label, warp_factor = map_mar_to_viseme(mar_value)
        else:
            # Use the viseme and warp factor from the script-based analysis
            viseme_label = override_viseme
            warp_factor = override_warp_factor
            if viseme_label == "REST":
                detection_only = True

        xs = [p[0] for p in mouth_points]
        ys = [p[1] for p in mouth_points]
        left, right = min(xs), max(xs)
        top, bottom = min(ys), max(ys)
        padding = int((bottom - top) * 0.2)
        top = max(0, top - padding)
        bottom = min(frame.shape[0], bottom + padding)
        mouth_rect = (left, top, right - left, bottom - top)

        # If in detection-only mode (i.e. no speech), simply draw the mouth bounding box.
        if detection_only:
            cv2.rectangle(frame_copy, (left, top), (right, bottom), (0, 255, 0), 2)
        else:
            # Otherwise, extract and warp the mouth region for animation.
            mouth_roi = frame[top:bottom, left:right]
            if mouth_roi.size > 0:
                new_height = int((bottom - top) * warp_factor)
                try:
                    warped_mouth = cv2.resize(mouth_roi, (right - left, new_height))
                    center_y = top + (bottom - top) // 2
                    new_top = max(0, center_y - new_height // 2)
                    new_bottom = min(frame.shape[0], new_top + new_height)
                    if new_bottom > new_top:
                        if warped_mouth.shape[0] != new_bottom - new_top:
                            warped_mouth = cv2.resize(warped_mouth, (right - left, new_bottom - new_top))
                        frame_copy[new_top:new_bottom, left:right] = warped_mouth
                except Exception as e:
                    print(f"Error warping mouth: {e}")

    info_text = f"MAR: {mar_value:.2f} | Viseme: {viseme_label} | Warp: {warp_factor:.1f}"
    cv2.putText(frame_copy, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return frame_copy, viseme_label, mar_value


# --------------- Basic Phoneme Analyzer Implementation ---------------
class PhonemeAnalyzer:
    def __init__(self):
        # Simple dictionary; you can expand this or integrate CMU dict/forced aligner later.
        self.phoneme_dict = {
            "hello": ["HH", "AH", "L", "OW"],
            "world": ["W", "ER", "L", "D"],
            "silence": []  # Indicates silence.
        }
        # Default durations for phonemes (in seconds)
        self.phoneme_durations = {
            "HH": 0.07, "AH": 0.10, "L": 0.08, "OW": 0.12,
            "W": 0.07, "ER": 0.15, "D": 0.06
        }
        self.default_duration = 0.08

    def analyze_text(self, text, estimate_duration=True, duration=None):
        phonemes = self.get_phoneme_sequence(text)
        if not phonemes:
            print("Error: No phonemes generated from text")
            return []
        if duration is None:
            if estimate_duration:
                total_duration = sum(self.phoneme_durations.get(p, self.default_duration) for p in phonemes)
                speaking_rate_factor = 1.2
                audio_duration = total_duration * speaking_rate_factor
            else:
                audio_duration = len(phonemes) * 0.1
        else:
            audio_duration = duration
        return self.assign_timings_weighted(phonemes, audio_duration)

    def get_phoneme_sequence(self, text):
        words = re.findall(r'\b\w+\b', text.lower())
        phonemes = []
        for word in words:
            if word in self.phoneme_dict:
                phonemes.extend(self.phoneme_dict[word])
            else:
                # Fallback: each letter becomes a phoneme (simple approximation)
                for char in word:
                    phonemes.append("AH" if char in 'aeiou' else char.upper())
        return phonemes

    def assign_timings_weighted(self, phonemes, audio_duration):
        total_relative = sum(self.phoneme_durations.get(p, self.default_duration) for p in phonemes)
        scale = audio_duration / total_relative if total_relative > 0 else 1.0
        timings = []
        current_time = 0
        for phoneme in phonemes:
            duration = self.phoneme_durations.get(phoneme, self.default_duration) * scale
            timings.append((phoneme, current_time, current_time + duration))
            current_time += duration
        return timings


# --------------- Basic Viseme Mapper Implementation ---------------
class VisemeMapper:
    def __init__(self):
        self.viseme_params = {
            "REST": (0.0, 0.0, 0.5),
            "A": (0.7, 0.0, 0.7),
            "E": (0.5, 0.2, 0.7),
            "I": (0.3, 0.0, 0.8),
            "O": (0.5, 0.8, 0.6),
            "U": (0.3, 0.8, 0.4),
            "Slight": (0.3, 0.0, 0.6),
            "Wide": (0.8, 0.0, 0.8)
        }
        self.phoneme_to_viseme = {
            "HH": "REST",
            "AH": "A",
            "L": "REST",
            "OW": "O",
            "W": "REST",
            "ER": "E",
            "D": "REST"
        }

    def get_viseme_for_phoneme(self, phoneme):
        if phoneme in self.phoneme_to_viseme:
            viseme_name = self.phoneme_to_viseme[phoneme]
            return viseme_name, self.viseme_params.get(viseme_name, self.viseme_params["REST"])
        else:
            return "REST", self.viseme_params["REST"]

    def map_to_viseme_sequence(self, phoneme_timings):
        """
        Given phoneme timings (list of tuples: (phoneme, start, end)),
        map each phoneme to its corresponding viseme and create a sequence.
        """
        viseme_sequence = []
        for phoneme, start_time, end_time in phoneme_timings:
            viseme_name, viseme_params = self.get_viseme_for_phoneme(phoneme)
            viseme_sequence.append((viseme_name, viseme_params, start_time, end_time))
        # Ensure the sequence starts and ends with REST
        if viseme_sequence and viseme_sequence[0][2] > 0:
            viseme_sequence.insert(0, ("REST", self.viseme_params["REST"], 0, viseme_sequence[0][2]))
        if viseme_sequence:
            last_end = viseme_sequence[-1][3]
            viseme_sequence.append(("REST", self.viseme_params["REST"], last_end, last_end + 0.5))
        return viseme_sequence


# --------------- Main Function: Video, Audio, & Pygame Integration ---------------
def main():
    video_path = input("Enter path to video file: ")
    audio_path = input("Enter path to audio file: ")

    if not os.path.exists(video_path):
        print(f"Error: Video file '{video_path}' not found.")
        return

    if not os.path.exists(audio_path):
        print(f"Error: Audio file '{audio_path}' not found.")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video file!")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video: {width}x{height}, {fps} FPS, {frame_count} frames")

    preview = input("Preview face detection first? (y/n): ").lower() == 'y'
    if preview:
        print("Previewing face detection. Press 'q' to continue to processing.")
        while True:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            processed_frame, _, _ = process_frame(frame, detection_only=True)
            cv2.imshow('Face Detection Preview', processed_frame)
            key = cv2.waitKey(int(1000 / fps)) & 0xFF
            if key == ord('q'):
                break
        cv2.destroyAllWindows()
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    print("Analyzing audio file for lip sync...")
    audio_mars = analyze_audio(audio_path, fps)
    if audio_mars is None or len(audio_mars) == 0:
        print("Failed to analyze audio. Using video-based lip sync only.")
        audio_mars = None
    else:
        print(f"Audio analysis complete: {len(audio_mars)} MAR values extracted")

    # Initialize Pygame and set up display and audio playback
    pygame.init()
    pygame.mixer.init()
    pygame.display.set_caption("Enhanced Audio-Driven Lip Sync")
    screen = pygame.display.set_mode((width, height))

    # Ask the user for an optional script text for phoneme-viseme mapping.
    script_text = input("Enter script text for phoneme-viseme mapping (or press ENTER to skip): ")
    viseme_sequence = None
    analyzer = PhonemeAnalyzer()
    mapper = VisemeMapper()
    if script_text.strip() != "":
        phoneme_timings = analyzer.analyze_text(script_text)
        viseme_sequence = mapper.map_to_viseme_sequence(phoneme_timings)
        print("Viseme sequence generated from script:")
        for seg in viseme_sequence:
            print(seg)

    start_time = None
    paused = True
    frame_idx = 0

    print("\nControls:\n  SPACE: Play/Pause\n  ESC: Quit\nPress SPACE to start playback...\n")
    pygame.mixer.music.load(audio_path)
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    if paused:
                        pygame.mixer.music.play()
                        start_time = time.time()
                        paused = False
                        print("Playback started")
                    else:
                        pygame.mixer.music.pause()
                        paused = True
                        print("Playback paused")

        if not paused:
            elapsed_time = time.time() - start_time
            target_frame = int(elapsed_time * fps)
            if target_frame > frame_idx + 1:
                skip_frames = target_frame - frame_idx
                print(f"Audio/video sync: Skipping {skip_frames} frames to catch up")
                frame_idx = target_frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

            ret, frame = cap.read()
            if not ret:
                print("End of video reached")
                pygame.mixer.music.stop()
                break

            # Determine if this frame is silent (using the silence threshold)
            current_audio_mar = None
            if audio_mars is not None and frame_idx < len(audio_mars):
                current_audio_mar = audio_mars[frame_idx]

            # Force detection-only mode when audio is silent.
            force_detection = False
            if current_audio_mar is not None and current_audio_mar < 0.21:
                force_detection = True

            # If a viseme sequence exists from script text, override the viseme mapping.
            override_viseme = None
            override_warp_factor = None
            if viseme_sequence:
                for segment in viseme_sequence:
                    # Each segment is: (viseme_name, viseme_params, start_time, end_time)
                    if segment[2] <= elapsed_time < segment[3]:
                        override_viseme = segment[0]
                        # Use the 3rd element of the viseme_params tuple as the warp factor.
                        override_warp_factor = segment[1][2]
                        break

            processed_frame, viseme, mar = process_frame(frame, audio_mar=current_audio_mar,
                                                         detection_only=force_detection,
                                                         override_viseme=override_viseme,
                                                         override_warp_factor=override_warp_factor)
            processed_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            pygame_frame = pygame.surfarray.make_surface(processed_rgb.swapaxes(0, 1))
            screen.blit(pygame_frame, (0, 0))
            pygame.display.update()
            frame_idx += 1
            if frame_idx >= frame_count:
                print("End of video reached")
                pygame.mixer.music.stop()
                break

        pygame.time.wait(10)

    cap.release()
    pygame.quit()
    cv2.destroyAllWindows()
    print("Processing complete")


if __name__ == "__main__":
    main()
