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

# ---------------------------
# Utility: Download Shape Predictor if Missing
# ---------------------------
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

# Define predictor path and download if it is missing.
predictor_path = "shape_predictor_68_face_landmarks.dat"
if not os.path.exists(predictor_path):
    download_shape_predictor(predictor_path)

# ---------------------------
# Initialize Face Detector and Predictor
# ---------------------------
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

# ---------------------------
# Audio Analysis Function: Compute RMS-based MAR values (normalized between 0.2 and 0.6)
# ---------------------------
def analyze_audio(audio_path, fps):
    """
    Analyze an audio file and compute RMS energy values per frame.
    Returns an array of normalized values where 0.2 indicates silence.
    """
    print(f"Analyzing audio file: {audio_path}")
    try:
        y, sr = librosa.load(audio_path, sr=None)
        hop_length = int(sr / fps)  # Align with video frames
        rms = librosa.feature.rms(y=y, frame_length=hop_length, hop_length=hop_length)[0]
        max_rms = np.max(rms) if np.max(rms) > 0 else 1
        normalized_rms = 0.2 + (rms / max_rms) * 0.4  # Values from 0.2 to 0.6
        smoothed_rms = gaussian_filter1d(normalized_rms, sigma=1)
        print(f"Audio analysis complete: {len(smoothed_rms)} frames processed")
        return smoothed_rms
    except Exception as e:
        print(f"Error analyzing audio: {e}")
        return None

# ---------------------------
# Utility: Compute Mouth Aspect Ratio (MAR) from facial landmarks.
# ---------------------------
def mouth_aspect_ratio(mouth_points):
    """
    Compute the Mouth Aspect Ratio (MAR) using inner mouth landmarks.
    """
    A = np.linalg.norm(np.array(mouth_points[14]) - np.array(mouth_points[18]))  # p62-p66
    B = np.linalg.norm(np.array(mouth_points[15]) - np.array(mouth_points[17]))  # p63-p65
    C = np.linalg.norm(np.array(mouth_points[12]) - np.array(mouth_points[16]))  # p60-p64
    mar = (A + B) / (2.0 * C) if C > 0 else 0.2
    return mar

# ---------------------------
# Fallback: Map the computed MAR value to a viseme and warp factor.
# ---------------------------
def map_mar_to_viseme(mar):
    """
    Map the MAR value to a discrete viseme label and a corresponding warp factor.
    This is used if no text-based (override) viseme is provided.
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

# ---------------------------
# Process a Single Video Frame: Detect face, compute MAR, and apply audio/text overrides.
# ---------------------------
def process_frame(frame, audio_mar=None, detection_only=False, blend_factor=0.7,
                  silence_threshold=0.21, override_viseme=None, override_warp_factor=None):
    """
    Process a single frame: detect face landmarks, compute mouth aspect ratio (MAR),
    and then choose a viseme (and apply warping) either based on MAR or text-based override.
    If audio_mar is below a silence threshold, the function forces a closed mouth.
    """
    frame_copy = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    viseme_label = "None"
    warp_factor = 1.0
    mar_value = 0.0

    if len(faces) > 0:
        face = faces[0]
        shape = predictor(gray, face)
        mouth_points = []
        for i in range(48, 68):  # Mouth landmarks
            x = shape.part(i).x
            y = shape.part(i).y
            mouth_points.append((x, y))
            cv2.circle(frame_copy, (x, y), 2, (0, 0, 255), -1)

        # Compute video-based MAR
        video_mar = mouth_aspect_ratio(mouth_points)

        # Use audio analysis to decide whether to animate or hold "REST"
        if audio_mar is not None and audio_mar < silence_threshold:
            mar_value = 0.2  # Force closed mouth (silence)
            detection_only = True
        else:
            if audio_mar is not None:
                # Blend audio-derived MAR with the video-derived MAR
                mar_value = (audio_mar * blend_factor) + (video_mar * (1 - blend_factor))
            else:
                mar_value = video_mar

        # If no override is provided, use MAR mapping; otherwise, use text-based viseme parameters.
        if override_viseme is None or override_warp_factor is None:
            viseme_label, warp_factor = map_mar_to_viseme(mar_value)
        else:
            viseme_label = override_viseme
            warp_factor = override_warp_factor
            if viseme_label == "REST":
                detection_only = True

        # Define mouth region and apply warping if not in detection-only mode.
        xs = [p[0] for p in mouth_points]
        ys = [p[1] for p in mouth_points]
        left, right = min(xs), max(xs)
        top, bottom = min(ys), max(ys)
        padding = int((bottom - top) * 0.2)
        top = max(0, top - padding)
        bottom = min(frame.shape[0], bottom + padding)

        if detection_only:
            cv2.rectangle(frame_copy, (left, top), (right, bottom), (0, 255, 0), 2)
        else:
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

# ---------------------------
# Phoneme Analyzer (Text -> Phoneme Timings)
# ---------------------------
class PhonemeAnalyzer:
    def __init__(self):
        self.phoneme_dict = {
            "hello": ["HH", "AH", "L", "OW"],
            "world": ["W", "ER", "L", "D"],
            "silence": []
        }
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
                for char in word:
                    phonemes.append("AH" if char in 'aeiou' else char.upper())
        return phonemes

    def assign_timings_weighted(self, phonemes, audio_duration):
        total_relative = sum(self.phoneme_durations.get(p, self.default_duration) for p in phonemes)
        scale = audio_duration / total_relative if total_relative > 0 else 1.0
        timings = []
        current_time = 0
        for phoneme in phonemes:
            d = self.phoneme_durations.get(phoneme, self.default_duration) * scale
            timings.append((phoneme, current_time, current_time + d))
            current_time += d
        return timings

# ---------------------------
# Viseme Mapper (Phoneme Timings -> Viseme Sequence)
# ---------------------------
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
            viseme = self.phoneme_to_viseme[phoneme]
            return viseme, self.viseme_params.get(viseme, self.viseme_params["REST"])
        else:
            return "REST", self.viseme_params["REST"]

    def map_to_viseme_sequence(self, phoneme_timings):
        viseme_sequence = []
        for phoneme, start_time, end_time in phoneme_timings:
            viseme, params = self.get_viseme_for_phoneme(phoneme)
            viseme_sequence.append((viseme, params, start_time, end_time))
        if viseme_sequence and viseme_sequence[0][2] > 0:
            viseme_sequence.insert(0, ("REST", self.viseme_params["REST"], 0, viseme_sequence[0][2]))
        if viseme_sequence:
            last_end = viseme_sequence[-1][3]
            viseme_sequence.append(("REST", self.viseme_params["REST"], last_end, last_end + 0.5))
        return viseme_sequence

# ---------------------------
# Main Function: Video, Audio, and Lip Sync Integration
# ---------------------------
def main():
    # Get file paths from user
    video_path = input("Enter path to video file: ").strip()
    audio_path = input("Enter path to audio file: ").strip()

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

    # Optional preview of face detection
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
            if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Analyze audio for lip sync data.
    print("Analyzing audio file for lip sync...")
    audio_mars = analyze_audio(audio_path, fps)
    if audio_mars is None or len(audio_mars) == 0:
        print("Failed to analyze audio. Using video-based lip sync only.")
        audio_mars = None
    else:
        print(f"Audio analysis complete: {len(audio_mars)} MAR values extracted")

    # Ask for optional transcript text for phoneme/viseme mapping.
    script_text = input("Enter transcript text for phoneme-viseme mapping (or press ENTER to skip): ").strip()
    viseme_sequence = None
    analyzer = PhonemeAnalyzer()
    mapper = VisemeMapper()
    if script_text != "":
        phoneme_timings = analyzer.analyze_text(script_text)
        viseme_sequence = mapper.map_to_viseme_sequence(phoneme_timings)
        print("Generated viseme sequence from transcript:")
        for seg in viseme_sequence:
            print(seg)
    else:
        print("No transcript provided. Using audio-based lip sync only.")

    # Initialize Pygame for audio playback and video display.
    pygame.init()
    pygame.mixer.init()
    pygame.display.set_caption("Audio-Driven Lip Sync")
    screen = pygame.display.set_mode((width, height))

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
                        if start_time is None:
                            pygame.mixer.music.play()
                            start_time = time.time()
                        else:
                            pygame.mixer.music.unpause()
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

            # Retrieve current audio MAR for this frame.
            current_audio_mar = None
            if audio_mars is not None and frame_idx < len(audio_mars):
                current_audio_mar = audio_mars[frame_idx]

            # Force detection-only mode if audio is below silence threshold.
            force_detection = False
            if current_audio_mar is not None and current_audio_mar < 0.21:
                force_detection = True

            # If a viseme sequence exists from transcript, override viseme mapping.
            override_viseme = None
            override_warp_factor = None
            if viseme_sequence:
                for segment in viseme_sequence:
                    # Each segment: (viseme_name, params, start_time, end_time)
                    if segment[2] <= elapsed_time < segment[3]:
                        override_viseme = segment[0]
                        # Use the third element of the parameter tuple as the warp factor.
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
