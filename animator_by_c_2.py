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

        # New: Store detailed mouth shapes for each frame
        self.speaking_mouth_shapes = []
        self.silence_mouth_shapes = []

    def extract_and_map_viseme_images(self, video_path, output_folder="mouth_images", json_file="viseme_map.json"):
        """
        Extract mouth images from video and create a mapping between phonemes, visemes, and image files.
        Stores the mapping in a JSON file for easy access.
        """
        import json

        print(f"Extracting and mapping viseme images from: {video_path}")

        # Create output folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)

        # Open the video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return False

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Initialize mapping dictionaries
        viseme_to_images = {}  # Maps viseme labels to lists of image filenames
        phoneme_to_images = {}  # Maps phonemes to lists of image filenames
        best_examples = {}  # Maps visemes to their best example images

        frame_idx = 0
        processed_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

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

                    # Determine viseme based on MAR
                    viseme_label, _ = self.map_mar_to_viseme(mar_value)

                    # Calculate mouth bounding box with padding
                    xs = [p[0] for p in mouth_points]
                    ys = [p[1] for p in mouth_points]
                    left, right = min(xs), max(xs)
                    top, bottom = min(ys), max(ys)

                    # Add padding
                    padding = int((bottom - top) * 0.5)  # 50% padding
                    top = max(0, top - padding)
                    bottom = min(frame.shape[0], bottom + padding)
                    left = max(0, left - padding)
                    right = min(frame.shape[1], right + padding)

                    # Extract mouth region
                    mouth_img = frame[top:bottom, left:right]

                    # Save the mouth image
                    filename = f"viseme_{viseme_label}_frame_{frame_idx:06d}_{mar_value:.2f}.jpg"
                    filepath = os.path.join(output_folder, filename)
                    cv2.imwrite(filepath, mouth_img)

                    # Update the viseme-to-images mapping
                    if viseme_label not in viseme_to_images:
                        viseme_to_images[viseme_label] = []
                    viseme_to_images[viseme_label].append(filename)

                    # Update best examples (choose images with MAR closest to the ideal for each viseme)
                    if viseme_label not in best_examples or self.is_better_example(viseme_label, mar_value,
                                                                                   best_examples.get(viseme_label,
                                                                                                     {}).get('mar', 0)):
                        best_examples[viseme_label] = {
                            'filename': filename,
                            'mar': mar_value,
                            'frame_idx': frame_idx
                        }

                    processed_count += 1
                else:
                    print(f"No face detected in frame {frame_idx}")

            except Exception as e:
                print(f"Error processing frame {frame_idx}: {e}")

            frame_idx += 1

            # Print progress
            if frame_idx % 100 == 0 or frame_idx == frame_count:
                print(f"Processed {frame_idx}/{frame_count} frames")

        cap.release()

        # Create phoneme-to-images mapping using the viseme mapper
        for phoneme, viseme in self.viseme_mapper.phoneme_to_viseme.items():
            if viseme in viseme_to_images:
                phoneme_to_images[phoneme] = viseme_to_images[viseme]

        # Create the final mapping dictionary
        mapping = {
            'video_source': video_path,
            'frame_count': frame_count,
            'fps': fps,
            'processed_frames': processed_count,
            'viseme_to_images': viseme_to_images,
            'phoneme_to_images': phoneme_to_images,
            'best_examples': best_examples
        }

        # Save the mapping to a JSON file
        with open(json_file, 'w') as f:
            json.dump(mapping, f, indent=2)

        print(f"Extracted {processed_count} mouth images to {output_folder}")
        print(f"Created viseme mapping in {json_file}")
        print(f"Visemes found: {list(viseme_to_images.keys())}")

        return True

    def generate_script_from_audio(self, audio_file):
        """
        Convert audio file to text script using speech recognition.
        Requires the SpeechRecognition library.
        """
        try:
            import speech_recognition as sr
            print(f"Converting audio to text: {audio_file}")

            # Initialize recognizer
            r = sr.Recognizer()

            # Load the audio file
            with sr.AudioFile(audio_file) as source:
                # Read the audio data
                audio_data = r.record(source)

                # Convert speech to text
                print("Recognizing speech...")
                text = r.recognize_google(audio_data)

                print(f"Generated script: '{text}'")
                return text
        except ImportError:
            print("Error: SpeechRecognition library not installed.")
            print("Install it with: pip install SpeechRecognition")
            return None
        except Exception as e:
            print(f"Error converting audio to text: {e}")
            return None

    def is_better_example(self, viseme, mar, current_best_mar):
        """
        Determine if this is a better example of the viseme based on MAR values.
        Different visemes have different ideal MAR values.
        """
        # Define ideal MAR values for each viseme
        ideal_mars = {
            "REST": 0.2,
            "A": 0.45,
            "E": 0.35,
            "I": 0.3,
            "O": 0.42,
            "U": 0.28,
            "F": 0.25,
            "M": 0.2,
            "L": 0.32,
            "S": 0.25,
            "T": 0.3,
            "SH": 0.28,
            "Wide": 0.5
        }

        # Get the ideal MAR for this viseme, default to 0.3 if not defined
        ideal = ideal_mars.get(viseme, 0.3)

        # Calculate how close this MAR is to the ideal
        current_distance = abs(current_best_mar - ideal)
        new_distance = abs(mar - ideal)

        # Return True if this is closer to the ideal
        return new_distance < current_distance

    def extract_mouth_images_from_video(self, video_path, output_folder="mouth_images"):
        """
        Extract actual mouth images from a video and save them with metadata.
        This allows us to reuse the exact mouth shapes for lip sync.
        """
        print(f"Extracting mouth images from video: {video_path}")

        # Create output folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)

        # Open the video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return False

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Create a metadata file
        metadata_file = os.path.join(output_folder, "metadata.txt")
        with open(metadata_file, 'w') as f:
            f.write(f"Video: {video_path}\n")
            f.write(f"Frames: {frame_count}\n")
            f.write(f"FPS: {fps}\n")
            f.write("frame_idx,mar,viseme,filename\n")

        frame_idx = 0
        mouth_data = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

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

                    # Determine viseme based on MAR
                    viseme_label, _ = self.map_mar_to_viseme(mar_value)

                    # Calculate mouth bounding box with padding
                    xs = [p[0] for p in mouth_points]
                    ys = [p[1] for p in mouth_points]
                    left, right = min(xs), max(xs)
                    top, bottom = min(ys), max(ys)

                    # Add padding
                    padding = int((bottom - top) * 0.5)  # 50% padding
                    top = max(0, top - padding)
                    bottom = min(frame.shape[0], bottom + padding)
                    left = max(0, left - padding)
                    right = min(frame.shape[1], right + padding)

                    # Extract mouth region
                    mouth_img = frame[top:bottom, left:right]

                    # Save the mouth image
                    filename = f"frame_{frame_idx:06d}_{viseme_label}_{mar_value:.2f}.jpg"
                    filepath = os.path.join(output_folder, filename)
                    cv2.imwrite(filepath, mouth_img)

                    # Store metadata
                    mouth_data.append((frame_idx, mar_value, viseme_label, filename))

                    # Write to metadata file
                    with open(metadata_file, 'a') as f:
                        f.write(f"{frame_idx},{mar_value:.4f},{viseme_label},{filename}\n")

                else:
                    print(f"No face detected in frame {frame_idx}")

            except Exception as e:
                print(f"Error processing frame {frame_idx}: {e}")

            frame_idx += 1

            # Print progress
            if frame_idx % 100 == 0 or frame_idx == frame_count:
                print(f"Processed {frame_idx}/{frame_count} frames")

        cap.release()
        print(f"Extracted {len(mouth_data)} mouth images to {output_folder}")
        return True

    def extract_detailed_mouth_data(self, video_path, is_silence=False):
        """
        Extract detailed mouth shape data from video frames.
        Captures the exact shape of the lips for more realistic animation.
        """
        print(f"Processing video for detailed mouth shapes: {video_path}")

        # Check if video file exists
        if not os.path.exists(video_path):
            print(f"Error: Video file {video_path} not found")
            return []

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return []

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_frames = []
        mouth_shapes = []
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

                    # Extract inner mouth points (60-67)
                    inner_mouth_points = []
                    for i in range(60, 68):
                        x = shape.part(i).x
                        y = shape.part(i).y
                        inner_mouth_points.append((x, y))

                    # Calculate mouth bounding box
                    xs = [p[0] for p in mouth_points]
                    ys = [p[1] for p in mouth_points]
                    left, right = min(xs), max(xs)
                    top, bottom = min(ys), max(ys)

                    # Calculate mouth center
                    center_x = (left + right) // 2
                    center_y = (top + bottom) // 2

                    # Calculate mouth width and height
                    width = right - left
                    height = bottom - top

                    # Store detailed mouth shape data
                    mouth_shape = {
                        'frame_idx': frame_idx,
                        'mar': mar_value,
                        'outer_points': mouth_points,
                        'inner_points': inner_mouth_points,
                        'bbox': (left, top, width, height),
                        'center': (center_x, center_y),
                        'viseme': self.map_mar_to_viseme(mar_value)[0]
                    }

                    mouth_shapes.append(mouth_shape)
                else:
                    # No face detected, use default values
                    mouth_shapes.append({
                        'frame_idx': frame_idx,
                        'mar': 0.2,
                        'outer_points': None,
                        'inner_points': None,
                        'bbox': None,
                        'center': None,
                        'viseme': "REST"
                    })
            except Exception as e:
                print(f"Error processing frame {frame_idx}: {e}")
                # Add default data for this frame
                mouth_shapes.append({
                    'frame_idx': frame_idx,
                    'mar': 0.2,
                    'outer_points': None,
                    'inner_points': None,
                    'bbox': None,
                    'center': None,
                    'viseme': "REST"
                })

            frame_idx += 1

            # Print progress
            if frame_idx % 100 == 0 or frame_idx == frame_count:
                print(f"Processed {frame_idx}/{frame_count} frames from {video_path}")

        cap.release()

        # Store the frames and mouth shapes
        if is_silence:
            self.silence_video_frames = video_frames
            self.silence_mouth_shapes = mouth_shapes
        else:
            self.speaking_video_frames = video_frames
            self.speaking_mouth_shapes = mouth_shapes

        print(f"Completed processing {video_path}: {len(mouth_shapes)} frames with detailed mouth data")
        return mouth_shapes

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


    def animate_with_mapped_visemes(self, audio_file, speaking_video_path, silence_video_path,
                                    json_mapping_file="viseme_map.json", script_text=None):
        """
        Enhanced animation that uses the JSON mapping between phonemes, visemes, and images.
        If script_text is None, it will attempt to generate it from the audio file.
        """
        import json

        # Load the JSON mapping
        try:
            with open(json_mapping_file, 'r') as f:
                mapping = json.load(f)
            print(f"Loaded viseme mapping from {json_mapping_file}")
        except Exception as e:
            print(f"Error loading mapping file: {e}")
            print("Falling back to standard animation")
            self.animate_with_videos(audio_file, speaking_video_path, silence_video_path, script_text)
            return

        # Process videos
        if not self.process_videos(speaking_video_path, silence_video_path):
            print("Error: Failed to process videos.")
            return

        # If no script text provided, try to generate it from audio
        if script_text is None:
            script_text = self.generate_script_from_audio(audio_file)
            if not script_text:
                print("Warning: Could not generate script from audio.")
                print("Using default animation without script.")

        # Load audio file
        try:
            pygame.mixer.music.load(audio_file)
            print(f"Successfully loaded audio file: {audio_file}")
        except Exception as e:
            print(f"Error loading audio {audio_file}: {e}")
            return

        # Get video properties
        cap = cv2.VideoCapture(speaking_video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {speaking_video_path}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        # Resize pygame window to match video dimensions
        if width > 0 and height > 0:
            self.width = width
            self.height = height
            self.screen = pygame.display.set_mode((width, height))

        # Generate viseme sequence from script text if provided
        if script_text:
            # Get audio duration
            import librosa
            y, sr = librosa.load(audio_file, sr=None)
            audio_duration = len(y) / sr

            # Generate phoneme timings
            phoneme_timings = self.phoneme_analyzer.analyze_text(script_text, estimate_duration=True,
                                                                 duration=audio_duration)

            # Convert to viseme sequence with image filenames
            viseme_sequence = []
            for phoneme, start, end in phoneme_timings:
                # Get the viseme for this phoneme
                viseme = self.viseme_mapper.phoneme_to_viseme.get(phoneme, "REST")

                # Get the best example image for this viseme
                image_filename = None
                if viseme in mapping['best_examples']:
                    image_filename = mapping['best_examples'][viseme]['filename']

                # Add to sequence
                viseme_sequence.append((viseme, image_filename, start, end))

            print(f"Generated viseme sequence with {len(viseme_sequence)} segments")
        else:
            viseme_sequence = None

        # Animation loop setup
        running = True
        paused = True
        start_time = None
        current_time = 0
        frame_idx = 0

        font = pygame.font.SysFont('Arial', 18)
        control_text = font.render('Space: Play/Pause, Esc: Quit, R: Restart', True, (255, 255, 255))
        control_rect = control_text.get_rect(topleft=(10, 10))

        print("Ready to animate with mapped visemes. Press SPACE to start playback.")

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
                                pygame.mixer.music.play()
                                start_time = time.time()
                                print("Starting playback")
                            else:
                                pygame.mixer.music.unpause()
                                print("Resuming playback")
                            paused = False
                        else:
                            pygame.mixer.music.pause()
                            paused = True
                            print("Paused playback")
                    elif event.key == pygame.K_r:
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
                    frame_idx = target_frame

                # Get current viseme and image filename from sequence
                current_viseme = "REST"
                current_image = None

                if viseme_sequence:
                    for viseme, image_filename, start, end in viseme_sequence:
                        if start <= current_time < end:
                            current_viseme = viseme
                            current_image = image_filename
                            break

                # Determine whether to use speaking or silence frame
                use_silence = (current_viseme == "REST")

                # Get the base frame from the appropriate video
                if self.video_extractor.speaking_video_frames and not use_silence:
                    frame_to_use = frame_idx % len(self.video_extractor.speaking_video_frames)
                    base_frame = self.video_extractor.get_frame(frame_to_use, False)
                elif self.video_extractor.silence_video_frames:
                    frame_to_use = frame_idx % len(self.video_extractor.silence_video_frames)
                    base_frame = self.video_extractor.get_frame(frame_to_use, True)
                else:
                    base_frame = None

                if base_frame is not None:
                    # Create a copy of the base frame to work with
                    frame_copy = base_frame.copy()

                    # If we have a mapped image for this viseme, use it
                    if current_image:
                        # Load the mouth image
                        mouth_img_path = os.path.join(os.path.dirname(json_mapping_file), current_image)
                        if os.path.exists(mouth_img_path):
                            mouth_img = cv2.imread(mouth_img_path)

                            # Find face in the base frame
                            gray = cv2.cvtColor(base_frame, cv2.COLOR_BGR2GRAY)
                            faces = self.video_extractor.detector(gray)

                            if len(faces) > 0 and mouth_img is not None:
                                face = faces[0]
                                shape = self.video_extractor.predictor(gray, face)

                                # Extract mouth landmarks
                                mouth_points = []
                                for i in range(48, 68):
                                    x = shape.part(i).x
                                    y = shape.part(i).y
                                    mouth_points.append((x, y))

                                # Calculate mouth bounding box
                                xs = [p[0] for p in mouth_points]
                                ys = [p[1] for p in mouth_points]
                                left, right = min(xs), max(xs)
                                top, bottom = min(ys), max(ys)

                                # Add padding
                                padding = int((bottom - top) * 0.2)
                                top = max(0, top - padding)
                                bottom = min(base_frame.shape[0], bottom + padding)
                                left = max(0, left - padding)
                                right = min(base_frame.shape[1], right + padding)

                                # Resize mouth image to fit the detected mouth region
                                try:
                                    mouth_img_resized = cv2.resize(mouth_img, (right - left, bottom - top))

                                    # Create a mask for smooth blending
                                    mask = np.zeros((bottom - top, right - left), dtype=np.float32)
                                    cv2.ellipse(mask,
                                                (mask.shape[1] // 2, mask.shape[0] // 2),
                                                (mask.shape[1] // 2 - 5, mask.shape[0] // 2 - 5),
                                                0, 0, 360, 1, -1)

                                    # Blur the mask edges for smoother transition
                                    mask = cv2.GaussianBlur(mask, (15, 15), 0)

                                    # Extract the region from the base frame
                                    roi = frame_copy[top:bottom, left:right]

                                    # Blend the mouth image with the ROI using the mask
                                    for c in range(0, 3):
                                        roi[:, :, c] = roi[:, :, c] * (1 - mask) + mouth_img_resized[:, :, c] * mask

                                    # Put the blended ROI back into the frame
                                    frame_copy[top:bottom, left:right] = roi

                                    # Draw a rectangle around the mouth region for debugging
                                    cv2.rectangle(frame_copy, (left, top), (right, bottom), (0, 255, 0), 1)
                                except Exception as e:
                                    print(f"Error blending mouth image: {e}")

                    # Convert OpenCV BGR to RGB for pygame
                    frame_rgb = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)

                    # Create pygame surface from numpy array
                    frame_surface = pygame.surfarray.make_surface(frame_rgb.swapaxes(0, 1))

                    # Scale to fit screen if needed
                    if frame_surface.get_width() != self.width or frame_surface.get_height() != self.height:
                        frame_surface = pygame.transform.scale(frame_surface, (self.width, self.height))

                    # Display the frame
                    self.screen.blit(frame_surface, (0, 0))

                    # Display current information
                    time_text = font.render(f'Time: {current_time:.2f}s | Frame: {frame_idx}', True, (255, 255, 255))
                    self.screen.blit(time_text, (10, self.height - 80))

                    viseme_text = font.render(f'Viseme: {current_viseme}', True, (255, 255, 255))
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

    def load_mouth_images(self, folder="mouth_images"):
        """
        Load the extracted mouth images and their metadata.
        """
        metadata_file = os.path.join(folder, "metadata.txt")
        if not os.path.exists(metadata_file):
            print(f"Error: Metadata file not found at {metadata_file}")
            return {}

        # Read metadata
        mouth_images = {}
        with open(metadata_file, 'r') as f:
            # Skip header lines
            for _ in range(3):
                f.readline()

            # Skip column headers
            f.readline()

            # Read data
            for line in f:
                parts = line.strip().split(',')
                if len(parts) == 4:
                    frame_idx, mar, viseme, filename = parts
                    frame_idx = int(frame_idx)
                    mar = float(mar)

                    # Load the image
                    img_path = os.path.join(folder, filename)
                    if os.path.exists(img_path):
                        img = cv2.imread(img_path)
                        mouth_images[frame_idx] = {
                            'image': img,
                            'mar': mar,
                            'viseme': viseme,
                            'filename': filename
                        }

        # Group by viseme for easier lookup
        viseme_groups = {}
        for frame_idx, data in mouth_images.items():
            viseme = data['viseme']
            if viseme not in viseme_groups:
                viseme_groups[viseme] = []
            viseme_groups[viseme].append((frame_idx, data))

        print(f"Loaded {len(mouth_images)} mouth images")
        print(f"Visemes available: {list(viseme_groups.keys())}")

        return {
            'by_frame': mouth_images,
            'by_viseme': viseme_groups
        }

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

    def animate_with_real_mouth_images(self, audio_file, script_text=None, mouth_images_folder="mouth_images"):
        """
        Animate lip sync using pre-extracted mouth images.
        """
        # Load the mouth images
        mouth_data = self.load_mouth_images(mouth_images_folder)
        if not mouth_data or not mouth_data['by_frame']:
            print("Error: No mouth images loaded.")
            return

        # Load audio file
        try:
            pygame.mixer.music.load(audio_file)
        except Exception as e:
            print(f"Error loading audio {audio_file}: {e}")
            return

        # Generate viseme sequence from script text if provided
        if script_text:
            # Get audio duration
            import librosa
            y, sr = librosa.load(audio_file, sr=None)
            audio_duration = len(y) / sr

            self.viseme_sequence = self.generate_viseme_sequence_from_text(script_text, audio_duration)
            if not self.viseme_sequence:
                print("Error: Failed to generate viseme sequence.")
                return
        else:
            self.viseme_sequence = None

        # Animation loop setup
        running = True
        paused = True
        start_time = None
        current_time = 0
        fps = 30

        font = pygame.font.SysFont('Arial', 18)
        control_text = font.render('Space: Play/Pause, Esc: Quit, R: Restart', True, (255, 255, 255))
        control_rect = control_text.get_rect(topleft=(10, 10))

        # Main animation loop
        while running:
            # Handle events
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
                        else:
                            pygame.mixer.music.pause()
                            paused = True
                    elif event.key == pygame.K_r:
                        pygame.mixer.music.stop()
                        pygame.mixer.music.play()
                        start_time = time.time()
                        paused = False

            # Update animation if playing
            if start_time is not None and not paused:
                current_time = time.time() - start_time

                # Get current viseme
                current_viseme = "REST"
                if self.viseme_sequence:
                    for viseme, _, start, end in self.viseme_sequence:
                        if start <= current_time < end:
                            current_viseme = viseme
                            break

                # Find a mouth image for this viseme
                mouth_img = None
                if current_viseme in mouth_data['by_viseme']:
                    # Get a random mouth image for this viseme
                    import random
                    options = mouth_data['by_viseme'][current_viseme]
                    mouth_img = random.choice(options)[1]['image']

                # Display the mouth image
                if mouth_img is not None:
                    # Convert to pygame surface
                    mouth_img_rgb = cv2.cvtColor(mouth_img, cv2.COLOR_BGR2RGB)
                    mouth_surface = pygame.surfarray.make_surface(mouth_img_rgb.swapaxes(0, 1))

                    # Position in center of screen
                    mouth_rect = mouth_surface.get_rect(center=(self.width // 2, self.height // 2))
                    self.screen.fill(self.bg_color)
                    self.screen.blit(mouth_surface, mouth_rect)

                    # Display info
                    viseme_text = font.render(f'Viseme: {current_viseme} | Time: {current_time:.2f}s', True,
                                              (255, 255, 255))
                    self.screen.blit(viseme_text, (10, self.height - 30))
                else:
                    self.screen.fill(self.bg_color)
                    no_img_text = font.render(f'No image for viseme: {current_viseme}', True, (255, 255, 255))
                    self.screen.blit(no_img_text, (10, self.height // 2))

                # Check if audio has finished
                if not pygame.mixer.music.get_busy() and not paused:
                    paused = True
            else:
                # Show paused state
                self.screen.fill(self.bg_color)
                if start_time is None:
                    msg = "Press SPACE to start"
                else:
                    msg = "PAUSED - Press SPACE to resume"

                msg_text = font.render(msg, True, (255, 255, 255))
                msg_rect = msg_text.get_rect(center=(self.width // 2, self.height // 2))
                self.screen.blit(msg_text, msg_rect)

            # Always show controls
            self.screen.blit(control_text, control_rect)
            pygame.display.flip()
            self.clock.tick(fps)

        pygame.mixer.music.stop()
        print("Animation complete")

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
                    self.viseme_map["REST"].jaw_open, self.viseme_map["REST"].lip_round,
                    self.viseme_map["REST"].lip_width),
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
                    self.viseme_map["REST"].jaw_open, self.viseme_map["REST"].lip_round,
                    self.viseme_map["REST"].lip_width),
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
                    self.viseme_map["REST"].jaw_open, self.viseme_map["REST"].lip_round,
                    self.viseme_map["REST"].lip_width),
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
                    self.viseme_map["REST"].jaw_open, self.viseme_map["REST"].lip_round,
                    self.viseme_map["REST"].lip_width),
                 3.0, duration)
            ]

        print(f"Generated {len(viseme_sequence)} viseme segments")
        return viseme_sequence

    def animate_with_videos_and_mouth_images(self, audio_file, speaking_video_path, silence_video_path,
                                             mouth_images_folder="mouth_images", script_text=None):
        """
        Enhanced animation that uses pre-extracted mouth images overlaid on the original video.
        This provides more realistic lip sync by using actual mouth shapes from the video.
        """
        # Process videos first
        if not self.process_videos(speaking_video_path, silence_video_path):
            print("Error: Failed to process videos.")
            return

        # Load the mouth images
        mouth_data = self.load_mouth_images(mouth_images_folder)
        if not mouth_data or not mouth_data['by_viseme']:
            print("Warning: No mouth images loaded. Falling back to standard animation.")
            # If no mouth images, fall back to regular animation
            self.animate_with_videos(audio_file, speaking_video_path, silence_video_path, script_text)
            return

        # Load audio file
        try:
            pygame.mixer.music.load(audio_file)
            print(f"Successfully loaded audio file: {audio_file}")
        except Exception as e:
            print(f"Error loading audio {audio_file}: {e}")
            return

        # Get video properties
        cap = cv2.VideoCapture(speaking_video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {speaking_video_path}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        # Resize pygame window to match video dimensions
        if width > 0 and height > 0:
            self.width = width
            self.height = height
            self.screen = pygame.display.set_mode((width, height))

        # Analyze audio to get MAR values
        audio_mars = self.analyze_audio(audio_file, fps)

        # Generate viseme sequence from script text if provided
        if script_text:
            audio_duration = len(audio_mars) / fps if audio_mars is not None else None
            self.viseme_sequence = self.generate_viseme_sequence_from_text(script_text, audio_duration)
            if not self.viseme_sequence:
                print("Error: Failed to generate viseme sequence. Using audio analysis only.")
                self.viseme_sequence = []
            else:
                print(f"Generated viseme sequence with {len(self.viseme_sequence)} segments")
        else:
            self.viseme_sequence = []

        # Animation loop setup
        running = True
        paused = True
        start_time = None
        current_time = 0
        frame_idx = 0

        font = pygame.font.SysFont('Arial', 18)
        control_text = font.render('Space: Play/Pause, Esc: Quit, R: Restart', True, (255, 255, 255))
        control_rect = control_text.get_rect(topleft=(10, 10))

        print("Ready to animate with mouth images. Press SPACE to start playback.")

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
                                pygame.mixer.music.play()
                                start_time = time.time()
                                print("Starting playback")
                            else:
                                pygame.mixer.music.unpause()
                                print("Resuming playback")
                            paused = False
                        else:
                            pygame.mixer.music.pause()
                            paused = True
                            print("Paused playback")
                    elif event.key == pygame.K_r:
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
                    frame_idx = target_frame

                # Get current viseme from sequence if available
                current_viseme = "REST"

                if self.viseme_sequence:
                    for viseme_name, _, start, end in self.viseme_sequence:
                        if start <= current_time < end:
                            current_viseme = viseme_name
                            break
                elif audio_mars is not None and frame_idx < len(audio_mars):
                    # Use audio analysis to determine mouth openness
                    mar_value = audio_mars[frame_idx]

                    # Map MAR to viseme
                    if mar_value < 0.25:
                        current_viseme = "REST"
                    elif mar_value < 0.32:
                        current_viseme = "I"
                    elif mar_value < 0.40:
                        current_viseme = "A"
                    elif mar_value < 0.48:
                        current_viseme = "O"
                    else:
                        current_viseme = "Wide"

                # Determine whether to use speaking or silence frame
                use_silence = False
                if self.viseme_sequence:
                    use_silence = (current_viseme == "REST")
                elif audio_mars is not None and frame_idx < len(audio_mars):
                    use_silence = (audio_mars[frame_idx] < self.silence_threshold)

                # Get the base frame from the appropriate video
                if self.video_extractor.speaking_video_frames and not use_silence:
                    frame_to_use = frame_idx % len(self.video_extractor.speaking_video_frames)
                    base_frame = self.video_extractor.get_frame(frame_to_use, False)
                elif self.video_extractor.silence_video_frames:
                    frame_to_use = frame_idx % len(self.video_extractor.silence_video_frames)
                    base_frame = self.video_extractor.get_frame(frame_to_use, True)
                else:
                    base_frame = None

                if base_frame is not None:
                    # Create a copy of the base frame to work with
                    frame_copy = base_frame.copy()

                    # Find a mouth image for the current viseme
                    mouth_img = None
                    if current_viseme in mouth_data['by_viseme'] and mouth_data['by_viseme'][current_viseme]:
                        # Get a mouth image for this viseme
                        import random
                        options = mouth_data['by_viseme'][current_viseme]
                        mouth_data_entry = random.choice(options)[1]
                        mouth_img = mouth_data_entry['image']

                    # If we have a mouth image, overlay it on the base frame
                    if mouth_img is not None:
                        # Find face in the base frame
                        gray = cv2.cvtColor(base_frame, cv2.COLOR_BGR2GRAY)
                        faces = self.video_extractor.detector(gray)

                        if len(faces) > 0:
                            face = faces[0]
                            shape = self.video_extractor.predictor(gray, face)

                            # Extract mouth landmarks
                            mouth_points = []
                            for i in range(48, 68):
                                x = shape.part(i).x
                                y = shape.part(i).y
                                mouth_points.append((x, y))

                            # Calculate mouth bounding box
                            xs = [p[0] for p in mouth_points]
                            ys = [p[1] for p in mouth_points]
                            left, right = min(xs), max(xs)
                            top, bottom = min(ys), max(ys)

                            # Add padding
                            padding = int((bottom - top) * 0.2)
                            top = max(0, top - padding)
                            bottom = min(base_frame.shape[0], bottom + padding)
                            left = max(0, left - padding)
                            right = min(base_frame.shape[1], right + padding)

                            # Resize mouth image to fit the detected mouth region
                            mouth_img_resized = cv2.resize(mouth_img, (right - left, bottom - top))

                            # Create a mask for smooth blending
                            mask = np.zeros((bottom - top, right - left), dtype=np.float32)
                            cv2.ellipse(mask,
                                        (mask.shape[1] // 2, mask.shape[0] // 2),
                                        (mask.shape[1] // 2 - 5, mask.shape[0] // 2 - 5),
                                        0, 0, 360, 1, -1)

                            # Blur the mask edges for smoother transition
                            mask = cv2.GaussianBlur(mask, (15, 15), 0)

                            # Extract the region from the base frame
                            roi = frame_copy[top:bottom, left:right]

                            # Blend the mouth image with the ROI using the mask
                            for c in range(0, 3):
                                roi[:, :, c] = roi[:, :, c] * (1 - mask) + mouth_img_resized[:, :, c] * mask

                            # Put the blended ROI back into the frame
                            frame_copy[top:bottom, left:right] = roi

                            # Draw a rectangle around the mouth region for debugging
                            cv2.rectangle(frame_copy, (left, top), (right, bottom), (0, 255, 0), 1)

                    # Convert OpenCV BGR to RGB for pygame
                    frame_rgb = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)

                    # Create pygame surface from numpy array
                    frame_surface = pygame.surfarray.make_surface(frame_rgb.swapaxes(0, 1))

                    # Scale to fit screen if needed
                    if frame_surface.get_width() != self.width or frame_surface.get_height() != self.height:
                        frame_surface = pygame.transform.scale(frame_surface, (self.width, self.height))

                    # Display the frame
                    self.screen.blit(frame_surface, (0, 0))

                    # Display current information
                    time_text = font.render(f'Time: {current_time:.2f}s | Frame: {frame_idx}', True, (255, 255, 255))
                    self.screen.blit(time_text, (10, self.height - 80))

                    viseme_text = font.render(f'Viseme: {current_viseme}', True, (255, 255, 255))
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

    def animate_with_videos(self, audio_file, speaking_video_path, silence_video_path, script_text=None):
        """
        Animate lip sync using data extracted from videos and synchronized with audio.
        Enhanced with detailed mouth shape extraction and morphing for realistic lip sync.
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
        last_viseme = "REST"
        last_mouth_shape = None

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
                        last_viseme = "REST"
                        last_mouth_shape = None
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
                transition_blend = 0.0

                if self.viseme_sequence and len(self.viseme_sequence) > 0:
                    # Find the current viseme based on time
                    current_viseme_idx = None
                    for i, (viseme_name, viseme_params, start, end) in enumerate(self.viseme_sequence):
                        if start <= current_time < end:
                            current_viseme_idx = i
                            current_viseme_name = viseme_name

                            # Check for transitions
                            if current_time >= end - self.transition_time and i < len(self.viseme_sequence) - 1:
                                # Transitioning to next viseme
                                next_viseme = self.viseme_sequence[i + 1][0]
                                transition_blend = (current_time - (end - self.transition_time)) / self.transition_time
                                transition_blend = max(0, min(1, transition_blend))
                                current_viseme_name = f"{viseme_name}->{next_viseme}"
                            elif current_time <= start + self.transition_time and i > 0:
                                # Transitioning from previous viseme
                                prev_viseme = self.viseme_sequence[i - 1][0]
                                transition_blend = 1.0 - (current_time - start) / self.transition_time
                                transition_blend = max(0, min(1, transition_blend))
                                current_viseme_name = f"{prev_viseme}->{viseme_name}"
                            break

                    if current_viseme_idx is None:
                        # If we're before the first viseme or after the last one
                        if current_time < self.viseme_sequence[0][2]:
                            current_viseme_name = self.viseme_sequence[0][0]
                        else:
                            current_viseme_name = self.viseme_sequence[-1][0]
                elif audio_mars is not None and frame_idx < len(audio_mars):
                    # Use audio analysis to determine mouth openness
                    mar_value = audio_mars[frame_idx]

                    # Map MAR to viseme
                    if mar_value < 0.25:
                        current_viseme_name = "REST"
                    elif mar_value < 0.32:
                        current_viseme_name = "I"
                    elif mar_value < 0.40:
                        current_viseme_name = "A"
                    elif mar_value < 0.48:
                        current_viseme_name = "O"
                    else:
                        current_viseme_name = "Wide"

                # Determine whether to use speaking or silence frame
                use_silence = False
                if "->" in current_viseme_name:
                    # For transitions, use speaking frames
                    use_silence = False
                    base_viseme = current_viseme_name.split("->")[1]
                else:
                    base_viseme = current_viseme_name
                    # Use silence frames only for REST visemes
                    use_silence = (base_viseme == "REST")

                    # Also check audio level if available
                    if audio_mars is not None and frame_idx < len(audio_mars):
                        use_silence = use_silence or (audio_mars[frame_idx] < self.silence_threshold)

                # Get the appropriate mouth shape for the current viseme
                current_mouth_shape = self.get_mouth_shape_for_viseme(base_viseme, frame_idx, use_silence)

                # For transitions, also get the other viseme's mouth shape
                transition_mouth_shape = None
                if "->" in current_viseme_name:
                    other_viseme = current_viseme_name.split("->")[0]
                    transition_mouth_shape = self.get_mouth_shape_for_viseme(other_viseme, frame_idx, use_silence)

                # Get the base frame to work with
                if self.video_extractor.speaking_video_frames and not use_silence:
                    frame_to_use = frame_idx % len(self.video_extractor.speaking_video_frames)
                    video_frame = self.video_extractor.get_frame(frame_to_use, False)
                elif self.video_extractor.silence_video_frames:
                    frame_to_use = frame_idx % len(self.video_extractor.silence_video_frames)
                    video_frame = self.video_extractor.get_frame(frame_to_use, True)
                else:
                    video_frame = None

                if video_frame is not None and current_mouth_shape is not None:
                    # Apply the mouth shape to the frame
                    if transition_mouth_shape is not None and transition_blend > 0:
                        # For transitions, blend between two mouth shapes
                        processed_frame = self.apply_mouth_shape(
                            video_frame,
                            current_mouth_shape,
                            transition_mouth_shape if transition_blend > 0.5 else None
                        )
                    else:
                        # For single visemes, just apply the current mouth shape
                        processed_frame = self.apply_mouth_shape(
                            video_frame,
                            current_mouth_shape,
                            last_mouth_shape if last_viseme != base_viseme else None
                        )

                    # Update last viseme and mouth shape for next frame
                    last_viseme = base_viseme
                    last_mouth_shape = current_mouth_shape

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
                    # Get the REST mouth shape
                    rest_mouth_shape = self.get_mouth_shape_for_viseme("REST", 0, True)

                    if rest_mouth_shape is not None:
                        # Apply the REST mouth shape
                        processed_silence = self.apply_mouth_shape(silence_frame, rest_mouth_shape)
                        silence_frame_rgb = cv2.cvtColor(processed_silence, cv2.COLOR_BGR2RGB)
                    else:
                        # Just use the frame as is
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

    def process_videos(self, speaking_video_path, silence_video_path):
        """
        Process both speaking and silence videos to extract detailed mouth data.
        """
        print("Processing speaking video for detailed mouth shapes...")
        speaking_shapes = self.video_extractor.extract_detailed_mouth_data(speaking_video_path, False)
        if not speaking_shapes:
            print("Warning: No mouth shape data extracted from speaking video")
            return False

        print("Processing silence video for detailed mouth shapes...")
        silence_shapes = self.video_extractor.extract_detailed_mouth_data(silence_video_path, True)
        if not silence_shapes:
            print("Warning: No mouth shape data extracted from silence video")
            return False

        print("Video processing complete with detailed mouth shapes.")
        return True

    def apply_mouth_shape(self, frame, target_shape, source_shape=None):
        """
        Apply a target mouth shape to a frame.
        If source_shape is provided, it will morph from source to target.
        """
        if target_shape['outer_points'] is None:
            return frame  # No valid mouth shape data

        frame_copy = frame.copy()

        # Extract the mouth region based on bounding box
        left, top, width, height = target_shape['bbox']

        # Add padding to the mouth region
        padding = int(height * 0.2)
        top = max(0, top - padding)
        bottom = min(frame.shape[0], top + height + 2 * padding)
        left = max(0, left - padding)
        right = min(frame.shape[1], left + width + 2 * padding)

        # Get the mouth region
        mouth_roi = frame[top:bottom, left:right]

        # If we have a source shape, we'll morph between the two
        if source_shape is not None and source_shape['outer_points'] is not None:
            # Create a mask for the mouth region
            mask = np.zeros_like(mouth_roi)

            # Draw the source mouth shape on the mask
            source_points = np.array(source_shape['outer_points'])
            source_points = source_points - np.array([left, top])  # Adjust to ROI coordinates
            cv2.fillPoly(mask, [source_points.astype(np.int32)], (255, 255, 255))

            # Draw the target mouth shape
            target_points = np.array(target_shape['outer_points'])
            target_points = target_points - np.array([left, top])  # Adjust to ROI coordinates

            # Create a transformation matrix
            transformation = cv2.estimateAffinePartial2D(
                source_points.astype(np.float32),
                target_points.astype(np.float32)
            )[0]

            if transformation is not None:
                # Apply the transformation to the mouth region
                warped_roi = cv2.warpAffine(
                    mouth_roi,
                    transformation,
                    (mouth_roi.shape[1], mouth_roi.shape[0])
                )

                # Blend the warped region with the original
                alpha = 0.7  # Blend factor
                blended_roi = cv2.addWeighted(warped_roi, alpha, mouth_roi, 1 - alpha, 0)

                # Place the blended region back into the frame
                frame_copy[top:bottom, left:right] = blended_roi
        else:
            # Just highlight the mouth region for visualization
            cv2.rectangle(frame_copy, (left, top), (right, bottom), (0, 255, 0), 2)

            # Draw the mouth points
            for point in target_shape['outer_points']:
                cv2.circle(frame_copy, point, 2, (0, 0, 255), -1)

            if target_shape['inner_points']:
                for point in target_shape['inner_points']:
                    cv2.circle(frame_copy, point, 2, (255, 0, 0), -1)

        # Add viseme info to the frame
        info_text = f"Viseme: {target_shape['viseme']} | MAR: {target_shape['mar']:.2f}"
        cv2.putText(frame_copy, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return frame_copy

    def get_mouth_shape_for_viseme(self, viseme_name, frame_idx, use_silence=False):
        """
        Find the best matching mouth shape for a given viseme from the video frames.
        """
        # Get the appropriate mouth shapes list
        mouth_shapes = self.video_extractor.silence_mouth_shapes if use_silence else self.video_extractor.speaking_mouth_shapes

        if not mouth_shapes:
            return None

        # First try to find a frame with the exact viseme
        matching_shapes = [shape for shape in mouth_shapes if shape['viseme'] == viseme_name]

        if matching_shapes:
            # Find the closest frame to the current frame index to maintain temporal coherence
            closest_idx = min(range(len(matching_shapes)),
                              key=lambda i: abs(matching_shapes[i]['frame_idx'] - frame_idx % len(mouth_shapes)))
            return matching_shapes[closest_idx]
        else:
            # If no exact match, find the closest viseme based on mouth openness
            if viseme_name in ["REST", "M", "P"]:
                # Closed mouth visemes
                target_mar = 0.2
            elif viseme_name in ["I", "F", "S", "T"]:
                # Slightly open visemes
                target_mar = 0.3
            elif viseme_name in ["E", "L"]:
                # Medium open visemes
                target_mar = 0.4
            elif viseme_name in ["O", "U", "SH"]:
                # Rounded visemes
                target_mar = 0.45
            else:  # "A", "Wide"
                # Wide open visemes
                target_mar = 0.6

            # Find the shape with the closest MAR value
            closest_idx = min(range(len(mouth_shapes)),
                              key=lambda i: abs(mouth_shapes[i]['mar'] - target_mar))
            return mouth_shapes[closest_idx]

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

    print("Choose an option:")
    print("1. Run video-based lip sync (original method)")
    print("2. Extract mouth images from video")
    print("3. Run enhanced lip sync with extracted mouth images")

    choice = input("Enter your choice (1-3): ")

    if choice == "1":
        # Original method
        audio_file = input("Enter path to audio file (default: sample_audio.wav): ") or "sample_audio.wav"
        speaking_video = input("Enter path to speaking video file (default: ./speaking.mp4): ") or "./speaking.mp4"
        silence_video = input("Enter path to silence video file (default: ./silence.mp4): ") or "./silence.mp4"
        script_text = input("Enter script text for phoneme-viseme mapping (default: Hello World): ") or "Hello World"

        animator.run_video_based_demo(audio_file, speaking_video, silence_video, script_text)

    elif choice == "2":
        # Extract mouth images
        video_path = input("Enter path to video file to extract mouth images: ")
        output_folder = input("Enter output folder name (default: mouth_images): ") or "mouth_images"

        if not os.path.exists(video_path):
            print(f"Error: Video file {video_path} not found.")
            return

        animator.video_extractor.extract_mouth_images_from_video(video_path, output_folder)
        print(f"Mouth images extracted to {output_folder}")

    elif choice == "3":
        # Run enhanced lip sync with extracted mouth images
        audio_file = input("Enter path to audio file (default: sample_audio.wav): ") or "sample_audio.wav"
        speaking_video = input("Enter path to speaking video file (default: ./speaking.mp4): ") or "./speaking.mp4"
        silence_video = input("Enter path to silence video file (default: ./silence.mp4): ") or "./silence.mp4"
        mouth_images_folder = input("Enter mouth images folder (default: mouth_images): ") or "mouth_images"
        script_text = input("Enter script text for phoneme-viseme mapping (default: Hello World): ") or "Hello World"

        if not os.path.exists(audio_file):
            print(f"Error: Audio file {audio_file} not found.")
            return

        if not os.path.exists(speaking_video):
            print(f"Error: Speaking video file {speaking_video} not found.")
            return

        if not os.path.exists(silence_video):
            print(f"Error: Silence video file {silence_video} not found.")
            return

        if not os.path.exists(mouth_images_folder):
            print(f"Warning: Mouth images folder {mouth_images_folder} not found.")
            create_folder = input("Do you want to extract mouth images from the speaking video first? (y/n): ")
            if create_folder.lower() == 'y':
                animator.video_extractor.extract_mouth_images_from_video(speaking_video, mouth_images_folder)
            else:
                print("Cannot proceed without mouth images.")
                return

        animator.animate_with_videos_and_mouth_images(audio_file, speaking_video, silence_video,
                                                      mouth_images_folder, script_text)

    else:
        print("Invalid choice. Please run again and select 1, 2, or 3.")


if __name__ == "__main__":
    main()
