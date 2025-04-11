import cv2
import numpy as np
import random
import os
import dlib
from phonemizer.backend import EspeakBackend
from moviepy.editor import VideoFileClip, AudioFileClip, concatenate_videoclips
from pydub import AudioSegment
import speech_recognition as sr
import subprocess
from moviepy.config import get_setting


def convert_video(input_file, output_file):
    """Convert the video to a different format using FFmpeg."""
    input_file = os.path.abspath(input_file)
    output_file = os.path.abspath(output_file)

    print(f"Converting video from {input_file} to {output_file}")

    cmd = [
        get_setting("FFMPEG_BINARY"),
        "-i", input_file,
        "-c:v", "libx264",
        "-preset", "medium",
        "-pix_fmt", "yuv420p",
        "-r", "30",
        output_file
    ]

    popen_params = {
        "bufsize": 10 ** 5,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "stdin": subprocess.DEVNULL,
    }

    if os.name == "nt":
        popen_params["creationflags"] = 0x08000000

    try:
        proc = subprocess.Popen(cmd, **popen_params)
        stdout, stderr = proc.communicate()

        if proc.returncode != 0:
            raise RuntimeError(f"Video conversion failed with error:\n{stderr.decode()}")

        print(f"Video conversion successful: {output_file}")
        return output_file
    except Exception as e:
        print(f"An error occurred during video conversion: {e}")
        return None


def resize_viseme_to_mouth(viseme_img, mouth_region, frame):
    """
    Resize the viseme image to fit the mouth region based on its detected size.

    Args:
        viseme_img (numpy.ndarray): Viseme image to be resized.
        mouth_region (dict): Mouth region with width and height information.
        frame (numpy.ndarray): The current video frame.

    Returns:
        numpy.ndarray: Resized viseme image to fit the mouth region.
    """
    mouth_width = mouth_region['width']
    mouth_height = mouth_region['height']

    resized_viseme = cv2.resize(viseme_img, (mouth_width, mouth_height))
    return resized_viseme


class LipSyncAnimator:
    def __init__(self, input_video, audio_file, viseme_folder, output_video, predictor_path, fps=30, resolution=None):
        self.input_video = input_video
        self.audio_file = audio_file
        self.viseme_folder = viseme_folder
        self.output_video = output_video
        self.fps = fps
        self.predictor_path = predictor_path

        # Initialize facial landmark detector
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor_path)

        # Initialize phonemizer
        self.backend = EspeakBackend('en-us')

        # Video and audio info
        self.video = cv2.VideoCapture(input_video)
        self.audio = AudioSegment.from_file(audio_file)

        # Get video properties
        self.width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if resolution:
            self.resolution = resolution
        elif self.width > 0 and self.height > 0:
            self.resolution = (self.width, self.height)
        else:
            self.resolution = (640, 480)

        if not os.path.exists(viseme_folder):
            raise ValueError(f"Viseme folder {viseme_folder} not found!")

        self.viseme_images = self.load_viseme_images()
        print("Available viseme categories:", list(self.viseme_images.keys()))

    def load_viseme_images(self):
        """Load all viseme images from the viseme folder."""
        viseme_images = {}

        if not os.path.exists(self.viseme_folder):
            print(f"Error: Viseme folder {self.viseme_folder} does not exist.")
            return viseme_images

        for category in os.listdir(self.viseme_folder):
            category_path = os.path.join(self.viseme_folder, category)
            if not os.path.isdir(category_path):
                continue

            image_files = [
                os.path.join(category_path, f)
                for f in os.listdir(category_path)
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))
            ]

            if image_files:
                viseme_images[category] = image_files
                print(f"Loaded {len(image_files)} images for viseme '{category}'")

        if not viseme_images:
            print("Warning: No viseme images found. Check your viseme folder structure.")
        return viseme_images

    def get_mouth_landmarks(self, frame):
        """Detect face and extract mouth landmarks using dlib."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray, 0)

        if len(faces) == 0:
            return None

        if len(faces) > 1:
            largest_area = 0
            largest_face_idx = 0
            for i, face in enumerate(faces):
                area = (face.right() - face.left()) * (face.bottom() - face.top())
                if area > largest_area:
                    largest_area = area
                    largest_face_idx = i
            face = faces[largest_face_idx]
        else:
            face = faces[0]

        landmarks = self.predictor(gray, face)

        mouth_points = []
        for i in range(48, 68):
            point = landmarks.part(i)
            mouth_points.append((point.x, point.y))

        top_lip_center = ((landmarks.part(51).x + landmarks.part(62).x) // 2,
                          (landmarks.part(51).y + landmarks.part(62).y) // 2)
        bottom_lip_center = ((landmarks.part(57).x + landmarks.part(66).x) // 2,
                             (landmarks.part(57).y + landmarks.part(66).y) // 2)

        mouth_height = np.sqrt((top_lip_center[0] - bottom_lip_center[0]) ** 2 +
                               (top_lip_center[1] - bottom_lip_center[1]) ** 2)

        mouth_width = np.sqrt((landmarks.part(48).x - landmarks.part(54).x) ** 2 +
                              (landmarks.part(48).y - landmarks.part(54).y) ** 2)

        mouth_aspect_ratio = mouth_height / max(mouth_width, 1)

        return {
            'landmarks': mouth_points,
            'aspect_ratio': mouth_aspect_ratio,
            'open_ratio': mouth_height / max(mouth_width, 1),
            'face_bbox': (face.left(), face.top(), face.right(), face.bottom())
        }

    def get_mouth_region(self, mouth_data):
        """Calculate mouth region (rectangle) from landmarks with dynamic padding."""
        if not mouth_data or 'landmarks' not in mouth_data:
            return None

        points = np.array(mouth_data['landmarks'])
        x_min = np.min(points[:, 0])
        y_min = np.min(points[:, 1])
        x_max = np.max(points[:, 0])
        y_max = np.max(points[:, 1])

        face_width = mouth_data['face_bbox'][2] - mouth_data['face_bbox'][0]
        face_height = mouth_data['face_bbox'][3] - mouth_data['face_bbox'][1]

        padding_x = int(face_width * 0.05)
        padding_y = int(face_height * 0.05)
        bottom_padding = int(padding_y * 1.5)

        x_min = max(0, x_min - padding_x)
        y_min = max(0, y_min - padding_y)
        x_max = min(self.width, x_max + padding_x)
        y_max = min(self.height, y_max + bottom_padding)

        return {
            'x': x_min,
            'y': y_min,
            'width': x_max - x_min,
            'height': y_max - y_min,
            'center_x': (x_min + x_max) // 2,
            'center_y': (y_min + y_max) // 2,
            'aspect_ratio': mouth_data.get('aspect_ratio', 1.0),
            'open_ratio': mouth_data.get('open_ratio', 0.5)
        }

    def phoneme_to_viseme(self, phoneme, previous_viseme=None):
        """
        Convert a phoneme to its corresponding viseme category with coarticulation awareness.

        Args:
            phoneme (str): The phoneme to convert
            previous_viseme (str): The previous viseme for context

        Returns:
            str: The viseme category
        """
        phoneme = phoneme.upper().replace('.', '').strip()
        phoneme = ''.join([c for c in phoneme if not c.isdigit()])

        phoneme_mapping = {
            'AA': 'A', 'AE': 'A', 'AH': 'A', 'AO': 'O', 'AW': 'A', 'AY': 'A',
            'EH': 'E', 'ER': 'E', 'EY': 'E', 'IH': 'I', 'IY': 'I',
            'UH': 'U', 'UW': 'U', 'OW': 'O', 'OY': 'O',
            'B': 'BMP', 'M': 'BMP', 'P': 'BMP',
            'CH': 'CH-J-SH', 'JH': 'CH-J-SH', 'SH': 'CH-J-SH', 'ZH': 'CH-J-SH',
            'D': 'D-N-T', 'N': 'D-N-T', 'T': 'D-N-T',
            'DH': 'TH', 'TH': 'TH',
            'F': 'FV', 'V': 'FV',
            'G': 'G-K-NG', 'K': 'G-K-NG', 'NG': 'G-K-NG',
            'HH': 'H-Y', 'Y': 'H-Y',
            'L': 'L', 'R': 'R',
            'S': 'S-Z', 'Z': 'S-Z',
            'W': 'U',
            'SIL': 'Rest', 'SP': 'Rest'
        }

        viseme = phoneme_mapping.get(phoneme, 'Rest')

        if viseme not in self.viseme_images:
            for key in self.viseme_images.keys():
                if key.lower() == viseme.lower():
                    return key

            if viseme == 'Rest' and 'Rest' not in self.viseme_images:
                alternatives = ['A', 'E', 'I', 'O', 'U', 'Rest']
                for alt in alternatives:
                    if alt in self.viseme_images:
                        print(f"Using '{alt}' as fallback for 'Rest'")
                        return alt

            if self.viseme_images:
                first_key = list(self.viseme_images.keys())[0]
                print(f"Viseme '{viseme}' not found, using '{first_key}' as fallback")
                return first_key

        return viseme

    def select_appropriate_viseme(self, viseme_category, mouth_data, blend_factor=None, next_viseme=None):
        """
        Select the most appropriate viseme image based on mouth characteristics with blending support.
        """
        is_transition = '-' in viseme_category

        # Fallback handling (same as before) ...
        if viseme_category not in self.viseme_images or not self.viseme_images[viseme_category]:
            # (Fallback code omitted for brevity)
            return None, None, None

        viseme_images = self.viseme_images[viseme_category]
        candidate_images = []
        best_match_diff = float('inf')
        target_open_ratio = 0.5
        if mouth_data and 'open_ratio' in mouth_data:
            target_open_ratio = mouth_data['open_ratio']

        # Evaluate each candidate image for how well it matches the target
        for img_path in viseme_images:
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                continue

            h, w = img.shape[:2]
            middle_section = img[h // 3:2 * h // 3, w // 3:2 * w // 3]
            brightness = np.mean(middle_section)
            openness = brightness / 255.0
            diff = abs(openness - target_open_ratio)

            # Keep track of the best difference found
            best_match_diff = min(best_match_diff, diff)
            # If the diff is within a threshold of the best match, add it as a candidate.
            print(diff - best_match_diff)
            if diff - best_match_diff < 0.1:
                candidate_images.append(img_path)

        # If multiple candidates exist, randomly choose one for added variety.
        if candidate_images:
            chosen_path = random.choice(candidate_images)
            chosen_img = cv2.imread(chosen_path, cv2.IMREAD_UNCHANGED)
            blend_info = {
                'intensity': 1.0 if blend_factor is None else blend_factor,
                'is_transition': is_transition,
                'next_viseme': next_viseme
            }
            return chosen_path, chosen_img, blend_info

        # Fallback to selecting randomly from all images if no candidates met threshold.
        selected_path = random.choice(viseme_images)
        selected_img = cv2.imread(selected_path, cv2.IMREAD_UNCHANGED)
        return selected_path, selected_img, {
            'intensity': 1.0 if blend_factor is None else blend_factor,
            'is_transition': is_transition,
            'next_viseme': next_viseme
        }

    def create_viseme_sequence(self, phonemes, fps=30):
        """
        Create a natural viseme sequence with realistic timing and transitions.

        Args:
            phonemes (dict): Dictionary mapping timestamps to phonemes
            fps (int): Frames per second for the output

        Returns:
            dict: Frame to viseme mapping
        """
        min_viseme_duration = 0.08
        transition_duration = 0.04

        viseme_sequence = {}
        prev_viseme = 'Rest'
        timestamps = sorted(phonemes.keys())

        for i, ts in enumerate(timestamps):
            phoneme = phonemes[ts]
            current_viseme = self.phoneme_to_viseme(phoneme, prev_viseme)
            viseme_sequence[ts] = current_viseme

            if i < len(timestamps) - 1:
                next_ts = timestamps[i + 1]
                gap = next_ts - ts
                if gap > transition_duration * 2:
                    transition_ts = ts + (gap * 0.4)
                    transition_key = f"{current_viseme}-{self.phoneme_to_viseme(phonemes[next_ts])}"
                    viseme_sequence[transition_ts] = transition_key
                else:
                    viseme_sequence[ts + gap * 0.5] = current_viseme

            prev_viseme = current_viseme

        return viseme_sequence

    def render_frame_with_viseme(self, frame, frame_number, viseme_sequence, mouth_motion_data):
        """
        Render a frame with the appropriate viseme, applying natural blending.

        Args:
            frame (numpy.ndarray): The video frame to render on
            frame_number (int): Current frame number
            viseme_sequence (dict): Mapping of frames to visemes
            mouth_motion_data (dict): Motion data for natural mouth movement

        Returns:
            numpy.ndarray: The rendered frame
        """
        current_viseme = viseme_sequence.get(frame_number, 'Rest')
        if '+' in current_viseme:
            viseme1, viseme2 = current_viseme.split('+')
            blend_factor = self._ease_in_out(frame_number)
            mouth_data = mouth_motion_data.get(frame_number, {'open_ratio': 0.5, 'velocity': 0.0})
            path1, img1, blend_info1 = self.select_appropriate_viseme(viseme1, mouth_data, 1.0 - blend_factor, viseme2)
            path2, img2, blend_info2 = self.select_appropriate_viseme(viseme2, mouth_data, blend_factor, viseme1)

            if img1 is not None and img2 is not None:
                result_img = self._blend_viseme_images(img1, img2, blend_factor)
                return self._overlay_viseme_on_frame(frame, result_img)

        mouth_data = mouth_motion_data.get(frame_number, {'open_ratio': 0.5, 'velocity': 0.0})
        path, img, _ = self.select_appropriate_viseme(current_viseme, mouth_data)
        if img is not None:
            return self._overlay_viseme_on_frame(frame, img)
        return frame

    def _ease_in_out(self, x):
        """Apply cubic ease-in-out function for natural motion."""
        # Normalize x to a 0-1 range if needed.
        if x < 0.5:
            return 4 * x * x * x
        else:
            return 1 - pow(-2 * x + 2, 3) / 2

    def _blend_viseme_images(self, img1, img2, blend_factor):
        """Blend two viseme images with the given factor."""
        if img1.shape != img2.shape:
            h = max(img1.shape[0], img2.shape[0])
            w = max(img1.shape[1], img2.shape[1])
            if img1.shape[0] != h or img1.shape[1] != w:
                img1 = cv2.resize(img1, (w, h), interpolation=cv2.INTER_LANCZOS4)
            if img2.shape[0] != h or img2.shape[1] != w:
                img2 = cv2.resize(img2, (w, h), interpolation=cv2.INTER_LANCZOS4)

        if len(img1.shape) == 3 and img1.shape[2] == 4:
            result = cv2.addWeighted(
                img1[:, :, :3], 1.0 - blend_factor,
                img2[:, :, :3], blend_factor,
                0
            )
            if len(img2.shape) == 3 and img2.shape[2] == 4:
                alpha = cv2.addWeighted(
                    img1[:, :, 3:], 1.0 - blend_factor,
                    img2[:, :, 3:], blend_factor,
                    0
                )
                result = np.concatenate((result, alpha), axis=2)
            elif len(img1.shape) == 3 and img1.shape[2] == 4:
                alpha = img1[:, :, 3:] * (1.0 - blend_factor)
                result = np.concatenate((result, alpha), axis=2)
        else:
            result = cv2.addWeighted(img1, 1.0 - blend_factor, img2, blend_factor, 0)
        return result

    def _overlay_viseme_on_frame(self, frame, viseme_img):
        """Overlay the viseme image on the frame with proper positioning."""
        frame_h, frame_w = frame.shape[:2]
        viseme_h, viseme_w = viseme_img.shape[:2]
        x = (frame_w - viseme_w) // 2
        y = int(frame_h * 0.6)
        x = max(0, min(x, frame_w - viseme_w))
        y = max(0, min(y, frame_h - viseme_h))
        roi = frame[y:y + viseme_h, x:x + viseme_w]

        if len(viseme_img.shape) == 3 and viseme_img.shape[2] == 4:
            viseme_rgb = viseme_img[:, :, :3]
            viseme_alpha = viseme_img[:, :, 3].astype(float) / 255.0
            mask = viseme_alpha.copy()
            mask = cv2.GaussianBlur(mask, (5, 5), 0)
            for c in range(3):
                roi[:, :, c] = (1 - mask) * roi[:, :, c] + mask * viseme_rgb[:, :, c]
            frame[y:y + viseme_h, x:x + viseme_w] = roi
        else:
            gray_viseme = cv2.cvtColor(viseme_img, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray_viseme, 10, 255, cv2.THRESH_BINARY)
            mask = mask.astype(float) / 255
            mask = cv2.GaussianBlur(mask, (5, 5), 0)
            mask = np.expand_dims(mask, axis=2)
            blended = (1 - mask) * roi + mask * viseme_img
            frame[y:y + viseme_h, x:x + viseme_w] = blended.astype(np.uint8)
        return frame

    def extract_audio_transcript(self):
        """Extract the transcript from the audio file using SpeechRecognition."""
        recognizer = sr.Recognizer()
        audio = sr.AudioFile(self.audio_file)

        with audio as source:
            audio_data = recognizer.record(source)

        try:
            transcript = recognizer.recognize_google(audio_data)
            print(f"Transcription: {transcript}")
            return transcript
        except sr.UnknownValueError:
            print("Google Speech Recognition could not understand the audio")
            return None
        except sr.RequestError as e:
            print(f"Could not request results from Google Speech Recognition service; {e}")
            return None

    def extract_phoneme_timing(self, transcript):
        """Extract phoneme timing from transcript with silence detection."""
        try:
            phonemes = self.backend.phonemize([transcript], strip=True)[0].split()
            print(f"Phonemized text: {phonemes}")

            phoneme_timing = []
            audio_duration_ms = len(self.audio)
            audio_duration_sec = audio_duration_ms / 1000.0

            silence_threshold = -35
            chunk_size = 100
            silence_min_duration = 300

            silence_regions = []
            current_silence_start = None

            for i in range(0, len(self.audio), chunk_size):
                chunk = self.audio[i:i + chunk_size]
                if len(chunk) == 0:
                    continue

                if chunk.dBFS < silence_threshold:
                    if current_silence_start is None:
                        current_silence_start = i / 1000.0
                else:
                    if current_silence_start is not None:
                        silence_end = i / 1000.0
                        if (silence_end - current_silence_start) >= (silence_min_duration / 1000.0):
                            silence_regions.append((current_silence_start, silence_end))
                        current_silence_start = None

            if current_silence_start is not None:
                silence_end = audio_duration_sec
                if (silence_end - current_silence_start) >= (silence_min_duration / 1000.0):
                    silence_regions.append((current_silence_start, silence_end))

            print(f"Detected {len(silence_regions)} silence regions: {silence_regions}")

            speech_regions = []
            last_end = 0

            for start, end in sorted(silence_regions):
                if start > last_end:
                    speech_regions.append((last_end, start))
                last_end = end

            if last_end < audio_duration_sec:
                speech_regions.append((last_end, audio_duration_sec))

            print(f"Calculated {len(speech_regions)} speech regions: {speech_regions}")

            if not speech_regions:
                speech_regions = [(0, audio_duration_sec)]

            total_speech_duration = sum(end - start for start, end in speech_regions)
            phonemes_per_second = len(phonemes) / max(total_speech_duration, 0.1)

            phoneme_idx = 0
            for start, end in speech_regions:
                region_duration = end - start
                num_phonemes_in_region = max(1, int(round(region_duration * phonemes_per_second)))
                num_phonemes_in_region = min(num_phonemes_in_region, len(phonemes) - phoneme_idx)

                if num_phonemes_in_region <= 0:
                    continue

                phoneme_duration = region_duration / num_phonemes_in_region

                for i in range(num_phonemes_in_region):
                    if phoneme_idx >= len(phonemes):
                        break

                    phoneme = phonemes[phoneme_idx]
                    phoneme_start = start + (i * phoneme_duration)
                    phoneme_end = phoneme_start + phoneme_duration

                    viseme = self.phoneme_to_viseme(phoneme)
                    phoneme_timing.append({
                        'phoneme': phoneme,
                        'viseme': viseme,
                        'start': phoneme_start,
                        'end': phoneme_end
                    })

                    phoneme_idx += 1

            print(f"Audio duration: {audio_duration_sec} seconds")
            print(f"Total phonemes: {len(phonemes)}")
            print(f"Phonemes assigned: {phoneme_idx}")
            print(f"Phoneme timing entries: {len(phoneme_timing)}")
            return phoneme_timing
        except Exception as e:
            print(f"Error during phoneme timing extraction: {e}")
            return self.create_fallback_timing(transcript)

    def create_fallback_timing(self, transcript):
        """Create a simple fallback timing when phonemization fails."""
        print("Using fallback timing generation...")
        words = transcript.split()
        total_duration_ms = len(self.audio)
        word_duration = total_duration_ms / max(1, len(words)) / 1000.0

        timing = []
        start_time = 0.0

        for word in words:
            end_time = start_time + word_duration
            first_char = word[0].upper() if word else 'A'
            if first_char in 'AEIOU':
                viseme = first_char
            elif first_char in 'BMP':
                viseme = 'BMP'
            elif first_char in 'DTN':
                viseme = 'D-N-T'
            elif first_char in 'GKNG':
                viseme = 'G-K-NG'
            else:
                viseme = 'A'
            viseme = self.phoneme_to_viseme(viseme)
            timing.append({
                'phoneme': word,
                'viseme': viseme,
                'start': start_time,
                'end': end_time
            })
            start_time = end_time

        return timing

    def apply_viseme_to_frame(self, frame, viseme_img, mouth_region, mouth_data):
        """
        Apply the viseme image to the frame with advanced blending techniques.
        """
        viseme_resized = resize_viseme_to_mouth(viseme_img, mouth_region, frame)
        mouth_x = mouth_region['x']
        mouth_y = mouth_region['y']
        mouth_width = mouth_region['width']
        mouth_height = mouth_region['height']
        roi = frame[mouth_y:mouth_y + mouth_height, mouth_x:mouth_x + mouth_width]

        if viseme_resized.shape[2] == 4:
            alpha = viseme_resized[:, :, 3] / 255.0
            mask = alpha.copy()
            mask = cv2.GaussianBlur(mask, (5, 5), 0)
            viseme_rgb = viseme_resized[:, :, :3]
            for c in range(3):
                roi[:, :, c] = (1 - mask) * roi[:, :, c] + mask * viseme_rgb[:, :, c]
            frame[mouth_y:mouth_y + mouth_height, mouth_x:mouth_x + mouth_width] = roi
        else:
            gray_viseme = cv2.cvtColor(viseme_resized, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray_viseme, 10, 255, cv2.THRESH_BINARY)
            mask = mask.astype(float) / 255
            mask = cv2.GaussianBlur(mask, (5, 5), 0)
            mask = np.expand_dims(mask, axis=2)
            blended = (1 - mask) * roi + mask * viseme_resized
            frame[mouth_y:mouth_y + mouth_height, mouth_x:mouth_x + mouth_width] = blended.astype(np.uint8)

    def generate_video_from_visemes(self, phoneme_timing):
        """Generate the video by overlaying viseme mouth images onto the original silent video frames."""
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_dir = os.path.dirname(self.output_video)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        writer = cv2.VideoWriter(self.output_video, fourcc, self.fps, self.resolution)

        if not writer.isOpened():
            print(f"Error: Could not open video writer for {self.output_video}")
            return False

        total_frames = int(self.audio.duration_seconds * self.fps)
        self.video.set(cv2.CAP_PROP_POS_FRAMES, 0)
        print(f"Generating {total_frames} frames in the video...")

        epsilon = 0.001
        last_valid_mouth_region = None
        last_valid_mouth_data = None

        for frame_idx in range(total_frames):
            current_time = frame_idx / self.fps
            ret, original_frame = self.video.read()
            if not ret:
                self.video.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, original_frame = self.video.read()
                if not ret:
                    print("Error: Could not read frame from video.")
                    break

            frame = original_frame.copy()
            mouth_data = self.get_mouth_landmarks(frame)
            if mouth_data:
                mouth_region = self.get_mouth_region(mouth_data)
                if mouth_region:
                    last_valid_mouth_region = mouth_region
                    last_valid_mouth_data = mouth_data
            else:
                mouth_region = last_valid_mouth_region
                mouth_data = last_valid_mouth_data

            if not mouth_region:
                print(f"Warning: No face detected in frame {frame_idx}")
                writer.write(frame)
                continue

            is_in_speech = False
            current_viseme = None
            current_phoneme = None

            for timing in phoneme_timing:
                if timing['start'] - epsilon <= current_time <= timing['end'] + epsilon:
                    is_in_speech = True
                    current_viseme = timing['viseme']
                    current_phoneme = timing['phoneme']
                    break

            if is_in_speech and current_viseme:
                if current_viseme in self.viseme_images and self.viseme_images[current_viseme]:
                    viseme_path, viseme_img, _ = self.select_appropriate_viseme(current_viseme, mouth_data)
                    if viseme_img is not None:
                        self.apply_viseme_to_frame(frame, viseme_img, mouth_region, mouth_data)
                    else:
                        print(f"Warning: Could not load viseme image for {current_viseme}")

            writer.write(frame)

        writer.release()
        print(f"Video written successfully to {self.output_video}")
        return True

    def merge_video_and_audio(self):
        """Merge the generated video with the original audio to create the final lip sync video."""
        try:
            print(f"Loading video file: {self.output_video}")
            video_clip = VideoFileClip(self.output_video)
            print(f"Loading audio file: {self.audio_file}")
            audio_clip = AudioFileClip(self.audio_file)

            if video_clip.duration != audio_clip.duration:
                print(
                    f"Video duration ({video_clip.duration}s) doesn't match audio duration ({audio_clip.duration}s). Adjusting...")
                if video_clip.duration > audio_clip.duration:
                    video_clip = video_clip.subclip(0, audio_clip.duration)
                else:
                    loops_needed = int(audio_clip.duration / video_clip.duration) + 1
                    video_clip = concatenate_videoclips([video_clip] * loops_needed).subclip(0, audio_clip.duration)

            final_clip = video_clip.set_audio(audio_clip)
            final_output = f"final_{self.output_video}"
            print(f"Writing final video to {final_output}")
            final_clip.write_videofile(
                final_output,
                codec="libx264",
                audio_codec="aac",
                fps=self.fps,
                preset="medium",
                ffmpeg_params=["-pix_fmt", "yuv420p"]
            )

            print(f"Final video saved as {final_output}")
            return True

        except Exception as e:
            print(f"Error during video and audio merging: {e}")
            return False


def main():
    """
    Main function to run the lip sync animation.
    Uses silence.mp4 as the base video and overlays viseme mouth images during speech parts.
    """
    input_video = "silence.mp4"
    audio_file = "ashish_audio.wav"
    viseme_folder = "../visemes"
    output_video = "lip_sync_output_4.mp4"
    predictor_path = "shape_predictor_68_face_landmarks.dat"

    print("Starting lip sync animation process...")
    print(f"Using input video: {input_video}")
    print(f"Using audio file: {audio_file}")
    print(f"Using viseme folder: {viseme_folder}")
    print(f"Using facial landmark predictor: {predictor_path}")
    print(f"Output will be saved to: {output_video}")

    try:
        animator = LipSyncAnimator(
            input_video,
            audio_file,
            viseme_folder,
            output_video,
            predictor_path
        )

        print("Extracting transcript from audio...")
        transcript_text = animator.extract_audio_transcript()

        if not transcript_text:
            print("Speech recognition failed. Using fallback transcript...")
            transcript_text = "thank you for contacting us all lines are currently busy you call is very important to us"
        else:
            print(f"Successfully extracted transcript: '{transcript_text}'")

        print("Extracting phoneme timings...")
        phoneme_timing = animator.extract_phoneme_timing(transcript_text)
        print(f"Generated {len(phoneme_timing)} phoneme timing entries")

        print("Generating lip-sync video with viseme overlays...")
        if animator.generate_video_from_visemes(phoneme_timing):
            print("Lip-sync video generated successfully!")
            print("Merging video with audio...")
            if animator.merge_video_and_audio():
                print("Lip sync video created successfully!")
                print(f"Final video saved as final_{output_video}")
            else:
                print("Error: Failed to merge video and audio.")
        else:
            print("Error: Failed to generate lip-sync video.")

    except Exception as e:
        print(f"An error occurred during the lip sync process: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()



































import os
import cv2
import dlib
import numpy as np
from scipy.io import wavfile
import librosa
import librosa.display
import matplotlib.pyplot as plt
from pydub import AudioSegment
import phonemizer
from phonemizer.backend import EspeakBackend
import time
from collections import deque


class VisemePhonemeExtractor:
    def __init__(self, video_path, output_dir="visemes"):
        """
        Initialize the Viseme and Phoneme extractor with improved realism.

        Args:
            video_path (str): Path to the input video file
            output_dir (str): Directory to save output viseme images
        """
        self.video_path = video_path
        self.output_dir = output_dir

        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Define viseme categories with more granular transitions
        # We'll add transition visemes to make speech more realistic
        self.viseme_categories = {
            'A': ['AA', 'AE', 'AH'],  # as in "car", "cat", "cut"
            'E': ['EH', 'ER', 'EY'],  # as in "met", "bird", "say"
            'I': ['IH', 'IY'],  # as in "sit", "see"
            'O': ['AO', 'OW'],  # as in "dog", "go"
            'U': ['UH', 'UW'],  # as in "book", "too"
            'BMP': ['B', 'M', 'P'],  # Bilabial consonants
            'FV': ['F', 'V'],  # Labiodental consonants
            'CH-J-SH': ['CH', 'JH', 'SH', 'ZH'],  # Postalveolar consonants
            'TH': ['DH', 'TH'],  # Dental consonants
            'L': ['L'],  # Alveolar lateral approximant
            'R': ['R'],  # Alveolar approximant
            'S-Z': ['S', 'Z'],  # Alveolar sibilants
            'D-N-T': ['D', 'N', 'T'],  # Alveolar plosives and nasal
            'G-K-NG': ['G', 'K', 'NG'],  # Velar consonants
            'H-Y': ['HH', 'Y'],  # Glottal and palatal
            'Rest': ['sil', 'sp']  # Silence/rest position
        }

        # Add transition visemes for more realism
        self.transition_visemes = {
            'A-E': ['A', 'E'],       # Transition from A to E
            'A-O': ['A', 'O'],       # Transition from A to O
            'E-I': ['E', 'I'],       # Transition from E to I
            'I-O': ['I', 'O'],       # Transition from I to O
            'BMP-A': ['BMP', 'A'],   # Transition from BMP to A
            'FV-O': ['FV', 'O'],     # Transition from FV to O
            'S-Z-I': ['S-Z', 'I'],   # Transition from S-Z to I
            'Rest-A': ['Rest', 'A'], # Transition from Rest to A
        }

        # Combine all viseme categories
        self.all_viseme_categories = {**self.viseme_categories, **self.transition_visemes}

        # Create all required directories
        for category in self.all_viseme_categories:
            category_dir = os.path.join(output_dir, category)
            if not os.path.exists(category_dir):
                os.makedirs(category_dir)

        # Initialize face detector and landmark predictor
        self.detector = dlib.get_frontal_face_detector()

        # Try different paths for the shape predictor
        predictor_paths = [
            "flask_app/shape_predictor_68_face_landmarks.dat",
            "./shape_predictor_68_face_landmarks.dat",
            os.path.join(os.path.dirname(__file__), "shape_predictor_68_face_landmarks.dat"),
            os.path.abspath("flask_app/shape_predictor_68_face_landmarks.dat"),
            "shape_predictor_68_face_landmarks",
            "./shape_predictor_68_face_landmarks"
        ]

        predictor_path = None
        for path in predictor_paths:
            if os.path.exists(path):
                predictor_path = path
                print(f"Found predictor file at: {predictor_path}")
                break

        if predictor_path is None:
            raise FileNotFoundError(
                "Could not find the shape predictor file. Please download it from "
                "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2, "
                "extract it, and place it in the same directory as this script."
            )

        try:
            self.predictor = dlib.shape_predictor(predictor_path)
        except RuntimeError as e:
            raise RuntimeError(f"Error loading the shape predictor file: {e}")

        # Initialize phonemizer
        self.backend = EspeakBackend('en-us')

        # Settings for realistic viseme timing
        self.min_viseme_duration = 0.08  # Minimum duration for a viseme in seconds
        self.transition_duration = 0.04   # Duration for transition between visemes
        self.natural_pause = 0.2          # Natural pause duration between phrases

        # Queue for smooth viseme transitions
        self.viseme_history = deque(maxlen=3)

    def extract_audio(self):
        """Extract audio from video file and save it"""
        video = cv2.VideoCapture(self.video_path)

        # Check if video opened successfully
        if not video.isOpened():
            print("Error: Could not open video.")
            return None

        # Get video properties
        fps = video.get(cv2.CAP_PROP_FPS)

        # Release video capture
        video.release()

        # Use pydub to extract audio
        audio_path = os.path.join(self.output_dir, "extracted_audio.wav")
        video_audio = AudioSegment.from_file(self.video_path)
        video_audio.export(audio_path, format="wav")

        return audio_path, fps

    def analyze_audio(self, audio_path):
        """
        Enhanced audio analysis with more natural speech patterns

        Args:
            audio_path (str): Path to the extracted audio file

        Returns:
            tuple: Dictionary mapping timestamps to phonemes and a list of speech segments
        """
        # Load audio file
        y, sr = librosa.load(audio_path)

        # Calculate energy for speech detection
        energy = librosa.feature.rms(y=y)[0]
        frames = range(len(energy))
        frame_time = librosa.frames_to_time(frames, sr=sr)

        # Normalize energy
        energy_norm = (energy - np.min(energy)) / (np.max(energy) - np.min(energy) + 1e-10)

        # Use adaptive thresholding for better speech detection
        energy_mean = np.mean(energy_norm)
        threshold = max(0.15, energy_mean * 0.5)  # Adaptive threshold
        speech_frames = energy_norm > threshold

        # Create segments of speech and silence with more realistic timing
        segments = []
        in_speech = False
        start_idx = 0
        min_speech_duration = 0.1  # Minimum speech segment in seconds
        min_silence_duration = 0.08  # Minimum silence segment in seconds

        for i, is_speech in enumerate(speech_frames):
            if is_speech and not in_speech:
                # Start of speech
                start_idx = i
                in_speech = True
            elif not is_speech and in_speech:
                # End of speech
                end_idx = i
                duration = frame_time[end_idx] - frame_time[start_idx]

                # Only add if segment is long enough
                if duration >= min_speech_duration:
                    segments.append((frame_time[start_idx], frame_time[end_idx], "speech"))
                in_speech = False
            elif i == len(speech_frames) - 1 and in_speech:
                # End of audio while still in speech
                duration = frame_time[i] - frame_time[start_idx]
                if duration >= min_speech_duration:
                    segments.append((frame_time[start_idx], frame_time[i], "speech"))

        # Add silence segments between speech segments
        silence_segments = []
        for i in range(1, len(segments)):
            prev_end = segments[i-1][1]
            curr_start = segments[i][0]
            silence_duration = curr_start - prev_end

            if silence_duration >= min_silence_duration:
                silence_segments.append((prev_end, curr_start, "silence"))

        # Combine and sort all segments
        all_segments = segments + silence_segments
        all_segments.sort(key=lambda x: x[0])

        # For each speech segment, analyze phonemes with more realistic timing
        phoneme_map = {}

        # Text corpus for phoneme extraction - more natural sentences with various phonetic contexts
        corpus = [
            "Hello, how are you doing today?",
            "The quick brown fox jumps over the lazy dog.",
            "Please speak clearly into the microphone.",
            "We need to make this sound more natural and realistic.",
            "Articulation is important for clear speech production.",
            "Visemes combine to form realistic mouth movements.",
            "Coarticulation affects how we transition between sounds."
        ]

        # Get phonemes for the corpus
        phoneme_pool = []
        for text in corpus:
            phonemes = self.backend.phonemize([text], strip=True)[0].split()
            phoneme_pool.extend(phonemes)

        # For each speech segment, distribute phonemes with realistic timing
        for start_time, end_time, segment_type in all_segments:
            if segment_type == "speech":
                segment_duration = end_time - start_time

                # Calculate how many phonemes for this segment based on realistic speech rate
                # Average syllable duration ~0.2-0.3 seconds
                phoneme_count = max(1, int(segment_duration / 0.15))

                # Select phonemes for this segment
                # Use sequential phonemes from the pool for more natural sequences
                start_idx = np.random.randint(0, max(1, len(phoneme_pool) - phoneme_count))
                segment_phonemes = phoneme_pool[start_idx:start_idx + phoneme_count]

                # If we don't have enough, just use random selection
                if len(segment_phonemes) < phoneme_count:
                    segment_phonemes = np.random.choice(phoneme_pool, size=phoneme_count)

                # Distribute phonemes with realistic timing
                # Include slight variations in duration
                base_duration = segment_duration / phoneme_count

                # Create a slightly uneven distribution for more natural rhythm
                durations = np.random.normal(base_duration, base_duration * 0.2, phoneme_count)
                durations = np.clip(durations, base_duration * 0.6, base_duration * 1.4)
                durations = durations / np.sum(durations) * segment_duration

                time_point = start_time
                for i, phoneme in enumerate(segment_phonemes):
                    phoneme_map[time_point] = phoneme
                    time_point += durations[i]

            elif segment_type == "silence":
                # Add silence/rest phonemes for non-speech segments
                phoneme_map[start_time] = 'sil'

                # For longer silences, add multiple silence markers
                if end_time - start_time > 0.3:
                    midpoint = (start_time + end_time) / 2
                    phoneme_map[midpoint] = 'sil'

        return phoneme_map, all_segments

    def phoneme_to_viseme(self, phoneme, previous_viseme=None):
        """
        Convert a phoneme to its corresponding viseme category with context awareness

        Args:
            phoneme (str): The phoneme to convert
            previous_viseme (str): The previous viseme for context

        Returns:
            str: The viseme category
        """
        # Clean up the phoneme
        phoneme = phoneme.upper().replace('.', '').strip()

        # Remove stress markers (numbers)
        phoneme = ''.join([c for c in phoneme if not c.isdigit()])

        # Handle espeak specific notation
        phoneme_mapping = {
            'A:': 'AA', 'A': 'AE', 'V': 'AH', 'O:': 'AO', 'E': 'EH',
            'E@': 'ER', 'EI': 'EY', 'I': 'IH', 'I:': 'IY', 'O': 'OW',
            'U': 'UH', 'U:': 'UW', 'AI': 'AY', 'OI': 'OY', 'AU': 'AW'
        }

        if phoneme in phoneme_mapping:
            phoneme = phoneme_mapping[phoneme]

        # First check for exact matches in core visemes
        current_viseme = None
        for viseme, phoneme_list in self.viseme_categories.items():
            if phoneme in phoneme_list:
                current_viseme = viseme
                break

        # If no exact match, check partial matches
        if current_viseme is None:
            for viseme, phoneme_list in self.viseme_categories.items():
                for p in phoneme_list:
                    if p in phoneme or phoneme in p:
                        current_viseme = viseme
                        break
                if current_viseme:
                    break

        # Special case handling
        if current_viseme is None:
            if any(p in phoneme for p in ['SIL', 'SP', 'PAUSE', '_', '-']):
                current_viseme = 'Rest'
            else:
                # Distribute to vowel categories more naturally
                first_char = phoneme[0] if phoneme else ''

                vowel_map = {
                    'A': 'A', 'E': 'E', 'I': 'I', 'O': 'O', 'U': 'U',
                    'B': 'BMP', 'M': 'BMP', 'P': 'BMP',
                    'F': 'FV', 'V': 'FV',
                    'S': 'S-Z', 'Z': 'S-Z',
                    'T': 'D-N-T', 'D': 'D-N-T', 'N': 'D-N-T',
                    'L': 'L', 'R': 'R',
                    'G': 'G-K-NG', 'K': 'G-K-NG',
                    'J': 'CH-J-SH', 'C': 'CH-J-SH'
                }

                current_viseme = vowel_map.get(first_char, 'A')  # Default to 'A' if unknown

        # If we have a previous viseme, check if we should use a transition viseme
        if previous_viseme and previous_viseme != current_viseme:
            # Check if we have a defined transition between these visemes
            transition_key = f"{previous_viseme}-{current_viseme}"
            if transition_key in self.transition_visemes:
                return transition_key

            # For common transitions between vowels, create dynamic transitions
            if previous_viseme in ['A', 'E', 'I', 'O', 'U'] and current_viseme in ['A', 'E', 'I', 'O', 'U']:
                return f"{previous_viseme}-{current_viseme}"

        return current_viseme

    def get_mouth_roi(self, frame, rect, landmarks):
        """
        Extract the region of interest (ROI) containing the mouth with added padding

        Args:
            frame (numpy.ndarray): Video frame
            rect (dlib.rectangle): Detected face rectangle
            landmarks (dlib.full_object_detection): Facial landmarks

        Returns:
            numpy.ndarray: Cropped image of the mouth region with padding
        """
        # Get all mouth points (both inner and outer lips for better context)
        mouth_points = []
        for i in range(48, 68):  # Points 48-67 represent the mouth region
            point = landmarks.part(i)
            mouth_points.append((point.x, point.y))

        # Find the bounding box of the mouth
        x_min = min(point[0] for point in mouth_points)
        y_min = min(point[1] for point in mouth_points)
        x_max = max(point[0] for point in mouth_points)
        y_max = max(point[1] for point in mouth_points)

        # Add padding for context (enough to include some of the surrounding face)
        # This helps maintain consistent scale and positioning
        width = x_max - x_min
        height = y_max - y_min

        padding_x = int(width * 0.3)  # 30% padding horizontally
        padding_y = int(height * 0.5)  # 50% padding vertically (more above the mouth)

        x_min = max(0, x_min - padding_x)
        y_min = max(0, y_min - padding_y)
        x_max = min(frame.shape[1], x_max + padding_x)
        y_max = min(frame.shape[0], y_max + padding_y)

        # Crop the mouth region
        mouth_roi = frame[y_min:y_max, x_min:x_max]

        return mouth_roi

    def process_video(self):
        """Process the video to extract visemes with improved realism"""
        # Extract audio
        audio_path, fps = self.extract_audio()
        if not audio_path:
            return

        # Analyze audio for phonemes with improved timing
        print("Analyzing audio for phonemes with realistic timing...")
        phoneme_map, speech_segments = self.analyze_audio(audio_path)

        if not phoneme_map:
            print("Warning: No phonemes detected in the audio. Check audio quality.")
            return

        print(f"Detected {len(phoneme_map)} phoneme timestamps in the audio")

        # Open video
        video = cv2.VideoCapture(self.video_path)

        if not video.isOpened():
            print("Error: Could not open video.")
            return

        # Get video properties
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Processing video with {total_frames} frames at {fps} fps")

        frame_count = 0
        viseme_count = {category: 0 for category in self.all_viseme_categories}
        face_detection_count = 0

        # Create debug directory
        debug_dir = os.path.join(self.output_dir, "debug")
        os.makedirs(debug_dir, exist_ok=True)

        # Previous viseme for tracking transitions
        prev_viseme = 'Rest'
        # Keep track of viseme duration for realistic timing
        current_viseme_duration = 0

        # Create timestamp to frame mapping
        timestamp_frames = {}

        # First pass: process all frames and determine timestamps
        while True:
            ret, _ = video.read()
            if not ret:
                break

            # Calculate timestamp for this frame
            timestamp = frame_count / fps
            timestamp_frames[timestamp] = frame_count
            frame_count += 1

        # Reset video for second pass
        video.release()
        video = cv2.VideoCapture(self.video_path)
        frame_count = 0

        # Add coarticulation: pre-process phoneme sequence to create smooth transitions
        phoneme_timestamps = sorted(phoneme_map.keys())
        smooth_viseme_map = {}

        # For each phoneme, determine viseme with context
        prev_viseme = 'Rest'
        for i, ts in enumerate(phoneme_timestamps):
            phoneme = phoneme_map[ts]
            viseme = self.phoneme_to_viseme(phoneme, prev_viseme)
            smooth_viseme_map[ts] = viseme

            # Add transition visemes between non-adjacent visemes
            if i < len(phoneme_timestamps) - 1:
                next_ts = phoneme_timestamps[i + 1]
                gap = next_ts - ts

                # If gap is large enough for a transition
                if gap > self.transition_duration * 2:
                    transition_ts = ts + (gap / 2)  # Midpoint
                    # Create transition or maintain current viseme
                    if viseme != prev_viseme:
                        transition_viseme = f"{prev_viseme}-{viseme}"
                        # Only use defined transitions
                        if transition_viseme in self.transition_visemes:
                            smooth_viseme_map[transition_ts] = transition_viseme

            prev_viseme = viseme

        # Process frames with smoothed viseme map
        while True:
            # Read frame
            ret, frame = video.read()
            if not ret:
                break

            # Calculate timestamp
            timestamp = frame_count / fps

            # Find nearest phoneme timestamp with improved algorithm
            nearest_ts = min(smooth_viseme_map.keys(), key=lambda x: abs(x - timestamp), default=None)

            if nearest_ts is not None:
                # Get the viseme at this timestamp
                viseme = smooth_viseme_map[nearest_ts]

                # Keep track of duration for this viseme
                current_viseme_duration += 1/fps

                # If we've been in this viseme long enough, consider changing it
                # This prevents visemes from changing too rapidly
                if current_viseme_duration >= self.min_viseme_duration:
                    # Add the current viseme to history for natural blending
                    self.viseme_history.append(viseme)

                    # Process frame in grayscale for better face detection
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                    # Detect faces
                    faces = self.detector(gray)

                    if len(faces) > 0:
                        face_detection_count += 1

                    for face in faces:
                        # Get facial landmarks
                        landmarks = self.predictor(gray, face)

                        # Extract mouth region
                        mouth_roi = self.get_mouth_roi(frame, face, landmarks)

                        if mouth_roi.size > 0:
                            # Save mouth image with category
                            viseme_count[viseme] += 1
                            output_path = os.path.join(
                                self.output_dir,
                                viseme,
                                f"{viseme}_{viseme_count[viseme]:04d}_{frame_count:06d}.jpg"
                            )
                            cv2.imwrite(output_path, mouth_roi)

                            # Save debug frame every 30 frames or at viseme changes
                            if frame_count % 30 == 0 or prev_viseme != viseme:
                                # Draw mouth landmarks on the original frame
                                debug_frame = frame.copy()

                                # Draw landmarks
                                for i in range(48, 68):
                                    point = landmarks.part(i)
                                    cv2.circle(debug_frame, (point.x, point.y), 2, (0, 255, 0), -1)

                                # Add text with viseme info
                                cv2.putText(debug_frame, f"Viseme: {viseme}", (10, 30),
                                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                                if '-' in viseme:  # It's a transition
                                    cv2.putText(debug_frame, "TRANSITION", (10, 60),
                                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

                                # Save debug image
                                debug_path = os.path.join(debug_dir, f"frame_{frame_count:06d}_{viseme}.jpg")
                                cv2.imwrite(debug_path, debug_frame)

                    # Print progress for viseme changes
                    if prev_viseme != viseme:
                        print(f"Frame {frame_count}/{total_frames}, " 
                              f"Time: {timestamp:.2f}s, Viseme change: {prev_viseme} → {viseme}")

                        # Reset duration counter for the new viseme
                        current_viseme_duration = 0
                        prev_viseme = viseme

            # Print progress periodically
            if frame_count % 100 == 0:
                print(f"Processed {frame_count}/{total_frames} frames "
                      f"({frame_count / total_frames * 100:.1f}%)")

            frame_count += 1

        # Release video
        video.release()

        # Print face detection stats
        print(f"Face detection success rate: {face_detection_count}/{frame_count} frames "
              f"({face_detection_count / frame_count * 100:.1f}%)")

        print(f"Viseme extraction complete. {sum(viseme_count.values())} visemes extracted.")
        for category, count in sorted(viseme_count.items()):
            if count > 0:  # Only show categories with images
                print(f"  {category}: {count} images")

    def visualize_phonemes(self, audio_path):
        """
        Create enhanced visualization of audio with phoneme and viseme annotations

        Args:
            audio_path (str): Path to the audio file
        """
        # Load audio
        y, sr = librosa.load(audio_path)

        # Get phoneme map and speech segments
        phoneme_map, speech_segments = self.analyze_audio(audio_path)

        # Create figure with more information
        plt.figure(figsize=(15, 12))

        # Plot waveform
        plt.subplot(3, 1, 1)
        librosa.display.waveshow(y, sr=sr)
        plt.title('Waveform with Phoneme & Viseme Annotations')

        # Add phoneme and viseme markers with better spacing
        prev_viseme = None
        phoneme_times = sorted(phoneme_map.keys())

        for i, timestamp in enumerate(phoneme_times):
            phoneme = phoneme_map[timestamp]
            viseme = self.phoneme_to_viseme(phoneme, prev_viseme)

            # Use different colors for different types
            plt.axvline(x=timestamp, color='r', linestyle='--', alpha=0.5)

            # Position text to avoid overlap
            vertical_position = 0.5 if i % 2 == 0 else -0.5

            plt.text(timestamp, vertical_position,
                     f"{phoneme}/{viseme}", fontsize=8,
                     bbox=dict(facecolor='white', alpha=0.7))

            prev_viseme = viseme

        # Add speech segment visualization
        for start, end, seg_type in speech_segments:
            color = 'green' if seg_type == 'speech' else 'gray'
            plt.axvspan(start, end, alpha=0.2, color=color)

        # Plot spectrogram
        plt.subplot(3, 1, 2)
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
        librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Spectrogram with Viseme Transitions')

        # Visualize viseme transitions
        prev_viseme = None
        for timestamp in sorted(phoneme_map.keys()):
            phoneme = phoneme_map[timestamp]
            viseme = self.phoneme_to_viseme(phoneme, prev_viseme)

            # Highlight transitions differently
            if '-' in viseme:  # It's a transition viseme
                plt.axvline(x=timestamp, color='blue', linestyle='-', alpha=0.7, linewidth=2)
            else:
                plt.axvline(x=timestamp, color='r', linestyle='--', alpha=0.5)

            plt.text(timestamp, sr/2, viseme, fontsize=8,
                    bbox=dict(facecolor='white', alpha=0.7))

            prev_viseme = viseme

        # Add a third plot showing viseme durations and transitions
        plt.subplot(3, 1, 3)
        plt.title('Viseme Timeline with Transitions')
        plt.xlabel('Time (s)')
        plt.ylabel('Viseme Category')

        # Get all unique visemes
        all_visemes = list(set([self.phoneme_to_viseme(phoneme_map[ts])
                               for ts in sorted(phoneme_map.keys())]))

        # Sort visemes for better visualization
        all_visemes.sort()

        # Create mapping for y-axis positions
        viseme_positions = {v: i for i, v in enumerate(all_visemes)}

        # Add a timeline for each viseme
        prev_viseme = None
        prev_time = 0

        for timestamp in sorted(phoneme_map.keys()):
            phoneme = phoneme_map[timestamp]
            viseme = self.phoneme_to_viseme(phoneme, prev_viseme)

            if prev_viseme is not None:
                # Draw a line for the duration of the previous viseme
                if prev_viseme in viseme_positions:
                    y_pos = viseme_positions[prev_viseme]
                    plt.hlines(y_pos, prev_time, timestamp, colors='blue', linewidth=4)

                    # If this is a transition, draw a diagonal line
                    if viseme in viseme_positions and viseme != prev_viseme:
                        plt.plot([timestamp, timestamp + 0.05],
                                [viseme_positions[prev_viseme], viseme_positions[viseme]],
                                'r-', linewidth=2, alpha=0.5)

            prev_viseme = viseme
            prev_time = timestamp

        # Set y-tick labels
        plt.yticks(range(len(all_visemes)), all_visemes)

        # Set time limits
        plt.xlim(0, y.shape[0]/sr)

        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'phoneme_visualization.png'))
        plt.close()

        def process_text_or_wav(self, input_path, output_fps=30):
            """
            Process text or wav file to extract realistic viseme sequences

            Args:
                input_path (str): Path to text file or WAV file
                output_fps (int): Frames per second for output timing

            Returns:
                dict: Dictionary mapping frame numbers to viseme categories
            """
            print(f"Processing input: {input_path}")

            # Determine if input is text or WAV file
            is_wav = input_path.lower().endswith('.wav')

            if is_wav:
                # For WAV file, use our audio analysis
                phoneme_map, _ = self.analyze_audio(input_path)

                # Calculate audio duration
                y, sr = librosa.load(input_path)
                audio_duration = len(y) / sr

                print(f"Audio duration: {audio_duration:.2f} seconds")
            else:
                # For text file, use phonemizer directly
                with open(input_path, 'r') as f:
                    text = f.read().strip()

                # Phonemize the text
                phonemes = self.backend.phonemize([text], strip=True)[0].split()

                # Create timestamps with realistic timing
                # Start with a small pause
                timestamp = 0.2
                phoneme_map = {}

                # Estimate realistic timing based on phoneme type
                for phoneme in phonemes:
                    phoneme_map[timestamp] = phoneme

                    # Vary durations based on phoneme type for more realism
                    if phoneme in ['sil', 'sp']:
                        # Longer for silences
                        timestamp += np.random.uniform(0.15, 0.3)
                    elif any(phoneme in p_list for p_list in [
                        self.viseme_categories['A'],
                        self.viseme_categories['E'],
                        self.viseme_categories['I'],
                        self.viseme_categories['O'],
                        self.viseme_categories['U']
                    ]):
                        # Vowels are typically longer
                        timestamp += np.random.uniform(0.08, 0.15)
                    else:
                        # Consonants are typically shorter
                        timestamp += np.random.uniform(0.05, 0.1)

                # Add a final pause
                phoneme_map[timestamp] = 'sil'
                timestamp += 0.3

                audio_duration = timestamp
                print(f"Estimated duration from text: {audio_duration:.2f} seconds")

            # Generate frame mapping with smooth transitions
            total_frames = int(audio_duration * output_fps)
            frame_to_viseme = {}

            # Add smooth transitions between visemes
            phoneme_times = sorted(phoneme_map.keys())
            smooth_viseme_map = {}

            # Add transition visemes
            prev_viseme = 'Rest'
            for i, ts in enumerate(phoneme_times):
                phoneme = phoneme_map[ts]
                viseme = self.phoneme_to_viseme(phoneme, prev_viseme)
                smooth_viseme_map[ts] = viseme

                # Add transition frames between visemes
                if i < len(phoneme_times) - 1:
                    next_ts = phoneme_times[i + 1]
                    gap = next_ts - ts

                    # If gap is large enough for a transition
                    if gap > self.transition_duration * 2:
                        # Add a transition point
                        transition_ts = ts + (gap * 0.4)  # 40% into the gap

                        # Next phoneme for context
                        next_phoneme = phoneme_map[next_ts]
                        next_viseme = self.phoneme_to_viseme(next_phoneme)

                        # Only add transition if visemes are different
                        if viseme != next_viseme:
                            # Try to use a defined transition viseme if available
                            transition_key = f"{viseme}-{next_viseme}"
                            if transition_key in self.transition_visemes:
                                smooth_viseme_map[transition_ts] = transition_key
                            else:
                                # Otherwise use intermediate blending
                                # We'll use the same viseme but mark it for blending in rendering
                                smooth_viseme_map[transition_ts] = f"{viseme}+{next_viseme}"

                prev_viseme = viseme

            # Map to frames with smooth transitions
            for frame_num in range(total_frames):
                # Convert frame to timestamp
                timestamp = frame_num / output_fps

                # Find nearest phoneme timestamp
                nearest_ts = min(smooth_viseme_map.keys(), key=lambda x: abs(x - timestamp), default=None)

                if nearest_ts is not None:
                    frame_to_viseme[frame_num] = smooth_viseme_map[nearest_ts]
                else:
                    # Default to rest position
                    frame_to_viseme[frame_num] = 'Rest'

            # Apply temporal smoothing to prevent rapid transitions
            smoothed_visemes = {}
            window_size = int(output_fps * 0.1)  # 100ms smoothing window

            for frame_num in range(total_frames):
                # Get visemes in window
                start = max(0, frame_num - window_size // 2)
                end = min(total_frames - 1, frame_num + window_size // 2)

                # Count viseme occurrences in window
                viseme_count = {}
                for i in range(start, end + 1):
                    viseme = frame_to_viseme.get(i, 'Rest')
                    if viseme not in viseme_count:
                        viseme_count[viseme] = 0
                    viseme_count[viseme] += 1

                # Use most common viseme in window
                if viseme_count:
                    most_common = max(viseme_count.items(), key=lambda x: x[1])[0]
                    smoothed_visemes[frame_num] = most_common
                else:
                    smoothed_visemes[frame_num] = 'Rest'

            # Apply minimum duration constraint to prevent flickering
            min_duration_frames = int(self.min_viseme_duration * output_fps)
            final_visemes = {}

            current_viseme = None
            viseme_start = 0

            for frame_num in range(total_frames):
                viseme = smoothed_visemes[frame_num]

                # If viseme changes
                if viseme != current_viseme:
                    # Check if previous viseme lasted long enough
                    if current_viseme is not None:
                        duration = frame_num - viseme_start

                        if duration < min_duration_frames:
                            # Too short, extend previous viseme
                            for i in range(viseme_start, viseme_start + min_duration_frames):
                                if i < total_frames:
                                    final_visemes[i] = current_viseme

                            # Update start for new viseme
                            viseme_start = viseme_start + min_duration_frames
                        else:
                            # Duration was sufficient
                            viseme_start = frame_num
                    else:
                        # First viseme
                        viseme_start = frame_num

                    current_viseme = viseme

                # Add current frame
                final_visemes[frame_num] = current_viseme

            # Print summary of viseme distribution
            viseme_distribution = {}
            for frame_num, viseme in final_visemes.items():
                if viseme not in viseme_distribution:
                    viseme_distribution[viseme] = 0
                viseme_distribution[viseme] += 1

            print("\nViseme distribution in frames:")
            for viseme, count in sorted(viseme_distribution.items()):
                percentage = count / total_frames * 100
                print(f"  {viseme}: {count} frames ({percentage:.1f}%)")

            return final_visemes

        def render_viseme_sequence(self, viseme_frames, output_path, fps=30):
            """
            Render a visualization of the viseme sequence

            Args:
                viseme_frames (dict): Dictionary mapping frame numbers to viseme categories
                output_path (str): Path to save the visualization
                fps (int): Frames per second for timing
            """
            if not viseme_frames:
                print("No viseme frames to render")
                return

            # Calculate total duration
            total_frames = max(viseme_frames.keys()) + 1
            duration = total_frames / fps

            # Create timeline figure
            plt.figure(figsize=(max(12, duration), 8))

            # Get unique visemes
            unique_visemes = sorted(set(viseme_frames.values()))

            # Create mapping for y-axis positions
            viseme_positions = {v: i for i, v in enumerate(unique_visemes)}

            # Plot timeline
            plt.subplot(2, 1, 1)
            plt.title(f'Viseme Sequence Timeline ({duration:.2f} seconds)')
            plt.xlabel('Time (s)')
            plt.ylabel('Viseme Category')

            # Plot each viseme segment
            current_viseme = None
            start_frame = 0

            for frame in range(total_frames):
                viseme = viseme_frames.get(frame, 'Rest')

                if viseme != current_viseme:
                    if current_viseme is not None:
                        # Draw previous segment
                        start_time = start_frame / fps
                        end_time = frame / fps

                        plt.hlines(
                            viseme_positions[current_viseme],
                            start_time,
                            end_time,
                            colors='blue',
                            linewidth=6
                        )

                        # Label longer segments
                        if (end_time - start_time) > 0.2:
                            plt.text(
                                (start_time + end_time) / 2,
                                viseme_positions[current_viseme],
                                current_viseme,
                                ha='center',
                                va='center',
                                bbox=dict(facecolor='white', alpha=0.7)
                            )

                    # Update for new segment
                    current_viseme = viseme
                    start_frame = frame

            # Draw final segment
            if current_viseme is not None:
                start_time = start_frame / fps
                end_time = total_frames / fps

                plt.hlines(
                    viseme_positions[current_viseme],
                    start_time,
                    end_time,
                    colors='blue',
                    linewidth=6
                )

                if (end_time - start_time) > 0.2:
                    plt.text(
                        (start_time + end_time) / 2,
                        viseme_positions[current_viseme],
                        current_viseme,
                        ha='center',
                        va='center',
                        bbox=dict(facecolor='white', alpha=0.7)
                    )

            # Set y-tick labels
            plt.yticks(range(len(unique_visemes)), unique_visemes)

            # Set x-axis limits
            plt.xlim(0, duration)

            # Plot viseme distribution
            plt.subplot(2, 1, 2)
            plt.title('Viseme Distribution')

            # Count viseme occurrences
            viseme_counts = {}
            for viseme in viseme_frames.values():
                if viseme not in viseme_counts:
                    viseme_counts[viseme] = 0
                viseme_counts[viseme] += 1

            # Convert to percentages
            viseme_percentages = {v: (count / total_frames * 100)
                                  for v, count in viseme_counts.items()}

            # Plot bar chart
            categories = list(viseme_percentages.keys())
            values = list(viseme_percentages.values())

            # Sort by percentage
            sorted_indices = np.argsort(values)[::-1]
            sorted_categories = [categories[i] for i in sorted_indices]
            sorted_values = [values[i] for i in sorted_indices]

            bars = plt.bar(sorted_categories, sorted_values)

            # Add percentage labels
            for bar in bars:
                height = bar.get_height()
                plt.text(
                    bar.get_x() + bar.get_width() / 2.,
                    height + 0.5,
                    f'{height:.1f}%',
                    ha='center',
                    va='bottom'
                )

            plt.ylabel('Percentage of Frames')
            plt.ylim(0, max(values) * 1.2)  # Add some space for labels

            # Rotate x-axis labels if needed
            plt.xticks(rotation=45, ha='right')

            # Save visualization
            plt.tight_layout()
            plt.savefig(output_path)
            plt.close()

            print(f"Viseme sequence visualization saved to {output_path}")

def main():
    """Main function to run the viseme extraction"""
    import argparse

    # parser = argparse.ArgumentParser(description='Extract visemes from video, text, or WAV file')
    # parser.add_argument('input', help='Path to input video, text, or WAV file')
    # parser.add_argument('--output', '-o', default='visemes', help='Output directory')
    # parser.add_argument('--fps', type=int, default=30, help='Output FPS for text/WAV processing')

    # args = parser.parse_args()
    #
    # # Check if input file exists
    # if not os.path.exists(args.input):
    #     print(f"Error: Input file '{args.input}' not found.")
    #     return
    video_path = "speaking.mp4"
    try:
        # Initialize extractor
        print(f"Initializing viseme extractor for: {video_path}")
        extractor = VisemePhonemeExtractor(video_path=video_path)

        # Determine processing mode based on file extension
        if video_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            # Process video
            print("Processing video to extract visemes...")
            extractor.process_video()

            # Visualize audio analysis
            audio_path = os.path.join(extractor.output_dir, "extracted_audio.wav")
            print("Generating phoneme visualization...")
            extractor.visualize_phonemes(audio_path)
        # else:
        #     # Process text or WAV file
        #     print(f"Processing {'WAV' if args.input.lower().endswith('.wav') else 'text'} file...")
        #     viseme_frames = extractor.process_text_or_wav(args.input, args.fps)
        #
        #     # Render viseme sequence
        #     visualization_path = os.path.join(args.output, "viseme_sequence.png")
        #     extractor.render_viseme_sequence(viseme_frames, visualization_path, args.fps)

        print(f"Results saved to {extractor.output_dir}")
    except Exception as e:
        print(f"Error processing input: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()