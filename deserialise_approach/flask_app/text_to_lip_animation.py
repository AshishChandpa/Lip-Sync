"""
Text_to_lip_animation.py with voice synthesis capabilities
"""
import os
import cv2
import dlib
import numpy as np
import time
import argparse
from pathlib import Path
import random
import phonemizer
from phonemizer.backend import EspeakBackend
import imageio
from tqdm import tqdm
import tempfile
import subprocess
import gtts
from pydub import AudioSegment
from moviepy.editor import VideoFileClip, AudioFileClip


class TextToLipAnimation:
    def __init__(self, input_video_path, viseme_folder, output_path="animated_output.mp4"):
        """
        Initialize the Text to Lip Animation generator with voice synthesis

        Args:
            input_video_path (str): Path to the video with a person not speaking
            viseme_folder (str): Path to the folder containing viseme images
            output_path (str): Path to save the output animated video
        """
        self.input_video_path = input_video_path
        self.viseme_folder = viseme_folder
        self.output_path = output_path
        self.temp_video_path = os.path.splitext(output_path)[0] + "_temp.mp4"
        self.temp_audio_path = os.path.splitext(output_path)[0] + "_speech.mp3"

        # Create the espeak backend for phonemizing text
        self.backend = EspeakBackend('en-us')

        # Viseme category mapping
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

        # Initialize dlib face detector and predictor
        self.detector = dlib.get_frontal_face_detector()
        try:
            predictor_path = "shape_predictor_68_face_landmarks.dat"
            if not os.path.exists(predictor_path):
                alternate_paths = [
                    "./shape_predictor_68_face_landmarks.dat",
                    os.path.join(os.path.dirname(__file__), "shape_predictor_68_face_landmarks.dat"),
                    os.path.abspath("shape_predictor_68_face_landmarks.dat")
                ]
                for alt_path in alternate_paths:
                    if os.path.exists(alt_path):
                        predictor_path = alt_path
                        print(f"Found predictor file at: {predictor_path}")
                        break
            self.predictor = dlib.shape_predictor(predictor_path)
        except Exception as e:
            print(f"Warning: Could not load facial landmark predictor: {e}")
            print("Continuing without facial landmark detection.")
            self.predictor = None

        # Load viseme images
        self.viseme_images = self.load_viseme_images()

    def load_viseme_images(self):
        """
        Load all viseme images from the viseme folder

        Returns:
            dict: Dictionary mapping viseme categories to lists of image paths
        """
        viseme_images = {}

        # Check if the viseme folder exists
        if not os.path.exists(self.viseme_folder):
            print(f"Error: Viseme folder {self.viseme_folder} does not exist.")
            return viseme_images

        # Loop through each viseme category folder
        for category in os.listdir(self.viseme_folder):
            category_path = os.path.join(self.viseme_folder, category)

            # Skip if not a directory
            if not os.path.isdir(category_path):
                continue

            # Get all image files in the category folder
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
            print(f"Expected structure: {self.viseme_folder}/[VISEME_CATEGORY]/[image_files]")

        return viseme_images

    def phoneme_to_viseme(self, phoneme):
        """
        Convert a phoneme to its corresponding viseme category

        Args:
            phoneme (str): The phoneme to convert

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

        # First check for exact matches
        for viseme, phoneme_list in self.viseme_categories.items():
            if phoneme in phoneme_list:
                return viseme

        # If no exact match, try partial matches
        for viseme, phoneme_list in self.viseme_categories.items():
            for p in phoneme_list:
                if p in phoneme or phoneme in p:
                    return viseme

        # Special case handling
        if any(p in phoneme for p in ['SIL', 'SP', 'PAUSE', '_', '-']):
            return 'Rest'

        # First character heuristic
        first_char = phoneme[0] if phoneme else ''

        if first_char in ['B', 'M', 'P']:
            return 'BMP'
        elif first_char in ['F', 'V']:
            return 'FV'
        elif first_char in ['S', 'Z']:
            return 'S-Z'
        elif first_char in ['T', 'D', 'N']:
            return 'D-N-T'
        elif first_char in ['A']:
            return 'A'
        elif first_char in ['E']:
            return 'E'
        elif first_char in ['I']:
            return 'I'
        elif first_char in ['O']:
            return 'O'
        elif first_char in ['U']:
            return 'U'

        # Default to Rest if no match
        return 'Rest'

    def text_to_phonemes(self, text):
        """
        Convert text to a sequence of phonemes with timing information

        Args:
            text (str): Input text to convert to phonemes

        Returns:
            list: List of (phoneme, duration) pairs
        """
        # Phonemize the text
        phonemes = self.backend.phonemize([text], strip=True)[0].split()

        # Estimate durations for each phoneme
        # These values are approximate - adjust for realistic speech timing
        phoneme_durations = []

        for phoneme in phonemes:
            # Assign duration based on phoneme type (vowels longer than consonants)
            if phoneme.upper() in ['A:', 'E:', 'I:', 'O:', 'U:', 'AA', 'AE', 'AH', 'AO', 'EH', 'ER', 'EY', 'IH', 'IY',
                                   'OW', 'UH', 'UW']:
                # Vowels are longer
                duration = random.uniform(0.1, 0.22)  # 100-220ms
            elif phoneme.upper() in ['SIL', 'SP', '.', ',', '?', '!']:
                # Pauses
                duration = random.uniform(0.2, 0.5)  # 200-500ms
            else:
                # Consonants are shorter
                duration = random.uniform(0.05, 0.15)  # 50-150ms

            phoneme_durations.append((phoneme, duration))

        return phoneme_durations

    def generate_speech(self, text, output_path):
        """
        Generate speech audio from text using gTTS

        Args:
            text (str): Text to convert to speech
            output_path (str): Path to save the generated audio file

        Returns:
            float: Duration of the generated audio in seconds
        """
        print(f"Generating speech audio for: {text[:50]}{'...' if len(text) > 50 else ''}")

        try:
            # Generate speech using Google Text-to-Speech
            tts = gtts.gTTS(text=text, lang='en', slow=False)
            tts.save(output_path)

            # Get the duration of the audio file
            audio = AudioSegment.from_file(output_path)
            duration = len(audio) / 1000.0  # Convert ms to seconds

            print(f"Speech generation complete. Duration: {duration:.2f} seconds")
            return duration
        except Exception as e:
            print(f"Error generating speech: {e}")
            # Create a silent audio file of appropriate length as fallback
            estimated_duration = len(text.split()) * 0.3  # Rough estimate: 0.3 seconds per word
            silence = AudioSegment.silent(duration=int(estimated_duration * 1000))
            silence.export(output_path, format="mp3")
            print(f"Created silent audio as fallback. Duration: {estimated_duration:.2f} seconds")
            return estimated_duration

    def get_face_landmarks(self, frame):
        """
        Get facial landmarks for a frame

        Args:
            frame: Video frame

        Returns:
            tuple: (face_rect, landmarks) or None if no face detected
        """
        if self.predictor is None:
            return None

        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = self.detector(gray)

        if not faces:
            return None

        # Get the first face
        face = faces[0]

        # Get facial landmarks
        landmarks = self.predictor(gray, face)

        return (face, landmarks)

    def get_mouth_roi(self, frame, landmarks):
        """
        Extract the region of interest (ROI) containing the mouth

        Args:
            frame: Video frame
            landmarks: Facial landmarks

        Returns:
            tuple: (mouth_roi, (x_min, y_min, x_max, y_max)) - ROI and its coordinates
        """
        # Get the mouth landmarks (points 48-68)
        mouth_points = []
        for i in range(48, 68):
            point = landmarks.part(i)
            mouth_points.append((point.x, point.y))

        # Find the bounding box of the mouth
        x_min = min(point[0] for point in mouth_points)
        y_min = min(point[1] for point in mouth_points)
        x_max = max(point[0] for point in mouth_points)
        y_max = max(point[1] for point in mouth_points)

        # Add some margin
        margin = 10
        x_min = max(0, x_min - margin)
        y_min = max(0, y_min - margin)
        x_max = min(frame.shape[1], x_max + margin)
        y_max = min(frame.shape[0], y_max + margin)

        # Crop the mouth region
        mouth_roi = frame[y_min:y_max, x_min:x_max]

        return mouth_roi, (x_min, y_min, x_max, y_max)

    def blend_mouth(self, frame, viseme_img, mouth_coords, alpha=0.7):
        """
        Blend a viseme mouth image onto the original frame

        Args:
            frame: Original video frame
            viseme_img: Viseme mouth image to blend
            mouth_coords: (x_min, y_min, x_max, y_max) of the mouth region
            alpha: Blending factor (0.0-1.0)

        Returns:
            numpy.ndarray: Frame with blended mouth
        """
        x_min, y_min, x_max, y_max = mouth_coords

        # Get dimensions
        h, w = y_max - y_min, x_max - x_min

        # Resize viseme image to match mouth dimensions
        try:
            resized_viseme = cv2.resize(viseme_img, (w, h), interpolation=cv2.INTER_LANCZOS4)
        except Exception as e:
            print(f"Error resizing viseme image: {e}")
            return frame

        # Create a mask for better blending (focus on mouth area)
        mask = np.zeros((h, w), dtype=np.float32)
        center = (w // 2, h // 2)
        cv2.ellipse(mask, center, (w // 2 - 5, h // 2 - 5), 0, 0, 360, 1, -1)
        mask = cv2.GaussianBlur(mask, (11, 11), 0)

        # Extract the mouth region from the original frame
        roi = frame[y_min:y_max, x_min:x_max].copy()

        # Apply mask-weighted blending
        for c in range(3):  # RGB channels
            roi[:, :, c] = (1 - mask * alpha) * roi[:, :, c] + mask * alpha * resized_viseme[:, :, c]

        # Put the blended region back into the frame
        result = frame.copy()
        result[y_min:y_max, x_min:x_max] = roi

        return result

    def animate_from_text(self, text, words_per_minute=150):
        """
        Create a lip-synced animation from text with speech

        Args:
            text (str): Text to animate
            words_per_minute (int): Speaking rate

        Returns:
            bool: True if animation was created successfully
        """
        # Check if we have viseme images
        if not self.viseme_images:
            print("Error: No viseme images available. Animation cannot be created.")
            return False

        # Generate speech audio
        audio_duration = self.generate_speech(text, self.temp_audio_path)

        # Convert text to phonemes with timing
        print("Converting text to phonemes...")
        phoneme_durations = self.text_to_phonemes(text)

        # Convert phonemes to visemes
        viseme_sequence = []
        estimated_duration = 0

        for phoneme, duration in phoneme_durations:
            viseme = self.phoneme_to_viseme(phoneme)
            viseme_sequence.append((viseme, duration))
            estimated_duration += duration

        print(f"Generated {len(viseme_sequence)} visemes with estimated duration of {estimated_duration:.2f} seconds")

        # Adjust phoneme durations to match actual audio duration
        if estimated_duration > 0 and audio_duration > 0:
            scale_factor = audio_duration / estimated_duration
            viseme_sequence = [(viseme, duration * scale_factor) for viseme, duration in viseme_sequence]
            print(f"Adjusted viseme durations to match audio duration of {audio_duration:.2f} seconds")
            estimated_duration = audio_duration

        # Open the input video
        print(f"Opening input video: {self.input_video_path}")
        video = cv2.VideoCapture(self.input_video_path)

        if not video.isOpened():
            print(f"Error: Could not open input video {self.input_video_path}")
            return False

        # Get video properties
        fps = video.get(cv2.CAP_PROP_FPS)
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"Video properties: {width}x{height}, {fps} fps, {frame_count} frames")

        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(self.temp_video_path, fourcc, fps, (width, height))

        # Calculate frames needed for the animation
        frames_needed = int(estimated_duration * fps)

        # Check if input video has enough frames
        if frames_needed > frame_count:
            print(f"Warning: Text animation requires {frames_needed} frames, but video only has {frame_count} frames.")
            print("The video will loop to accommodate the full animation.")

        # Process animation
        print("Creating animation...")
        current_frame = 0
        processed_frames = 0
        viseme_idx = 0
        time_in_current_viseme = 0

        # Default mouth coordinates if face detection fails
        default_mouth_coords = (width // 3, height // 2, 2 * width // 3, 3 * height // 4)
        mouth_coords = default_mouth_coords

        # Store face landmark detection result to reuse when detection fails
        last_successful_landmarks = None

        progress_bar = tqdm(total=frames_needed)

        while processed_frames < frames_needed:
            # Read frame
            video.set(cv2.CAP_PROP_POS_FRAMES, current_frame % frame_count)
            ret, frame = video.read()

            if not ret:
                print("Error reading frame. Exiting.")
                break

            # Get the current viseme
            current_viseme, viseme_duration = viseme_sequence[viseme_idx]

            # Try to detect face landmarks
            landmark_result = self.get_face_landmarks(frame)

            if landmark_result:
                face, landmarks = landmark_result
                last_successful_landmarks = landmarks
                _, mouth_coords = self.get_mouth_roi(frame, landmarks)
            elif last_successful_landmarks is not None:
                # Use the last successful detection
                _, mouth_coords = self.get_mouth_roi(frame, last_successful_landmarks)

            # Select a random viseme image for the current category
            if current_viseme in self.viseme_images and self.viseme_images[current_viseme]:
                viseme_img_path = random.choice(self.viseme_images[current_viseme])
                viseme_img = cv2.imread(viseme_img_path)

                if viseme_img is not None:
                    # Blend the viseme mouth onto the frame
                    frame = self.blend_mouth(frame, viseme_img, mouth_coords)
            else:
                # If no image available for this viseme, use 'Rest' or continue without modification
                if 'Rest' in self.viseme_images and self.viseme_images['Rest']:
                    viseme_img_path = random.choice(self.viseme_images['Rest'])
                    viseme_img = cv2.imread(viseme_img_path)

                    if viseme_img is not None:
                        frame = self.blend_mouth(frame, viseme_img, mouth_coords)

            # Write the frame
            writer.write(frame)

            # Update time and viseme index
            time_in_current_viseme += 1.0 / fps
            if time_in_current_viseme >= viseme_duration:
                viseme_idx = (viseme_idx + 1) % len(viseme_sequence)
                time_in_current_viseme = 0

            # Move to next frame
            current_frame += 1
            processed_frames += 1
            progress_bar.update(1)

        progress_bar.close()

        # Release resources
        video.release()
        writer.release()

        # Combine video and audio
        print("Combining animation with speech audio...")
        self.combine_video_and_audio(self.temp_video_path, self.temp_audio_path, self.output_path)

        # Clean up temporary files
        if os.path.exists(self.temp_video_path):
            os.remove(self.temp_video_path)

        print(f"Animation with speech complete! Output saved to {self.output_path}")
        return True

    def combine_video_and_audio(self, video_path, audio_path, output_path):
        """
        Combine video and audio into a single file

        Args:
            video_path (str): Path to video file
            audio_path (str): Path to audio file
            output_path (str): Path to save the combined file
        """
        try:
            # Load the video and audio clips
            video_clip = VideoFileClip(video_path)
            audio_clip = AudioFileClip(audio_path)

            # Set the audio of the video clip
            video_with_audio = video_clip.set_audio(audio_clip)

            # Write the result to a file
            video_with_audio.write_videofile(
                output_path,
                codec='libx264',
                audio_codec='aac',
                temp_audiofile='temp-audio.m4a',
                remove_temp=True
            )

            # Close the clips
            video_clip.close()
            audio_clip.close()

        except Exception as e:
            print(f"Error combining video and audio: {e}")
            print("Trying alternative method with FFmpeg...")

            try:
                # Alternative method using FFmpeg directly
                cmd = [
                    'ffmpeg', '-y',
                    '-i', video_path,
                    '-i', audio_path,
                    '-c:v', 'copy',
                    '-c:a', 'aac',
                    '-map', '0:v:0',
                    '-map', '1:a:0',
                    '-shortest',
                    output_path
                ]
                subprocess.run(cmd, check=True)
            except Exception as e2:
                print(f"Error with alternative method: {e2}")
                print("Keeping silent video as output.")
                # If all else fails, just rename the video file
                import shutil
                shutil.copy(video_path, output_path)

    def create_preview_gif(self, duration=5.0):
        """
        Create a preview GIF of the animation

        Args:
            duration (float): Duration of the preview in seconds

        Returns:
            str: Path to the created GIF file
        """
        # Check if output video exists
        if not os.path.exists(self.output_path):
            print("Error: Output video not found. Create animation first.")
            return None

        # Create output gif path
        gif_path = os.path.splitext(self.output_path)[0] + "_preview.gif"

        # Open the output video
        video = cv2.VideoCapture(self.output_path)

        if not video.isOpened():
            print("Error: Could not open output video.")
            return None

        # Get video properties
        fps = video.get(cv2.CAP_PROP_FPS)
        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        # Calculate frames to include in GIF
        frames_for_gif = min(int(duration * fps), frame_count)

        # Sample frames for GIF
        frames = []
        sampling_step = max(1, frame_count // (frames_for_gif))

        for i in range(0, frame_count, sampling_step):
            if len(frames) >= frames_for_gif:
                break

            video.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = video.read()

            if not ret:
                break

            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Resize for GIF (smaller file size)
            height, width = frame_rgb.shape[:2]
            new_width = min(width, 480)
            new_height = int(height * (new_width / width))
            frame_rgb = cv2.resize(frame_rgb, (new_width, new_height))

            frames.append(frame_rgb)

        video.release()

        if not frames:
            print("Error: No frames could be read from the video.")
            return None

        # Create GIF
        print(f"Creating preview GIF with {len(frames)} frames...")
        imageio.mimsave(gif_path, frames, duration=0.1)  # 10 FPS

        print(f"Preview GIF saved to {gif_path}")
        return gif_path


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Generate lip animation from text with speech')
    parser.add_argument('--video', required=True, help='Path to input video with a person')
    parser.add_argument('--visemes', required=True, help='Path to folder containing viseme images')
    parser.add_argument('--text', required=True, help='Text to animate and speak')
    parser.add_argument('--output', default='animated_speech.mp4', help='Output video path')
    parser.add_argument('--wpm', type=int, default=150, help='Words per minute speaking rate')
    parser.add_argument('--preview', action='store_true', help='Create a preview GIF')

    args = parser.parse_args()

    # Create the animator
    animator = TextToLipAnimation(
        input_video_path=args.video,
        viseme_folder=args.visemes,
        output_path=args.output
    )

    # Animate from text with speech
    success = animator.animate_from_text(args.text, words_per_minute=args.wpm)

    if success and args.preview:
        animator.create_preview_gif()


if __name__ == "__main__":
    main()