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
    # Ensure the paths are absolute
    input_file = os.path.abspath(input_file)
    output_file = os.path.abspath(output_file)

    print(f"Converting video from {input_file} to {output_file}")

    # Use a more robust FFmpeg command with explicit codec specifications
    cmd = [
        get_setting("FFMPEG_BINARY"),
        "-i", input_file,
        "-c:v", "libx264",  # Force video codec to H.264
        "-preset", "medium",  # Encoding preset (speed/quality tradeoff)
        "-pix_fmt", "yuv420p",  # Standard pixel format for better compatibility
        "-r", "30",  # Ensure the frame rate is set
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
        # Execute the conversion process
        proc = subprocess.Popen(cmd, **popen_params)
        stdout, stderr = proc.communicate()

        # Check if the process was successful
        if proc.returncode != 0:
            raise RuntimeError(f"Video conversion failed with error:\n{stderr.decode()}")

        print(f"Video conversion successful: {output_file}")
        return output_file
    except Exception as e:
        # Log any exceptions that occur
        print(f"An error occurred during video conversion: {e}")
        return None


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

        # Set the resolution to match the input video if not specified
        if resolution:
            self.resolution = resolution
        elif self.width > 0 and self.height > 0:
            self.resolution = (self.width, self.height)
        else:
            self.resolution = (640, 480)  # Default fallback

        # Check if the viseme folder exists
        if not os.path.exists(viseme_folder):
            raise ValueError(f"Viseme folder {viseme_folder} not found!")

        self.viseme_images = self.load_viseme_images()

        # Print available viseme categories after loading
        print("Available viseme categories:", list(self.viseme_images.keys()))

    def load_viseme_images(self):
        """Load all viseme images from the viseme folder."""
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
                # Normalize category name to uppercase for case-insensitive matching
                viseme_images[category] = image_files
                print(f"Loaded {len(image_files)} images for viseme '{category}'")

        if not viseme_images:
            print("Warning: No viseme images found. Check your viseme folder structure.")
        return viseme_images

    def get_mouth_landmarks(self, frame):
        """Detect face and extract mouth landmarks using dlib."""
        # Convert frame to grayscale for better face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscale frame
        faces = self.detector(gray, 0)

        # If no faces detected, return None
        if len(faces) == 0:
            return None

        # Get the first face (assuming only one face in the video)
        face = faces[0]

        # Predict facial landmarks
        landmarks = self.predictor(gray, face)

        # Extract mouth landmarks (points 48-68 in the 68-point model)
        mouth_points = []
        for i in range(48, 68):
            point = landmarks.part(i)
            mouth_points.append((point.x, point.y))

        return mouth_points

    def get_mouth_region(self, mouth_landmarks):
        """Calculate mouth region (rectangle) from landmarks."""
        if not mouth_landmarks:
            return None

        # Convert landmarks to numpy array
        points = np.array(mouth_landmarks)

        # Get the bounding rectangle with some padding
        x_min = np.min(points[:, 0])
        y_min = np.min(points[:, 1])
        x_max = np.max(points[:, 0])
        y_max = np.max(points[:, 1])

        # Add padding around the mouth (20% extra width and height)
        width = x_max - x_min
        height = y_max - y_min

        padding_x = int(width * 0.2)
        padding_y = int(height * 0.2)

        x_min = max(0, x_min - padding_x)
        y_min = max(0, y_min - padding_y)
        x_max = min(self.width, x_max + padding_x)
        y_max = min(self.height, y_max + padding_y)

        # Return dictionary with mouth region coordinates
        return {
            'x': x_min,
            'y': y_min,
            'width': x_max - x_min,
            'height': y_max - y_min,
            'center_x': (x_min + x_max) // 2,
            'center_y': (y_min + y_max) // 2
        }

    def phoneme_to_viseme(self, phoneme):
        """Map phoneme to viseme based on the actual viseme folders available."""
        # Based on the loaded viseme categories from the error message
        phoneme_mapping = {
            # Vowels
            'AA': 'A', 'AE': 'A', 'AH': 'A', 'AO': 'O', 'AW': 'A', 'AY': 'A',
            'EH': 'E', 'ER': 'E', 'EY': 'E', 'IH': 'I', 'IY': 'I',
            'UH': 'U', 'UW': 'U', 'OW': 'O', 'OY': 'O',

            # Consonants
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

            # Silence and pauses
            'SIL': 'Rest', 'SP': 'Rest'
        }

        # Get the mapped viseme or use 'Rest' as fallback
        viseme = phoneme_mapping.get(phoneme, 'Rest')

        # Check if the mapped viseme exists in our loaded images, if not use a fallback
        if viseme not in self.viseme_images:
            # Try case-insensitive matching
            for key in self.viseme_images.keys():
                if key.lower() == viseme.lower():
                    return key

            # If still not found, use an alternative or default
            if viseme == 'Rest' and 'Rest' not in self.viseme_images:
                # Find a suitable alternative for 'Rest'
                alternatives = ['debug', 'A']  # Try these alternatives in order
                for alt in alternatives:
                    if alt in self.viseme_images:
                        print(f"Using '{alt}' as fallback for 'Rest'")
                        return alt

            # Default fallback to the first available viseme
            if self.viseme_images:
                first_key = list(self.viseme_images.keys())[0]
                print(f"Viseme '{viseme}' not found, using '{first_key}' as fallback")
                return first_key

        return viseme

    def extract_audio_transcript(self):
        """Extract the transcript from the audio file using SpeechRecognition."""
        recognizer = sr.Recognizer()

        # Load the audio file for transcription
        audio = sr.AudioFile(self.audio_file)

        with audio as source:
            audio_data = recognizer.record(source)

        try:
            # Use Google Web Speech API to transcribe the audio
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
            # Convert text to phonemes using phonemizer
            phonemes = self.backend.phonemize([transcript], strip=True)[0].split()
            print(f"Phonemized text: {phonemes}")

            phoneme_timing = []
            audio_duration_ms = len(self.audio)  # in milliseconds
            audio_duration_sec = audio_duration_ms / 1000.0  # Convert to seconds

            # Detect silent parts using audio energy
            silence_threshold = -35  # dB, adjust based on your audio characteristics
            chunk_size = 100  # ms
            silence_min_duration = 300  # ms minimum silence duration to consider

            # Process audio to find silence regions
            silence_regions = []
            current_silence_start = None

            for i in range(0, len(self.audio), chunk_size):
                chunk = self.audio[i:i + chunk_size]
                if len(chunk) == 0:
                    continue

                # Calculate dB level
                if chunk.dBFS < silence_threshold:
                    # This is silence
                    if current_silence_start is None:
                        current_silence_start = i / 1000.0  # convert to seconds
                else:
                    # This is speech
                    if current_silence_start is not None:
                        silence_end = i / 1000.0
                        if (silence_end - current_silence_start) >= (silence_min_duration / 1000.0):
                            silence_regions.append((current_silence_start, silence_end))
                        current_silence_start = None

            # Add final silence region if applicable
            if current_silence_start is not None:
                silence_end = audio_duration_sec
                if (silence_end - current_silence_start) >= (silence_min_duration / 1000.0):
                    silence_regions.append((current_silence_start, silence_end))

            print(f"Detected {len(silence_regions)} silence regions: {silence_regions}")

            # Calculate speech regions (inverse of silence regions)
            speech_regions = []
            last_end = 0

            for start, end in sorted(silence_regions):
                if start > last_end:
                    speech_regions.append((last_end, start))
                last_end = end

            # Add final speech region if needed
            if last_end < audio_duration_sec:
                speech_regions.append((last_end, audio_duration_sec))

            print(f"Calculated {len(speech_regions)} speech regions: {speech_regions}")

            # If no speech regions detected, fall back to using the entire audio
            if not speech_regions:
                speech_regions = [(0, audio_duration_sec)]

            # Distribute phonemes across speech regions
            total_speech_duration = sum(end - start for start, end in speech_regions)
            phonemes_per_second = len(phonemes) / max(total_speech_duration, 0.1)

            phoneme_idx = 0
            for start, end in speech_regions:
                region_duration = end - start
                # Calculate number of phonemes in this speech region
                num_phonemes_in_region = max(1, int(round(region_duration * phonemes_per_second)))
                num_phonemes_in_region = min(num_phonemes_in_region, len(phonemes) - phoneme_idx)

                if num_phonemes_in_region <= 0:
                    continue

                # Calculate phoneme duration in this region
                phoneme_duration = region_duration / num_phonemes_in_region

                # Assign timings to phonemes in this region
                for i in range(num_phonemes_in_region):
                    if phoneme_idx >= len(phonemes):
                        break

                    phoneme = phonemes[phoneme_idx]
                    phoneme_start = start + (i * phoneme_duration)
                    phoneme_end = phoneme_start + phoneme_duration

                    # Get the viseme for this phoneme
                    viseme = self.phoneme_to_viseme(phoneme)

                    phoneme_timing.append({
                        'phoneme': phoneme,
                        'viseme': viseme,
                        'start': phoneme_start,
                        'end': phoneme_end
                    })

                    phoneme_idx += 1

            # Print some timing info for debugging
            print(f"Audio duration: {audio_duration_sec} seconds")
            print(f"Total phonemes: {len(phonemes)}")
            print(f"Phonemes assigned: {phoneme_idx}")
            print(f"Phoneme timing entries: {len(phoneme_timing)}")

            return phoneme_timing
        except Exception as e:
            print(f"Error during phoneme timing extraction: {e}")
            # Return a simple fallback timing if phonemization failed
            return self.create_fallback_timing(transcript)

    def create_fallback_timing(self, transcript):
        """Create a simple fallback timing when phonemization fails."""
        print("Using fallback timing generation...")

        # Use words for timing if phonemization failed
        words = transcript.split()
        total_duration_ms = len(self.audio)
        word_duration = total_duration_ms / max(1, len(words)) / 1000.0  # in seconds

        timing = []
        start_time = 0.0

        for word in words:
            # For each word, assign a viseme
            end_time = start_time + word_duration

            # Choose a viseme based on the first letter of the word (simplified approach)
            first_char = word[0].upper() if word else 'A'
            if first_char in 'AEIOU':
                viseme = first_char  # Use vowel directly
            elif first_char in 'BMP':
                viseme = 'BMP'
            elif first_char in 'DTN':
                viseme = 'D-N-T'
            elif first_char in 'GKNG':
                viseme = 'G-K-NG'
            else:
                viseme = 'A'  # Default to 'A' as fallback

            # Ensure the viseme exists in our loaded images
            viseme = self.phoneme_to_viseme(viseme)

            timing.append({
                'phoneme': word,
                'viseme': viseme,
                'start': start_time,
                'end': end_time
            })

            start_time = end_time

        return timing

    def generate_video_from_visemes(self, phoneme_timing):
        """Generate the video by overlaying viseme mouth images onto the original silent video frames."""
        # Make sure to use a widely supported codec
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Video codec ('mp4v')

        # Ensure output directory exists
        output_dir = os.path.dirname(self.output_video)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        writer = cv2.VideoWriter(self.output_video, fourcc, self.fps, self.resolution)

        # Check if writer opened successfully
        if not writer.isOpened():
            print(f"Error: Could not open video writer for {self.output_video}")
            return False

        total_frames = int(self.audio.duration_seconds * self.fps)

        # Reset video capture to start
        self.video.set(cv2.CAP_PROP_POS_FRAMES, 0)

        print(f"Generating {total_frames} frames in the video...")

        # Add a small constant for floating point comparison safety
        epsilon = 0.001

        # Store the last detected mouth region as fallback
        last_valid_mouth_region = None

        # Process each frame
        for frame_idx in range(total_frames):
            # Get the current time for the frame
            current_time = frame_idx / self.fps

            # Read the corresponding frame from the silence video
            ret, original_frame = self.video.read()
            if not ret:
                # If we've reached the end of the input video, reset to the beginning
                self.video.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, original_frame = self.video.read()
                if not ret:
                    print(f"Error: Could not read frame from video.")
                    break  # End of video

            # Use original frame as the base frame
            frame = original_frame.copy()

            # Detect face and get mouth landmarks for this frame
            mouth_landmarks = self.get_mouth_landmarks(frame)

            # Get mouth region from landmarks or use the last valid region
            if mouth_landmarks:
                mouth_region = self.get_mouth_region(mouth_landmarks)
                # Save this region as fallback for future frames
                if mouth_region:
                    last_valid_mouth_region = mouth_region
            else:
                # Use the last valid mouth region if no face is detected in current frame
                mouth_region = last_valid_mouth_region

            # If we still don't have a valid mouth region, skip the frame
            if not mouth_region:
                print(f"Warning: No face detected in frame {frame_idx}")
                writer.write(frame)
                continue

            # Check if the current time is within any phoneme segment
            is_in_speech = False
            current_viseme = None
            current_phoneme = None

            for timing in phoneme_timing:
                if timing['start'] - epsilon <= current_time <= timing['end'] + epsilon:
                    is_in_speech = True
                    current_viseme = timing['viseme']
                    current_phoneme = timing['phoneme']
                    break

            # Only overlay viseme if we're in a speech segment
            if is_in_speech and current_viseme and current_viseme in self.viseme_images and self.viseme_images[
                current_viseme]:
                # Choose a random viseme image from the appropriate category
                viseme_image_path = random.choice(self.viseme_images[current_viseme])
                viseme_img = cv2.imread(viseme_image_path, cv2.IMREAD_UNCHANGED)  # Read with alpha if available

                if viseme_img is not None:
                    # Extract mouth coordinates from the detected region
                    mouth_x = mouth_region['x']
                    mouth_y = mouth_region['y']
                    mouth_width = mouth_region['width']
                    mouth_height = mouth_region['height']

                    # Resize viseme image to fit the mouth region
                    viseme_resized = cv2.resize(viseme_img, (mouth_width, mouth_height))

                    # Check if viseme has alpha channel (4 channels)
                    if viseme_resized.shape[2] == 4:
                        # Extract the alpha channel
                        alpha = viseme_resized[:, :, 3] / 255.0
                        # Extract BGR channels
                        viseme_rgb = viseme_resized[:, :, :3]

                        # Get the region of interest from the original frame
                        roi = frame[mouth_y:mouth_y + mouth_height, mouth_x:mouth_x + mouth_width]

                        # Ensure ROI dimensions match the viseme
                        if roi.shape[:2] == viseme_rgb.shape[:2]:
                            # For each color channel
                            for c in range(3):
                                # Blend the viseme and ROI using alpha
                                roi[:, :, c] = (1 - alpha) * roi[:, :, c] + alpha * viseme_rgb[:, :, c]

                            # Place the blended ROI back into the frame
                            frame[mouth_y:mouth_y + mouth_height, mouth_x:mouth_x + mouth_width] = roi
                        else:
                            print(
                                f"Warning: ROI dimensions {roi.shape[:2]} don't match viseme dimensions {viseme_rgb.shape[:2]}")
                    else:
                        # If no alpha channel, create a simple overlay with the viseme
                        try:
                            frame[mouth_y:mouth_y + mouth_height, mouth_x:mouth_x + mouth_width] = viseme_resized
                        except ValueError as e:
                            print(f"Error overlaying viseme: {e}")
                            # Continue with original frame

                    # Optional: Draw mouth landmarks for debugging
                    # if mouth_landmarks:
                    #     for point in mouth_landmarks:
                    #         cv2.circle(frame, point, 2, (0, 255, 0), -1)

                else:
                    print(f"Warning: Could not load viseme image from {viseme_image_path}")

            # Write the processed frame
            writer.write(frame)

        # Release the writer
        writer.release()
        print(f"Video written successfully to {self.output_video}")
        return True

    def merge_video_and_audio(self):
        """Merge the generated video with the original audio to create the final lip sync video."""
        try:
            # Skip the FFmpeg conversion step and directly use MoviePy
            print(f"Loading video file: {self.output_video}")
            video_clip = VideoFileClip(self.output_video)

            print(f"Loading audio file: {self.audio_file}")
            audio_clip = AudioFileClip(self.audio_file)

            # Adjust video duration to match audio if needed
            if video_clip.duration != audio_clip.duration:
                print(
                    f"Video duration ({video_clip.duration}s) doesn't match audio duration ({audio_clip.duration}s). Adjusting...")
                # Trim or loop the video if needed
                if video_clip.duration > audio_clip.duration:
                    video_clip = video_clip.subclip(0, audio_clip.duration)
                else:
                    # If video is shorter, loop it
                    loops_needed = int(audio_clip.duration / video_clip.duration) + 1
                    video_clip = concatenate_videoclips([video_clip] * loops_needed).subclip(0, audio_clip.duration)

            # Set the audio to the video
            final_clip = video_clip.set_audio(audio_clip)

            # Create output filename
            final_output = f"final_{self.output_video}"

            # Write the final output with explicit codec settings
            print(f"Writing final video to {final_output}")
            final_clip.write_videofile(
                final_output,
                codec="libx264",
                audio_codec="aac",
                fps=self.fps,
                preset="medium",
                ffmpeg_params=["-pix_fmt", "yuv420p"]  # Ensure compatible pixel format
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
    During silent gaps in the audio, it shows the original silence.mp4 frames only.
    """
    # Initialize the LipSyncAnimator
    input_video = "silence.mp4"  # The video of the person without sound
    audio_file = "ashish_audio.wav"  # The audio file that will sync to the video
    viseme_folder = "visemes"  # Folder containing subfolders for each viseme
    output_video = "lip_sync_output.mp4"  # Final output video file
    predictor_path = "shape_predictor_68_face_landmarks.dat"  # Path to dlib's face predictor model

    print("Starting lip sync animation process...")
    print(f"Using input video: {input_video}")
    print(f"Using audio file: {audio_file}")
    print(f"Using viseme folder: {viseme_folder}")
    print(f"Using facial landmark predictor: {predictor_path}")
    print(f"Output will be saved to: {output_video}")

    try:
        # Create the animator with dlib facial landmark detector
        animator = LipSyncAnimator(
            input_video,
            audio_file,
            viseme_folder,
            output_video,
            predictor_path
        )

        # Extract the transcript from the audio file
        print("Extracting transcript from audio...")
        transcript_text = animator.extract_audio_transcript()

        if not transcript_text:
            # If speech recognition fails, use a fallback transcript
            print("Speech recognition failed. Using fallback transcript...")
            transcript_text = "thank you for contacting us all lines are currently busy you call is very important to us"
        else:
            print(f"Successfully extracted transcript: '{transcript_text}'")

        # Extract phoneme timings and corresponding visemes
        print("Extracting phoneme timings...")
        phoneme_timing = animator.extract_phoneme_timing(transcript_text)
        print(f"Generated {len(phoneme_timing)} phoneme timing entries")

        # Generate the lip-sync video (with viseme overlays)
        print("Generating lip-sync video with viseme overlays...")
        if animator.generate_video_from_visemes(phoneme_timing):
            print("Lip-sync video generated successfully!")

            # Merge the video with the original audio
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