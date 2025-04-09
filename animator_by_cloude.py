import cv2
import dlib
import pygame
import numpy as np
import time
import os
import math
from collections import namedtuple
import urllib.request
import bz2
from scipy.ndimage import gaussian_filter1d

# Define viseme shapes more precisely
VisemeShape = namedtuple('VisemeShape', ['jaw_open', 'lip_round', 'lip_width', 'tongue_visible', 'teeth_visible'])


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
    A = np.linalg.norm(np.array(mouth_points[14]) - np.array(mouth_points[18]))  # p62-p66
    B = np.linalg.norm(np.array(mouth_points[15]) - np.array(mouth_points[17]))  # p63-p65
    C = np.linalg.norm(np.array(mouth_points[12]) - np.array(mouth_points[16]))  # p60-p64
    mar = (A + B) / (2.0 * C) if C > 0 else 0.2
    return mar


class VideoMouthExtractor:
    def __init__(self):
        # Path to the predictor file
        self.predictor_path = "shape_predictor_68_face_landmarks.dat"
        if not os.path.exists(self.predictor_path):
            download_shape_predictor(self.predictor_path)

        # Initialize dlib's detector and predictor
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(self.predictor_path)

        # Cache for processed frames
        self.processed_frames = {}
        self.silence_frames = {}

        # Store actual video frames
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

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return []

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        mouth_data = []
        video_frames = []
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Store the frame
            video_frames.append(frame.copy())

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
                mar_value = mouth_aspect_ratio(mouth_points)

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

            frame_idx += 1

            # Print progress
            if frame_idx % 100 == 0:
                print(f"Processed {frame_idx}/{frame_count} frames")

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

    def map_mar_to_viseme(self, mar):
        """
        Map the MAR value to a discrete viseme label and a corresponding warp factor.
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
        pygame.mixer.init()

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

    def get_mouth_data_for_frame(self, frame_idx, audio_mar=None):
        """
        Get mouth data for a specific frame index.
        Uses speaking video data when audio is active, silence video data otherwise.
        """
        # Determine if this frame should use silence data
        use_silence = False
        if audio_mar is not None and audio_mar < self.silence_threshold:
            use_silence = True

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
        self.speaking_mouth_data = self.video_extractor.extract_mouth_data_from_video(speaking_video_path, False)

        print("Processing silence video...")
        self.silence_mouth_data = self.video_extractor.extract_mouth_data_from_video(silence_video_path, True)

        print("Video processing complete.")
        return len(self.speaking_mouth_data) > 0 and len(self.silence_mouth_data) > 0

    def animate_with_videos(self, audio_file, speaking_video_path, silence_video_path):
        """
        Animate lip sync using data extracted from videos and synchronized with audio.
        """
        # Process videos first
        if not self.process_videos(speaking_video_path, silence_video_path):
            print("Error: Failed to process videos.")
            return

        # Load audio file
        try:
            pygame.mixer.music.load(audio_file)
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

        # Resize pygame window to match video dimensions
        if width > 0 and height > 0:
            self.width = width
            self.height = height
            self.screen = pygame.display.set_mode((width, height))

        # Analyze audio to get MAR values
        audio_mars = self.analyze_audio(audio_file, fps)
        if audio_mars is None:
            print("Warning: Failed to analyze audio. Using video-based lip sync only.")

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
                        else:
                            pygame.mixer.music.pause()
                            paused = True
                    elif event.key == pygame.K_r:
                        pygame.mixer.music.stop()
                        pygame.mixer.music.play()
                        start_time = time.time()
                        frame_idx = 0
                        paused = False

            if start_time is not None and not paused:
                # Calculate current time and frame index
                current_time = time.time() - start_time
                target_frame = int(current_time * fps)

                # Handle frame skipping if necessary
                if target_frame > frame_idx + 1:
                    skip_frames = target_frame - frame_idx
                    print(f"Audio/video sync: Skipping {skip_frames} frames to catch up")
                    frame_idx = target_frame

                # Get current audio MAR value if available
                current_audio_mar = None
                if audio_mars is not None and frame_idx < len(audio_mars):
                    current_audio_mar = audio_mars[frame_idx]

                # Get mouth data for the current frame
                mouth_data, use_silence = self.get_mouth_data_for_frame(frame_idx, current_audio_mar)
                _, mar_value, mouth_rect, viseme_label, warp_factor = mouth_data

                # Get the appropriate video frame
                video_frame = self.video_extractor.get_frame(
                    frame_idx % len(self.video_extractor.silence_video_frames) if use_silence
                    else frame_idx % len(self.video_extractor.speaking_video_frames),
                    use_silence)

                if video_frame is not None:
                    # Convert OpenCV BGR to RGB for pygame
                    video_frame_rgb = cv2.cvtColor(video_frame, cv2.COLOR_BGR2RGB)

                    # Create pygame surface from numpy array
                    video_surface = pygame.surfarray.make_surface(video_frame_rgb.swapaxes(0, 1))

                    # Scale to fit screen if needed
                    if video_surface.get_width() != self.width or video_surface.get_height() != self.height:
                        video_surface = pygame.transform.scale(video_surface, (self.width, self.height))

                    # Display the frame
                    self.screen.blit(video_surface, (0, 0))

                    # Highlight mouth area if available
                    if mouth_rect:
                        left, top, width, height = mouth_rect
                        # Scale coordinates if video was resized
                        scale_x = self.width / video_frame.shape[1]
                        scale_y = self.height / video_frame.shape[0]
                        rect = (int(left * scale_x), int(top * scale_y),
                                int(width * scale_x), int(height * scale_y))
                        pygame.draw.rect(self.screen, (0, 255, 0), rect, 2)
                else:
                    # If no frame is available, just fill with background color
                    self.screen.fill(self.bg_color)

                # Display current information
                time_text = font.render(f'Time: {current_time:.2f}s | Frame: {frame_idx}', True, (255, 255, 255))
                self.screen.blit(time_text, (self.width - 250, 10))

                viseme_text = font.render(f'Viseme: {viseme_label} | MAR: {mar_value:.2f}', True, (255, 255, 255))
                self.screen.blit(viseme_text, (self.width - 250, 40))

                status = 'Playing' if not paused else 'Paused'
                status_text = font.render(f'Status: {status}', True, (255, 255, 255))
                self.screen.blit(status_text, (self.width - 250, 70))

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

                # Always display control instructions
            self.screen.blit(control_text, control_rect)
            pygame.display.flip()
            self.clock.tick(self.fps)

        pygame.mixer.music.stop()
        print("Animation complete")

    def run_video_based_demo(self, audio_file, speaking_video, silence_video):
        """
        Run a demo using video-based mouth extraction.

        Parameters:
        - audio_file: Path to the audio file to play
        - speaking_video: Path to the video with speaking expressions
        - silence_video: Path to the video with silence/neutral expressions
        """
        if not os.path.exists(audio_file):
            print(f"Warning: Audio file {audio_file} not found. Please check the path.")
            return

        if not os.path.exists(speaking_video):
            print(f"Error: Speaking video file {speaking_video} not found.")
            return

        if not os.path.exists(silence_video):
            print(f"Error: Silence video file {silence_video} not found.")
            return

        self.animate_with_videos(audio_file, speaking_video, silence_video)

def main():
    # Default window size, will be adjusted to match video dimensions
    animator = RealisticLipSyncAnimator(width=800, height=600)

    # Ask user for input files
    audio_file = input("Enter path to audio file: ")
    speaking_video = input("Enter path to speaking video file: ")
    silence_video = input("Enter path to silence video file: ")

    animator.run_video_based_demo(audio_file, speaking_video, silence_video)

if __name__ == "__main__":
    main()