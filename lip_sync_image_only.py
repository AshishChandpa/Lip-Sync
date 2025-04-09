
import pygame
import numpy as np
import time
import os
import cv2  # OpenCV for image processing
import urllib.request
import bz2

# Make sure dlib is installed:
import dlib

# Placeholder imports (if you have these modules)
from phoneme_analyzer import PhonemeAnalyzer
from viseme_mapper import VisemeMapper


class ImageLipSyncAnimator:
    def __init__(self, width=800, height=600):
        # Initialize pygame
        pygame.init()
        pygame.mixer.init()

        # Setup display
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Interactive Avatar Lip Sync Animation")

        # Setup clock
        self.clock = pygame.time.Clock()
        self.fps = 60

        # Image and face parameters
        self.face_image = None
        self.original_face_image = None  # To preserve the unmodified face
        self.face_rect = None
        self.mouth_rect = None  # Detected mouth region (bounding box)
        self.mouth_points = None  # Facial landmark points for the mouth
        self.image_scale = 1.0

        # Animation parameters
        self.transition_time = 0.05  # seconds for blending between visemes

        # Background color
        self.bg_color = (240, 240, 240)

        # Viseme warp parameters: a simple mapping from viseme to vertical scaling factor.
        # (Values >1.0 will open the mouth wider; adjust as needed.)
        self.viseme_warp = {
            "REST": 1.0,
            "A": 1.5,
            "E": 1.3,
            "I": 1.2,
            "O": 1.6,
            "U": 1.4,
            "F": 1.1,
            "L": 1.1,
            "M": 1.0,
            "S": 1.1,
            "T": 1.0
        }

    def download_shape_predictor(self, predictor_path):
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

    def load_face_image(self, image_path):
        """Load the face image, scale it, and detect the mouth region."""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Face image not found: {image_path}")

        # Load the image as a pygame surface with alpha channel
        self.face_image = pygame.image.load(image_path).convert_alpha()

        # Scale image to fit the screen (80% of available width/height)
        img_width, img_height = self.face_image.get_size()
        scale_x = (self.width * 0.8) / img_width
        scale_y = (self.height * 0.8) / img_height
        self.image_scale = min(scale_x, scale_y)

        if self.image_scale < 1:
            new_width = int(img_width * self.image_scale)
            new_height = int(img_height * self.image_scale)
            self.face_image = pygame.transform.scale(self.face_image, (new_width, new_height))

        # Set face rectangle and center it on screen
        self.face_rect = self.face_image.get_rect()
        self.face_rect.center = (self.width // 2, self.height // 2)

        # Keep an original copy for later lip warping
        self.original_face_image = self.face_image.copy()

        # Automatically detect the mouth region using facial landmarks
        self.detect_mouth_region()

        return True

    def detect_mouth_region(self):
        """
        Detect facial landmarks on the face image and extract the mouth region.
        Automatically downloads the predictor if needed.
        """
        predictor_path = "shape_predictor_68_face_landmarks.dat"
        if not os.path.exists(predictor_path):
            print("shape_predictor_68_face_landmarks.dat not found!")
            self.download_shape_predictor(predictor_path)
            if not os.path.exists(predictor_path):
                print("Failed to obtain the shape predictor. Cannot detect mouth region.")
                return

        # Convert pygame surface to a NumPy array (RGB)
        face_array = pygame.surfarray.array3d(self.face_image)
        # Transpose from (width, height, channels) to (height, width, channels)
        face_array = np.transpose(face_array, (1, 0, 2))

        # Convert to grayscale for detection
        gray = cv2.cvtColor(face_array, cv2.COLOR_RGB2GRAY)

        # Initialize dlib's face detector and shape predictor
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(predictor_path)

        # Detect faces in the image
        faces = detector(gray)
        if len(faces) == 0:
            print("No face detected!")
            return

        # Use the first detected face
        face_rect_dlib = faces[0]
        shape = predictor(gray, face_rect_dlib)

        # Extract mouth landmarks (indices 48 to 67 from the 68-point model)
        mouth_points = []
        for i in range(48, 68):
            x = shape.part(i).x
            y = shape.part(i).y
            mouth_points.append((x, y))

        # Compute the bounding rectangle around the mouth landmarks
        xs = [p[0] for p in mouth_points]
        ys = [p[1] for p in mouth_points]
        left = min(xs)
        right = max(xs)
        top = min(ys)
        bottom = max(ys)

        self.mouth_rect = pygame.Rect(left, top, right - left, bottom - top)
        self.mouth_points = mouth_points
        print("Mouth region detected:", self.mouth_rect)

    def get_current_viseme(self, viseme_sequence, current_time):
        """
        Determine the current viseme from the sequence.
        If near a transition boundary, blend the warp factor between the two visemes.
        Returns either a viseme string or a tuple (viseme_name, custom_warp_factor).
        """
        if not viseme_sequence:
            return "REST"

        current_viseme_idx = None
        for i, (viseme, _, start, end) in enumerate(viseme_sequence):
            if start <= current_time < end:
                current_viseme_idx = i
                break

        if current_viseme_idx is None:
            if current_time < viseme_sequence[0][2]:
                return viseme_sequence[0][0]
            elif current_time >= viseme_sequence[-1][3]:
                return viseme_sequence[-1][0]
            else:
                return "REST"

        current_viseme, _, start, end = viseme_sequence[current_viseme_idx]

        # Check if transitioning to the next viseme
        if (current_viseme_idx < len(viseme_sequence) - 1 and
                current_time >= end - self.transition_time):
            next_viseme = viseme_sequence[current_viseme_idx + 1][0]
            blend = (current_time - (end - self.transition_time)) / self.transition_time
            warp1 = self.viseme_warp.get(current_viseme, 1.0)
            warp2 = self.viseme_warp.get(next_viseme, 1.0)
            blended_warp = warp1 * (1 - blend) + warp2 * blend
            return (current_viseme, blended_warp)

        # Check if transitioning from the previous viseme
        elif current_viseme_idx > 0 and current_time <= start + self.transition_time:
            prev_viseme = viseme_sequence[current_viseme_idx - 1][0]
            blend = 1 - (current_time - start) / self.transition_time
            warp1 = self.viseme_warp.get(current_viseme, 1.0)
            warp2 = self.viseme_warp.get(prev_viseme, 1.0)
            blended_warp = warp1 * (1 - blend) + warp2 * blend
            return (current_viseme, blended_warp)

        # No transition – return the current viseme name
        return current_viseme

    def get_warped_mouth(self, current_viseme):
        """
        Extract the mouth region from the original face image and apply a vertical scaling
        transformation based on the current viseme (or its blended warp factor).
        Returns the warped mouth surface and its rectangle.
        """
        if isinstance(current_viseme, tuple):
            viseme_name, custom_warp = current_viseme
            warp_factor = custom_warp
        else:
            viseme_name = current_viseme
            warp_factor = self.viseme_warp.get(viseme_name, 1.0)

        # If mouth detection failed, return None
        if self.mouth_rect is None:
            return None, None

        # Extract the mouth region from the original face image
        mouth_surface = self.original_face_image.subsurface(self.mouth_rect).copy()

        # Calculate the new size (keep width the same; scale height)
        new_width = self.mouth_rect.width
        new_height = max(1, int(self.mouth_rect.height * warp_factor))
        warped_mouth = pygame.transform.smoothscale(mouth_surface, (new_width, new_height))

        # Re-center the warped mouth on the original mouth region's center
        new_rect = warped_mouth.get_rect(center=self.mouth_rect.center)
        return warped_mouth, new_rect

    def animate(self, audio_file, viseme_sequence):
        """Main animation function: plays audio and animates the face with lip warping."""
        # Load audio
        pygame.mixer.music.load(audio_file)

        running = True
        paused = False
        start_time = None
        current_time = 0

        # Display controls
        font = pygame.font.SysFont('Arial', 18)
        control_text = font.render('Space: Play/Pause, Esc: Quit, R: Restart', True, (0, 0, 0))
        control_rect = control_text.get_rect(topleft=(10, 10))

        while running:
            # Event handling
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        if paused:
                            pygame.mixer.music.unpause()
                            paused = False
                        else:
                            if start_time is None:
                                pygame.mixer.music.play()
                                start_time = time.time()
                            else:
                                pygame.mixer.music.pause()
                                paused = True
                    elif event.key == pygame.K_r:
                        pygame.mixer.music.stop()
                        pygame.mixer.music.play()
                        start_time = time.time()
                        paused = False

            if start_time is not None and not paused:
                current_time = time.time() - start_time

            # Determine the current viseme based on audio time
            current_viseme = self.get_current_viseme(viseme_sequence, current_time)

            # Create a fresh copy of the original face image
            display_face = self.original_face_image.copy()

            # Get the warped mouth (i.e. the deformed lip region) and composite it
            warped_mouth, mouth_rect = self.get_warped_mouth(current_viseme)
            if warped_mouth and mouth_rect:
                display_face.blit(warped_mouth, mouth_rect)

            # Draw to the screen
            self.screen.fill(self.bg_color)
            self.screen.blit(display_face, self.face_rect)

            # Draw playback info
            if start_time is not None:
                time_text = font.render(f'Time: {current_time:.2f}s', True, (0, 0, 0))
                time_rect = time_text.get_rect(topright=(self.width - 10, 10))
                self.screen.blit(time_text, time_rect)

                status = 'Playing' if not paused else 'Paused'
                status_text = font.render(f'Status: {status}', True, (0, 0, 0))
                status_rect = status_text.get_rect(topright=(self.width - 10, 40))
                self.screen.blit(status_text, status_rect)

            # Display controls info
            self.screen.blit(control_text, control_rect)

            pygame.display.flip()
            self.clock.tick(self.fps)

            # If audio finished playing, pause the animation
            if start_time is not None and not pygame.mixer.music.get_busy() and not paused:
                paused = True

        pygame.mixer.music.stop()
        pygame.quit()

    def run_demo(self, face_image_path, audio_file, text=None):
        """Run a demo using the given face image and audio. Optionally generate viseme sequence from text."""
        # Load face image and detect mouth region
        try:
            self.load_face_image(face_image_path)
        except Exception as e:
            print(f"Error loading face image: {e}")
            return

        # Check if the audio file exists
        if not os.path.exists(audio_file):
            print(f"Warning: Audio file {audio_file} not found.")

        # Create a viseme sequence – either default or using phoneme/viseme analysis
        if text is None:
            viseme_sequence = [
                ("REST", (0.0, 0.0, 0.5), 0.0, 0.2),
                ("A", (0.7, 0.0, 0.7), 0.2, 0.5),
                ("O", (0.5, 0.8, 0.6), 0.5, 0.8),
                ("F", (0.1, 0.0, 0.8), 0.8, 1.0),
                ("REST", (0.0, 0.0, 0.5), 1.0, 1.2),
                ("L", (0.3, 0.0, 0.6), 1.2, 1.5),
                ("A", (0.7, 0.0, 0.7), 1.5, 1.8),
                ("REST", (0.0, 0.0, 0.5), 1.8, 2.0)
            ]
        else:
            try:
                analyzer = PhonemeAnalyzer()
                mapper = VisemeMapper()
                phoneme_timings = analyzer.analyze_text(text, estimate_duration=True)
                viseme_sequence = mapper.map_to_viseme_sequence(phoneme_timings)
            except Exception as e:
                print(f"Error creating viseme sequence: {e}")
                print("Using default viseme sequence instead.")
                viseme_sequence = [
                    ("REST", (0.0, 0.0, 0.5), 0.0, 0.5),
                    ("A", (0.7, 0.0, 0.7), 0.5, 1.0),
                    ("REST", (0.0, 0.0, 0.5), 1.0, 1.5)
                ]

        try:
            self.animate(audio_file, viseme_sequence)
        except Exception as e:
            print(f"Animation error: {e}")


def main():
    """Main function to demonstrate the interactive avatar lip sync system."""
    animator = ImageLipSyncAnimator()

    # Update these paths with your files
    face_image_path = "pexels-simon-robben-55958-614810.jpg"
    audio_file = "sample_audio.wav"
    example_text = "Hi, this is an example of an interactive avatar with lip sync."

    animator.run_demo(face_image_path, audio_file, example_text)


if __name__ == "__main__":
    main()

