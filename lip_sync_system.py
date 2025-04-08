# image_lip_sync_system.py
import pygame
import numpy as np
import time
import os
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
        pygame.display.set_caption("Image-Based Lip Sync Animation")

        # Setup clock
        self.clock = pygame.time.Clock()
        self.fps = 60

        # Animation parameters
        self.face_center = (width // 2, height // 2)
        self.transition_time = 0.05  # seconds for blending between visemes

        # Colors
        self.bg_color = (240, 240, 240)

        # Viseme image dictionaries
        self.mouth_images = {}
        self.face_image = None

    def load_face_image(self, image_path):
        """Load the base face image"""
        try:
            self.face_image = pygame.image.load(image_path).convert_alpha()
            # Scale if needed
            # self.face_image = pygame.transform.scale(self.face_image, (width, height))
        except pygame.error as e:
            print(f"Error loading face image: {e}")
            # Create a placeholder face image
            self.face_image = pygame.Surface((300, 300), pygame.SRCALPHA)
            pygame.draw.ellipse(self.face_image, (255, 220, 200), (0, 0, 300, 300))
            pygame.draw.ellipse(self.face_image, (0, 0, 0), (0, 0, 300, 300), 2)

    def load_mouth_images(self, viseme_image_dir):
        """Load mouth images for each viseme from a directory"""
        try:
            # Standard viseme names to look for
            viseme_names = [
                "REST", "A", "E", "I", "O", "U",
                "F", "V", "TH",
                "S", "SH",
                "P", "B", "M",
                "K", "G", "N", "L", "R"
            ]

            # Load each viseme image
            for viseme in viseme_names:
                image_path = os.path.join(viseme_image_dir, f"{viseme.lower()}.png")
                if os.path.exists(image_path):
                    self.mouth_images[viseme] = pygame.image.load(image_path).convert_alpha()
                else:
                    print(f"Warning: No image found for viseme {viseme}")

            # If no viseme images were found, create placeholder images
            if not self.mouth_images:
                print("No viseme images found. Creating placeholders.")
                self.create_placeholder_visemes()

        except Exception as e:
            print(f"Error loading mouth images: {e}")
            # Create placeholder viseme images
            self.create_placeholder_visemes()

    def create_placeholder_visemes(self):
        """Create placeholder images for visemes if real images are not available"""
        # Standard viseme dictionary with parameters (jaw_open, lip_round, lip_width)
        viseme_params = {
            "REST": (0.0, 0.0, 0.5),
            "A": (0.7, 0.0, 0.7),
            "E": (0.5, 0.2, 0.7),
            "I": (0.3, 0.2, 0.8),
            "O": (0.5, 0.8, 0.6),
            "U": (0.3, 0.8, 0.4),
            "F": (0.1, 0.0, 0.8),
            "V": (0.1, 0.0, 0.8),
            "TH": (0.2, 0.0, 0.7),
            "S": (0.2, 0.0, 0.8),
            "SH": (0.2, 0.0, 0.7),
            "P": (0.0, 0.0, 0.5),
            "B": (0.1, 0.0, 0.5),
            "M": (0.0, 0.0, 0.5),
            "K": (0.4, 0.0, 0.7),
            "G": (0.4, 0.0, 0.7),
            "N": (0.2, 0.0, 0.5),
            "L": (0.3, 0.0, 0.6),
            "R": (0.3, 0.3, 0.6)
        }

        # Create simple shape images for each viseme
        for viseme, params in viseme_params.items():
            jaw_open, lip_round, lip_width = params
            image = pygame.Surface((100, 80), pygame.SRCALPHA)

            # Base coordinates
            base_x = 50
            base_y = 40

            # Calculate mouth dimensions
            width = 60 * lip_width
            height = max(5, 60 * 0.3 * jaw_open)
            round_factor = 10 * lip_round

            # Draw placeholder mouth based on parameters
            if lip_round > 0.5:  # Rounded lips (O, U shapes)
                # For rounded lips, make a more circular shape
                lip_height = height * 0.8 + round_factor
                outer_width = width * (0.7 + 0.3 * lip_round)
                pygame.draw.ellipse(image, (180, 100, 100),
                                    (base_x - outer_width / 2, base_y - lip_height / 2,
                                     outer_width, lip_height))

                # Inner lip if the mouth is open
                if jaw_open > 0.05:
                    inner_width = max(2, width * 0.8 * jaw_open)
                    inner_height = max(2, height * 0.8 * jaw_open)
                    pygame.draw.ellipse(image, (50, 0, 0),
                                        (base_x - inner_width / 2, base_y - inner_height / 2,
                                         inner_width, inner_height))
            else:  # Regular lips
                # Upper lip
                upper_points = [
                    (base_x - width / 2, base_y),  # Left corner
                    (base_x - width / 4, base_y - height / 2 - round_factor / 2),  # Left upper
                    (base_x, base_y - height / 2 - round_factor),  # Center upper
                    (base_x + width / 4, base_y - height / 2 - round_factor / 2),  # Right upper
                    (base_x + width / 2, base_y),  # Right corner
                ]
                pygame.draw.polygon(image, (180, 100, 100), upper_points)
                pygame.draw.polygon(image, (0, 0, 0), upper_points, 1)

                # Lower lip
                lower_points = [
                    (base_x - width / 2, base_y),  # Left corner
                    (base_x - width / 4, base_y + height / 2 + round_factor / 2),  # Left lower
                    (base_x, base_y + height / 2 + round_factor),  # Center lower
                    (base_x + width / 4, base_y + height / 2 + round_factor / 2),  # Right lower
                    (base_x + width / 2, base_y),  # Right corner
                ]
                pygame.draw.polygon(image, (180, 100, 100), lower_points)
                pygame.draw.polygon(image, (0, 0, 0), lower_points, 1)

                # Draw mouth interior if open
                if jaw_open > 0.05:
                    pygame.draw.ellipse(image, (50, 0, 0),
                                        (base_x - width / 2, base_y - height / 2,
                                         width, height))

            # Add to the mouth images dictionary
            self.mouth_images[viseme] = image

    def interpolate_images(self, image1, image2, blend):
        """Blend two mouth images together based on blend factor"""
        if image1 is None or image2 is None:
            return image1 if image2 is None else image2

        # Create a new surface for the blended image
        blended = pygame.Surface(image1.get_size(), pygame.SRCALPHA)

        # Simple alpha blending
        image1_alpha = pygame.surfarray.pixels_alpha(image1.copy()) * (1 - blend)
        image2_alpha = pygame.surfarray.pixels_alpha(image2.copy()) * blend

        # Draw the images with their respective alpha values
        blended.blit(image1, (0, 0))
        blended.set_alpha(int(255 * (1 - blend)))

        temp = pygame.Surface(image2.get_size(), pygame.SRCALPHA)
        temp.blit(image2, (0, 0))
        temp.set_alpha(int(255 * blend))

        blended.blit(temp, (0, 0))

        return blended

    def get_current_mouth_image(self, viseme_sequence, current_time):
        """Get the current mouth image based on time"""
        # Handle empty sequence
        if not viseme_sequence:
            return self.mouth_images.get("REST", None)

        # Find the current viseme
        current_viseme_idx = None
        for i, (viseme_name, _, start, end) in enumerate(viseme_sequence):
            if start <= current_time < end:
                current_viseme_idx = i
                break

        # If not in any viseme range, find the closest one
        if current_viseme_idx is None:
            if current_time < viseme_sequence[0][2]:  # Before first viseme
                return self.mouth_images.get(viseme_sequence[0][0], None)
            elif current_time >= viseme_sequence[-1][3]:  # After last viseme
                return self.mouth_images.get(viseme_sequence[-1][0], None)
            else:
                # This shouldn't happen with properly structured viseme sequences
                return self.mouth_images.get("REST", None)

        current_viseme, _, start, end = viseme_sequence[current_viseme_idx]
        current_image = self.mouth_images.get(current_viseme, None)

        # Check if transitioning to next viseme
        if (current_viseme_idx < len(viseme_sequence) - 1 and
                current_time >= end - self.transition_time):
            # Blend with next viseme
            next_viseme = viseme_sequence[current_viseme_idx + 1][0]
            next_image = self.mouth_images.get(next_viseme, None)
            blend = (current_time - (end - self.transition_time)) / self.transition_time
            blend = max(0, min(1, blend))  # Clamp to [0,1]

            if current_image is not None and next_image is not None:
                return self.interpolate_images(current_image, next_image, blend)

        # Check if transitioning from previous viseme
        elif current_time <= start + self.transition_time and current_viseme_idx > 0:
            # Blend with previous viseme
            prev_viseme = viseme_sequence[current_viseme_idx - 1][0]
            prev_image = self.mouth_images.get(prev_viseme, None)
            blend = 1 - (current_time - start) / self.transition_time
            blend = max(0, min(1, blend))  # Clamp to [0,1]

            if current_image is not None and prev_image is not None:
                return self.interpolate_images(current_image, prev_image, blend)

        # No transition, use current viseme directly
        return current_image

    def animate(self, audio_file, viseme_sequence, mouth_position=None):
        """Main animation function that plays audio and animates the face"""
        # Default mouth position if not specified (center of screen)
        if mouth_position is None:
            mouth_position = (self.width // 2 - 50, self.height // 2)

        # Load audio
        pygame.mixer.music.load(audio_file)

        # Prepare animation
        running = True
        paused = False
        start_time = None
        current_time = 0

        # Display controls
        font = pygame.font.SysFont('Arial', 18)
        control_text = font.render('Space: Play/Pause, Esc: Quit, R: Restart', True, (0, 0, 0))
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
                            pygame.mixer.music.unpause()
                            paused = False
                        else:
                            if start_time is None:
                                # First play
                                pygame.mixer.music.play()
                                start_time = time.time()
                            else:
                                pygame.mixer.music.pause()
                                paused = True
                    elif event.key == pygame.K_r:
                        # Restart animation
                        pygame.mixer.music.stop()
                        pygame.mixer.music.play()
                        start_time = time.time()
                        paused = False

            # Update current time if playing
            if start_time is not None and not paused:
                current_time = time.time() - start_time

            # Get current mouth image
            mouth_image = self.get_current_mouth_image(viseme_sequence, current_time)

            # Draw everything
            self.screen.fill(self.bg_color)

            # Draw the face image if available
            if self.face_image:
                face_rect = self.face_image.get_rect(center=self.face_center)
                self.screen.blit(self.face_image, face_rect)

            # Draw the mouth image if available
            if mouth_image:
                self.screen.blit(mouth_image, mouth_position)

            # Draw playback info
            if start_time is not None:
                time_text = font.render(f'Time: {current_time:.2f}s', True, (0, 0, 0))
                time_rect = time_text.get_rect(topright=(self.width - 10, 10))
                self.screen.blit(time_text, time_rect)

                status = 'Playing' if not paused else 'Paused'
                status_text = font.render(f'Status: {status}', True, (0, 0, 0))
                status_rect = status_text.get_rect(topright=(self.width - 10, 40))
                self.screen.blit(status_text, status_rect)

            # Display controls
            self.screen.blit(control_text, control_rect)

            # Update display
            pygame.display.flip()
            self.clock.tick(self.fps)

            # Check if music finished playing
            if start_time is not None and not pygame.mixer.music.get_busy() and not paused:
                # Auto-restart option (comment out if not desired)
                # pygame.mixer.music.play()
                # start_time = time.time()
                paused = True

        # Clean up
        pygame.mixer.music.stop()
        pygame.quit()

    def run_demo(self, audio_file, face_image_path=None, viseme_image_dir=None, text=None):
        """Run a demo with either the provided text or a default sequence"""

        # Check if audio file exists
        import os
        if not os.path.exists(audio_file):
            print(f"Warning: Audio file {audio_file} not found. Using default silent audio.")
            # You could generate a silent audio file here or use a default one

        # Load face image if provided
        if face_image_path and os.path.exists(face_image_path):
            self.load_face_image(face_image_path)

        # Load mouth images if directory provided
        if viseme_image_dir and os.path.exists(viseme_image_dir):
            self.load_mouth_images(viseme_image_dir)
        else:
            print("No viseme image directory provided or not found. Using placeholder visemes.")
            self.create_placeholder_visemes()

        # Create a sample viseme sequence if text is not provided
        if text is None:
            # Default viseme sequence for testing
            # Format: (viseme_name, (jaw_open, lip_round, lip_width), start_time, end_time)
            viseme_sequence = [
                ("REST", (0.0, 0.0, 0.5), 0.0, 0.2),
                ("A", (0.7, 0.0, 0.7), 0.2, 0.5),
                ("O", (0.5, 0.8, 0.6), 0.5, 0.8),
                ("F", (0.1, 0.0, 0.8), 0.8, 1.0),
                ("P", (0.0, 0.0, 0.5), 1.0, 1.2),
                ("L", (0.3, 0.0, 0.6), 1.2, 1.5),
                ("A", (0.7, 0.0, 0.7), 1.5, 1.8),
                ("REST", (0.0, 0.0, 0.5), 1.8, 2.0)
            ]
        else:
            try:
                # Use PhonemeAnalyzer and VisemeMapper to create viseme sequence from text
                analyzer = PhonemeAnalyzer()
                mapper = VisemeMapper()

                # Get phoneme timings from the analyzer
                phoneme_timings = analyzer.analyze_text(text, estimate_duration=True)

                # Map phonemes to visemes
                viseme_sequence = mapper.map_to_viseme_sequence(phoneme_timings)
            except Exception as e:
                print(f"Error creating viseme sequence: {e}")
                print("Using default viseme sequence instead.")
                # Fallback to a simple sequence
                viseme_sequence = [
                    ("REST", (0.0, 0.0, 0.5), 0.0, 0.5),
                    ("A", (0.7, 0.0, 0.7), 0.5, 1.0),
                    ("REST", (0.0, 0.0, 0.5), 1.0, 1.5)
                ]

        # Calculate mouth position (can be customized)
        mouth_position = (
            self.width // 2 - 50,  # X position
            self.height // 2 + 20  # Y position
        )

        # Run the animation
        try:
            self.animate(audio_file, viseme_sequence, mouth_position)
        except Exception as e:
            print(f"Animation error: {e}")


def main():
    """Main function to demonstrate the image-based lip sync system"""
    # Create the animator
    animator = ImageLipSyncAnimator()

    # Example audio file - replace with your own
    audio_file = "sample_audio.wav"

    # Example text corresponding to the audio
    example_text = "Hello, this is an image-based lip sync demonstration."

    # Optional: Paths for face and viseme images
    face_image_path = "assets/face.png"
    viseme_image_dir = "assets/visemes/"

    # Run the animation
    animator.run_demo(audio_file, face_image_path, viseme_image_dir, example_text)


if __name__ == "__main__":
    main()