# Animator.py
import pygame
import numpy as np
import time
from phoneme_analyzer import PhonemeAnalyzer
from viseme_mapper import VisemeMapper


class LipSyncAnimator:
    def __init__(self, width=800, height=600):
        # Initialize pygame
        pygame.init()
        pygame.mixer.init()

        # Setup display
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Custom Lip Sync Animation")

        # Setup clock
        self.clock = pygame.time.Clock()
        self.fps = 60

        # Animation parameters
        self.face_center = (width // 2, height // 2)
        self.face_radius = 120
        self.mouth_base_y = self.face_center[1] + 20
        self.mouth_base_width = 80
        self.transition_time = 0.05  # seconds for blending between visemes

        # Colors
        self.bg_color = (240, 240, 240)
        self.face_color = (255, 220, 200)
        self.outline_color = (0, 0, 0)
        self.mouth_color = (180, 100, 100)
        self.teeth_color = (255, 255, 255)
        self.tongue_color = (255, 150, 150)

    def interpolate_params(self, params1, params2, blend):
        """Interpolate between two sets of viseme parameters"""
        return [p1 * (1 - blend) + p2 * blend for p1, p2 in zip(params1, params2)]

    def get_current_viseme_params(self, viseme_sequence, current_time):
        """Get the current viseme parameters based on time"""
        # Handle empty sequence
        if not viseme_sequence:
            # Return default REST viseme
            return (0.0, 0.0, 0.5)

        # Find the current viseme
        current_viseme_idx = None
        for i, (_, params, start, end) in enumerate(viseme_sequence):
            if start <= current_time < end:
                current_viseme_idx = i
                break

        # If not in any viseme range, find the closest one
        if current_viseme_idx is None:
            if current_time < viseme_sequence[0][2]:  # Before first viseme
                return viseme_sequence[0][1]  # Use first viseme params
            elif current_time >= viseme_sequence[-1][3]:  # After last viseme
                return viseme_sequence[-1][1]  # Use last viseme params
            else:
                # This shouldn't happen with properly structured viseme sequences
                return (0.0, 0.0, 0.5)  # Default REST

        current_viseme, current_params, start, end = viseme_sequence[current_viseme_idx]

        # Check if transitioning to next viseme
        if (current_viseme_idx < len(viseme_sequence) - 1 and
                current_time >= end - self.transition_time):
            # Blend with next viseme
            next_params = viseme_sequence[current_viseme_idx + 1][1]
            blend = (current_time - (end - self.transition_time)) / self.transition_time
            blend = max(0, min(1, blend))  # Clamp to [0,1]
            return self.interpolate_params(current_params, next_params, blend)

        # Check if transitioning from previous viseme
        elif current_time <= start + self.transition_time and current_viseme_idx > 0:
            # Blend with previous viseme
            prev_params = viseme_sequence[current_viseme_idx - 1][1]
            blend = 1 - (current_time - start) / self.transition_time
            blend = max(0, min(1, blend))  # Clamp to [0,1]
            return self.interpolate_params(current_params, prev_params, blend)

        # No transition, use current viseme directly
        return current_params

    def draw_mouth(self, params):
        """Draw the mouth based on viseme parameters"""
        jaw_open, lip_round, lip_width = params

        # Calculate mouth dimensions
        width = self.mouth_base_width * lip_width
        height = max(5, self.mouth_base_width * 0.3 * jaw_open)

        # Adjust for lip rounding
        round_factor = 10 * lip_round

        # Base coordinates
        base_x = self.face_center[0]
        base_y = self.mouth_base_y

        # Draw mouth interior if open
        if jaw_open > 0.05:
            # Draw teeth if mouth is open enough
            if jaw_open > 0.2:
                # Upper teeth
                pygame.draw.rect(self.screen, self.teeth_color,
                                 (base_x - width / 2, base_y - height / 2,
                                  width, height / 4))

                # Lower teeth
                pygame.draw.rect(self.screen, self.teeth_color,
                                 (base_x - width / 2, base_y + height / 4,
                                  width, height / 4))

            # Draw tongue if mouth is open wide
            if jaw_open > 0.5:
                pygame.draw.ellipse(self.screen, self.tongue_color,
                                    (base_x - width / 3, base_y + height / 8,
                                     width * 2 / 3, height / 2))

            # Mouth cavity
            pygame.draw.ellipse(self.screen, (50, 0, 0),
                                (base_x - width / 2, base_y - height / 2,
                                 width, height))

        # Lip points
        if lip_round > 0.5:  # Rounded lips (O, U shapes)
            # For rounded lips, make a more circular shape
            lip_height = height * 0.8 + round_factor
            outer_width = width * (0.7 + 0.3 * lip_round)
            pygame.draw.ellipse(self.screen, self.mouth_color,
                                (base_x - outer_width / 2, base_y - lip_height / 2,
                                 outer_width, lip_height))

            # Inner lip
            inner_width = max(2, width * 0.8 * jaw_open)
            inner_height = max(2, height * 0.8 * jaw_open)
            if jaw_open > 0.05:  # Only draw inner lip if mouth is open
                pygame.draw.ellipse(self.screen, (50, 0, 0),
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
            pygame.draw.polygon(self.screen, self.mouth_color, upper_points)
            pygame.draw.polygon(self.screen, self.outline_color, upper_points, 1)

            # Lower lip
            lower_points = [
                (base_x - width / 2, base_y),  # Left corner
                (base_x - width / 4, base_y + height / 2 + round_factor / 2),  # Left lower
                (base_x, base_y + height / 2 + round_factor),  # Center lower
                (base_x + width / 4, base_y + height / 2 + round_factor / 2),  # Right lower
                (base_x + width / 2, base_y),  # Right corner
            ]
            pygame.draw.polygon(self.screen, self.mouth_color, lower_points)
            pygame.draw.polygon(self.screen, self.outline_color, lower_points, 1)

    def draw_face(self):
        """Draw a simple face"""
        # Face circle
        pygame.draw.circle(self.screen, self.face_color, self.face_center, self.face_radius)
        pygame.draw.circle(self.screen, self.outline_color, self.face_center, self.face_radius, 2)

        # Eyes
        eye_y = self.face_center[1] - 30
        eye_distance = 40

        # Left eye
        left_eye_x = self.face_center[0] - eye_distance
        pygame.draw.circle(self.screen, (255, 255, 255), (left_eye_x, eye_y), 20)
        pygame.draw.circle(self.screen, (50, 100, 200), (left_eye_x, eye_y), 10)
        pygame.draw.circle(self.screen, (0, 0, 0), (left_eye_x, eye_y), 5)
        pygame.draw.circle(self.screen, self.outline_color, (left_eye_x, eye_y), 20, 1)

        # Right eye
        right_eye_x = self.face_center[0] + eye_distance
        pygame.draw.circle(self.screen, (255, 255, 255), (right_eye_x, eye_y), 20)
        pygame.draw.circle(self.screen, (50, 100, 200), (right_eye_x, eye_y), 10)
        pygame.draw.circle(self.screen, (0, 0, 0), (right_eye_x, eye_y), 5)
        pygame.draw.circle(self.screen, self.outline_color, (right_eye_x, eye_y), 20, 1)

        # Nose
        nose_top = (self.face_center[0], eye_y + 30)
        nose_width = 20
        nose_bottom = (self.face_center[0], self.mouth_base_y - 15)
        nose_left = (self.face_center[0] - nose_width / 2, nose_bottom[1] - 5)
        nose_right = (self.face_center[0] + nose_width / 2, nose_bottom[1] - 5)

        # Draw nose
        pygame.draw.line(self.screen, self.outline_color, nose_top, nose_bottom, 2)
        pygame.draw.line(self.screen, self.outline_color, nose_bottom, nose_left, 2)
        pygame.draw.line(self.screen, self.outline_color, nose_bottom, nose_right, 2)

    def animate(self, audio_file, viseme_sequence):
        """Main animation function that plays audio and animates the face"""
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

            # Get current viseme parameters
            params = self.get_current_viseme_params(viseme_sequence, current_time)

            # Draw everything
            self.screen.fill(self.bg_color)
            self.draw_face()
            self.draw_mouth(params)

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

    def run_demo(self, audio_file, text=None):
        """Run a demo with either the provided text or a default sequence"""

        # Check if audio file exists
        import os
        if not os.path.exists(audio_file):
            print(f"Warning: Audio file {audio_file} not found. Using default silent audio.")
            # You could generate a silent audio file here or use a default one
            # For now, we'll still use the provided filename but it will likely fail

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

        # Run the animation
        try:
            self.animate(audio_file, viseme_sequence)
        except Exception as e:
            print(f"Animation error: {e}")


def main():
    """Main function to demonstrate the lip sync system"""
    # Create the animator
    animator = LipSyncAnimator()

    # Example audio file - replace with your own
    audio_file = "sample_audio.1.wav"

    # Example text corresponding to the audio
    example_text = "Hii, This is Ashish"

    # Run the animation
    animator.run_demo(audio_file, example_text)


if __name__ == "__main__":
    main()