import io

import cairosvg
import pygame
import numpy as np
import time
import os
import math
from collections import namedtuple

# Define viseme shapes more precisely
VisemeShape = namedtuple('VisemeShape', ['jaw_open', 'lip_round', 'lip_width', 'tongue_visible', 'teeth_visible'])


def load_svg_asset(filename):
    """
    Convert an SVG file into a pygame Surface.
    Requires the cairosvg package: pip install cairosvg
    """
    # Convert the SVG file to PNG bytes.
    png_data = cairosvg.svg2png(url=filename)
    # Create a bytes stream from the PNG data.
    png_stream = io.BytesIO(png_data)
    # Load the PNG image into a pygame surface.
    return pygame.image.load(png_stream)


class RealisticLipSyncAnimator:
    def __init__(self, width=800, height=600, transition_time=0.08):
        # Initialize pygame
        pygame.init()
        pygame.mixer.init()

        # Setup display
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Realistic Lip Sync Animation")
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
        }
        # Setup clock
        self.clock = pygame.time.Clock()
        self.fps = 60

        # Animation parameters
        self.face_center = (width // 2, height // 2)
        self.transition_time = transition_time  # Faster transitions for realism
        self.bg_color = (240, 240, 240)
        self.load_facial_assets()
        # Blinking parameters
        self.blink_timer = 0
        self.next_blink = np.random.uniform(2.0, 5.0)  # Random blink interval
        self.is_blinking = False
        self.blink_duration = 0.15  # Blink lasts 0.15 seconds

        # Idle animation parameters
        self.idle_offset_x = 0
        self.idle_offset_y = 0
        self.idle_timer = 0

        # Define viseme parameters - now more detailed with tongue and teeth visibility

        # Load images for more realistic rendering
        # self.load_facial_assets()

    def load_facial_assets(self):
        """Load or create assets for facial features."""
        try:
            self.avatar_svg_original = load_svg_asset("animation.svg")
            self.avatar_svg = self.avatar_svg_original.copy()
        except Exception as e:
            print(f"Error loading SVG asset: {e}")
            self.avatar_svg = None
        self.face_base = pygame.Surface((300, 400), pygame.SRCALPHA)
        pygame.draw.ellipse(self.face_base, (255, 224, 196), (0, 0, 300, 400))
        pygame.draw.ellipse(self.face_base, (180, 140, 120), (0, 0, 300, 400), 2)

        # Eyes (open)
        self.eye_open = pygame.Surface((60, 30), pygame.SRCALPHA)
        pygame.draw.ellipse(self.eye_open, (255, 255, 255), (0, 0, 60, 30))  # White
        pygame.draw.ellipse(self.eye_open, (0, 0, 0), (0, 0, 60, 30), 1)  # Outline
        pygame.draw.circle(self.eye_open, (70, 140, 180), (30, 15), 12)  # Iris
        pygame.draw.circle(self.eye_open, (0, 0, 0), (30, 15), 6)  # Pupil
        pygame.draw.circle(self.eye_open, (255, 255, 255), (34, 12), 3)  # Highlight

        # Closed eye for blinking
        self.eye_closed = pygame.Surface((60, 30), pygame.SRCALPHA)
        pygame.draw.line(self.eye_closed, (0, 0, 0), (5, 15), (55, 15), 3)

        # Create different mouth shapes
        self.create_mouth_shapes()

        # Eyebrows
        self.eyebrow = pygame.Surface((70, 20), pygame.SRCALPHA)
        pygame.draw.arc(self.eyebrow, (100, 70, 40), (0, 0, 70, 40), 3.14, 2 * 3.14, 3)

        # Nose
        self.nose = pygame.Surface((40, 60), pygame.SRCALPHA)
        pygame.draw.line(self.nose, (180, 140, 120), (20, 0), (20, 40), 2)  # Bridge
        pygame.draw.arc(self.nose, (180, 140, 120), (10, 30, 20, 20), 0, 3.14, 2)  # Nostril

    def create_mouth_shapes(self):
        """Create surfaces for different mouth shapes based on visemes"""
        self.mouth_shapes = {}

        # For each viseme, create a corresponding mouth surface
        for viseme, shape in self.viseme_map.items():
            mouth_surf = pygame.Surface((160, 80), pygame.SRCALPHA)

            # Base mouth size parameters
            width = 100 * shape.lip_width
            height = max(5, 60 * shape.jaw_open)

            # Adjust for lip rounding
            round_factor = 10 * shape.lip_round

            # Draw mouth cavity if open
            if shape.jaw_open > 0.05:
                pygame.draw.ellipse(mouth_surf, (80, 30, 30),
                                    (80 - width / 2, 40 - height / 2, width, height))
                # Teeth if visible
                if shape.teeth_visible:
                    pygame.draw.rect(mouth_surf, (250, 250, 250),
                                     (80 - width / 2 + 5, 40 - height / 2 + 2, width - 10, 8))
                    pygame.draw.rect(mouth_surf, (250, 250, 250),
                                     (80 - width / 2 + 5, 40 + height / 2 - 10, width - 10, 8))
                # Tongue if visible
                if shape.tongue_visible:
                    pygame.draw.ellipse(mouth_surf, (230, 100, 100),
                                        (80 - width / 3, 40 + 5, width * 2 / 3, height / 2))

            # Draw lips based on rounding
            if shape.lip_round > 0.5:  # Rounded lips
                lip_height = height * 0.8 + round_factor
                outer_width = width * (0.7 + 0.3 * shape.lip_round)
                pygame.draw.ellipse(mouth_surf, (200, 100, 100),
                                    (80 - outer_width / 2, 40 - lip_height / 2, outer_width, lip_height))
                if shape.jaw_open > 0.05:
                    inner_width = max(2, width * 0.8 * shape.jaw_open)
                    inner_height = max(2, height * 0.8 * shape.jaw_open)
                    pygame.draw.ellipse(mouth_surf, (80, 30, 30),
                                        (80 - inner_width / 2, 40 - inner_height / 2, inner_width, inner_height))
            else:  # Normal lips
                upper_points = [
                    (80 - width / 2, 40),
                    (80 - width / 4, 40 - height / 2 - round_factor / 2),
                    (80, 40 - height / 2 - round_factor),
                    (80 + width / 4, 40 - height / 2 - round_factor / 2),
                    (80 + width / 2, 40),
                ]
                pygame.draw.polygon(mouth_surf, (200, 100, 100), upper_points)
                pygame.draw.polygon(mouth_surf, (150, 80, 80), upper_points, 1)
                lower_points = [
                    (80 - width / 2, 40),
                    (80 - width / 4, 40 + height / 2 + round_factor / 2),
                    (80, 40 + height / 2 + round_factor),
                    (80 + width / 4, 40 + height / 2 + round_factor / 2),
                    (80 + width / 2, 40),
                ]
                pygame.draw.polygon(mouth_surf, (200, 100, 100), lower_points)
                pygame.draw.polygon(mouth_surf, (150, 80, 80), lower_points, 1)

            self.mouth_shapes[viseme] = mouth_surf

    def interpolate_viseme_params(self, shape1, shape2, blend):
        """Interpolate between two viseme shapes"""
        return VisemeShape(
            shape1.jaw_open * (1 - blend) + shape2.jaw_open * blend,
            shape1.lip_round * (1 - blend) + shape2.lip_round * blend,
            shape1.lip_width * (1 - blend) + shape2.lip_width * blend,
            shape2.tongue_visible if blend > 0.5 else shape1.tongue_visible,
            shape2.teeth_visible if blend > 0.5 else shape1.teeth_visible
        )

    def get_current_viseme_params(self, viseme_sequence, current_time):
        """Get the current viseme parameters based on time"""
        if not viseme_sequence:
            return self.viseme_map["REST"]

        current_viseme_idx = None
        for i, (viseme_name, start, end) in enumerate(viseme_sequence):
            if start <= current_time < end:
                current_viseme_idx = i
                break

        if current_viseme_idx is None:
            if current_time < viseme_sequence[0][1]:
                return self.viseme_map[viseme_sequence[0][0]]
            elif current_time >= viseme_sequence[-1][2]:
                return self.viseme_map[viseme_sequence[-1][0]]
            else:
                return self.viseme_map["REST"]

        current_viseme, start, end = viseme_sequence[current_viseme_idx]
        current_params = self.viseme_map[current_viseme]

        # Transition to next viseme if near the end of the segment
        if (current_viseme_idx < len(viseme_sequence) - 1 and
                current_time >= end - self.transition_time):
            next_viseme = viseme_sequence[current_viseme_idx + 1][0]
            next_params = self.viseme_map[next_viseme]
            blend = (current_time - (end - self.transition_time)) / self.transition_time
            blend = max(0, min(1, blend))
            return self.interpolate_viseme_params(current_params, next_params, blend)
        elif current_time <= start + self.transition_time and current_viseme_idx > 0:
            prev_viseme = viseme_sequence[current_viseme_idx - 1][0]
            prev_params = self.viseme_map[prev_viseme]
            blend = 1 - (current_time - start) / self.transition_time
            blend = max(0, min(1, blend))
            return self.interpolate_viseme_params(current_params, prev_params, blend)

        return current_params

    def generate_mouth_surface(self, params):
        """Generate a custom mouth surface based on interpolated parameters"""
        mouth_surf = pygame.Surface((160, 80), pygame.SRCALPHA)
        width = 100 * params.lip_width
        height = max(5, 60 * params.jaw_open)
        round_factor = 10 * params.lip_round

        if params.jaw_open > 0.05:
            pygame.draw.ellipse(mouth_surf, (80, 30, 30),
                                (80 - width / 2, 40 - height / 2, width, height))
            if params.teeth_visible:
                pygame.draw.rect(mouth_surf, (250, 250, 250),
                                 (80 - width / 2 + 5, 40 - height / 2 + 2, width - 10, 8))
                pygame.draw.rect(mouth_surf, (250, 250, 250),
                                 (80 - width / 2 + 5, 40 + height / 2 - 10, width - 10, 8))
            if params.tongue_visible:
                pygame.draw.ellipse(mouth_surf, (230, 100, 100),
                                    (80 - width / 3, 40 + 5, width * 2 / 3, height / 2))

        if params.lip_round > 0.5:
            lip_height = height * 0.8 + round_factor
            outer_width = width * (0.7 + 0.3 * params.lip_round)
            pygame.draw.ellipse(mouth_surf, (200, 100, 100),
                                (80 - outer_width / 2, 40 - lip_height / 2, outer_width, lip_height))
            if params.jaw_open > 0.05:
                inner_width = max(2, width * 0.8 * params.jaw_open)
                inner_height = max(2, height * 0.8 * params.jaw_open)
                pygame.draw.ellipse(mouth_surf, (80, 30, 30),
                                    (80 - inner_width / 2, 40 - inner_height / 2, inner_width, inner_height))
        else:
            upper_points = [
                (80 - width / 2, 40),
                (80 - width / 4, 40 - height / 2 - round_factor / 2),
                (80, 40 - height / 2 - round_factor),
                (80 + width / 4, 40 - height / 2 - round_factor / 2),
                (80 + width / 2, 40),
            ]
            pygame.draw.polygon(mouth_surf, (200, 100, 100), upper_points)
            pygame.draw.polygon(mouth_surf, (150, 80, 80), upper_points, 1)
            lower_points = [
                (80 - width / 2, 40),
                (80 - width / 4, 40 + height / 2 + round_factor / 2),
                (80, 40 + height / 2 + round_factor),
                (80 + width / 4, 40 + height / 2 + round_factor / 2),
                (80 + width / 2, 40),
            ]
            pygame.draw.polygon(mouth_surf, (200, 100, 100), lower_points)
            pygame.draw.polygon(mouth_surf, (150, 80, 80), lower_points, 1)

        return mouth_surf

    def update_blink(self, delta_time):
        """Update blinking animation"""
        self.blink_timer += delta_time
        if self.is_blinking:
            if self.blink_timer >= self.blink_duration:
                self.is_blinking = False
                self.blink_timer = 0
                self.next_blink = np.random.uniform(2.0, 5.0)
        else:
            if self.blink_timer >= self.next_blink:
                self.is_blinking = True
                self.blink_timer = 0

    def update_idle_animation(self, delta_time):
        """Update subtle idle animations for realism"""
        self.idle_timer += delta_time
        breath = math.sin(self.idle_timer * 0.5) * 2
        self.idle_offset_x = math.sin(self.idle_timer * 0.3) * 2
        self.idle_offset_y = breath
        return self.idle_offset_x, self.idle_offset_y

    def draw_eye(screen, center_x, center_y, size, is_left):
        """Draws a more realistic eye."""
        # Eye white
        pygame.draw.ellipse(screen, (255, 255, 255), (center_x - size, center_y - size // 2, size * 2, size))
        # Iris gradient
        iris_color = (70, 140, 180)
        for i in range(size // 2, 0, -1):
            color = (iris_color[0] * i // (size // 2),
                     iris_color[1] * i // (size // 2),
                     iris_color[2] * i // (size // 2))
            pygame.draw.circle(screen, color, (center_x, center_y), i)
        # Pupil
        pygame.draw.circle(screen, (0, 0, 0), (center_x, center_y), size // 3)
        # Highlight
        pygame.draw.circle(screen, (255, 255, 255), (center_x + size // 4, center_y - size // 4), size // 6)
        if is_left:
            pygame.draw.ellipse(screen, (0, 0, 0), (center_x - size, center_y - size // 2, size * 2, size), 1)
        else:
            pygame.draw.ellipse(screen, (0, 0, 0), (center_x - size, center_y - size // 2, size * 2, size), 1)

    def draw_blink(screen, center_x, center_y, size):
        """Draws a closed eye for blinking."""
        pygame.draw.line(screen, (0, 0, 0), (center_x - size, center_y), (center_x + size, center_y), 3)

    def draw_face(self, mouth_params, delta_time):
        """
        Draws the face using the loaded SVG asset without re‑drawing its static features.
        Only the animated mouth is redrawn on top of the SVG.
        """
        # Update idle (and optional blinking) animations
        self.update_blink(delta_time)
        idle_x, idle_y = self.update_idle_animation(delta_time)

        # Calculate the current face center with idle adjustments
        face_x = self.face_center[0] + idle_x
        face_y = self.face_center[1] + idle_y

        # Blit the full SVG face asset onto the screen.
        if self.avatar_svg:
            scale_factor = 1.0  # Adjust to scale the SVG if needed.
            svg_width, svg_height = self.avatar_svg.get_size()
            desired_width = int(svg_width * scale_factor)
            desired_height = int(svg_height * scale_factor)
            scaled_svg = pygame.transform.scale(self.avatar_svg, (desired_width, desired_height))
            svg_rect = scaled_svg.get_rect(center=(face_x, face_y))
            self.screen.blit(scaled_svg, svg_rect)

        # Generate the animated mouth surface based on current viseme parameters.
        mouth_surf = self.generate_mouth_surface(mouth_params)

        # Determine the mouth's position relative to the face.
        # (For example, if your SVG was designed with a mouth center around (200, 310) in a 400x500 view,
        # then relative to the SVG's center, the mouth offset might be roughly (0, 60).)
        mouth_offset_x = 0
        mouth_offset_y = 60
        mouth_rect = mouth_surf.get_rect(center=(face_x + mouth_offset_x, face_y + mouth_offset_y))

        # Overlay the animated mouth on top of the SVG face.
        self.screen.blit(mouth_surf, mouth_rect)

    def map_phonemes_to_visemes(self, phoneme_sequence):
        """Map phoneme sequence to viseme sequence"""
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
            viseme_sequence.append((viseme, start, end))
        return viseme_sequence

    def simple_text_to_phoneme(self, text, duration=None):
        """
        A very simple text to phoneme converter for demonstration.
        This naive implementation splits the text into non-space characters
        and assigns each character an equal time slice.
        Letters are converted to uppercase; vowels remain as is, while consonants
        are mapped to a simplified set.
        """
        # Remove unwanted punctuation and trim spaces
        text = text.strip()
        if not text:
            return []

        # Define a simple mapping for consonants
        consonant_mapping = {
            'B': 'M', 'C': 'REST', 'D': 'T', 'F': 'F', 'G': 'T', 'H': 'REST',
            'J': 'SH', 'K': 'T', 'L': 'L', 'M': 'M', 'N': 'T', 'P': 'M',
            'Q': 'T', 'R': 'L', 'S': 'S', 'T': 'T', 'V': 'F', 'W': 'U',
            'X': 'S', 'Y': 'I', 'Z': 'S'
        }
        vowels = ['A', 'E', 'I', 'O', 'U']

        # Remove spaces and use the non-space characters
        phonemes = []
        non_space_chars = [ch for ch in text if not ch.isspace()]
        total_duration = duration if duration is not None else len(non_space_chars) * 0.15
        per_phoneme_duration = total_duration / len(non_space_chars)

        t = 0.0
        for ch in text:
            if ch.isspace():
                continue
            ch = ch.upper()
            if ch in vowels:
                phoneme = ch
            elif ch in consonant_mapping:
                phoneme = consonant_mapping[ch]
            else:
                phoneme = 'REST'
            phonemes.append((phoneme, t, t + per_phoneme_duration))
            t += per_phoneme_duration
        return phonemes

    def animate(self, audio_file, viseme_sequence):
        """Animate the realistic lip sync along with audio playback"""
        try:
            pygame.mixer.music.load(audio_file)
        except Exception as e:
            print(f"Error loading audio {audio_file}: {e}")
            return

        running = True
        paused = False
        start_time = None
        current_time = 0

        font = pygame.font.SysFont('Arial', 18)
        control_text = font.render('Space: Play/Pause, Esc: Quit, R: Restart', True, (0, 0, 0))
        control_rect = control_text.get_rect(topleft=(10, 10))

        while running:
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

            current_viseme_params = self.get_current_viseme_params(viseme_sequence, current_time)
            self.screen.fill(self.bg_color)
            self.draw_face(current_viseme_params, 1.0 / self.fps)

            if start_time is not None:
                time_text = font.render(f'Time: {current_time:.2f}s', True, (0, 0, 0))
                self.screen.blit(time_text, (self.width - 150, 10))
                status = 'Playing' if not paused else 'Paused'
                status_text = font.render(f'Status: {status}', True, (0, 0, 0))
                self.screen.blit(status_text, (self.width - 150, 40))

            self.screen.blit(control_text, control_rect)
            pygame.display.flip()
            self.clock.tick(self.fps)

            if start_time is not None and not pygame.mixer.music.get_busy() and not paused:
                paused = True

        pygame.mixer.music.stop()
        pygame.quit()

    def run_demo(self, audio_file, text=None):
        """
        Run the demo. If text is provided, convert it to phonemes and then map to visemes.
        Otherwise, use a default viseme sequence.
        """
        if not os.path.exists(audio_file):
            print(f"Warning: Audio file {audio_file} not found. Please check the path.")

        if text is None:
            # Default viseme sequence for testing (each tuple: viseme, start, end)
            viseme_sequence = [
                ("REST", 0.0, 0.3),
                ("A", 0.3, 0.6),
                ("O", 0.6, 0.9),
                ("F", 0.9, 1.2),
                ("M", 1.2, 1.5),
                ("L", 1.5, 1.8),
                ("A", 1.8, 2.1),
                ("REST", 2.1, 2.4)
            ]
        else:
            phoneme_sequence = self.simple_text_to_phoneme(text)
            viseme_sequence = self.map_phonemes_to_visemes(phoneme_sequence)

        self.animate(audio_file, viseme_sequence)


def main():
    animator = RealisticLipSyncAnimator()
    # Replace with your actual audio file path
    audio_file = "sample_audio.1.wav"
    example_text = "Hello, this is Ashish"
    animator.run_demo(audio_file, example_text)


if __name__ == "__main__":
    main()
