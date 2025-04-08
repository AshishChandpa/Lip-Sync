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
from scipy.signal import find_peaks

# pip install librosa soundfile scipy

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


# Path to the predictor file
predictor_path = "shape_predictor_68_face_landmarks.dat"
if not os.path.exists(predictor_path):
    download_shape_predictor(predictor_path)

# Initialize dlib's detector and predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)


# --------------- Audio Analysis Functions ---------------
def analyze_audio(audio_path, fps):
    """
    Analyze audio file to extract features for lip sync.
    Returns an array of mouth openness values aligned with video frames.
    """
    print(f"Analyzing audio file: {audio_path}")
    try:
        # Load audio with librosa
        y, sr = librosa.load(audio_path, sr=None)

        # Extract amplitude envelope
        hop_length = int(sr / fps)  # Align with video frames
        amplitude_envelope = np.array([max(abs(y[i:i + hop_length])) if i + hop_length < len(y) else 0
                                       for i in range(0, len(y), hop_length)])

        # Normalize to range suitable for mouth aspect ratio (0.2 - 0.6)
        max_amp = np.max(amplitude_envelope) if np.max(amplitude_envelope) > 0 else 1
        normalized_envelope = 0.2 + (amplitude_envelope / max_amp) * 0.4

        # Smooth the curve
        window_size = 3
        smoothed_envelope = np.convolve(normalized_envelope, np.ones(window_size) / window_size, mode='same')

        print(f"Audio analysis complete: {len(smoothed_envelope)} frames processed")
        return smoothed_envelope
    except Exception as e:
        print(f"Error analyzing audio: {e}")
        return None


# --------------- Utility: Compute Mouth Aspect Ratio (MAR) ---------------
def mouth_aspect_ratio(mouth_points):
    """
    Compute the Mouth Aspect Ratio (MAR) using the inner mouth landmarks.
    """
    A = np.linalg.norm(np.array(mouth_points[14]) - np.array(mouth_points[18]))  # p62-p66
    B = np.linalg.norm(np.array(mouth_points[15]) - np.array(mouth_points[17]))  # p63-p65
    C = np.linalg.norm(np.array(mouth_points[12]) - np.array(mouth_points[16]))  # p60-p64
    mar = (A + B) / (2.0 * C) if C > 0 else 0.2  # Default to closed mouth if division by zero
    return mar


# --------------- Utility: Map MAR to Viseme and Warp Factor ---------------
def map_mar_to_viseme(mar):
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


# --------------- Video Processing Function ---------------
def process_frame(frame, audio_mar=None, detection_only=False):
    """
    Process a single frame to detect face and mouth.
    If audio_mar is provided, use it to guide the mouth shape.
    If detection_only is True, only show the detection without modification.
    """
    frame_copy = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    viseme_label = "None"
    warp_factor = 1.0
    mar_value = 0.0
    mouth_rect = None

    # Process first detected face
    if len(faces) > 0:
        face = faces[0]
        shape = predictor(gray, face)
        mouth_points = []
        for i in range(48, 68):  # Mouth landmarks
            x = shape.part(i).x
            y = shape.part(i).y
            mouth_points.append((x, y))
            cv2.circle(frame_copy, (x, y), 2, (0, 0, 255), -1)

        # Calculate actual MAR from the video frame
        video_mar = mouth_aspect_ratio(mouth_points)

        # Decide which MAR to use
        if audio_mar is not None:
            # Use audio-derived MAR, but blend with video MAR for natural movement
            blend_factor = 0.7  # 0.7 audio influence, 0.3 video influence
            mar_value = (audio_mar * blend_factor) + (video_mar * (1 - blend_factor))
        else:
            mar_value = video_mar

        viseme_label, warp_factor = map_mar_to_viseme(mar_value)

        # Extract mouth region coordinates
        xs = [p[0] for p in mouth_points]
        ys = [p[1] for p in mouth_points]
        left, right = min(xs), max(xs)
        top, bottom = min(ys), max(ys)

        # Add padding around mouth for better visual results
        padding = int((bottom - top) * 0.2)
        top = max(0, top - padding)
        bottom = min(frame.shape[0], bottom + padding)

        mouth_rect = (left, top, right - left, bottom - top)

        # If we only want detection without modification, return here
        if detection_only:
            # Draw mouth bounding box
            cv2.rectangle(frame_copy, (left, top), (right, bottom), (0, 255, 0), 2)
        else:
            # Extract and warp mouth region
            mouth_roi = frame[top:bottom, left:right]
            if mouth_roi.size > 0:
                # Calculate new height based on warp factor
                new_height = int((bottom - top) * warp_factor)
                # Warp the mouth vertically
                try:
                    warped_mouth = cv2.resize(mouth_roi, (right - left, new_height))

                    # Center the warped mouth on the original position
                    center_y = top + (bottom - top) // 2
                    new_top = max(0, center_y - new_height // 2)
                    new_bottom = min(frame.shape[0], new_top + new_height)

                    # Ensure we don't exceed frame boundaries
                    if new_bottom > new_top:
                        # Resize again if needed to fit the available space
                        if warped_mouth.shape[0] != new_bottom - new_top:
                            warped_mouth = cv2.resize(warped_mouth, (right - left, new_bottom - new_top))

                        # Copy the warped mouth back to the frame
                        frame_copy[new_top:new_bottom, left:right] = warped_mouth
                except Exception as e:
                    print(f"Error warping mouth: {e}")

    # Overlay info text
    info_text = f"MAR: {mar_value:.2f} | Viseme: {viseme_label} | Warp: {warp_factor:.1f}"
    cv2.putText(frame_copy, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return frame_copy, viseme_label, mar_value


# --------------- Main Function: Video, Audio, & Pygame Integration ---------------
def main():
    # Get file paths from user
    video_path = input("Enter path to video file: ")
    audio_path = input("Enter path to audio file: ")

    # Validate file paths
    if not os.path.exists(video_path):
        print(f"Error: Video file '{video_path}' not found.")
        return

    if not os.path.exists(audio_path):
        print(f"Error: Audio file '{audio_path}' not found.")
        return

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video file!")
        return

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Video: {width}x{height}, {fps} FPS, {frame_count} frames")

    # Ask user if they want to preview face detection before processing
    preview = input("Preview face detection first? (y/n): ").lower() == 'y'
    if preview:
        print("Previewing face detection. Press 'q' to continue to processing.")
        while True:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Loop back to start
                continue

            processed_frame, _, _ = process_frame(frame, detection_only=True)
            cv2.imshow('Face Detection Preview', processed_frame)

            key = cv2.waitKey(int(1000 / fps)) & 0xFF
            if key == ord('q'):
                break

        cv2.destroyAllWindows()
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to beginning

    print("Analyzing audio file for lip sync...")
    # Analyze audio to get MAR values for each frame
    audio_mars = analyze_audio(audio_path, fps)

    if audio_mars is None or len(audio_mars) == 0:
        print("Failed to analyze audio. Using video-based lip sync only.")
        audio_mars = None
    else:
        print(f"Audio analysis complete: {len(audio_mars)} MAR values extracted")

    # Initialize Pygame for audio playback
    pygame.init()
    pygame.mixer.init()
    pygame.display.set_caption("Enhanced Audio-Driven Lip Sync")
    screen = pygame.display.set_mode((width, height))

    # Prepare for processing
    start_time = None
    paused = True  # Start paused so user can prepare
    frame_idx = 0

    print("\nControls:")
    print("  SPACE: Play/Pause")
    print("  ESC: Quit")
    print("\nPress SPACE to start playback...\n")

    # Load audio
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
                        pygame.mixer.music.play()
                        start_time = time.time()
                        paused = False
                        print("Playback started")
                    else:
                        pygame.mixer.music.pause()
                        paused = True
                        print("Playback paused")

        if not paused:
            # Calculate which frame we should be on based on elapsed time
            elapsed_time = time.time() - start_time
            target_frame = int(elapsed_time * fps)

            # If we need to catch up, skip frames
            if target_frame > frame_idx + 1:
                skip_frames = target_frame - frame_idx
                print(f"Audio/video sync: Skipping {skip_frames} frames to catch up")
                frame_idx = target_frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

            # Read the current frame
            ret, frame = cap.read()
            if not ret:
                print("End of video reached")
                pygame.mixer.music.stop()
                break

            # Get corresponding audio MAR value if available
            current_audio_mar = None
            if audio_mars is not None and frame_idx < len(audio_mars):
                current_audio_mar = audio_mars[frame_idx]

            # Process the frame with audio-guided lip sync
            processed_frame, viseme, mar = process_frame(frame, current_audio_mar)

            # Convert OpenCV frame to Pygame surface and display
            processed_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            pygame_frame = pygame.surfarray.make_surface(processed_rgb.swapaxes(0, 1))
            screen.blit(pygame_frame, (0, 0))
            pygame.display.update()

            # Move to next frame
            frame_idx += 1

            # Check if we've reached the end of the video
            if frame_idx >= frame_count:
                print("End of video reached")
                pygame.mixer.music.stop()
                break

        # Cap the frame rate
        pygame.time.wait(10)

    # Clean up
    cap.release()
    pygame.quit()
    cv2.destroyAllWindows()
    print("Processing complete")


if __name__ == "__main__":
    main()