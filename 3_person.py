import cv2
import pygame
import numpy as np
import time
import os
import librosa
from scipy.ndimage import gaussian_filter1d


# ---------------------------
# Audio Analysis Function: RMS-based Energy
# ---------------------------
def analyze_audio(audio_path, fps):
    """
    Analyze an audio file using RMS energy.
    This function loads the audio with librosa, computes the RMS per frame (with a
    hop length matching the video frame rate), and normalizes the values to range from
    0.2 (silence) to 0.6 (loud). A Gaussian filter is applied for smoothing.
    Returns a numpy array of length equal to the number of frames (or more) for use
    in synchronizing the video.
    """
    print(f"Analyzing audio file: {audio_path}")
    try:
        y, sr = librosa.load(audio_path, sr=None)
        hop_length = int(sr / fps)  # one audio frame per video frame
        rms = librosa.feature.rms(y=y, frame_length=hop_length, hop_length=hop_length)[0]

        # Avoid divide-by-zero; normalize RMS to [0.2, 0.6]
        max_rms = np.max(rms) if np.max(rms) > 0 else 1
        normalized_rms = 0.2 + (rms / max_rms) * 0.4
        smoothed_rms = gaussian_filter1d(normalized_rms, sigma=1)

        print(f"Audio analysis complete: {len(smoothed_rms)} frames processed")
        return smoothed_rms
    except Exception as e:
        print(f"Error analyzing audio: {e}")
        return None


# ---------------------------
# Main Function: Switching Between Speaking and Idle Videos
# ---------------------------
def main():
    # User inputs: paths for the two videos and audio file.
    # speaking_video_path = input("Enter path to the SPEAKING video file: ").strip()
    # idle_video_path = input("Enter path to the IDLE video file: ").strip()
    # audio_path = input("Enter path to the AUDIO file: ").strip()
    speaking_video_path = "sample_vid.mp4"
    idle_video_path = "sample_silence.mp4"
    audio_path = "sample_audio.wav"

    # Check that files exist.
    for file_path, label in zip([speaking_video_path, idle_video_path, audio_path],
                                ["Speaking video", "Idle video", "Audio file"]):
        if not os.path.exists(file_path):
            print(f"Error: {label} '{file_path}' not found.")
            return

    # Open both video files using OpenCV.
    speaking_cap = cv2.VideoCapture(speaking_video_path)
    idle_cap = cv2.VideoCapture(idle_video_path)
    if not speaking_cap.isOpened() or not idle_cap.isOpened():
        print("Error: Unable to open one of the video files.")
        return

    # Use properties from the speaking video as reference.
    fps = speaking_cap.get(cv2.CAP_PROP_FPS)
    width = int(speaking_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(speaking_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(speaking_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video properties: {width}x{height} at {fps:.2f} FPS; {frame_count} frames available")

    # Analyze audio; if analysis fails, default to silent (energy value 0.2).
    audio_values = analyze_audio(audio_path, fps)
    if audio_values is None or len(audio_values) == 0:
        print("Audio analysis failed; defaulting to silence for all frames.")
        audio_values = np.full((frame_count,), 0.2)

    # Silence threshold: values below indicate silence.
    silence_threshold = 0.21

    # ---------------------------
    # Pygame Setup for Display and Audio Playback
    # ---------------------------
    pygame.init()
    pygame.mixer.init()
    pygame.display.set_caption("Avatar Video Switching Based on Audio Activity")
    screen = pygame.display.set_mode((width, height))

    # Load the audio for playback (this will serve as the master clock).
    try:
        pygame.mixer.music.load(audio_path)
    except Exception as e:
        print(f"Error loading audio in Pygame: {e}")
        return

    print("\nControls:")
    print("  SPACE: Play/Pause")
    print("  ESC: Quit")
    print("Press SPACE to start playback...\n")

    running = True
    paused = True
    start_time = None
    frame_idx = 0

    while running:
        # Process Pygame events.
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
                        print("Playback started")
                    else:
                        pygame.mixer.music.pause()
                        paused = True
                        print("Playback paused")

        if not paused:
            # Calculate elapsed time and corresponding frame index.
            elapsed_time = time.time() - start_time
            target_frame = int(elapsed_time * fps)

            # If we've fallen behind, update frame index (and optionally log skipped frames).
            if target_frame > frame_idx:
                skip_frames = target_frame - frame_idx
                if skip_frames > 1:
                    print(f"Audio/video sync: Skipping {skip_frames} frames to catch up")
                frame_idx = target_frame

            # Retrieve the current audio energy for this frame.
            if frame_idx < len(audio_values):
                current_energy = audio_values[frame_idx]
            else:
                current_energy = 0.2  # default to silence if past audio analysis length

            # Decide which video to use.
            if current_energy >= silence_threshold:
                # Use speaking video.
                ret, frame = speaking_cap.read()
                video_mode = "SPEAKING"
                if not ret:
                    # If at end, loop the speaking video.
                    speaking_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = speaking_cap.read()
            else:
                # Use idle video.
                ret, frame = idle_cap.read()
                video_mode = "IDLE"
                if not ret:
                    idle_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = idle_cap.read()

            if frame is None:
                print("No frame retrieved; ending playback.")
                running = False
                break

            # Optional: Overlay text with current state.
            overlay_text = f"{video_mode} | Energy: {current_energy:.2f} | Frame: {frame_idx}"
            cv2.putText(frame, overlay_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (255, 255, 255), 2, cv2.LINE_AA)

            # Convert the frame from BGR to RGB for Pygame.
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pygame_frame = pygame.surfarray.make_surface(frame_rgb.swapaxes(0, 1))
            screen.blit(pygame_frame, (0, 0))
            pygame.display.update()

            # Maintain a short wait to approximately preserve the display frame rate.
            pygame.time.wait(10)
        else:
            pygame.time.wait(10)

    # Cleanup resources.
    speaking_cap.release()
    idle_cap.release()
    pygame.mixer.music.stop()
    pygame.quit()
    cv2.destroyAllWindows()
    print("Playback finished.")


if __name__ == "__main__":
    main()
