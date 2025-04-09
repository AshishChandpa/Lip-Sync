import cv2
import dlib
import os
import numpy as np
import json
from collections import defaultdict


def download_shape_predictor(predictor_path):
    """Download and decompress the dlib shape predictor file."""
    import urllib.request
    import bz2
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


def mouth_aspect_ratio(mouth_points):
    """Compute the Mouth Aspect Ratio (MAR) using inner mouth landmarks."""
    try:
        A = np.linalg.norm(np.array(mouth_points[14]) - np.array(mouth_points[18]))  # p62-p66
        B = np.linalg.norm(np.array(mouth_points[15]) - np.array(mouth_points[17]))  # p63-p65
        C = np.linalg.norm(np.array(mouth_points[12]) - np.array(mouth_points[16]))  # p60-p64
        mar = (A + B) / (2.0 * C) if C > 0 else 0.2
        return mar
    except (IndexError, ZeroDivisionError) as e:
        print(f"Error calculating MAR: {e}")
        return 0.2


def map_mar_to_viseme(mar):
    """Map the MAR value to a viseme label."""
    if mar < 0.25:
        return "REST"
    elif mar < 0.32:
        return "I"
    elif mar < 0.40:
        return "A"
    elif mar < 0.48:
        return "O"
    else:
        return "Wide"


def extract_viseme_images(video_path, output_dir="viseme_images", max_per_viseme=5):
    """
    Extract mouth images for each viseme from a video.

    Args:
        video_path: Path to the video file
        output_dir: Directory to save extracted images
        max_per_viseme: Maximum number of images to extract per viseme

    Returns:
        Dictionary mapping viseme names to lists of image paths
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Initialize dlib's face detector and shape predictor
    detector = dlib.get_frontal_face_detector()
    predictor_path = "shape_predictor_68_face_landmarks.dat"

    if not os.path.exists(predictor_path):
        download_shape_predictor(predictor_path)

    predictor = dlib.shape_predictor(predictor_path)

    # Open the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return {}

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Processing {frame_count} frames from {video_path}")

    # Define all viseme categories we want to extract
    # This is the comprehensive list from the animator
    all_visemes = [
        "REST", "A", "E", "I", "O", "U",
        "F", "M", "L", "S", "T", "SH", "Wide", "Slight"
    ]

    # Keep track of extracted visemes
    extracted_counts = {viseme: 0 for viseme in all_visemes}
    viseme_to_images = {viseme: [] for viseme in all_visemes}

    # Sample interval to avoid processing every frame
    sample_interval = max(1, frame_count // 200)  # Process about 200 frames

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process only every sample_interval frames
        if frame_idx % sample_interval != 0:
            frame_idx += 1
            continue

        try:
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector(gray)

            if len(faces) > 0:
                face = faces[0]
                shape = predictor(gray, face)

                # Extract mouth landmarks (points 48-67)
                mouth_points = []
                for i in range(48, 68):
                    x = shape.part(i).x
                    y = shape.part(i).y
                    mouth_points.append((x, y))

                # Calculate mouth aspect ratio
                mar_value = mouth_aspect_ratio(mouth_points)

                # Determine viseme based on MAR
                viseme = map_mar_to_viseme(mar_value)

                # If we haven't collected enough examples of this viseme yet
                if extracted_counts[viseme] < max_per_viseme:
                    # Calculate mouth bounding box with padding
                    xs = [p[0] for p in mouth_points]
                    ys = [p[1] for p in mouth_points]
                    left, right = min(xs), max(xs)
                    top, bottom = min(ys), max(ys)

                    # Add padding
                    padding = int((bottom - top) * 0.5)  # 50% padding
                    top = max(0, top - padding)
                    bottom = min(frame.shape[0], bottom + padding)
                    left = max(0, left - padding)
                    right = min(frame.shape[1], right + padding)

                    # Extract mouth region
                    mouth_img = frame[top:bottom, left:right]

                    if mouth_img.size > 0:
                        # Save the mouth image
                        filename = f"{viseme}_{extracted_counts[viseme]:03d}_mar_{mar_value:.2f}.png"
                        filepath = os.path.join(output_dir, filename)
                        cv2.imwrite(filepath, mouth_img)

                        # Add to our mapping
                        viseme_to_images[viseme].append(filepath)

                        # Increment counter
                        extracted_counts[viseme] += 1
                        print(f"Extracted {viseme} viseme (MAR: {mar_value:.2f})")

        except Exception as e:
            print(f"Error processing frame {frame_idx}: {e}")

        frame_idx += 1

        # Check if we've collected enough examples of all visemes
        if all(count >= max_per_viseme for viseme, count in extracted_counts.items() if
               viseme in map_mar_to_viseme(0.99)):
            print("Collected enough examples of all basic visemes")
            # Continue to try to collect the other visemes

        # Print progress occasionally
        if frame_idx % 100 == 0:
            print(f"Processed {frame_idx}/{frame_count} frames")

    cap.release()

    # Save the mapping to a JSON file
    mapping_file = os.path.join(output_dir, "viseme_mapping.json")
    with open(mapping_file, 'w') as f:
        json.dump(viseme_to_images, f, indent=2)

    print(f"Saved viseme mapping to {mapping_file}")

    # Print summary
    print("\nExtraction complete. Summary:")
    for viseme, count in extracted_counts.items():
        print(f"{viseme}: {count} images")

    return viseme_to_images


if __name__ == "__main__":
    video_path = input("Enter path to speaking video file (default: ./speaking.mp4): ") or "./speaking.mp4"
    output_dir = input("Enter output directory for viseme images (default: ./viseme_images): ") or "./viseme_images"
    max_per_viseme = int(input("Enter maximum number of images per viseme (default: 10): ") or "10")

    viseme_mapping = extract_viseme_images(video_path, output_dir, max_per_viseme)
    print(f"Extracted {sum(len(imgs) for imgs in viseme_mapping.values())} images total")
