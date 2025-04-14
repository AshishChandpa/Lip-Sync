import os
import subprocess


# Youtube_download
def download_youtube_video(url, output_path=None):
    """
    Download a YouTube video using yt-dlp.
    """
    try:
        if output_path is None:
            output_path = os.getcwd()
        os.makedirs(output_path, exist_ok=True)
        cmd = ["yt-dlp", "-f", "best", "-o", f"{output_path}/%(title)s.%(ext)s", url]
        print("Downloading...")
        result = subprocess.run(cmd, check=True, text=True)
        print("Download completed!")
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")
        return e.returncode
    except Exception as e:
        print(f"An error occurred: {e}")
        return -1


# Cut video
def cut_video(input_path, output_path, start_time, end_time):
    try:
        # Try to import moviepy - if it fails, install it first
        # pip install moviepy==1.0.3
        try:
            from moviepy.editor import VideoFileClip
        except ImportError:
            print("MoviePy not found. Installing...")
            subprocess.run(["pip", "install", "moviepy"], check=True)
            # Try importing again after installation
            from moviepy.editor import VideoFileClip

        # Convert time strings to float if needed
        if isinstance(start_time, str) and ':' in start_time:
            parts = start_time.split(':')
            if len(parts) == 3:  # hh:mm:ss
                h, m, s = parts
                start_time = int(h) * 3600 + int(m) * 60 + float(s)
            elif len(parts) == 2:  # mm:ss
                m, s = parts
                start_time = int(m) * 60 + float(s)
        else:
            start_time = float(start_time)

        if isinstance(end_time, str) and ':' in end_time:
            parts = end_time.split(':')
            if len(parts) == 3:  # hh:mm:ss
                h, m, s = parts
                end_time = int(h) * 3600 + int(m) * 60 + float(s)
            elif len(parts) == 2:  # mm:ss
                m, s = parts
                end_time = int(m) * 60 + float(s)
        else:
            end_time = float(end_time)

        print(f"Cutting video from {start_time} to {end_time} seconds...")
        # Create video clip and cut it
        video = VideoFileClip(input_path)
        video_cut = video.subclip(start_time, end_time)

        print(f"Writing output file to {output_path}...")
        video_cut.write_videofile(output_path, codec="libx264", audio_codec="aac")
        video.close()
        video_cut.close()
        print("Video successfully trimmed and saved.")
    except Exception as e:
        print(f"An error occurred while cutting the video: {e}")


if __name__ == "__main__":
    print("\nChoose an operation:")
    print("1. Download YouTube Video")
    print("2. Cut a Local Video")
    choice = input("Enter your choice (1 or 2): ").strip()

    if choice == "1":
        # Ensure yt-dlp is installed
        try:
            subprocess.run(["yt-dlp", "--version"], check=True, capture_output=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("yt-dlp not found. Installing...")
            subprocess.run(["pip", "install", "yt-dlp"], check=True)

        video_url = input("Enter the YouTube video URL: ")
        download_dir = input("Enter download directory (press Enter for current directory): ")
        if download_dir.strip() == "":
            download_dir = None
        download_youtube_video(video_url, download_dir)

    elif choice == "2":
        # We'll check for moviepy in the cut_video function
        input_video = input("Enter the path to the video file: ")
        output_video = input("Enter the name/path for the trimmed video: ")
        start_time = input("Enter the start time (in seconds or hh:mm:ss): ")
        end_time = input("Enter the end time (in seconds or hh:mm:ss): ")

        cut_video(input_video, output_video, start_time, end_time)
    else:
        print("Invalid choice. Please enter 1 or 2.")