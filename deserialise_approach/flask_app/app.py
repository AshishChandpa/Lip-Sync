import os
import uuid
import time
from flask import Flask, request, render_template, redirect, url_for, flash, send_from_directory
from werkzeug.utils import secure_filename
from text_to_lip_animation import TextToLipAnimation

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'dev_key_for_flask_session')
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['VISEME_FOLDER'] = 'static/visemes'
app.config['OUTPUT_FOLDER'] = 'static/outputs'
app.config['ALLOWED_EXTENSIONS'] = {'mp4', 'webm', 'mov'}
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max upload

# Create required directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


@app.route('/')
def index():
    # Get list of sample videos
    sample_videos = [f for f in os.listdir('static/samples')
                     if f.lower().endswith(('.mp4', '.webm', '.mov'))]

    # Get list of recent outputs
    recent_outputs = []
    if os.path.exists(app.config['OUTPUT_FOLDER']):
        output_files = [(f, os.path.getmtime(os.path.join(app.config['OUTPUT_FOLDER'], f)))
                        for f in os.listdir(app.config['OUTPUT_FOLDER'])
                        if f.lower().endswith('.mp4')]
        output_files.sort(key=lambda x: x[1], reverse=True)  # Sort by modification time
        recent_outputs = [f[0] for f in output_files[:5]]  # Get 5 most recent

    return render_template('index.html',
                           sample_videos=sample_videos,
                           recent_outputs=recent_outputs)


@app.route('/process', methods=['POST'])
def process_animation():
    # Check if it's a sample video or uploaded file
    video_source = request.form.get('video_source', 'upload')
    text_to_speak = request.form.get('text', '')
    wpm = int(request.form.get('wpm', 150))

    if not text_to_speak:
        flash('Please enter text for the animation.')
        return redirect(url_for('index'))

    # Generate a unique ID for this job
    job_id = str(uuid.uuid4())
    output_filename = f"{job_id}.mp4"
    output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)

    if video_source == 'sample':
        sample_video = request.form.get('sample_video', '')
        if not sample_video:
            flash('Please select a sample video.')
            return redirect(url_for('index'))

        input_video_path = os.path.join('static/samples', sample_video)
    else:
        # Check if a file was uploaded
        if 'video_file' not in request.files:
            flash('No file part')
            return redirect(url_for('index'))

        file = request.files['video_file']
        if file.filename == '':
            flash('No selected file')
            return redirect(url_for('index'))

        if not file or not allowed_file(file.filename):
            flash('Invalid file type. Please upload a video file.')
            return redirect(url_for('index'))

        # Save the uploaded file
        filename = secure_filename(file.filename)
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{job_id}_{filename}")
        file.save(upload_path)
        input_video_path = upload_path

    # Create a status file to indicate processing is ongoing
    with open(os.path.join(app.config['OUTPUT_FOLDER'], f"{job_id}_status.txt"), 'w') as f:
        f.write('processing')

    # Create a background task for processing
    # In a production environment, you would use Celery or similar
    # For simplicity, we'll use a separate process
    import threading

    def process_job():
        try:
            # Initialize the animator
            animator = TextToLipAnimation(
                input_video_path=input_video_path,
                viseme_folder=app.config['VISEME_FOLDER'],
                output_path=output_path
            )

            # Animate from text
            success = animator.animate_from_text(text_to_speak, words_per_minute=wpm)

            # Create a preview GIF if successful
            if success:
                animator.create_preview_gif()

            # Update status file
            with open(os.path.join(app.config['OUTPUT_FOLDER'], f"{job_id}_status.txt"), 'w') as f:
                f.write('completed' if success else 'failed')

        except Exception as e:
            # Log the error
            with open(os.path.join(app.config['OUTPUT_FOLDER'], f"{job_id}_error.txt"), 'w') as f:
                f.write(str(e))

            # Update status file
            with open(os.path.join(app.config['OUTPUT_FOLDER'], f"{job_id}_status.txt"), 'w') as f:
                f.write('failed')

    # Start the processing thread
    thread = threading.Thread(target=process_job)
    thread.daemon = True
    thread.start()

    # Redirect to the status page
    return redirect(url_for('job_status', job_id=job_id))


@app.route('/status/<job_id>')
def job_status(job_id):
    status_file = os.path.join(app.config['OUTPUT_FOLDER'], f"{job_id}_status.txt")

    if os.path.exists(status_file):
        with open(status_file, 'r') as f:
            status = f.read().strip()
    else:
        status = 'not_found'

    output_filename = f"{job_id}.mp4"
    preview_filename = f"{job_id}_preview.gif"

    return render_template('status.html',
                           job_id=job_id,
                           status=status,
                           output_filename=output_filename,
                           preview_filename=preview_filename)


@app.route('/outputs/<filename>')
def output_file(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)


if __name__ == '__main__':
    app.run(debug=True, port=1111)