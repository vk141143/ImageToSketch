from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import cv2
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Folders to store uploaded and output images
UPLOAD_FOLDER = 'uploads/'
OUTPUT_FOLDER = 'output/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

# Ensure directories exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

# Allowed image extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Function to check allowed file types
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    # Check if the post request has the file part
    if 'image' not in request.files:
        print("No file part in the request.")
        return redirect(request.url)

    file = request.files['image']

    # If no file is selected
    if file.filename == '':
        print("No selected file.")
        return redirect(request.url)

    # If file is allowed
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Create a sketch from the uploaded image
        sketch_image = create_realistic_sketch(filepath)
        if sketch_image:
            # Pass the image file paths to the template
            return render_template('index.html', original_image=filename, sketch_image=sketch_image)
        else:
            print("Failed to create sketch.")
            return redirect(request.url)

    return redirect(request.url)

# Function to create realistic sketch from an image
def create_realistic_sketch(filepath):
    # Check if file exists
    if not os.path.exists(filepath):
        print("File not found:", filepath)
        return None

    # Load the image
    image = cv2.imread(filepath)
    
    # Check if the image was successfully loaded
    if image is None:
        print("Failed to load image:", filepath)
        return None

    try:
        # Convert to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Invert the grayscale image
        inverted_image = 255 - gray_image

        # Apply Gaussian blur to smooth transitions
        blurred = cv2.GaussianBlur(inverted_image, (31, 31), 0)

        # Invert the blurred image
        inverted_blurred = 255 - blurred

        # Blend the grayscale image with the inverted blurred image
        sketch = cv2.divide(gray_image, inverted_blurred, scale=256.0)

        # Apply bilateral filter for smoothing
        smoothed_sketch = cv2.bilateralFilter(sketch, 9, 75, 75)

        # Save the sketch
        sketch_filename = 'realistic_sketch_' + os.path.basename(filepath)
        sketch_filepath = os.path.join(app.config['OUTPUT_FOLDER'], sketch_filename)
        cv2.imwrite(sketch_filepath, smoothed_sketch)

        return sketch_filename
    except Exception as e:
        print("Error processing image:", e)
        return None

# Routes to serve images
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/output/<filename>')
def output_file(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
