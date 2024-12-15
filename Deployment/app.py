import os
from flask import Flask, render_template, request, jsonify
import pickle
from PIL import Image
import numpy as np

app = Flask(__name__)

# Path to the model
MODEL_PATH = 'model/kubis_model.pkl'

# Load the model
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

with open(MODEL_PATH, 'rb') as file:
    model = pickle.load(file)

# Ensure upload folder exists
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Allowed extensions for image files
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Route for homepage
@app.route('/')
def index():
    return render_template('index.html')

# Route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'imageUpload' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['imageUpload']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Unsupported file type. Allowed types are jpg, jpeg, png.'}), 400

    # Save the uploaded image
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    try:
        # Preprocess the image
        img = Image.open(file_path).convert('RGB')
        img = img.resize((224, 224))  # Resize image to 224x224
        img_array = np.array(img)  # Convert image to numpy array
        img_array = img_array / 255.0  # Normalize pixel values
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Predict using the model
        prediction = model.predict(img_array)

        # Jika output berupa array multidimensi, ratakan array-nya
        prediction = prediction.flatten()

        # Ambil elemen pertama untuk mendapatkan nilai prediksi
        prediction_value = prediction[0]

        # Determine class based on prediction
        predicted_class = 'Sehat' if prediction_value > 0.5 else 'Tidak Sehat'

        # Cleanup: Remove the uploaded file after prediction
        os.remove(file_path)

        return jsonify({'prediction': predicted_class})

    except Exception as e:
        # Handle errors during processing or prediction
        os.remove(file_path)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
