import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import os
import numpy as np
import pandas as pd
from flask import Flask, render_template, redirect, url_for, flash, request, session, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
import joblib
from sklearn.preprocessing import LabelEncoder
import random
from statsmodels.tsa.statespace.sarimax import SARIMAX
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from datetime import datetime
import matplotlib.pyplot as plt
import os
from flask import jsonify
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.preprocessing import image
import torch
from torchvision import transforms
from PIL import Image
import os
from tensorflow.keras.models import load_model

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Image transformations
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define the model class (same as the one used during training)
class MobileNetModel(nn.Module):
    def __init__(self, num_classes):
        super(MobileNetModel, self).__init__()
        self.mobilenet = models.mobilenet_v2(pretrained=True)
        num_features = self.mobilenet.classifier[1].in_features
        self.mobilenet.classifier[1] = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.mobilenet(x)

# Load the trained model
model = MobileNetModel(num_classes=2)
model.load_state_dict(torch.load("mobilenet_irrelevent.pt"))
model = model.to(device)
model.eval()


# Function to predict image relevance
def predict_image(image_path):
    # Load and preprocess the image
    image = Image.open(image_path).convert('RGB')
    image = image_transform(image).unsqueeze(0)  # Add batch dimension
    image = image.to(device)

    # Perform the prediction
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)

    return predicted.item()

# Helper function to map the prediction to label
def map_prediction_to_label(prediction):
    label_mapping = {0: "irrelevent", 1: "relevent"}
    return label_mapping.get(prediction, "Unknown")

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'

db = SQLAlchemy(app)
bcrypt = Bcrypt(app)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)
    age = db.Column(db.Integer, nullable=False)
    gender = db.Column(db.String(1), nullable=False)
    mobile = db.Column(db.String(15), nullable=False)

# Load the feature extractor model (MobileNet-based model)
feature_extractor = load_model('feature_extractor.h5')

# Load the trained Random Forest model
rf_model = joblib.load('rf_model.pkl')

# Define the class labels
class_labels = ['deforestation', 'forest']

# Function to preprocess the input image
def preprocess_image(img_path):
    # Load the image with target size as (224, 224) to match MobileNet input size
    img = image.load_img(img_path, target_size=(224, 224))
    # Convert the image to an array
    img_array = image.img_to_array(img)
    # Rescale the image by dividing by 255 (normalization)
    img_array = img_array / 255.0
    # Add a batch dimension since the model expects batches of images
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Function to predict the class of a single image
def predict_single_image(img_path):
    # Preprocess the input image
    preprocessed_image = preprocess_image(img_path)
    
    # Extract features using the MobileNet-based feature extractor
    features = feature_extractor.predict(preprocessed_image)
    
    # Predict the class using the Random Forest classifier
    predicted_class_index = rf_model.predict(features)[0]
    
    # Convert the predicted class index to an integer (since it's a numpy.float64)
    predicted_class_index = int(predicted_class_index)
    
    # Convert the class index to the class label
    predicted_class_label = class_labels[predicted_class_index]
    
    return predicted_class_label

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')

        user = User.query.filter_by(email=email).first()
        if user and bcrypt.check_password_hash(user.password, password):
            session['user_id'] = user.id  # Store user ID in session
            return redirect(url_for('home'))
        else:
            flash('Invalid email or password.', 'danger')
    return render_template('auth.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        age = request.form.get('age')
        gender = request.form.get('gender')
        mobile = request.form.get('mobile')
        
        if len(mobile) != 10 or not mobile.isdigit():
            flash('Mobile number must be exactly 10 digits.', 'danger')
            return render_template('auth.html')

        if User.query.filter_by(email=email).first():
            flash('Email address already in use. Please choose a different one.', 'danger')
            return render_template('auth.html')
        
        if User.query.filter_by(username=username).first():
            flash('Username is already taken. Please choose a different one.', 'danger')
            return render_template('auth.html')

        if password != confirm_password:
            flash('Passwords do not match.', 'danger')
            return render_template('auth.html')
        
        if len(password) < 8:
            flash('Password must be at least 8 characters long.', 'danger')
            return render_template('auth.html')

        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
        new_user = User(username=username, email=email, password=hashed_password, age=age, gender=gender, mobile=mobile)
        
        db.session.add(new_user)
        db.session.commit()

        flash('Registration successful! You can now log in.', 'success')
        return redirect(url_for('login'))
    return render_template('auth.html')

@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    predicted_class = None  # Initialize the predicted_class variable

    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part', 'danger')
            return redirect(request.url)

        myfile = request.files['file']

        if myfile.filename == '':
            flash('No selected file', 'danger')
            return redirect(request.url)

        fn = myfile.filename
        mypath = os.path.join(r'static/saved_images', fn)
        myfile.save(mypath)

        # Predict image relevance
        prediction = predict_image(mypath)
        predicted_label = map_prediction_to_label(prediction)

        if predicted_label == "relevent":
            # Pass the image to the RandomForestClassifier for deforestation/forest prediction
            predicted_class = predict_single_image(mypath)
        else:
            predicted_class = "Not relevant"

    # Pass the predicted_class and image file name to the template
    return render_template('prediction.html', predicted_class=predicted_class, image_path=fn if 'fn' in locals() else None)

if __name__ == '__main__':
    with app.app_context():
        db.create_all()  # Create the database tables
    app.run(debug=True)