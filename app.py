import os
from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
import requests
from io import BytesIO
import logging


# Start the app
app = Flask(__name__)

# Class labels for prediction
disease_label_from_category = {
    0: "Mild Demented",
    1: "Moderate Demented",
    2: "Non Demented",
    3: "Very Mild Demented",
}

N_CLASSES = len(disease_label_from_category)

# Define your model architecture here
class TunedCNN(nn.Module):
    def __init__(self):
        super(TunedCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.batchnorm1 = nn.BatchNorm2d(num_features=32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 32 * 32, 128)
        self.drop1 = nn.Dropout(p=0.2)
        self.out = nn.Linear(128, N_CLASSES)

    def forward(self, x):
        x = F.mish(self.conv1(x))
        x = self.pool1(x)
        x = self.batchnorm1(x)
        x = F.mish(self.conv2(x))
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        leaky = nn.LeakyReLU(0.01)
        x = leaky(x)
        x = self.drop1(x)
        x = self.out(x)
        return x


# Load the model
model = TunedCNN()

# Load the model weights
model.load_state_dict(torch.load('tuned_model.pt', map_location=torch.device('cpu')))

model.eval()  # Set the model to evaluation mode

# Define the preprocessing transformations
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Ensure image is grayscale
        transforms.Resize((128, 128)),  # Resize to match model input size
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize(mean=[0.485], std=[0.229]),  # Normalize
    ])
    return transform(image)


@app.route('/predict', methods=['POST'])
def predict():
    # Get the image URL from the request
    data = request.json
    imageUrl = data.get('imageUrl')  # Assuming the URL is passed as part of the JSON request
    
    if not imageUrl:
        return jsonify({"error": "No image URL provided"}), 400

    try:
        # Download the image from the URL
        response = requests.get(imageUrl)
        response.raise_for_status()  # Raise an error for bad HTTP responses

        # Open the image using PIL
        img = Image.open(BytesIO(response.content))

        # Apply preprocessing
        img = preprocess_image(img)

        # Add a batch dimension
        img = img.unsqueeze(0)

        # Make prediction with the model
        with torch.no_grad():
            output = model(img) 

        # Apply softmax to get probabilities
        probabilities = F.softmax(output, dim=1)
        
        # Get the predicted class index
        predicted_class_idx = torch.argmax(probabilities, dim=1).item()

        # Get the predicted label from the dictionary
        predicted_label = disease_label_from_category.get(predicted_class_idx, "Unknown")
        
        # Return the predicted class label and probability
        return jsonify({
            "predicted_class": predicted_label,
            "probability": probabilities[0][predicted_class_idx].item()
        })
    
    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Failed to fetch image from URL: {e}"}), 400

port = int(os.environ.get("PORT", 5000))
if __name__ == '__main__':
    # Configure logging to see the output
    logging.basicConfig(level=logging.DEBUG)
    app.run(host='0.0.0.0', port=port)


