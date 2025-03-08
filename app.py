import os
import torch
import torchvision.transforms as transforms
from flask import Flask, request, render_template
from PIL import Image
from models.cnn import CNN  # Import your trained model

app = Flask(__name__)

# Create an 'uploads' folder if not exists
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN()
model.load_state_dict(torch.load("models/cifar10_cnn.pth", map_location=device))  # Load your trained model
model.to(device)
model.eval()  # Set model to evaluation mode

# CIFAR-10 Classes
CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

# Define image transformation (same as training)
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # CIFAR-10 images are 32x32
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return "No file uploaded"

    file = request.files["file"]
    if file.filename == "":
        return "No selected file"

    # Save uploaded image
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    # Process Image
    image = Image.open(filepath).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)  # Add batch dimension

    # Make Prediction
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        class_name = CLASSES[predicted.item()]

    return render_template("result.html", image_path=filepath, prediction=class_name)

if __name__ == "__main__":
    app.run(debug=True)
