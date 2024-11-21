from flask import Flask, render_template, request, jsonify
import torch
from model.mnist_model import CompactMNIST
from torchvision import transforms
from PIL import Image
import io
import base64

app = Flask(__name__)

# Load model
model = CompactMNIST()
model.load_state_dict(torch.load('mnist_model.pth'))
model.eval()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get image data from request
    image_data = request.json['image']
    image_data = base64.b64decode(image_data.split(',')[1])
    image = Image.open(io.BytesIO(image_data)).convert('L')
    
    # Transform image
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    image_tensor = transform(image).unsqueeze(0)
    
    # Make prediction
    with torch.no_grad():
        output = model(image_tensor)
        prediction = output.argmax(dim=1).item()
    
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True) 