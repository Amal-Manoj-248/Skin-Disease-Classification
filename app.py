import torch
import torch.nn as nn
import numpy as np
import cv2
from torchvision.transforms.functional import to_pil_image
import streamlit as st
from torchvision import models, transforms
from torchcam.methods import SmoothGradCAMpp
from PIL import Image

# Class labels for skin diseases
class_labels = ['Actinic keratoses', 'Basal cell carcinoma', 'Benign keratosis-like lesions ', 'Dermatofibroma', 'Melanocytic nevi', 'Melanoma', 'Vascular lesions']

# Define Model
class MobileNetV2Model(nn.Module):
    def __init__(self, num_classes):
        super(MobileNetV2Model, self).__init__()
        self.model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

        # Freeze feature extraction layers
        for param in self.model.features.parameters():
            param.requires_grad = False

        # Unfreeze last conv block for Grad-CAM
        for param in self.model.features[18].parameters():
            param.requires_grad = True  

        # Modify classifier for custom classification
        in_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.model(x)

# Initialize Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = len(class_labels)  # Number of classes
model = MobileNetV2Model(num_classes=num_classes).to(device)

# Load trained weights
model.load_state_dict(torch.load("mobilenet_skin_disease.pth", map_location=device))
model.eval()

# Initialize Grad-CAM++ before inference
target_layer = model.model.features[18][0]  # Last convolutional layer
cam_extractor = SmoothGradCAMpp(model.model, target_layer=target_layer)

# Define preprocessing (consistent with validation transform)
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.76352966, 0.5461343, 0.5705373], std=[0.11806747, 0.1419266, 0.15766038])  
])

# Function to denormalize image
def denormalize(tensor, mean, std):
    tensor = tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)  # Convert to HWC format
    tensor = tensor * std + mean  # Reverse normalization
    tensor = np.clip(tensor * 255, 0, 255).astype(np.uint8)  # Convert to uint8
    return tensor

# Streamlit UI
st.title("Skin Disease Classification with Smooth Grad-CAM++")

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # **Step 1: Read and display the uploaded image**
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)  # Show uploaded image
    except Exception as e:
        st.error(f"Error loading image: {e}")
        st.stop()

    # Preprocess the image
    input_tensor = transform(image).unsqueeze(0).to(device)

    # **Step 2: Forward Pass**
    output = model(input_tensor)  # Forward pass
    prediction_idx = torch.argmax(output, dim=1).item()
    predicted_label = class_labels[prediction_idx]  # Get class label

    # **Step 3: Generate Grad-CAM Heatmap**
    activation_map = cam_extractor(prediction_idx, output)  # Extract heatmap

    # Normalize and resize heatmap
    heatmap = activation_map[0].cpu().numpy()
    heatmap = np.max(heatmap, axis=0)
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-10)
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(cv2.resize(heatmap, (128, 128)), cv2.COLORMAP_JET)

    # **Step 4: Ensure correct image format before overlay**
    mean = np.array([0.76352966, 0.5461343, 0.5705373])
    std = np.array([0.11806747, 0.1419266, 0.15766038])

    # **Convert the tensor image back to original format (denormalization)**
    original_image = denormalize(input_tensor, mean, std)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)  # Convert to BGR

    # Overlay heatmap
    overlayed_image = cv2.addWeighted(original_image, 0.3, heatmap, 0.7, 0)
    overlayed_image = cv2.cvtColor(overlayed_image, cv2.COLOR_BGR2RGB)  # Convert back to RGB for Streamlit

    # **Step 5: Display Prediction and Overlayed Image**
    st.subheader(f"Predicted Label: **{predicted_label}**")
    st.image(overlayed_image, caption="Grad-CAM Overlayed Image", use_container_width=True)
