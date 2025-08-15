import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import os
import cv2
import gdown
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# =========================
# CONFIGURATION
# =========================
st.set_page_config(page_title="Betel Leaf Disease Detection", layout="wide")
DOWNLOAD_DIR = "models"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

# Google Drive IDs for models
MODEL_LINKS = {
    "efficientnet_v2_s_best.pth": "1Ed3I67acsRHzTgQ9zm4oJFdu29temOhd",
    "convnext_tiny_best.pth": "1YuaDFLVtpkXHcZ_TAowrnPprDB0nXVVh",
    "densenet121_best.pth": "1tsSXYGaIq8L_bBoQq9IJ2LE9iy5yPgBv",
    "regnet_y_8gf_best.pth": "16DrG_tcjq269iQXaSCWQ7MzG6Tx7_1Kj",
    "vit_b_16_best.pth": "1vX_iYomMYqufCxX2nrty2KQbcXbntf5M",
    "custom_cnn_best.pth": "11x1SajFXhofI0K17yDx1N1h8Gl7SyOhS",
}

# =========================
# MODEL DOWNLOAD
# =========================
def download_model(filename, file_id):
    path = os.path.join(DOWNLOAD_DIR, filename)
    if not os.path.exists(path):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, path, quiet=False)
    return path

# =========================
# MODEL BUILDER
# =========================
class ConvBlock(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
    def forward(self, x): return self.block(x)

class CustomCNN(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3,32), ConvBlock(32,64), ConvBlock(64,128), ConvBlock(128,256)
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(256,num_classes)
        )
    def forward(self, x): return self.classifier(self.features(x))

def build_model(arch, num_classes):
    if arch == "custom_cnn":
        return CustomCNN(num_classes)
    elif arch == "efficientnet_v2_s":
        m = models.efficientnet_v2_s(weights=None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
        return m
    elif arch == "convnext_tiny":
        m = models.convnext_tiny(weights=None)
        m.classifier[2] = nn.Linear(m.classifier[2].in_features, num_classes)
        return m
    elif arch == "densenet121":
        m = models.densenet121(weights=None)
        m.classifier = nn.Linear(m.classifier.in_features, num_classes)
        return m
    elif arch == "regnet_y_8gf":
        m = models.regnet_y_8gf(weights=None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m
    elif arch == "vit_b_16":
        m = models.vit_b_16(weights=None)
        m.heads.head = nn.Linear(m.heads.head.in_features, num_classes)
        return m
    else:
        raise ValueError(f"Unsupported arch: {arch}")

# =========================
# TARGET LAYER DETECTION
# =========================
def get_target_layer(model, arch):
    if arch == "custom_cnn":
        return model.features[-1]
    elif arch in ["efficientnet_v2_s", "densenet121"]:
        return model.features[-1]
    elif arch == "convnext_tiny":
        return model.features[-1][-1]
    elif arch == "regnet_y_8gf":
        return model.trunk_output
    elif arch == "vit_b_16":
        return model.encoder.layers[-1].mlp[0]
    else:
        raise ValueError(f"No target layer for {arch}")

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model_from_drive(filename):
    path = download_model(filename, MODEL_LINKS[filename])
    ckpt = torch.load(path, map_location="cpu")
    labels = ckpt["classes"]
    num_classes = len(labels)
    model = build_model(ckpt["arch"], num_classes)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, labels, ckpt["arch"]

# =========================
# IMAGE PREPROCESS
# =========================
def preprocess_image(img):
    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    return tfm(img).unsqueeze(0)

# =========================
# CAM GENERATION (SAFE)
# =========================
def generate_cam(cam_class, model, target_layer, input_tensor):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    input_tensor = input_tensor.to(device)
    cam = cam_class(model=model, target_layers=[target_layer])
    grayscale_cam = cam(input_tensor=input_tensor, eigen_smooth=True)
    return grayscale_cam[0, :]

def safe_cam(cam_class, model, target_layer, input_tensor):
    try:
        return generate_cam(cam_class, model, target_layer, input_tensor)
    except Exception as e:
        st.warning(f"{cam_class.__name__} failed: {e}")
        return np.zeros((224, 224))

# =========================
# STREAMLIT UI
# =========================
st.title("🌿 Betel Leaf Disease Detection with Explainability")

selected_model_name = st.selectbox("Choose a model", list(MODEL_LINKS.keys()))
model, labels, arch = load_model_from_drive(selected_model_name)

uploaded_file = st.file_uploader("Upload a betel leaf image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    input_tensor = preprocess_image(img)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        pred_idx = torch.argmax(probs).item()
        pred_class = labels[pred_idx] if isinstance(labels, dict) else labels[str(pred_idx)]
        st.markdown(f"**Prediction:** {pred_class} ({probs[pred_idx]*100:.2f}%)")

    rgb_img = np.array(img.resize((224, 224))) / 255.0
    target_layer = get_target_layer(model, arch)

    gradcam_mask = safe_cam(GradCAM, model, target_layer, input_tensor)
    gradcampp_mask = safe_cam(GradCAMPlusPlus, model, target_layer, input_tensor)
    eigencam_mask = safe_cam(EigenCAM, model, target_layer, input_tensor)

    gradcam_img = show_cam_on_image(rgb_img, gradcam_mask, use_rgb=True)
    gradcampp_img = show_cam_on_image(rgb_img, gradcampp_mask, use_rgb=True)
    eigencam_img = show_cam_on_image(rgb_img, eigencam_mask, use_rgb=True)

    st.subheader("Explainability Visualizations")
    col1, col2, col3 = st.columns(3)
    col1.image(gradcam_img, caption="Grad-CAM", use_column_width=True)
    col2.image(gradcampp_img, caption="Grad-CAM++", use_column_width=True)
    col3.image(eigencam_img, caption="Eigen-CAM", use_column_width=True)