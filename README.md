# 🌿 Betel Leaf Disease Detection with Explainable AI

This project presents a web-based Streamlit application to detect diseases in betel leaves using multiple deep learning models. It enhances model interpretability with integrated explainable AI (XAI) techniques.

---

## 🚀 Live Demo

Access the deployed app here:  
**🔗 [Streamlit Cloud App](https://br7p2jppwvs8c4qhiirkrh.streamlit.app/)**

---

## 🧠 Models Included

The following models are supported via dropdown selection in the app:

| Model Architecture    | Source         |
|-----------------------|----------------|
| EfficientNet V2 S     | torchvision    |
| ConvNeXt Tiny         | torchvision    |
| DenseNet121           | torchvision    |
| RegNet Y-8GF          | torchvision    |
| ViT B-16              | torchvision    |
| Custom CNN            | self-built     |

Each model is trained to classify betel leaf conditions across **4 classes** and includes an attached `.pth` checkpoint.

---

## 🔍 Explainability Techniques (XAI)

| Method         | Description                                         |
|----------------|-----------------------------------------------------|
| Grad-CAM       | Highlights regions relevant to prediction          |
| Grad-CAM++     | Refines Grad-CAM for improved localization         |
| Eigen-CAM      | Uses principal components of feature maps          |
| *(Ablation-CAM & LIME not implemented yet, but planned)* |

Each method displays a heatmap over the input image for interpretability.

---

## 📸 Sample Screenshots
EfficientNet V2 S: <img width="1865" height="710" alt="image" src="https://github.com/user-attachments/assets/3110b58c-0a54-460b-ad01-9712842bdc10" />
densenet121: <img width="1873" height="725" alt="image" src="https://github.com/user-attachments/assets/bcd79812-e3b5-4935-bbc1-a87c8244150e" />
regnet_y_8gf: <img width="1839" height="737" alt="image" src="https://github.com/user-attachments/assets/5b06e5a7-86d1-4c03-8799-89976ade9a9c" />




---
## Model Weights  
If you're running locally, download model weights manually and place them in the models/ folder
OR
The app will automatically download them from Google Drive using gdown.

🔗 Google Drive Folder:
📂 All Model Weights: https://drive.google.com/drive/folders/1Z02X8VTI_sJ0kJUyk1y3E5bmJ1zZzPej?usp=sharing

## 📂 Project Structure

```bash
betel-leaf-disease/
├── app.py                      # Main Streamlit app
├── requirements.txt            # Dependency list
├── models/                     # Folder for model weights (if local)
├── README.md                   # This file
└── screenshots/                # UI preview images




