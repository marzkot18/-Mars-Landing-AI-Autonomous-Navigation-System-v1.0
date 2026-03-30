# ==========================================
# AUTONOMOUS LANDING (RESNET VERSION)
# ==========================================
import requests
import os
import cv2
import torch
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import Dataset
from torchvision import models
MODEL_PATH = "model.pth"
MODEL_URL = "https://drive.google.com/uc?export=download&id=1BRpK4BcXiBaJ2fG2UCobo0mTJxwJ2CdT"
if not os.path.exists(MODEL_PATH):
    st.info("Downloading AI model... (first time only)")
    
    r = requests.get(MODEL_URL, stream=True)
    with open(MODEL_PATH, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

    st.success("Model ready!")

# ==========================================
# 1. DATASET
# ==========================================
class MarsDataset(Dataset):
    def __init__(self, root_dir):
        self.data = []
        self.labels = []
        self.class_names = os.listdir(root_dir)

        for idx, cls in enumerate(self.class_names):
            folder = os.path.join(root_dir, cls)
            for img_name in os.listdir(folder):
                self.data.append(os.path.join(folder, img_name))
                self.labels.append(idx)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = cv2.imread(self.data[idx])
        img = cv2.resize(img, (224, 224))  # match ResNet input size
        img = img / 255.0
        img = np.transpose(img, (2, 0, 1))
        return torch.tensor(img, dtype=torch.float32), self.labels[idx]

# ==========================================
# 2. RISK MAPPING
# ==========================================
risk_map = {0:0.5, 1:1.0, 2:0.6, 3:1.0, 4:0.2, 5:0.7, 6:0.8, 7:0.9}

# ==========================================
# 3. PATCH-BASED HAZARD DETECTION
# ==========================================
def generate_hazard_map(model, image):
    image = cv2.resize(image, (256, 256))
    patch_size = 32
    hazard_map = np.zeros((256, 256))

    for i in range(0, 256, patch_size):
        for j in range(0, 256, patch_size):
            patch = image[i:i+patch_size, j:j+patch_size]
            patch_resized = cv2.resize(patch, (224, 224))
            patch_resized = patch_resized / 255.0
            patch_resized = np.transpose(patch_resized, (2, 0, 1))
            patch_tensor = torch.tensor(patch_resized, dtype=torch.float32).unsqueeze(0)

            with torch.no_grad():
                pred = model(patch_tensor)
                cls = torch.argmax(pred).item()

            hazard_map[i:i+patch_size, j:j+patch_size] = risk_map.get(cls, 0.5)
    return hazard_map

# ==========================================
# 4. LANDING ZONE SELECTION
# ==========================================
def find_safe_zone(hazard_map):
    window_size = 40
    best_score = 9999
    best_coord = (0,0)
    for i in range(0, hazard_map.shape[0]-window_size):
        for j in range(0, hazard_map.shape[1]-window_size):
            window = hazard_map[i:i+window_size, j:j+window_size]
            score = np.mean(window)
            if score < best_score:
                best_score = score
                best_coord = (i,j)
    return best_coord, best_score

# ==========================================
# 5. STREAMLIT APP
# ==========================================
st.set_page_config(page_title="Mars Landing AI", layout="wide")
st.markdown("""
<style>
.stApp {background: radial-gradient(circle at top, #1a1f2b, #0e1117); color:white;}
h1,h2,h3 {color:#ff4b4b;}
.stButton>button {background:linear-gradient(90deg,#ff4b4b,#ff7b00); color:white; border-radius:12px; font-weight:bold;}
.block-container {padding-top:2rem;}
</style>
""", unsafe_allow_html=True)

st.sidebar.title("🛰 Mission Control")
uploaded_file = st.sidebar.file_uploader("Upload Mars Image", type=["jpg","png"])
patch_size = st.sidebar.slider("Patch Size", 16, 64, 32)
run_button = st.sidebar.button("🚀 Initiate Landing Scan")
st.sidebar.markdown("---")
st.sidebar.warning("⚠️ Use ONLY top-down satellite images")

st.markdown("# 🚀 Mars Autonomous Landing System")
st.caption("AI-powered terrain hazard detection & optimal landing selection")
st.error("⚠️ IMPORTANT: Only top-down Mars images work properly")

# ------------------------------------------
# LOAD MODEL
# ------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, 8)
model = model.to(device)

if os.path.exists("model.pth"):
    model.load_state_dict(torch.load("model.pth", map_location=device))
    model.eval()
else:
    st.error("⚠️ Model not loaded")

# ------------------------------------------
# MAIN UI
# ------------------------------------------
col1, col2 = st.columns([1,1])
if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    with col1:
        st.markdown("### 📡 Live Feed")
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_container_width=True)
    if run_button:
        progress = st.progress(0)
        with st.spinner("🧠 Scanning terrain..."):
            for i in range(100):
                import time; time.sleep(0.01); progress.progress(i+1)
            hazard_map = generate_hazard_map(model, image)
            coord, score = find_safe_zone(hazard_map)

        base = cv2.resize(image, (256,256))
        heat = (hazard_map - hazard_map.min()) / (hazard_map.max() - hazard_map.min())
        heat = (heat*255).astype(np.uint8)
        heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(base, 0.6, heat, 0.4, 0)
        x,y = coord
        cv2.rectangle(overlay, (y,x), (y+40,x+40), (0,255,0), 3)
        with col2:
            st.markdown("### 🔥 Hazard Overlay")
            st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), use_container_width=True)

        st.markdown("## 📊 Mission Telemetry")
        safe_score = 1 - score
        colA, colB, colC = st.columns(3)
        colA.metric("Safety Score", f"{safe_score:.2f}")
        colB.metric("Landing X", coord[0])
        colC.metric("Landing Y", coord[1])
        st.progress(float(safe_score))
        if safe_score > 0.75: st.success("🟢 Landing Zone Optimal")
        elif safe_score > 0.4: st.warning("🟡 Caution: Moderate Risk")
        else: st.error("🔴 Abort: Unsafe Terrain")

        st.markdown("### 🎨 Hazard Legend")
        st.markdown("""
        - 🔵 Low Risk (Safe Terrain)  
        - 🟢 Moderate Risk  
        - 🔴 High Risk (Craters / Hazards)  
        """)

with st.expander("ℹ️ How the AI Works"):
    st.markdown("""
    - Image divided into patches  
    - CNN classifies each patch  
    - Each terrain type has a risk score  
    - System scans for lowest-risk landing zone  
    ⚠️ Works best on top-down images
    """)

st.markdown("---")
st.caption("🚀 Mars Landing AI • Autonomous Navigation System v1.0")
