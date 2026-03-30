# 🚀 Mars Autonomous Landing AI

## 📌 Overview

This project uses a pretrained **ResNet18 convolutional neural network** to analyze Mars terrain images and identify the safest landing zone. It classifies terrain patches, assigns risk scores, and visualizes hazard maps through an interactive Streamlit app.

---

## 🧠 Features

* Terrain classification using ResNet18 (pretrained on ImageNet)
* Patch-based hazard detection
* Risk scoring system for terrain types
* Automatic safe landing zone selection
* Interactive UI with Streamlit
* Heatmap overlay visualization

---

## 📂 Project Structure

```
├── main.py        # Training script
├── app.py         # Streamlit app (inference + UI)
├── model.pth      # Saved trained model (generated after training)
└── dataset/       # Mars terrain dataset (not included)
```

---

## ⚙️ Requirements

Install dependencies before running:

```
pip install torch torchvision opencv-python streamlit matplotlib numpy
```

---

## 🏋️ Training the Model

1. Update dataset path in `main.py`
2. Run:

```
python main.py
```

3. Best model will be saved as:

```
model.pth
```

---

## 🖥 Running the App

Launch the Streamlit interface:

```
streamlit run app.py
```

Then:

* Upload a **top-down Mars image**
* Click **"Initiate Landing Scan"**
* View hazard map + safest landing zone

---

## ⚠️ Important Notes

* Only works well with **top-down satellite images**
* Dataset must be organized into class folders
* Model expects **224x224 input (ResNet standard)**

---

## 📊 How It Works

1. Image split into patches
2. Each patch classified by CNN
3. Risk score assigned per class
4. Sliding window finds lowest-risk zone
5. Heatmap + safe landing box displayed

---

## 🎯 Output

* Hazard heatmap overlay
* Safety score (0–1)
* Optimal landing coordinates

---

## 🚧 Future Improvements

* Use larger models (ResNet50, EfficientNet)
* Real-time video input
* Better dataset / labeling
* GPU optimization

---

## 👨‍💻 Author

Amaar Kothari
---

## 📜 License

This project is for educational/research purposes.
