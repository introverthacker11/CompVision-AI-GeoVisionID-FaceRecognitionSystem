![AI-GeoVisionID](https://upload.wikimedia.org/wikipedia/commons/e/ef/Face_detection.jpg)

# 🌍 AI-GeoVisionID

**Face Recognition of World Leaders using YOLOv8 + ArcFace**

## App link: https://compvision-ai-geovisionid-facerecognitionsystem-3fzlspcrvy7ayv.streamlit.app/

---

## 📌 Project Overview

AI-GeoVisionID is an interactive computer vision application that detects and recognizes the faces of world leaders in images and videos. The system combines **YOLOv8** for fast and accurate face detection with **ArcFace embeddings** (via DeepFace) for identity recognition. Unknown faces are handled using a similarity threshold.

Key features:

- Detect faces in **images and videos**.
- Recognize world leaders using **ArcFace embeddings**.
- Display identity metadata: **Name, Position, Age, Nationality**.
- Handle unknown faces with **cosine similarity thresholding**.
- Streamlit interface for **interactive uploads and visualizations**.
- Export processed videos with annotated bounding boxes and labels.

---

## 🛠️ Tech Stack

- **YOLOv8 (Ultralytics)** → Real-time face detection
- **ArcFace (via DeepFace)** → Facial embeddings for recognition
- **OpenCV** → Image and video processing, annotation, video export
- **NumPy** → Array operations and similarity computation
- **Streamlit** → Web interface for uploads and visualization
- **TensorFlow** → Backend for ArcFace embeddings
- **Python Standard Libraries** → File handling, temporary storage

---

## 🎯 Usage

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/AI-GeoVisionID.git
cd AI-GeoVisionID
```
```
AI-GeoVisionID/
├── app.py                # Streamlit main app
├── yolov8s-face.pt       # YOLOv8 face detection model
├── embeddings/           # Precomputed ArcFace embeddings (.npy)
├── requirements.txt      # Python dependencies
├── detectedfaces/          # Sample images/videos
└── README.md
```
