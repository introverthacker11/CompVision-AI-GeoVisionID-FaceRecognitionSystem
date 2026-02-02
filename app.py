import streamlit as st
import cv2
import numpy as np
import os
import tempfile
from ultralytics import YOLO
from deepface import DeepFace

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI-GeoVisionID",
    page_icon="🌍",
    layout="wide"
)

st.markdown("""
<h1 style="text-align: center;">🌍 AI-GeoVisionID</h1>
<p style="text-align: center; font-size: 18px;">
    Face Recognition of World Leaders using <b>YOLOv8</b> + <b>ArcFace</b>
</p>
<p style="text-align: center; color: lightgray; font-size: 14px;">
    Developed by Rayyan Ahmed<br>
""", unsafe_allow_html=True)

st.markdown('''---''')
# ---------------- Background -----------------

# ---------------- Background ----------------
st.markdown("""
<style>
.stApp {
    background-image: linear-gradient(rgba(0, 0, 0, 0.7), rgba(0, 0, 0, 0.7)),
                      url("https://media.licdn.com/dms/image/v2/D4D12AQG0D3XyjigtrA/article-cover_image-shrink_720_1280/B4DZWbSJkDHIAI-/0/1742066984144?e=2147483647&v=beta&t=vohjxDKSYtE8nuLDFAdWrBSOUur1H0p94JooC3CQLSQ");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: fixed;
    color: white;
}
h1 { color: #FFD700; text-align: center; }
</style>
""", unsafe_allow_html=True)

# ---------------- Sidebar ------------------

# ---------------- Sidebar Styling ----------------
st.markdown("""
<style>
[data-testid="stSidebar"] {
    background-color: rgba(0, 0.7, 0, 0.6);
    color: white;
}
[data-testid="stSidebar"] h1, h2, h3 { color: #00BFFF; }
::-webkit-scrollbar-thumb { background: #FFD700; border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# ---------------- Sidebar Info ----------------
with st.sidebar.expander("📌 Project Overview"):
    st.markdown("""
    - Perform **real-time face detection and recognition** in images and videos  
    - Detect faces using **YOLOv8 Face Detection**  
    - Identify global leaders using **ArcFace embeddings (DeepFace)**  
    - Display **identity metadata** (name, position, age, nationality)  
    - Handle **unknown faces** using similarity thresholding  
    - Support **image & video uploads** via an interactive Streamlit UI  
    - Designed for **AI Engineer / Computer Vision portfolio showcase**  
    """)


with st.sidebar.expander("👨‍💻 Developer"):
    st.markdown("""
    - **Rayyan Ahmed**
    - **Google Certified AI Prompt Specialist**
    - **IBM Certified LLM Fine Tuner** 
    - **Google Certified Business Intelligence Professional**  
    - **Expert in ML, DL, CV, NLP, LLM**  
    """)

with st.sidebar.expander("🛠️ Tech Stack Used"):
    st.markdown("""
    - 👁️ **YOLOv8 (Ultralytics)** → High-accuracy real-time face detection in images & videos  
    - 🧠 **ArcFace (DeepFace)** → Deep facial embeddings for identity recognition  
    - 🖼️ **OpenCV** → Frame processing, face cropping, annotations, and video rendering  
    - ⚙️ **NumPy** → Vector operations and cosine similarity computation  
    - 🌐 **Streamlit** → Interactive web interface for image & video uploads  
    - 🎥 **Video I/O (OpenCV)** → Video decoding, frame-wise inference, and export  
    - 🧪 **Python Ecosystem** → File handling, caching, and model integration  
    """)

# ---------------- LOAD MODELS ----------------
@st.cache_resource
def load_models():
    yolo = YOLO("yolov8s-face.pt")

    embeddings = {}
    for file in os.listdir("embeddings"):
        name = file.replace(".npy", "")
        embeddings[name] = np.load(os.path.join("embeddings", file))

    return yolo, embeddings

model, known_embeddings = load_models()

# ---------------- LEADER INFO ----------------
leader_info = {
    "Emmanuel Macron": {
        "Position": "President of France",
        "Age": 47,
        "Nationality": "French"
    },
    "Giorgia Meloni": {
        "Position": "Prime Minister of Italy",
        "Age": 48,
        "Nationality": "Italian"
    },
    "Recep Tayyip Erdoğan": {
        "Position": "President of Türkiye",
        "Age": 71,
        "Nationality": "Turkish"
    },
    "António Guterres": {
        "Position": "UN Secretary-General",
        "Age": 76,
        "Nationality": "Portuguese"
    },
    "Keir Starmer": {
        "Position": "Prime Minister of UK",
        "Age": 63,
        "Nationality": "British"
    },
    "Luiz Inácio Lula da Silva": {
        "Position": "President of Brazil",
        "Age": 80,
        "Nationality": "Brazilian"
    },
    "Cyril Ramaphosa": {
        "Position": "President of South Africa",
        "Age": 73,
        "Nationality": "South African"
    }
}

# ---------------- UTILS ----------------
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

THRESHOLD = 0.36

def recognize_face(face):
    face = cv2.resize(face, (200, 200))

    embedding = DeepFace.represent(
        img_path=face,
        model_name="ArcFace",
        detector_backend="skip",
        enforce_detection=False
    )[0]["embedding"]

    best_match, best_score = None, -1

    for person, embeddings in known_embeddings.items():
        for db_emb in embeddings:
            score = cosine_similarity(embedding, db_emb)
            if score > best_score:
                best_score = score
                best_match = person

    if best_score < THRESHOLD:
        return "Unknown", best_score

    return best_match, best_score

mode = st.sidebar.radio(
    "🎯 Select Mode",
    ["Image Recognition", "Video Recognition"]
)

if mode == "Image Recognition":

    st.subheader("🖼 Image Face Recognition")

    img_file = st.file_uploader(
        "Upload an image",
        type=["jpg", "png", "jpeg"]
    )

    if img_file:
        img = np.frombuffer(img_file.read(), np.uint8)
        frame = cv2.imdecode(img, cv2.IMREAD_COLOR)

        results = model(frame)

        for box in results[0].boxes.xyxy:
            x1, y1, x2, y2 = map(int, box)
            face = frame[y1:y2, x1:x2]

            name, score = recognize_face(face)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            label = name
            if name in leader_info:
                info = leader_info[name]

                y_offset = y1 - 10

                # Line 1: Name + confidence
                cv2.putText(
                    frame,
                    f"{name} ({score:.2f})",
                    (x1, y_offset - 18),
                    cv2.FONT_HERSHEY_PLAIN,
                    0.7,
                    (0, 255, 0),
                    1
                )

                # Line 2: Position
                cv2.putText(
                    frame,
                    info["Position"],
                    (x1, y_offset),
                    cv2.FONT_HERSHEY_PLAIN,
                    0.7,
                    (0, 255, 0),
                    1
                )

                cv2.putText(
                    frame,
                    (f"{info['Age']} | {info['Nationality']}"),
                    (x1, y_offset - 9),
                    cv2.FONT_HERSHEY_PLAIN,
                    0.7,
                    (0, 255, 0),
                    1
                )


        col1, col2 = st.columns([1, 1])

        with col1:
            st.image(
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                caption="Recognition Output",
                width=450
            )

if mode == "Video Recognition":

    st.subheader("🎥 Video Face Recognition")

    video_file = st.file_uploader(
        "Upload a video",
        type=["mp4"]
    )

    if video_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())

        cap = cv2.VideoCapture(tfile.name)

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        output_path = "AI_GeoVisionID_Output.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        progress = st.progress(0)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        processed = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame)

            for box in results[0].boxes.xyxy:
                x1, y1, x2, y2 = map(int, box)
                face = frame[y1:y2, x1:x2]

                name, score = recognize_face(face)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                info = leader_info[name]
                line_gap = 22
                base_y = y1 - 10  # starting point just above the box

                # Line 1: Name + confidence (top)
                cv2.putText(
                    frame,
                    f"{name} ({score:.2f})",
                    (x1, base_y - 2 * line_gap),
                    cv2.FONT_HERSHEY_PLAIN,
                    1.4,
                    (0, 255, 0),
                    2
                )

                # Line 2: Position (middle)
                cv2.putText(
                    frame,
                    info["Position"],
                    (x1, base_y - line_gap),
                    cv2.FONT_HERSHEY_PLAIN,
                    1.3,
                    (0, 255, 0),
                    2
                )

                # Line 3: Age | Nationality (bottom)
                cv2.putText(
                    frame,
                    f"{info['Age']} | {info['Nationality']}",
                    (x1, base_y),
                    cv2.FONT_HERSHEY_PLAIN,
                    1.2,
                    (0, 255, 0),
                    2
                )


            out.write(frame)
            processed += 1
            progress.progress(min(processed / total_frames, 1.0))

        cap.release()
        out.release()

        st.success("✅ Video processing completed!")

        # SHOW VIDEO
        #st.video(output_path)

        # DOWNLOAD BUTTON
        with open(output_path, "rb") as f:
            st.download_button(
                label="⬇️ Download Processed Video",
                data=f,
                file_name="AI_GeoVisionID_Output.mp4",
                mime="video/mp4"
            )

        video_file = None
