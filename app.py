import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
import pandas as pd
import sqlite3
import pickle
from PIL import Image, ImageDraw, ImageFont
import io
from datetime import datetime
import os

# ====================== DATABASE SETUP ======================
DB_PATH = "attendance_system.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users 
                 (id INTEGER PRIMARY KEY, name TEXT UNIQUE, landmarks BLOB)''')
    c.execute('''CREATE TABLE IF NOT EXISTS attendance 
                 (id INTEGER PRIMARY KEY, name TEXT, date TEXT, time TEXT)''')
    conn.commit()
    conn.close()

def load_known_faces():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT name, landmarks FROM users", conn)
    conn.close()
    if df.empty:
        return [], []
    known_names = df['name'].tolist()
    known_landmarks = [pickle.loads(blob) for blob in df['landmarks']]
    return known_names, known_landmarks

def mark_attendance_in_db(name: str):
    today = datetime.now().strftime('%Y-%m-%d')
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id FROM attendance WHERE name=? AND date=?", (name, today))
    if c.fetchone() is None:
        time_now = datetime.now().strftime('%H:%M:%S')
        c.execute("INSERT INTO attendance (name, date, time) VALUES (?, ?, ?)", (name, today, time_now))
        conn.commit()
        st.success(f"✅ Attendance marked for **{name}** at {time_now}")
    else:
        st.info(f"ℹ️ {name} already marked today.")
    conn.close()

# ====================== MediaPipe SETUP ======================
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh  # for landmarks

face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=5, min_detection_confidence=0.5)

# ====================== HELPER FUNCTIONS ======================
def extract_landmarks(image_np):
    results = face_mesh.process(image_np)
    if not results.multi_face_landmarks:
        return None
    # Take first face for simplicity (or average multiple later)
    landmarks = results.multi_face_landmarks[0].landmark
    vec = np.array([[lm.x, lm.y, lm.z] for lm in landmarks]).flatten()
    # Normalize (important for scale/rotation invariance)
    vec = (vec - vec.mean()) / (vec.std() + 1e-8)
    return vec

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)

def process_image_for_recognition(uploaded_file):
    image_pil = Image.open(io.BytesIO(uploaded_file.getvalue()))
    image_np = np.array(image_pil.convert('RGB'))
    return image_np, image_pil

def draw_recognized_faces(image_pil, face_locations, names):
    draw = ImageDraw.Draw(image_pil)
    try:
        font = ImageFont.truetype("arial.ttf", 24)
    except:
        font = ImageFont.load_default()
    
    for (ymin, xmin, ymax, xmax), name in zip(face_locations, names):  # MediaPipe uses normalized coords
        h, w = image_pil.size[1], image_pil.size[0]
        left, top, right, bottom = int(xmin * w), int(ymin * h), int(xmax * w), int(ymax * h)
        draw.rectangle(((left, top), (right, bottom)), outline="lime", width=4)
        text = f" {name} "
        bbox = draw.textbbox((left, top-30), text, font=font)
        draw.rectangle(bbox, fill="lime")
        draw.text((left, top-30), text, fill="black", font=font)
    return image_pil

# ====================== STREAMLIT APP ======================
init_db()
st.set_page_config(page_title="Face Attendance – MediaPipe", layout="wide")
st.title("🧑‍💼 AI Face Attendance System (MediaPipe Edition)")
st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs(["📸 Register", "🔍 Mark Attendance", "👥 Users", "📊 Logs"])

with tab1:
    st.header("Register New User")
    name = st.text_input("Full Name (unique)")
    uploaded_images = st.file_uploader("Upload 3–8 clear face photos", accept_multiple_files=True, type=["jpg", "jpeg", "png"])
    
    if st.button("🚀 Register") and name and uploaded_images:
        landmarks_list = []
        progress = st.progress(0)
        
        for i, file in enumerate(uploaded_images):
            img_np, _ = process_image_for_recognition(file)
            vec = extract_landmarks(img_np)
            if vec is not None:
                landmarks_list.append(vec)
            progress.progress((i+1)/len(uploaded_images))
        
        if landmarks_list:
            avg_vec = np.mean(landmarks_list, axis=0)
            blob = pickle.dumps(avg_vec)
            
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            try:
                c.execute("INSERT INTO users (name, landmarks) VALUES (?, ?)", (name, blob))
                conn.commit()
                st.success(f"🎉 **{name}** registered!")
            except sqlite3.IntegrityError:
                st.error("Name already exists!")
            conn.close()
        else:
            st.error("No faces detected in uploads.")

with tab2:
    st.header("Mark Attendance")
    source = st.radio("Input", ["📷 Camera", "📤 Upload"])
    
    if source == "📷 Camera":
        captured = st.camera_input("Take photo")
        if captured:
            uploaded_file = captured
    else:
        uploaded_file = st.file_uploader("Upload photo", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        image_np, image_pil = process_image_for_recognition(uploaded_file)
        
        # Detection first (bounding boxes)
        detection_results = face_detection.process(image_np)
        face_locations = []
        if detection_results.detections:
            for det in detection_results.detections:
                bbox = det.location_data.relative_bounding_box
                face_locations.append((bbox.ymin, bbox.xmin, bbox.ymin + bbox.height, bbox.xmin + bbox.width))
        
        face_vecs = []
        for loc in face_locations:
            # Crop face roughly (improve later)
            h, w, _ = image_np.shape
            ymin, xmin, ymax, xmax = [int(x * dim) for x, dim in zip(loc, [h, w, h, w])]
            face_crop = image_np[max(0, ymin-20):min(h, ymax+20), max(0, xmin-20):min(w, xmax+20)]
            vec = extract_landmarks(face_crop)
            if vec is not None:
                face_vecs.append(vec)
        
        known_names, known_vecs = load_known_faces()
        recognized_names = []
        
        for vec in face_vecs:
            name = "Unknown"
            if known_vecs:
                sims = [cosine_similarity(vec, kv) for kv in known_vecs]
                best_idx = np.argmax(sims)
                if sims[best_idx] > 0.92:  # Tune: 0.90–0.95 range
                    name = known_names[best_idx]
                    mark_attendance_in_db(name)
            recognized_names.append(name)
        
        final_image = draw_recognized_faces(image_pil, face_locations, recognized_names)
        st.image(final_image, use_column_width=True)
        
        if any(n != "Unknown" for n in recognized_names):
            st.subheader("Recognized")
            for n in set(n for n in recognized_names if n != "Unknown"):
                st.success(f"✅ {n}")

with tab3:
    st.header("Registered Users")
    known_names, _ = load_known_faces()
    if known_names:
        st.dataframe(pd.DataFrame({"Name": known_names}))
    else:
        st.info("No users yet.")

with tab4:
    st.header("Attendance Logs")
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT name, date, time FROM attendance ORDER BY date DESC, time DESC", conn)
    conn.close()
    if not df.empty:
        st.dataframe(df)
        st.download_button("Download CSV", df.to_csv(index=False), "attendance.csv")
    else:
        st.info("No records yet.")
