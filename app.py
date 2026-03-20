import streamlit as st
import face_recognition
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
                 (id INTEGER PRIMARY KEY, name TEXT UNIQUE, encoding BLOB)''')
    c.execute('''CREATE TABLE IF NOT EXISTS attendance 
                 (id INTEGER PRIMARY KEY, name TEXT, date TEXT, time TEXT)''')
    conn.commit()
    conn.close()

def load_known_faces():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT name, encoding FROM users", conn)
    conn.close()
    if df.empty:
        return [], []
    known_names = df['name'].tolist()
    known_encodings = [pickle.loads(blob) for blob in df['encoding']]
    return known_names, known_encodings

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

# ====================== HELPER FUNCTIONS ======================
def process_image_for_recognition(uploaded_file):
    image_pil = Image.open(io.BytesIO(uploaded_file.getvalue()))
    image_np = np.array(image_pil.convert('RGB'))
    return image_np, image_pil

def draw_recognized_faces(image_pil, face_locations, names):
    draw = ImageDraw.Draw(image_pil)
    try:
        font = ImageFont.truetype("arial.ttf", 30)
    except:
        font = ImageFont.load_default()
    
    for (top, right, bottom, left), name in zip(face_locations, names):
        # Draw box
        draw.rectangle(((left, top), (right, bottom)), outline="lime", width=4)
        # Draw label
        text = f" {name} "
        bbox = draw.textbbox((left, top-40), text, font=font)
        draw.rectangle(bbox, fill="lime")
        draw.text((left, top-40), text, fill="black", font=font)
    return image_pil

# ====================== STREAMLIT APP ======================
init_db()
st.set_page_config(page_title="Face Attendance System", layout="wide")
st.title("🧑‍💼 AI Face Recognition Attendance System")
st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs(["📸 Register New User", "🔍 Mark Attendance", "👥 Registered Users", "📊 Attendance Logs"])

# ====================== TAB 1: REGISTER ======================
with tab1:
    st.header("Register New User")
    name = st.text_input("Full Name (must be unique)")
    uploaded_images = st.file_uploader("Upload 3–10 clear face photos (JPG/PNG)", 
                                      accept_multiple_files=True, type=["jpg", "jpeg", "png"])
    
    if st.button("🚀 Register User") and name and uploaded_images:
        encodings_list = []
        progress_bar = st.progress(0)
        
        for i, file in enumerate(uploaded_images):
            img_np, _ = process_image_for_recognition(file)
            enc = face_recognition.face_encodings(img_np)
            if enc:
                encodings_list.append(enc[0])
            progress_bar.progress((i+1)/len(uploaded_images))
        
        if encodings_list:
            avg_encoding = np.mean(encodings_list, axis=0)
            encoding_blob = pickle.dumps(avg_encoding)
            
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            try:
                c.execute("INSERT INTO users (name, encoding) VALUES (?, ?)", (name, encoding_blob))
                conn.commit()
                st.success(f"🎉 User **{name}** registered successfully!")
                st.balloons()
            except sqlite3.IntegrityError:
                st.error("❌ Name already exists!")
            conn.close()
        else:
            st.error("❌ No face detected in any uploaded image.")

# ====================== TAB 2: MARK ATTENDANCE ======================
with tab2:
    st.header("Mark Attendance")
    source = st.radio("Choose input source", ["📷 Camera (Live)", "📤 Upload Image"])
    
    if source == "📷 Camera (Live)":
        captured = st.camera_input("Take a photo")
        if captured:
            uploaded_file = captured
    else:
        uploaded_file = st.file_uploader("Upload photo containing face(s)", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        st.subheader("Processing...")
        image_np, image_pil = process_image_for_recognition(uploaded_file)
        
        face_locations = face_recognition.face_locations(image_np)
        face_encodings = face_recognition.face_encodings(image_np, face_locations)
        
        known_names, known_encodings = load_known_faces()
        
        recognized_names = []
        
        for face_encoding, location in zip(face_encodings, face_locations):
            name = "Unknown"
            if known_encodings:
                distances = face_recognition.face_distance(known_encodings, face_encoding)
                best_match_index = np.argmin(distances)
                if distances[best_match_index] <= 0.55:  # Strict tolerance
                    name = known_names[best_match_index]
                    mark_attendance_in_db(name)
            recognized_names.append(name)
        
        # Draw boxes & labels
        final_image = draw_recognized_faces(image_pil, face_locations, recognized_names)
        st.image(final_image, caption="Detected & Recognized Faces", use_column_width=True)
        
        if recognized_names:
            st.subheader("Recognized People")
            for n in set([x for x in recognized_names if x != "Unknown"]):
                st.success(f"✅ {n}")

# ====================== TAB 3: REGISTERED USERS ======================
with tab3:
    st.header("Registered Users")
    known_names, _ = load_known_faces()
    if known_names:
        df_users = pd.DataFrame({"Name": known_names})
        st.dataframe(df_users, use_container_width=True)
        st.write(f"**Total registered users: {len(known_names)}**")
    else:
        st.info("No users registered yet.")

# ====================== TAB 4: ATTENDANCE LOGS ======================
with tab4:
    st.header("Attendance Logs")
    conn = sqlite3.connect(DB_PATH)
    df_att = pd.read_sql_query("SELECT name, date, time FROM attendance ORDER BY date DESC, time DESC", conn)
    conn.close()
    
    if not df_att.empty:
        st.dataframe(df_att, use_container_width=True)
        st.download_button("Download CSV", df_att.to_csv(index=False), "attendance_log.csv")
    else:
        st.info("No attendance records yet.")

st.caption("Built with ❤️ using face_recognition + Streamlit + SQLite")
