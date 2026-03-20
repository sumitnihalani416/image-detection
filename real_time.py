import cv2
import face_recognition
import numpy as np
import sqlite3
import pickle
from datetime import datetime

DB_PATH = "attendance_system.db"

def load_known_faces():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT name, encoding FROM users", conn)  # pandas already imported in app
    conn.close()
    if df.empty:
        return [], []
    return df['name'].tolist(), [pickle.loads(b) for b in df['encoding']]

def mark_attendance(name):
    today = datetime.now().strftime('%Y-%m-%d')
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id FROM attendance WHERE name=? AND date=?", (name, today))
    if c.fetchone() is None:
        time = datetime.now().strftime('%H:%M:%S')
        c.execute("INSERT INTO attendance (name, date, time) VALUES (?, ?, ?)", (name, today, time))
        conn.commit()
        print(f"✅ Attendance marked: {name} at {time}")
    conn.close()

print("🔴 Starting Real-Time Face Recognition Attendance...")
cap = cv2.VideoCapture(0)
known_names, known_encodings = load_known_faces()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        name = "Unknown"
        if known_encodings:
            distances = face_recognition.face_distance(known_encodings, face_encoding)
            best_idx = np.argmin(distances)
            if distances[best_idx] <= 0.55:
                name = known_names[best_idx]
                mark_attendance(name)
        
        # Draw box + label
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 3)
        cv2.putText(frame, name, (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    cv2.imshow("Real-Time Face Attendance System", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Session ended.")
