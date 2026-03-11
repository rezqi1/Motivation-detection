import streamlit as st
from deepface import DeepFace
from transformers import pipeline
import cv2
import numpy as np
from PIL import Image
import tempfile
import time

# --- Load Text Model ---
text_classifier = pipeline("sentiment-analysis")

# --- Sidebar ---
st.sidebar.title("Motivation Detection")
app_mode = st.sidebar.radio("Choose Detection Mode", ["Video", "Image", "Text"])

# Function for color-coded motivation
def color_motivation(level):
    if level == "HIGH":
        return f"🟢 HIGH"
    elif level == "MEDIUM":
        return f"🟡 MEDIUM"
    else:
        return f"🔴 LOW"

# Function to map DeepFace emotions to motivation
def emotion_to_motivation(emotion):
    if emotion in ["happy","surprise"]:
        return "HIGH"
    elif emotion=="neutral":
        return "MEDIUM"
    else:
        return "LOW"

# ----------------- VIDEO -----------------
if app_mode == "Video":
    st.header("Video Motivation Detection")
    uploaded_file = st.file_uploader("Upload your video", type=["mp4","avi"])
    
    if uploaded_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        cap = cv2.VideoCapture(tfile.name)
        
        frame_count = 0
        motivations = []
        confidences = []

        st.write("Processing video frames (~1 per second)...")
        progress_bar = st.progress(0)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        processed_frames = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            # Process ~1 frame per second assuming 30fps
            if frame_count % 30 != 0:
                continue

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            try:
                result = DeepFace.analyze(rgb_frame, actions=['emotion'], enforce_detection=False)
                emotion = result[0]['dominant_emotion']
                confidence = result[0]['emotion'][emotion]
                motivation = emotion_to_motivation(emotion)
                
                motivations.append(motivation)
                confidences.append(confidence)
            except:
                pass

            processed_frames += 1
            progress_bar.progress(min(processed_frames/((total_frames//30)+1),1.0))

        cap.release()
        progress_bar.empty()

        if motivations:
            # Count motivation levels
            high = motivations.count("HIGH")
            med = motivations.count("MEDIUM")
            low = motivations.count("LOW")
            st.write("Motivation summary for video:")
            st.write(f"🟢 HIGH: {high}, 🟡 MEDIUM: {med}, 🔴 LOW: {low}")
            dominant = max(set(motivations), key=motivations.count)
            avg_confidence = sum(confidences)/len(confidences)
            st.write("Dominant Motivation Level:", color_motivation(dominant))
            st.write(f"Average Confidence: {avg_confidence:.2f}%")
        else:
            st.write("No faces detected in video.")

# ----------------- IMAGE -----------------
elif app_mode == "Image":
    st.header("Image Motivation Detection")
    uploaded_file = st.file_uploader("Upload your image", type=["jpg","png"])
    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image", width=600)
        img_np = np.array(img)

        with st.spinner("Analyzing image..."):
            time.sleep(0.5)  # Simulate loading
            try:
                result = DeepFace.analyze(img_np, actions=['emotion'], enforce_detection=False)
                emotion = result[0]['dominant_emotion']
                confidence = result[0]['emotion'][emotion]
                motivation = emotion_to_motivation(emotion)

                st.write("Detected Emotion:", emotion)
                st.write("Motivation Level:", color_motivation(motivation))
                st.write(f"Confidence: {confidence:.2f}%")
            except:
                st.write("No face detected in image.")

# ----------------- TEXT -----------------
elif app_mode == "Text":
    st.header("Text Motivation Detection")
    user_text = st.text_area("Enter text here")
    if user_text:
        with st.spinner("Analyzing text..."):
            time.sleep(0.3)
            result_text = text_classifier(user_text)[0]
            label = result_text['label']
            confidence = result_text['score']*100
            motivation = "HIGH" if label=="POSITIVE" else "LOW"

            st.write("Text Sentiment:", label)
            st.write("Motivation Level:", color_motivation(motivation))
            st.write(f"Confidence: {confidence:.2f}%")