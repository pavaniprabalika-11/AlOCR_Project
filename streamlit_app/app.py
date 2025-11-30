import os
from datetime import datetime

import streamlit as st
import pandas as pd
from ultralytics import YOLO
from PIL import Image

# ------------------------------------------------
# 🔧 PAGE CONFIG
# ------------------------------------------------
st.set_page_config(
    page_title="Aadhar Document Verification",
    layout="wide",
    page_icon="🧬",
)

# ------------------------------------------------
# 🎨 LOAD CUSTOM CSS
# ------------------------------------------------
BASE_DIR = os.path.dirname(__file__)
CSS_PATH = os.path.join(BASE_DIR, "styles.css")

with open(CSS_PATH, "r", encoding="utf-8") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ------------------------------------------------
# 🧠 LOAD MODELS
# ------------------------------------------------
MODEL_DIR = os.path.join(os.path.dirname(BASE_DIR), "models")
DETECTOR_PATH = os.path.join(MODEL_DIR, "detector.pt")
CLASSIFIER_PATH = os.path.join(MODEL_DIR, "classifier.pt")

@st.cache_resource
def load_models():
    return YOLO(DETECTOR_PATH), YOLO(CLASSIFIER_PATH)

detector, classifier = load_models()

# ------------------------------------------------
# 📦 SESSION STATE
# ------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []
if "last_report" not in st.session_state:
    st.session_state.last_report = None

# ------------------------------------------------
# 🧭 SIDEBAR
# ------------------------------------------------
with st.sidebar:
    st.markdown("<h2 class='sidebar-title'>🧬 Aadhar Verifier</h2>", unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        ["Verify Document", "About"],
        index=0,
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("""
        <div class="sidebar-footer">
            <p>Model: YOLOv8 (Detect + Classify)</p>
            <p class="tiny">Designed by Pavani ✨</p>
        </div>
    """, unsafe_allow_html=True)

# ------------------------------------------------
# 🔍 VERIFY PAGE
# ------------------------------------------------
if page == "Verify Document":

    st.markdown("""
        <div class="floating-orb orb-1"></div>
        <div class="floating-orb orb-2"></div>
        <div class="floating-orb orb-3"></div>
    """, unsafe_allow_html=True)

    st.markdown("<h1 class='title'>Aadhar Document Verification System</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Upload your Aadhar image to check if it is REAL or FAKE</p>", unsafe_allow_html=True)

    st.markdown("<div class='neon-card upload-card'>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Upload Aadhar Image",
        type=["jpg", "jpeg", "png"],
        key="aadhar_upload",
    )
    st.markdown("</div>", unsafe_allow_html=True)

    if uploaded_file:

        img = Image.open(uploaded_file).convert("RGB")

        col_img, col_info = st.columns([2.2, 1])

        with col_img:
            st.markdown("<p class='section-title'>📷 Uploaded Image</p>", unsafe_allow_html=True)
            st.image(img, use_container_width=True)

        with col_info:
            st.markdown("<p class='section-title'>🧾 File Details</p>", unsafe_allow_html=True)
            st.markdown(f"""
                <div class="neon-card small-card">
                    <p><b>Name:</b> {uploaded_file.name}</p>
                    <p><b>Type:</b> {uploaded_file.type}</p>
                    <p><b>Size:</b> {round(len(uploaded_file.getvalue())/1024, 1)} KB</p>
                </div>
            """, unsafe_allow_html=True)

            verify_btn = st.button("🔍 Verify Document", use_container_width=True)

        if verify_btn:

            progress = st.progress(0, text="Starting verification…")
            status_placeholder = st.empty()

            # Save temp file
            tmp_dir = os.path.join(BASE_DIR, "tmp")
            os.makedirs(tmp_dir, exist_ok=True)
            tmp_path = os.path.join(tmp_dir, "temp_upload.jpg")
            img.save(tmp_path)

            # ----------------------------------------------------
            # 1️⃣ DETECTOR — CORRECTED & FULLY WORKING
            # ----------------------------------------------------
            status_placeholder.info("Step 1/2 – Detecting Aadhar card…")
            progress.progress(35)

            det_results = detector(tmp_path)[0]  # CORRECT CALL
            boxes = det_results.boxes

            has_aadhar = boxes is not None and len(boxes) > 0

            # Save bounding box output image
            det_image_path = os.path.join(tmp_dir, "detected.jpg")
            det_results.save(filename=det_image_path)

            if os.path.exists(det_image_path):
                st.markdown("<p class='section-title'>🔍 Detector Output</p>", unsafe_allow_html=True)
                st.image(det_image_path, caption="Detected Aadhar", use_container_width=True)

            # ----------------------------------------------------
            # 2️⃣ CLASSIFIER
            # ----------------------------------------------------
            status_placeholder.info("Step 2/2 – Classifying document…")
            progress.progress(70)

            class_results = classifier(tmp_path, verbose=False)[0]
            probs = class_results.probs.data.cpu().numpy()

            fake_prob = float(probs[0])
            real_prob = float(probs[1])

            label = "REAL" if real_prob > fake_prob else "FAKE"
            confidence = max(real_prob, fake_prob)

            progress.progress(100)
            status_placeholder.success("Verification complete!")

            # Clean temp
            try: os.remove(tmp_path)
            except: pass

            # ----------------------------------------------------
            # RESULT CARD
            # ----------------------------------------------------
            st.markdown("<h2 class='result-title'>Verification Result</h2>", unsafe_allow_html=True)

            if not has_aadhar and confidence < 0.80:
                st.markdown("""
                    <div class="neon-card warn-card">
                        <p>⚠ Aadhar not clearly detected.<br>Please upload a clearer image.</p>
                    </div>
                """, unsafe_allow_html=True)

            if label == "REAL":
                st.markdown(f"""
                    <div class="result-card success-neon">
                        <h3>✔ REAL DOCUMENT</h3>
                        <p>Confidence: <span>{real_prob:.2f}</span></p>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div class="result-card error-neon">
                        <h3>✘ FAKE DOCUMENT</h3>
                        <p>Confidence: <span>{fake_prob:.2f}</span></p>
                    </div>
                """, unsafe_allow_html=True)

            # ----------------------------------------------------
            # HISTORY UPDATE
            # ----------------------------------------------------
            record = {
                "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "File": uploaded_file.name,
                "Detector Found Aadhar": "Yes" if has_aadhar else "No",
                "Prediction": label,
                "Confidence": round(confidence, 3),
            }

            st.session_state.history.append(record)

    # ----------------------------------------------------
    # HISTORY TABLE
    # ----------------------------------------------------
    if st.session_state.history:
        st.markdown("<h2 class='history-title'>Verification History (this session)</h2>", unsafe_allow_html=True)

        df = pd.DataFrame(st.session_state.history)
        st.dataframe(df, use_container_width=True)

# ------------------------------------------------
# ℹ️ ABOUT PAGE
# ------------------------------------------------
else:
    st.markdown("<h1 class='title'>About This System</h1>", unsafe_allow_html=True)
    st.markdown("""
        <div class="neon-card about-card">
            <p>This AI-based system verifies Aadhar documents using:</p>
            <ul>
                <li>YOLOv8 Detector</li>
                <li>YOLOv8 Classifier</li>
                <li>Cyber Neon UI</li>
            </ul>
            <p>UID ✨</p>
        </div>
    """, unsafe_allow_html=True)
