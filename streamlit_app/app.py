import os
import time
from datetime import datetime
import re

import streamlit as st
import pandas as pd
from ultralytics import YOLO
from PIL import Image
import matplotlib.pyplot as plt
import easyocr
import cv2
import numpy as np

# ==========================================================
# 🔧 PAGE CONFIG
# ==========================================================
st.set_page_config(
    page_title="Aadhar Document Verification",
    layout="wide",
    page_icon="🧬",
)

# ==========================================================
# 🎨 LOAD CSS
# ==========================================================
BASE_DIR = os.path.dirname(__file__)
CSS_PATH = os.path.join(BASE_DIR, "styles.css")

if os.path.exists(CSS_PATH):
    with open(CSS_PATH, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# keep tabs centered
st.markdown("""
<style>
.stTabs [data-baseweb="tab-list"]{ justify-content:center; }
</style>
""", unsafe_allow_html=True)

# ==========================================================
# 📌 MODEL + OCR LOADING
# ==========================================================
MODEL_DIR = os.path.join(os.path.dirname(BASE_DIR), "models")
DETECTOR_PATH = os.path.join(MODEL_DIR, "detector.pt")
CLASSIFIER_PATH = os.path.join(MODEL_DIR, "classifier.pt")

@st.cache_resource
def load_models_and_ocr():
    detector_model = YOLO(DETECTOR_PATH)
    classifier_model = YOLO(CLASSIFIER_PATH)
    ocr_reader = easyocr.Reader(['en'], gpu=False)
    return detector_model, classifier_model, ocr_reader

if not os.path.exists(DETECTOR_PATH) or not os.path.exists(CLASSIFIER_PATH):
    st.error("❌ Place `detector.pt` & `classifier.pt` inside models folder!")
    st.stop()

detector, classifier, ocr_reader = load_models_and_ocr()

# ==========================================================
# 🧠 OCR HELPERS
# ==========================================================
def ocr_extract(pil_image):
    img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    results = ocr_reader.readtext(img)
    return [r[1] for r in results if len(r) >= 2]

def extract_aadhaar(texts):
    joined = " ".join(texts)
    clean = joined.replace("-", " ")
    match = re.search(r'\b\d{4}\s?\d{4}\s?\d{4}\b', clean)
    if match:
        num = re.sub(r'\s+', "", match.group(0))
        return f"{num[0:4]} {num[4:8]} {num[8:12]}" if len(num) == 12 else "N/A"
    return "N/A"

def extract_name(texts):
    BAD = ["GOVERNMENT","AADHAR","AADHAAR","INDIA","YEAR","YOB","DOB","MALE","FEMALE"]
    for t in texts:
        clean = re.sub(r'[^A-Za-z\s]', ' ', t).strip()
        if len(clean.split()) >= 2 and all(i not in clean.upper() for i in BAD):
            return clean.title()
    return "N/A"

# ==========================================================
# 🧠 MAIN VERIFICATION (FIXED)
# ==========================================================
def run_verification(uploaded_file, source="Single"):

    start = time.time()
    img = Image.open(uploaded_file).convert("RGB")

    tmp = os.path.join(BASE_DIR, "tmp"); os.makedirs(tmp, exist_ok=True)
    ts = int(time.time()*1000)

    temp_path = os.path.join(tmp, f"temp_{ts}.jpg")
    det_path = os.path.join(tmp, f"det_{ts}.jpg")
    img.save(temp_path)

    # ---------------- DETECTOR ----------------
    det_out = detector(temp_path)[0]
    has_aadhar = det_out.boxes is not None and len(det_out.boxes) > 0

    annotated = det_out.plot()
    cv2.imwrite(det_path, annotated)

    # ---------------- CLASSIFIER FIXED ----------------
    img_np = np.array(img)                   # 🔥 key fix
    cls_out = classifier(img_np, verbose=False)[0]   # 🔥 not file path
    p = cls_out.probs.data.cpu().numpy()
    label = "REAL" if p[1] > p[0] else "FAKE"
    conf = float(max(p))

    # ---------------- OCR ----------------
    aadhaar = name = "N/A"
    if label == "REAL":
        try:
            t = ocr_extract(img)
            aadhaar = extract_aadhaar(t)
            name = extract_name(t)
        except:
            pass

    et = round(time.time()-start, 3)
    try: os.remove(temp_path)
    except: pass

    record = {
        "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "File Name": uploaded_file.name,
        "Detector Flag": "Detected" if has_aadhar else "No",
        "Decision": label,
        "Confidence": round(conf,4),
        "Processing Time (s)": et,
        "Aadhaar Number": aadhaar,
        "Name": name,
        "Source": source,
    }

    return record, det_path, img, label, has_aadhar, et, aadhaar, name


# ==========================================================
# 🔥 UI START
# ==========================================================
if "history" not in st.session_state: st.session_state.history=[]

st.markdown("<h1 class='main-title'>Aadhar Verification System</h1>", unsafe_allow_html=True)
tab1,tab2,tab3,tab4,tab5 = st.tabs(["🔍 Verify","📊 Dashboard","📦 Bulk","📂 History","ℹ About"])

# ==========================================================
# TAB 1 — SINGLE VERIFY
# ==========================================================
with tab1:
    up = st.file_uploader("Upload Aadhar",type=["jpg","jpeg","png"])
    if st.button("Verify") and up:

        r,p,im,l,h,t,a,n = run_verification(up)
        st.session_state.history.append(r)

        st.image(p,width=400) if p else st.image(im,width=400)
        st.success(f"Result → **{l}** ( {t}s )")

        if l=="REAL":
            st.write("📌 Aadhaar:",a)
            st.write("👤 Name:",n)

# ==========================================================
# TAB2 — DASHBOARD (unchanged from your version)
# ==========================================================
with tab2:
    if not st.session_state.history: st.info("No records yet.")
    else:
        df=pd.DataFrame(st.session_state.history)
        st.dataframe(df,use_container_width=True)

# ==========================================================
# TAB3 — BULK
# ==========================================================
with tab3:
    files=st.file_uploader("Upload multiple",accept_multiple_files=True)
    if st.button("Run Bulk"):
        if not files: st.warning("Upload images")
        else:
            out=[]
            for f in files:
                try:
                    r,*_ = run_verification(f,source="Bulk")
                    st.session_state.history.append(r); out.append(r)
                except: continue
            st.success(f"{len(out)} verified")
            st.dataframe(pd.DataFrame(out))

# ==========================================================
# TAB4 — HISTORY
# ==========================================================
with tab4:
    if st.session_state.history:
        df=pd.DataFrame(st.session_state.history)
        st.dataframe(df,use_container_width=True)
    else: st.info("No history yet")

# ==========================================================
# TAB5 — ABOUT
# ==========================================================
with tab5:
    st.write("""
Prototype for **Aadhar Verification using YOLO + OCR**
- Card detection
- Real/Fake classification
- Number & Name extraction
- Dashboard + Bulk Processing
    """)
