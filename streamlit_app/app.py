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

# Extra tiny CSS (does NOT change your file) just to ensure tabs are centered
st.markdown(
    """
    <style>
    .stTabs [data-baseweb="tab-list"]{
        justify-content:center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

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
    st.error("❌ Model files missing — place detector.pt and classifier.pt inside the 'models' folder.")
    st.stop()

detector, classifier, ocr_reader = load_models_and_ocr()

# ==========================================================
# 📦 SESSION STATE
# ==========================================================
if "history" not in st.session_state:
    st.session_state.history = []  # list of dicts


# ==========================================================
# 🔎 OCR HELPERS (Aadhaar Number + Name)
# ==========================================================
def ocr_extract_texts(pil_image):
    """Run EasyOCR on a PIL image and return list of text strings."""
    img_cv = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    results = ocr_reader.readtext(img_cv)
    texts = []
    for item in results:
        # item format: [bbox, text, confidence]
        if len(item) >= 2:
            texts.append(str(item[1]))
    return texts


def extract_aadhaar_number(texts):
    """
    Find a 12-digit Aadhaar-style number in OCR result.
    Returns formatted 'XXXX XXXX XXXX' or 'N/A'.
    """
    if not texts:
        return "N/A"
    joined = " ".join(texts)
    # remove hyphens etc.
    cleaned = joined.replace("-", " ")
    match = re.search(r'\b\d{4}\s?\d{4}\s?\d{4}\b', cleaned)
    if not match:
        return "N/A"
    num = re.sub(r'\s+', "", match.group(0))
    if len(num) != 12:
        return "N/A"
    return f"{num[0:4]} {num[4:8]} {num[8:12]}"


def extract_name(texts):
    """
    Very simple heuristic to guess holder name:
    pick the first text line that looks like a human name.
    """
    if not texts:
        return "N/A"

    EXCLUDE = [
        "GOVERNMENT", "GOVT", "INDIA", "UNION", "AUTHORITY",
        "UNIQUE", "IDENTIFICATION", "AADHAAR", "AADHAR", "CARD",
        "YEAR", "BIRTH", "YOB", "DOB", "FEMALE", "MALE",
        "ADDRESS", "OF", "REPUBLIC"
    ]

    for t in texts:
        # keep letters and spaces only
        clean = re.sub(r'[^A-Za-z\s]', ' ', t).strip()
        if len(clean.split()) < 2:
            continue
        upper = clean.upper()
        if any(word in upper for word in EXCLUDE):
            continue
        # looks like a candidate name
        return clean.title()

    return "N/A"


# ==========================================================
# 🧠 CORE VERIFICATION FUNCTION
# ==========================================================
def run_verification(uploaded_file, source="Single"):
    """
    Runs detector + classifier + (for REAL) OCR on one uploaded image.
    Returns:
      record (dict for history),
      preview_path (str),
      base_pil (PIL.Image),
      label (REAL/FAKE),
      has_aadhar (bool),
      elapsed_seconds (float),
      aadhaar_number (str),
      holder_name (str)
    """
    start_time = time.time()

    img = Image.open(uploaded_file).convert("RGB")
    tmp_dir = os.path.join(BASE_DIR, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    ts = int(time.time() * 1000)
    tmp_path = os.path.join(tmp_dir, f"temp_{ts}.jpg")
    det_image_path = os.path.join(tmp_dir, f"detected_{ts}.jpg")
    img.save(tmp_path)

    # 1️⃣ DETECTION
    det_results = detector(tmp_path)[0]
    boxes = det_results.boxes
    has_aadhar = boxes is not None and len(boxes) > 0
    det_results.save(filename=det_image_path)

    # 2️⃣ CLASSIFICATION
    cls_results = classifier(tmp_path, verbose=False)[0]
    probs = cls_results.probs.data.cpu().numpy()
    fake_prob = float(probs[0])
    real_prob = float(probs[1])
    label = "REAL" if real_prob > fake_prob else "FAKE"
    confidence = max(real_prob, fake_prob)

    # 3️⃣ OCR (only if model thinks REAL)
    aadhaar_number = "N/A"
    holder_name = "N/A"
    if label == "REAL":
        try:
            texts = ocr_extract_texts(img)
            aadhaar_number = extract_aadhaar_number(texts)
            holder_name = extract_name(texts)
        except Exception:
            # don't break pipeline if OCR fails
            aadhaar_number = "N/A"
            holder_name = "N/A"

    elapsed_seconds = time.time() - start_time

    # Clean temp input (keep detected for preview)
    try:
        os.remove(tmp_path)
    except Exception:
        pass

    record = {
        "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "File Name": uploaded_file.name,
        "Detector Flag": "Detected" if has_aadhar else "Not detected",
        "Decision": label,
        "Model Confidence": float(f"{confidence:.4f}"),
        "Processing Time (s)": float(f"{elapsed_seconds:.4f}"),
        "Aadhaar Number": aadhaar_number,
        "Name": holder_name,
        "Source": source,
    }

    return (
        record,
        det_image_path if os.path.exists(det_image_path) else None,
        img,
        label,
        has_aadhar,
        elapsed_seconds,
        aadhaar_number,
        holder_name,
    )


# ==========================================================
# 🏷 MAIN TITLE
# ==========================================================
st.markdown("<h1 class='main-title'>Aadhar Document Verification System</h1>", unsafe_allow_html=True)

# ==========================================================
# 📑 TABS (TOP NAVIGATION)
# ==========================================================
tab_verify, tab_dashboard, tab_bulk, tab_history, tab_about = st.tabs(
    ["🔍 Verify Document", "📊 Dashboard", "📦 Bulk Upload", "📂 History", "ℹ About"]
)

# ==========================================================
# TAB 1 — VERIFY DOCUMENT
# ==========================================================
with tab_verify:
    st.markdown("### Upload and verify Aadhar document")

    uploaded_file = st.file_uploader(
        "Upload Aadhar Image",
        type=["jpg", "jpeg", "png"],
        key="aadhar_upload_single",
    )

    verify_btn = st.button("Run Verification", type="primary", key="btn_single_verify")

    if uploaded_file and verify_btn:
        (
            record,
            preview_path,
            base_img,
            label,
            has_aadhar,
            elapsed_seconds,
            aadhaar_number,
            holder_name,
        ) = run_verification(uploaded_file, source="Single")

        # Save to global history
        st.session_state.history.append(record)

        st.markdown("---")
        st.markdown("### Result")

        # ➤ Image preview
        if preview_path is not None:
            st.image(preview_path, caption="Detected Aadhar region", width=420)
        else:
            st.image(base_img, caption="Uploaded image", width=420)

        # ➤ Circle Badge
        if label == "REAL":
            st.markdown("<div class='result-circle real-badge'>REAL AADHAR</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='result-circle fake-badge'>FAKE AADHAR</div>", unsafe_allow_html=True)

        # ➤ Summary Box
        summary_html = f"""
        <div style='margin:15px auto;max-width:420px;
                    background:#0f223a;border-radius:10px;
                    padding:12px 18px;border:1px solid #ffffff22;'>
            <p><b>File:</b> {record['File Name']}</p>
            <p><b>Detector:</b> {record['Detector Flag']}</p>
            <p><b>Decision:</b> {record['Decision']} Aadhar (AI-based)</p>
            <p><b>Processing Time:</b> {record['Processing Time (s)']:.2f} seconds</p>
        </div>
        """
        st.markdown(summary_html, unsafe_allow_html=True)

        # ➤ Extracted Aadhaar Info (only for REAL)
        if record["Decision"] == "REAL":
            info_html = f"""
            <div style='margin:10px auto;max-width:420px;
                        background:#120021cc;border-radius:10px;
                        padding:10px 16px;border:1px solid #ff7bff44;'>
                <p><b>Aadhaar Number:</b> {aadhaar_number}</p>
                <p><b>Name:</b> {holder_name}</p>
            </div>
            """
            st.markdown(info_html, unsafe_allow_html=True)

    elif verify_btn and not uploaded_file:
        st.warning("Please upload an Aadhar image before running verification.")


# ==========================================================
# TAB 2 — DASHBOARD
# ==========================================================
with tab_dashboard:
    st.markdown("### System dashboard and analytics")

    if not st.session_state.history:
        st.info("No verifications yet. Run at least one check in Verify Document or Bulk Upload.")
    else:
        df = pd.DataFrame(st.session_state.history)

        # --- basic fields safety ---
        if "Source" not in df.columns:
            df["Source"] = "Single"
        if "Aadhaar Number" not in df.columns:
            df["Aadhaar Number"] = "N/A"

        # ----------------- BASIC NUMBERS -----------------
        total_docs = len(df)
        real_docs = (df["Decision"] == "REAL").sum()
        fake_docs = (df["Decision"] == "FAKE").sum()
        real_rate = (real_docs / total_docs) * 100 if total_docs > 0 else 0

        # Processing time stats
        if "Processing Time (s)" in df.columns:
            proc_times = df["Processing Time (s)"].astype(float)
            avg_t = proc_times.mean()
            min_t = proc_times.min()
            max_t = proc_times.max()
        else:
            avg_t = min_t = max_t = 0.0

        # Unique Aadhaar stats (only well-formed)
        valid_ids = df["Aadhaar Number"].dropna()
        valid_ids = valid_ids[valid_ids != "N/A"]
        unique_ids = valid_ids.nunique()

        # Suspicious usage: Aadhaar appearing >= 3 times
        suspicious_count = 0
        if not valid_ids.empty:
            counts = valid_ids.value_counts()
            suspicious_ids = counts[counts >= 3]
            suspicious_count = len(suspicious_ids)

        # Single vs Bulk summary
        single_total = (df["Source"] == "Single").sum()
        bulk_total = (df["Source"] == "Bulk").sum()

        # ----------------- KPI CARDS (DOCS + TIME) -----------------
        st.markdown(
            f"""
            <div class='stats-container'>
                <div class='card'><h3>Total Documents</h3><p>{total_docs}</p></div>
                <div class='card'><h3>Real Documents</h3><p>{real_docs}</p></div>
                <div class='card'><h3>Fake Documents</h3><p>{fake_docs}</p></div>
                <div class='card'><h3>Real Verification Rate</h3><p>{real_rate:.1f}%</p></div>
                <div class='card'><h3>Avg Processing Time</h3><p>{avg_t:.2f}s</p></div>
                <div class='card'><h3>Fastest</h3><p>{min_t:.2f}s</p></div>
                <div class='card'><h3>Slowest</h3><p>{max_t:.2f}s</p></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ----------------- SECOND ROW: BULK / UNIQUE / SUSPICIOUS -----------------
        st.markdown(
            f"""
            <div class='stats-container'>
                <div class='card'><h3>Single Verifications</h3><p>{single_total}</p></div>
                <div class='card'><h3>Bulk Verifications</h3><p>{bulk_total}</p></div>
                <div class='card'><h3>Unique Aadhaar Numbers</h3><p>{unique_ids}</p></div>
                <div class='card'><h3>Suspicious Aadhaar IDs (≥3 uses)</h3><p>{suspicious_count}</p></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("---")

        # ----------------- REAL vs FAKE BAR -----------------
        st.markdown("#### Real vs Fake distribution")
        fig1, ax1 = plt.subplots(figsize=(4, 3))
        ax1.bar(["Real", "Fake"], [real_docs, fake_docs], color=["#00D97E", "#FF4D4D"])
        ax1.set_facecolor("#0f223a")
        fig1.patch.set_facecolor("#0f223a")
        st.pyplot(fig1, use_container_width=False)

        # ----------------- DAILY VOLUME -----------------
        st.markdown("#### Verification activity over time")
        df_time = df.copy()
        df_time["Time"] = pd.to_datetime(df_time["Time"])
        df_time["Date"] = df_time["Time"].dt.date
        daily_counts = df_time.groupby("Date").size()
        fig2, ax2 = plt.subplots(figsize=(4, 3))
        ax2.plot(daily_counts.index, daily_counts.values, marker="o")
        ax2.set_facecolor("#0f223a")
        fig2.patch.set_facecolor("#0f223a")
        plt.setp(ax2.get_xticklabels(), rotation=45, ha="right")
        st.pyplot(fig2, use_container_width=False)

        st.markdown("---")

        # ----------------- SUSPICIOUS AADHAAR TABLE -----------------
        if suspicious_count > 0:
            st.markdown("#### Suspicious Aadhaar usage (same ID used multiple times)")
            sus_df = df[df["Aadhaar Number"].isin(suspicious_ids.index)]
            show_cols = ["Time", "File Name", "Aadhaar Number", "Decision", "Source"]
            st.dataframe(sus_df[show_cols], use_container_width=True)
        else:
            st.success("No Aadhaar numbers with unusually frequent usage detected so far.")

        st.markdown("---")

        # ----------------- DOWNLOAD FULL DASHBOARD DATA -----------------
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇ Download full verification history (CSV)",
            csv_bytes,
            "aadhar_verification_history.csv",
            "text/csv",
        )

# ==========================================================
# TAB 3 — BULK UPLOAD
# ==========================================================
with tab_bulk:
    st.markdown("### Bulk verification (multiple documents)")
    bulk_files = st.file_uploader(
        "Upload multiple Aadhar images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key="bulk_upload",
    )

    run_bulk = st.button("Run Bulk Verification", type="primary", key="btn_bulk")

    if run_bulk:
        if not bulk_files:
            st.warning("Please upload at least one image for bulk verification.")
        else:
            results = []
            for file in bulk_files:
                try:
                    (
                        record,
                        _preview_path,
                        _base_img,
                        _label,
                        _has_aadhar,
                        _elapsed,
                        _aadhaar_number,
                        _holder_name,
                    ) = run_verification(file, source="Bulk")

                    st.session_state.history.append(record)
                    results.append(record)
                except Exception as e:
                    # For safety, you can also log/print e if needed
                    continue

            if results:
                st.success(f"Bulk verification completed for {len(results)} document(s).")
                df_bulk = pd.DataFrame(results)
                show_cols = ["Time", "File Name", "Decision", "Detector Flag", "Aadhaar Number", "Name"]
                st.dataframe(df_bulk[show_cols], use_container_width=True)

# ==========================================================
# TAB 4 — HISTORY
# ==========================================================
with tab_history:
    st.markdown("### Verification history")

    if not st.session_state.history:
        st.info("No verification history yet.")
    else:
        df = pd.DataFrame(st.session_state.history)
        if "Source" not in df.columns:
            df["Source"] = "Single"
        if "Aadhaar Number" not in df.columns:
            df["Aadhaar Number"] = "N/A"
        if "Name" not in df.columns:
            df["Name"] = "N/A"

        display_cols = ["Time", "File Name", "Detector Flag", "Decision", "Aadhaar Number", "Name", "Source"]
        if "Processing Time (s)" in df.columns:
            display_cols.append("Processing Time (s)")

        st.dataframe(df[display_cols], use_container_width=True)

        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇ Download full history as CSV",
            csv_bytes,
            "aadhar_verification_history.csv",
            "text/csv",
        )

# ==========================================================
# TAB 5 — ABOUT
# ==========================================================
with tab_about:
    st.markdown("### About this system")
    st.markdown(
        """
        This application is a **prototype for Aadhar document verification**.

        It provides:
        - **YOLOv8 Detector** to locate the Aadhar card region in the uploaded document  
        - **YOLOv8 Classifier** to determine whether the card is likely **REAL** or **FAKE**  
        - **Aadhaar number & name extraction** from REAL documents using OCR (for audit and linkage)  
        - A **Dashboard** showing document volumes, real/fake ratios, processing times, and suspicious usage patterns  
        - A **Bulk upload** workflow for batch verifications  
        - A **History log** with exportable CSV for downstream compliance, KYC, or reporting tools  

        The focus is to demonstrate how computer-vision-based models can support
        **KYC, identity validation, and fraud monitoring** in a structured and auditable way.
        """
    )                     