🌐 AI Powered Fraud Management System for UID Aadhaar

An end-to-end Aadhaar Document Verification system using YOLO-based detection, classification, OCR extraction, and analytics dashboard.
The system verifies Aadhaar authenticity, detects fraudulent documents, extracts Aadhaar details, supports bulk verification, and maintains audit history.

🔍 Overview

This project automates Aadhaar document verification using Computer Vision + OCR + Fraud Classification.
The system validates document authenticity, detects fake/misprinted cards, extracts Aadhaar details, and visualizes fraud patterns — making it suitable for KYC, banking, telecom, government onboarding systems, verification desks & automated kiosks.

| Feature                               | Status | Description                         |
| ------------------------------------- | ------ | ----------------------------------- |
| 🔹 Real-Time Aadhaar Verification     | ✔️     | Upload → Detect → Verify → Classify |
| 🔹 Deep Learning Fraud Classification | ✔️     | Detects *Real vs Fake* using YOLOv8 |
| 🔹 OCR Extraction for Authentic Cards | ✔️     | Reads Aadhaar Number + Name         |
| 🔹 Bulk Verification Mode             | ✔️     | 10+ Cards → One-Click Processing    |
| 🔹 Analytics Dashboard                | ✔️     | Charts + Metrics + Trends           |
| 🔹 Verification History + Export      | ✔️     | Save & Download CSV Logs            |

🧠 System Architecture

User → Upload Aadhaar → YOLO Detector → Image Crop → 
      → YOLO Classifier (Real/Fake)
      → OCR (If Real) → Aadhaar Number + Name Extraction
      → Dashboard + Logs Storage




📁 Project Structure



<img width="646" height="500" alt="Screenshot 2025-12-06 190001" src="https://github.com/user-attachments/assets/c5d083c2-8744-4dfc-a807-f414060ac7b6" />


📸 Preview UI


<img width="1907" height="773" alt="Screenshot 2025-12-06 184744" src="https://github.com/user-attachments/assets/83d7b763-e1d5-4f00-9a24-cd7299a1cf4f" />


<img width="1896" height="795" alt="Screenshot 2025-12-06 185035" src="https://github.com/user-attachments/assets/6bf0b0a9-45ce-4022-93f7-2d550d3dbc97" />


<img width="1896" height="837" alt="Screenshot 2025-12-06 185121" src="https://github.com/user-attachments/assets/05618e6d-a10f-4ff6-9678-21930f34e4cb" />


<img width="1892" height="841" alt="Screenshot 2025-12-06 185150" src="https://github.com/user-attachments/assets/e342ae78-4f6d-41d2-94a9-9b65b11696df" />


<img width="1877" height="742" alt="Screenshot 2025-12-06 185211" src="https://github.com/user-attachments/assets/4ee5c9b4-41f6-4f14-800b-833cc566167a" />


🏗 Installation & Run

git clone https://github.com/pavaniprabalika-11/AlOCR_Project.git
cd AlOCR_Project

python -m venv venv
.\venv\Scripts\Activate.ps1    # Windows


pip install -r requirements.txt
streamlit run streamlit_app/app.py


🔥 Future Roadmap (Optional to Implement)


Feature	Benefit
Face-Match With Aadhaar Photo	KYC Person-to-Document Match
QR-Code Extraction + UID Verification	UID Validation Against DB
Mobile Upload + API Access	Government/Bank Integration
Cloud Deployment on Azure/AWS	Scalable Real-World Rollout

DEPLOYMENT LINK
https://alocrproject-7ad34e4ojdveq6xhnw4fsx.streamlit.app/


⭐ Contribution

Contributions, feature requests & issue reports are always welcome!
Start by starring ⭐ the repo to support continued development.










