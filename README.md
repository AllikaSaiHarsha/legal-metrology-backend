# Legal Metrology Compliance - Backend API

This repository contains the backend service for the Legal Metrology Compliance verification system. It exposes a robust REST API built with FastAPI that processes image payloads, applies computer vision techniques, and extracts text via OCR to validate product packaging against statutory metrology regulations.

---

## 🚀 Tech Stack

* **API Framework:** [FastAPI](https://fastapi.tiangolo.com/) (Python)
* **Server:** Uvicorn
* **Computer Vision:** OpenCV
* **Text Extraction:** Tesseract OCR
* **Data Processing:** Pandas / NumPy (as applicable for data structuring)

---

## ✨ Core Capabilities

* **Image Processing Engine:** Accepts image uploads of product labels and applies OpenCV preprocessing (noise reduction, grayscale conversion, and contrast thresholding) to optimize them for text extraction.
* **Optical Character Recognition (OCR):** Integrates Tesseract OCR to scan processed images for critical required fields such as MRP, Net Quantity, Manufacturer Details, and Date of Manufacture.
* **Compliance Rules Engine:** Evaluates the extracted text against predefined Legal Metrology guidelines, returning structured JSON responses highlighting compliance status and missing data.
* **Fast & Asynchronous:** Built on FastAPI to handle concurrent API requests efficiently.

---

## 🛠️ Getting Started

### Prerequisites
* Python 3.9+
* Tesseract OCR installed locally (Ensure the `tesseract` executable is added to your system's PATH).

### Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/AllikaSaiHarsha/legal-metrology-backend.git](https://github.com/AllikaSaiHarsha/legal-metrology-backend.git)
   cd legal-metrology-backend
