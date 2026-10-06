# Legal Metrology Vision Backend API 🧠⚡

FastAPI microservice powering multimodal AI vision inspection for the **Legal Metrology (Packaged Commodities) Rules, 2011 (Rule 6)**.

---

## 🚀 True Architecture & Tech Stack

Contrary to legacy templates that relied on local OpenCV and Tesseract OCR, this service runs on **Google Gemini Multimodal Vision AI**. Local OCR engines fail frequently on complex retail packaging (curved bottles, reflective foils, stylized fonts) and cause Out-Of-Memory (OOM) crashes on cloud containers. Gemini Vision performs zero-shot optical text extraction, semantic validation, and 2D spatial bounding box localization in a single pass.

* **API Framework:** [FastAPI](https://fastapi.tiangolo.com/) (Python 3.10+) with Uvicorn ASGI
* **Multimodal AI Engine:** **Google Gemini Vision API** (`gemini-2.5-flash`, `gemini-3.5-flash`) via `google-genai` SDK
* **Image Processing & Normalization:** Python Pillow (PIL)
* **Key & Quota Management:** Automatic cascading multi-key rotation to eliminate `429 RESOURCE_EXHAUSTED` errors
* **Cloud Deployment:** Render (Web Service) / Docker

---

## ✨ Core Capabilities

* **Multimodal Label Analysis:** Extracts all mandatory Rule 6 declarations (MRP, USP, Net Weight, Manufacturer Details, Expiry Date, Consumer Care, Country of Origin).
* **Spatial Bounding Box Localization:** Directly generates normalized 2D coordinates `[ymin, xmin, ymax, xmax]` for each detected declaration for visual overlays on the frontend.
* **Low-Memory Footprint:** 100% cloud-native inference without heavy C++ system binary dependencies (`tesseract-ocr`, `libgl1`), ensuring lightning-fast startup and stable deployment on Render.
* **Synchronous Web Mirroring:** Automatically mirrors uploaded evidence images to the connected Next.js dashboard uploads directory.

---

## 🛠️ Getting Started

### Prerequisites
* Python 3.10+
* Google Gemini API Key(s) from [Google AI Studio](https://aistudio.google.com/)

### Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/AllikaSaiHarsha/legal-metrology-backend.git
   cd legal-metrology-backend
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   # Windows:
   .\venv\Scripts\activate
   # Linux / macOS:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Environment Variables:**
   Create a `.env` file:
   ```env
   GEMINI_API_KEYS="your_api_key_1,your_api_key_2"
   ```

5. **Start the API server:**
   ```bash
   python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```

6. **Interactive Documentation:**
   * Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)
   * Health Check: [http://localhost:8000/api/v1/health](http://localhost:8000/api/v1/health)
