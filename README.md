# Legal Metrology Vision Backend API 🧠⚡

[![Backend CI](https://github.com/AllikaSaiHarsha/legal-metrology-backend/actions/workflows/ci.yml/badge.svg)](https://github.com/AllikaSaiHarsha/legal-metrology-backend/actions/workflows/ci.yml)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Google Gemini](https://img.shields.io/badge/Google_Gemini-4285F4?style=flat-square&logo=google&logoColor=white)](https://ai.google.dev)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](LICENSE)

> 📦 **Legal Metrology Vision Ecosystem**  
> 📱 [Mobile Field Scanner App](https://github.com/AllikaSaiHarsha/legal-metrology-app) • 💻 [Web Command Center](https://github.com/AllikaSaiHarsha/legal-metrology-web) • 🧠 [AI Vision Backend](https://github.com/AllikaSaiHarsha/legal-metrology-backend)

FastAPI microservice powering multimodal AI vision inspection for the **Legal Metrology (Packaged Commodities) Rules, 2011 (Rule 6)**.

---

## 🚀 True Architecture & Tech Stack

Contrary to legacy templates that relied on local OpenCV and Tesseract OCR, this service runs on **Google Gemini Multimodal Vision AI**. Local OCR engines fail frequently on complex retail packaging (curved bottles, reflective foils, stylized fonts) and cause Out-Of-Memory (OOM) crashes on cloud containers. Gemini Vision performs zero-shot optical text extraction, semantic validation, and 2D spatial bounding box localization in a single pass.

* **API Framework:** [FastAPI](https://fastapi.tiangolo.com/) with Uvicorn ASGI
* **Multimodal AI Engine:** **Google Gemini Vision API** (`gemini-3.5-flash`, `gemini-3.5-flash-lite`, `gemini-2.5-flash`) via official `google-genai` SDK
* **Image Processing & Normalization:** Python Pillow (PIL)
* **Key & Quota Management:** Automatic cascading multi-key rotation to eliminate `429 RESOURCE_EXHAUSTED` errors
* **Cloud Deployment:** Render (Web Service) / Docker / Hugging Face Spaces

---

## 📂 Modular Architecture

```
legal-metrology-backend/
├── app/
│   ├── api/
│   │   └── v1/
│   │       ├── endpoints/
│   │       │   ├── health.py        # /api/v1/health status
│   │       │   └── analyze.py       # /api/v1/analyze multimodal label analysis
│   │       └── router.py            # API v1 router aggregation
│   ├── core/
│   │   └── config.py                # App configuration, models, & key manager
│   ├── schemas/
│   │   └── compliance.py            # Pydantic data schemas & response models
│   ├── services/
│   │   ├── gemini_service.py        # Gemini client, prompt engineering, key rotation
│   │   └── image_service.py         # Image dimensions, storage, & coordinate parsing
│   └── main.py                      # FastAPI application factory & middleware
├── tests/
│   └── test_api.py                  # Pytest test suite for endpoint health
├── .github/workflows/
│   └── ci.yml                       # Automated GitHub Actions CI workflow
├── .env.example                     # Environment configuration template
├── Dockerfile                       # Production container setup
├── requirements.txt                 # Application dependencies
└── main.py                          # Minimal root entry point (uvicorn main:app)
```

---

## ✨ Core Capabilities

* **Multimodal Label Analysis:** Extracts all mandatory Rule 6 declarations (MRP, USP, Net Weight, Manufacturer Details, Expiry Date, Consumer Care, Country of Origin).
* **Spatial Bounding Box Localization:** Directly generates normalized 2D coordinates `[ymin, xmin, ymax, xmax]` for each detected declaration for visual overlays on the frontend.
* **Low-Memory Footprint:** 100% cloud-native inference without heavy C++ system binary dependencies, ensuring lightning-fast startup and stable deployment on Render.
* **Synchronous Web Mirroring:** Automatically mirrors uploaded evidence images to the connected Next.js dashboard uploads directory.

---

## 🛠️ Getting Started

### Prerequisites
* Python 3.11+
* Google Gemini API Key from [Google AI Studio](https://aistudio.google.com/)

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
   ```bash
   cp .env.example .env
   # Edit .env with your actual GEMINI_API_KEY
   ```

5. **Start the API server:**
   ```bash
   python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```

6. **Interactive Documentation:**
   * Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)
   * Health Check: [http://localhost:8000/api/v1/health](http://localhost:8000/api/v1/health)

7. **Run Tests:**
   ```bash
   pytest tests/
   ```

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
