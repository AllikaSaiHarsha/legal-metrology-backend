from typing import Optional
from fastapi import FastAPI, File, UploadFile, Request, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
import logging
import os
import io
import json
import uuid
import base64
from google import genai
from google.genai import types

load_dotenv()

app = FastAPI(title="Legal Metrology Vision API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Ensure uploads directory exists
os.makedirs("uploads", exist_ok=True)
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

gemini_client = None
if os.getenv("GEMINI_API_KEY"):
    try:
        gemini_client = genai.Client()
        logger.info("Gemini AI loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize Gemini: {e}")

@app.get("/api/v1/health")
async def health_check():
    return {"status": "ok"}

@app.post("/api/v1/analyze")
async def analyze_image(request: Request, file: UploadFile = File(...)):
    if not gemini_client:
        raise HTTPException(status_code=500, detail="Gemini API Key missing")

    try:
        contents = await file.read()
        
        # Save image for UI rendering
        ext = ".jpg"
        if file.filename:
            _, file_ext = os.path.splitext(file.filename)
            if file_ext:
                ext = file_ext
        
        file_id = str(uuid.uuid4())
        filename = f"{file_id}{ext}"
        filepath = os.path.join("uploads", filename)
        
        with open(filepath, "wb") as f:
            f.write(contents)

        base_url = str(request.base_url).rstrip("/")
        image_url = f"{base_url}/uploads/{filename}"
        
        # Encode for Gemini
        encoded_image = base64.b64encode(contents).decode("utf-8")
        
        prompt = """
        You are an expert Indian Legal Metrology package inspector. 
        Analyze the packaging label and extract the following details precisely as JSON.
        We need to check for compliance with Rule 6 of Legal Metrology (Packaged Commodities) Rules, 2011.

        Return valid JSON ONLY. No markdown formatting blocks like ```json.
        {
          "product_name": "Extracted brand and product name",
          "manufacturer": "Extracted manufacturer details",
          "detections": [
            {
              "category": "MRP",
              "label": "Extracted MRP text exactly as shown",
              "status": "Passed" // Passed if clearly readable, Failed if missing/obscured
            },
            {
              "category": "Net Weight",
              "label": "Extracted net weight exactly as shown",
              "status": "Passed"
            },
            {
              "category": "Manufacture Date",
              "label": "Extracted date exactly as shown",
              "status": "Passed"
            },
            {
              "category": "Expire Date",
              "label": "Extracted expiry date exactly as shown",
              "status": "Passed"
            },
            {
              "category": "Customer Care",
              "label": "Extracted contact info exactly as shown",
              "status": "Passed"
            }
          ]
        }
        """

        logger.info("Sending to Gemini API...")
        response = gemini_client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[
                types.Part.from_bytes(data=contents, mime_type="image/jpeg"),
                prompt
            ]
        )

        response_text = response.text.replace("```json", "").replace("```", "").strip()
        data = json.loads(response_text)
        
        # Inject standard required metadata
        data["filename"] = filename
        data["original_width"] = 800
        data["original_height"] = 800
        data["image_url"] = image_url
        
        # Provide default fake boxes for the frontend bounding box viewer since Gemini text mode doesn't return exact pixels easily
        for idx, det in enumerate(data.get("detections", [])):
            det["box"] = {
                "x": 50,
                "y": 50 + (idx * 60),
                "width": 300,
                "height": 40
            }

        return data

    except Exception as e:
        logger.error(f"Error during analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))