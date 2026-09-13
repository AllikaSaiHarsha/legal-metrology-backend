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
import time
from google import genai
from google.genai import types

from PIL import Image

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

def get_api_keys():
    # Reload .env so newly pasted keys take effect immediately
    load_dotenv(override=True)
    raw = os.getenv("GEMINI_API_KEYS") or os.getenv("GEMINI_API_KEY") or ""
    keys = [k.strip() for k in raw.replace("\n", ",").split(",") if k.strip()]
    return keys

def get_gemini_client(api_key: str):
    return genai.Client(api_key=api_key)

@app.get("/api/v1/health")
async def health_check():
    keys = get_api_keys()
    return {"status": "ok", "keys_configured": len(keys)}

@app.post("/api/v1/analyze")
async def analyze_image(request: Request, file: UploadFile = File(...)):
    api_keys = get_api_keys()
    if not api_keys:
        raise HTTPException(status_code=500, detail="Gemini API Key missing in .env (set GEMINI_API_KEY or GEMINI_API_KEYS)")

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

        # Mirror image to Next.js public/uploads if folder exists
        web_uploads = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "legal-metrology-web", "public", "uploads"))
        if os.path.exists(web_uploads):
            try:
                with open(os.path.join(web_uploads, filename), "wb") as wf:
                    wf.write(contents)
            except Exception as e:
                logger.warning(f"Could not mirror to web public/uploads: {e}")

        base_url = str(request.base_url).rstrip("/")
        image_url = f"{base_url}/uploads/{filename}"

        # Get actual image dimensions
        try:
            with Image.open(io.BytesIO(contents)) as pil_img:
                real_width, real_height = pil_img.size
        except Exception:
            real_width, real_height = 1000, 1000
        
        prompt = """
        You are an expert Indian Legal Metrology package inspector. 
        Analyze the packaging label and extract the statutory declarations required under Rule 6 of the Legal Metrology (Packaged Commodities) Rules, 2011:
        1. Product Name / Generic Name
        2. Net Quantity / Net Weight
        3. Retail Sale Price (MRP, inclusive of all taxes)
        4. Date of Manufacture / Packing / Import
        5. Expiry Date / Best Before Date (if applicable)
        6. Consumer Care / Customer Care contact details (email, phone, address)
        7. Manufacturer / Packer / Importer Name & Address
        8. Country of Origin (for imported goods)
        9. Unit Sale Price (USP)

        For EACH detection:
        - "category": Standardize to one of: "MRP", "Net Weight", "Manufacture Date", "Expire Date", "Customer Care", "Manufacturer", "Country of Origin", "Unit Sale Price", or "Product Name"
        - "label": The extracted text verbatim as printed on the package
        - "status": "Passed" if clearly legible and compliant, "Failed" if missing, non-compliant, or obscured
        - "box_2d": Accurate 2D bounding box coordinates [ymin, xmin, ymax, xmax] normalized to integer values between 0 and 1000 (where [0, 0] is top-left and [1000, 1000] is bottom-right of the image). If a declaration is missing, failed, or not visible on the packaging, set "box_2d": [0, 0, 0, 0].

        Return valid JSON ONLY without any markdown code block formatting (no ```json):
        {
          "product_name": "Extracted brand and product name",
          "manufacturer": "Extracted manufacturer details",
          "detections": [
            {
              "category": "MRP",
              "label": "MRP ₹ 150.00 (Incl. of all taxes)",
              "status": "Passed",
              "box_2d": [ymin, xmin, ymax, xmax]
            }
          ]
        }
        """

        candidate_models = ["gemini-3.5-flash", "gemini-3.5-flash-lite", "gemini-2.5-flash"]
        response = None
        last_exception = None

        for key_idx, key in enumerate(api_keys):
            try:
                client = get_gemini_client(key)
            except Exception as ce:
                logger.error(f"Failed to create Gemini client with key #{key_idx + 1}: {ce}")
                continue

            for model_name in candidate_models:
                try:
                    logger.info(f"Analyzing with key #{key_idx + 1} on {model_name}...")
                    response = client.models.generate_content(
                        model=model_name,
                        contents=[
                            types.Part.from_bytes(data=contents, mime_type="image/jpeg"),
                            prompt
                        ]
                    )
                    if response and response.text:
                        logger.info(f"Successfully analyzed image with key #{key_idx + 1} on {model_name}")
                        break
                except Exception as ge:
                    last_exception = ge
                    err_str = str(ge)
                    logger.warning(f"Key #{key_idx + 1} on {model_name} error: {err_str[:120]}")
                    if "RESOURCE_EXHAUSTED" in err_str or "429" in err_str:
                        # Key quota exhausted! Break from models and rotate to next key immediately
                        break
                    elif "503" in err_str or "UNAVAILABLE" in err_str:
                        time.sleep(1)
                        continue
                    else:
                        continue
            if response and response.text:
                break

        if not response or not response.text:
            raise HTTPException(
                status_code=429 if ("RESOURCE_EXHAUSTED" in str(last_exception) or "429" in str(last_exception)) else 500,
                detail=f"All configured Gemini API keys or models reached their quota/limit: {last_exception}"
            )

        response_text = response.text.strip()
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        elif response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        response_text = response_text.strip()

        data = json.loads(response_text)
        
        # Inject standard required metadata
        data["filename"] = filename
        data["original_width"] = 1000
        data["original_height"] = 1000
        data["real_width"] = real_width
        data["real_height"] = real_height
        data["image_url"] = image_url
        
        # Parse real Gemini 2D bounding boxes into x, y, width, height (normalized 0-1000 scale)
        for det in data.get("detections", []):
            box_2d = det.get("box_2d") or [0, 0, 0, 0]
            if isinstance(box_2d, list) and len(box_2d) == 4:
                ymin, xmin, ymax, xmax = box_2d
                if (xmax > xmin or ymax > ymin) and (ymin > 0 or xmin > 0 or ymax > 0 or xmax > 0):
                    det["box"] = {
                        "x": max(0, min(1000, int(xmin))),
                        "y": max(0, min(1000, int(ymin))),
                        "width": max(0, min(1000, int(xmax - xmin))),
                        "height": max(0, min(1000, int(ymax - ymin))),
                    }
                else:
                    det["box"] = {"x": 0, "y": 0, "width": 0, "height": 0}
            else:
                det["box"] = {"x": 0, "y": 0, "width": 0, "height": 0}

        return data

    except Exception as e:
        logger.error(f"Error during analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))