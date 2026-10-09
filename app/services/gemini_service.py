import json
import logging
import time
from typing import List, Dict, Any, Optional
from fastapi import HTTPException
from google import genai
from google.genai import types

from app.core.config import settings

logger = logging.getLogger(__name__)

RULE_6_INSPECTION_PROMPT = """
You are an expert Indian Legal Metrology package inspector. 
Analyze the packaging label and extract the statutory declarations required under Rule 6 of the Legal Metrology (Packaged Commodities) Rules, 2011:
1. Product Name / Generic Name
2. Net Quantity / Net Weight
3. Retail Sale Price (MRP, inclusive of all taxes)
4. Date of Manufacture / Packing / Import
5. Expiry Date / Best Before Date (if applicable)
6. Consumer Care / Customer Care contact details (email, phone, address)
7. Manufacturer / Packer / Importer Name & Address
8. Country of Origin (Only required for imported goods; for domestic goods or if not visible, do not mark as Failed)
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


class GeminiVisionService:
    @staticmethod
    def get_client(api_key: str) -> genai.Client:
        return genai.Client(api_key=api_key)

    @classmethod
    def analyze_label(cls, contents: bytes, api_keys: List[str]) -> Dict[str, Any]:
        """
        Analyze packaging label using Gemini Multimodal AI.
        Rotates across available API keys and candidate models on 429 rate limit.
        """
        if not api_keys:
            raise HTTPException(
                status_code=500,
                detail="Gemini API Key missing in environment (set GEMINI_API_KEY or GEMINI_API_KEYS)"
            )

        response = None
        last_exception = None

        for key_idx, key in enumerate(api_keys):
            try:
                client = cls.get_client(key)
            except Exception as ce:
                logger.error(f"Failed to create Gemini client with key #{key_idx + 1}: {ce}")
                continue

            for model_name in settings.CANDIDATE_MODELS:
                try:
                    logger.info(f"Analyzing with key #{key_idx + 1} on model: {model_name}...")
                    response = client.models.generate_content(
                        model=model_name,
                        contents=[
                            types.Part.from_bytes(data=contents, mime_type="image/jpeg"),
                            RULE_6_INSPECTION_PROMPT
                        ]
                    )
                    if response and response.text:
                        logger.info(f"Successfully analyzed label with key #{key_idx + 1} on {model_name}")
                        break
                except Exception as ge:
                    last_exception = ge
                    err_str = str(ge)
                    logger.warning(f"Key #{key_idx + 1} on {model_name} error: {err_str[:120]}")
                    if "RESOURCE_EXHAUSTED" in err_str or "429" in err_str:
                        # Move to next API key immediately
                        break
                    elif "503" in err_str or "UNAVAILABLE" in err_str:
                        time.sleep(1)
                        continue
                    else:
                        continue

            if response and response.text:
                break

        if not response or not response.text:
            status_code = 429 if ("RESOURCE_EXHAUSTED" in str(last_exception) or "429" in str(last_exception)) else 500
            raise HTTPException(
                status_code=status_code,
                detail=f"All configured Gemini API keys or models reached their limit: {last_exception}"
            )

        response_text = response.text.strip()
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        elif response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        response_text = response_text.strip()

        try:
            return json.loads(response_text)
        except json.JSONDecodeError as jde:
            logger.error(f"Failed to parse Gemini output as JSON: {jde}. Raw text: {response_text[:300]}")
            raise HTTPException(status_code=502, detail="Failed to parse model response as structured JSON")
