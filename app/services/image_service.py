import os
import io
import uuid
import logging
from typing import Tuple, List, Dict, Any, Optional

try:
    from PIL import Image
except ImportError:
    Image = None

from app.core.config import settings

logger = logging.getLogger(__name__)


class ImageService:
    @staticmethod
    def save_upload(contents: bytes, original_filename: Optional[str] = None) -> Tuple[str, str]:
        """
        Saves the uploaded file to the configured uploads directory and returns (filename, filepath).
        """
        os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
        
        ext = ".jpg"
        if original_filename:
            _, file_ext = os.path.splitext(original_filename)
            if file_ext:
                ext = file_ext

        file_id = str(uuid.uuid4())
        filename = f"{file_id}{ext}"
        filepath = os.path.join(settings.UPLOAD_DIR, filename)

        with open(filepath, "wb") as f:
            f.write(contents)

        # Mirror image to Next.js public/uploads if sibling folder exists
        ImageService._mirror_to_web_dashboard(contents, filename)

        return filename, filepath

    @staticmethod
    def _mirror_to_web_dashboard(contents: bytes, filename: str) -> None:
        try:
            web_uploads = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "..", "..", "legal-metrology-web", "public", "uploads")
            )
            if os.path.exists(web_uploads):
                with open(os.path.join(web_uploads, filename), "wb") as wf:
                    wf.write(contents)
        except Exception as e:
            logger.warning(f"Could not mirror image to web public/uploads: {e}")

    @staticmethod
    def get_image_dimensions(contents: bytes) -> Tuple[int, int]:
        """
        Calculates pixel width and height from bytes.
        """
        try:
            if Image is not None:
                with Image.open(io.BytesIO(contents)) as pil_img:
                    return pil_img.size
        except Exception:
            pass
        return 1000, 1000

    @staticmethod
    def sanitize_detections(detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Converts 0-1000 normalized 2D boxes [ymin, xmin, ymax, xmax] into {x, y, width, height}
        and applies Legal Metrology compliance rules (e.g., skips false alarms for domestic goods).
        """
        sanitized = []
        for det in detections:
            cat = (det.get("category") or "").strip().lower()
            lbl = (det.get("label") or "").strip().lower()
            status = (det.get("status") or "").strip()

            # Ignore Country of Origin error if absent on domestic goods
            if "country of origin" in cat or "country of origin" in lbl:
                if status == "Failed" or "missing" in lbl or "not visible" in lbl:
                    continue

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

            sanitized.append(det)

        return sanitized
