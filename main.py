import os
import json
import re
import cv2
import numpy as np
import easyocr
import uuid
import logging
import time
import asyncio
from typing import Optional
from fastapi import FastAPI, File, UploadFile, Request, HTTPException, BackgroundTasks, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# Load environment variables (like GEMINI_API_KEY)
load_dotenv()

gemini_client = None
if os.getenv("GEMINI_API_KEY"):
    try:
        from google import genai
        from google.genai import types
        gemini_client = genai.Client()
        logger.info("Gemini AI fallback enabled.")
    except ImportError:
        logger.warning("google-genai not installed. AI fallback disabled.")

from contextlib import asynccontextmanager

# Initialize EasyOCR globally to avoid loading it on every request
reader = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global reader
    logger.info("Initializing EasyOCR Model...")
    reader = await asyncio.to_thread(easyocr.Reader, ['en'], gpu=False)
    yield
    logger.info("Shutting down...")

app = FastAPI(title="Legal Metrology Compliance OCR API", version="2.1.0", lifespan=lifespan)

# Create uploads directory if it doesn't exist
os.makedirs("uploads", exist_ok=True)
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

# Allow Next.js frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def extract_packaging_lines(img, package_height_cm=None):
    """
    Extracts text using EasyOCR and reconstructs visual rows.
    
    Instead of returning raw OCR blocks (which split 'MRP' and '₹ 120' into 
    separate items), this function:
    1. Gets all text blocks from EasyOCR
    2. Groups blocks into visual rows by Y-coordinate proximity
    3. Merges blocks within each row left-to-right
    4. Returns complete lines like 'MRP : ₹ 120' with merged bounding boxes
    """
    orig_h, orig_w = img.shape[:2]

    # Smart resize
    max_dim = max(orig_h, orig_w)
    scale = 1.0
    if max_dim > 1200:
        scale = 1200.0 / max_dim
        proc_img = cv2.resize(img, (int(orig_w * scale), int(orig_h * scale)), interpolation=cv2.INTER_AREA)
    else:
        proc_img = img.copy()

    inv_scale = 1.0 / scale if scale != 1.0 else 1.0

    # Read text with EasyOCR
    results = reader.readtext(proc_img, paragraph=False)

    # Collect all blocks with bounding boxes (scaled back to original image coords)
    blocks = []
    for bbox, text, conf in results:
        if conf < 0.15 or not text.strip():
            continue
        xs = [pt[0] for pt in bbox]
        ys = [pt[1] for pt in bbox]
        min_x = min(xs)
        max_x = max(xs)
        min_y = min(ys)
        max_y = max(ys)
        blocks.append({
            "text": text.strip(),
            "x": int(min_x * inv_scale),
            "y": int(min_y * inv_scale),
            "width": int(max(max_x - min_x, 2) * inv_scale),
            "height": int(max(max_y - min_y, 2) * inv_scale),
            "cy": int(((min_y + max_y) / 2) * inv_scale),
        })

    if not blocks:
        return []

    # Sort blocks top-to-bottom, then left-to-right
    blocks.sort(key=lambda b: (b["cy"], b["x"]))

    # --- Row Reconstruction ---
    # Group blocks into visual rows: blocks whose vertical centers are within 
    # a tolerance are considered part of the same row
    rows = []
    current_row = [blocks[0]]

    for block in blocks[1:]:
        # Calculate row tolerance based on the average height of blocks in current row
        avg_height = sum(b["height"] for b in current_row) / len(current_row)
        tolerance = max(avg_height * 0.75, 15)

        # Compare vertical center of this block with the average cy of current row
        row_avg_cy = sum(b["cy"] for b in current_row) / len(current_row)

        if abs(block["cy"] - row_avg_cy) <= tolerance:
            # Same row
            current_row.append(block)
        else:
            # New row — flush the current one
            rows.append(current_row)
            current_row = [block]

    rows.append(current_row)  # flush last row

    # --- Merge blocks within each row into single line items ---
    line_items = []
    
    pixels_per_mm = None
    if package_height_cm and package_height_cm > 0:
        pixels_per_cm = orig_h / package_height_cm
        pixels_per_mm = pixels_per_cm / 10.0

    for row in rows:
        # Sort left-to-right within each row
        row.sort(key=lambda b: b["x"])

        # Merge text and bounding boxes
        merged_text = " ".join(b["text"] for b in row)
        min_x = min(b["x"] for b in row)
        min_y = min(b["y"] for b in row)
        max_x = max(b["x"] + b["width"] for b in row)
        max_y = max(b["y"] + b["height"] for b in row)
        
        box_height = max(max_y - min_y, 2)
        font_size_mm = None
        if pixels_per_mm:
            font_size_mm = round(box_height / pixels_per_mm, 2)

        line_items.append({
            "text": merged_text,
            "box": {
                "x": min_x,
                "y": min_y,
                "width": max(max_x - min_x, 2),
                "height": box_height,
            },
            "estimated_font_size_mm": font_size_mm
        })

    return line_items


def evaluate_declaration(text: str):
    """
    Evaluates text against the 5 mandatory Legal Metrology categories.
    Covers ALL known keywords, short forms, OCR misreads, and value formats
    found on Indian product packaging.
    Returns (category_name, status, status_message) if matched, or None.
    """
    t = text.upper()

    # =====================================================================
    # 1. MRP (Maximum Retail Price & Taxes)
    # =====================================================================
    # Keywords: MRP, M.R.P., M.R.P, M R P, Max Retail Price, Maximum Retail
    #   Price, Max. Ret. Price, Retail Price, RSP, Retail Selling Price
    # Tax phrases: Incl. of all taxes, Inclusive of all taxes, Including all
    #   taxes, Inc. all taxes, Tax Inclusive, All taxes included, (incl tax),
    #   incl. GST, Including GST, incl. of all taxes
    # Price formats: ₹120, Rs.120, Rs 120.00, INR 120, 120/-, ₹1,200.50
    # =====================================================================
    mrp_kw = re.search(
        r"\b(M[\s.]*R[\s.]*P\.?|MAX(?:IMUM)?[\s.]*RET(?:AIL)?[\s.]*PRICE|RETAIL\s*(?:SELLING\s*)?PRICE|RSP)(?:\s*[:\-=])?\b",
        t,
    )
    has_price = re.search(
        r"(?:RS\.?|₹|INR|RUPEES?)[\s.:]*(\d[\d,]*(?:\.\d{1,2})?)|(\d[\d,]*(?:\.\d{1,2})?)\s*/\-",
        t,
    )
    has_tax = bool(re.search(
        r"(INCL(?:USIVE|UDING|\.)?[\s.,]*(?:OF\s*)?(?:ALL\s*)?(?:TAX(?:ES)?|GST)|TAX\s*INCLU(?:SIVE|DED)|ALL\s*TAX(?:ES)?\s*INCL(?:UDED|USIVE)?|INCL\.?\s*GST)",
        t,
    ))

    if mrp_kw or (has_price and has_tax):
        price_val = has_price.group(1) or has_price.group(2) if has_price else None
        price_str = f" ₹{price_val}" if price_val else ""
        if has_tax and price_val:
            return (
                "MRP",
                "Passed",
                f"Passed: Valid MRP{price_str} with 'Inclusive of all taxes' declaration",
            )
        elif has_tax:
            return (
                "MRP",
                "Failed",
                "Failed: 'Inclusive of all taxes' found, but missing MRP price",
            )
        elif price_val:
            return (
                "MRP",
                "Failed",
                f"Failed: MRP{price_str} found, but missing mandatory 'Inclusive of all taxes' declaration",
            )
        else:
            return (
                "MRP",
                "Failed",
                "Failed: MRP keyword found, but missing price and 'Inclusive of all taxes' declaration",
            )

    # =====================================================================
    # 2. Expire Date / Best Before / Use By
    # =====================================================================
    # Keywords: Expiry Date, Exp Date, Exp. Date, Exp Dt, Exp. Dt., EXPD,
    #   EXP, Date of Expiry, Best Before, Best Before Date, Best Before End,
    #   BBE, BB, B.B., B.B, Use By, Use By Date, Use Before, Consume By,
    #   Consume Before, Valid Till, Valid Until, Validity, Shelf Life,
    #   Good Till, Good Until
    # Date formats: DD/MM/YYYY, DD-MM-YYYY, DD.MM.YYYY, DD MM YYYY,
    #   MM/YYYY, MM-YYYY, MMM YYYY, DD MMM YYYY, MMM-YY, YYYY-MM-DD
    # Duration: 6 months, 12 months from mfg, 180 days, 2 years
    # =====================================================================
    exp_kw = re.search(
        r"\b(EXP(?:IRY)?[\s.]*(?:DATE|DT\.?)?|EXPD\.?|DATE\s*(?:OF\s*)?EXP(?:IRY)?|"
        r"BEST\s*BEFORE(?:\s*(?:DATE|END))?|BBE|B[\s.]*B[\s.]*(?:DATE|DT\.?)?|"
        r"USE[\s.]*(?:BY|BEFORE)(?:\s*DATE)?|CONSUME[\s.]*(?:BY|BEFORE)|"
        r"VALID\s*(?:TILL|UNTIL|UPTO|UP\s*TO)|VALIDITY|SHELF\s*LIFE|"
        r"GOOD\s*(?:TILL|UNTIL)|NOT\s*FOR\s*(?:USE|SALE)\s*AFTER)(?:\s*[:\-=])?\b",
        t,
    )
    date_val = re.search(
        r"\b(\d{1,2}[\s./\-]+\d{1,2}[\s./\-]+(?:\d{2}|\d{4})\b|"               # DD/MM/YYYY or DD/MM/YY
        r"\d{4}[\s./\-]+\d{1,2}[\s./\-]+\d{1,2}\b|"                              # YYYY-MM-DD (ISO)
        r"\d{1,2}[\s./\-]+(?:\d{2}|\d{4})\b|"                                    # MM/YYYY or MM/YY
        r"(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)[A-Z]*[\s./\-]+(?:\d{2}|\d{4})\b|"  # MMM YYYY
        r"\d{1,2}[\s./\-]*(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)[A-Z]*[\s./\-]*(?:\d{2}|\d{4})\b)",  # DD MMM YYYY
        t,
    )
    exp_dur = re.search(
        r"(\d+\s*(?:MONTHS?|DAYS?|WEEKS?|YEARS?|YRS?)(?:\s*(?:FROM|AFTER|OF)\s*(?:MFG|MFD|PKD|PACK\w*|MANUFACTUR\w*|PRODUCTION|DATE|MAKING))?)",
        t,
    )

    if exp_kw:
        val = (
            date_val.group(1)
            if date_val
            else (exp_dur.group(1) if exp_dur else None)
        )
        if val:
            return (
                "Expire Date",
                "Passed",
                f"Passed: Valid Expiry / Best Before declaration found ({val})",
            )
        else:
            return (
                "Expire Date",
                "Failed",
                "Failed: Expiry keyword present but date or shelf life duration is missing/unreadable",
            )

    # =====================================================================
    # 3. Manufacture Date / Packing Date / Production Date
    # =====================================================================
    # Keywords: MFG, MFG., Mfg Date, Mfg. Date, Mfg Dt, MFD, MFD.,
    #   Date of Mfg, Date of Manufacturing, Date of Manufacture,
    #   Manufactured On, Manufactured Date, PKD, PKD., Packed Date,
    #   Packed On, Packing Date, Date of Packing, Date of Pack,
    #   DOM, D.O.M., D.O.M, DOP, D.O.P., Production Date, PRD,
    #   Date of Production, Filling Date, Date of Filling,
    #   OCR misreads: MEG, MFC, MPG, MED
    # =====================================================================
    mfg_kw = re.search(
        r"\b(?:M[EF][GCD]\b|M[EF][GCD][\s.]*(?:DATE|DT\.?)\b|M[FE]D\b|M[FE]D[\s.]*(?:DATE|DT\.?)\b|"
        r"DATE\s*(?:OF\s*)?M(?:FG|FD|ANUFACTUR\w*)|MANUFACTUR\w*\s*(?:ON|DATE|DT\.?)|"
        r"P[KR]D[\s.]*(?:DATE|DT\.?)?|PACK(?:ED|ING)?\s*(?:ON|DATE|DT\.?)|DATE\s*(?:OF\s*)?PACK\w*|"
        r"D[\s.]*O[\s.]*M\b|D[\s.]*O[\s.]*P\b|"
        r"PROD(?:UCTION)?[\s.]*(?:DATE|DT\.?)?|DATE\s*(?:OF\s*)?PROD(?:UCTION)?|"
        r"FILL(?:ING|ED)?[\s.]*(?:DATE|DT\.?|ON)?|DATE\s*(?:OF\s*)?FILL\w*)(?:\s*[:\-=])?\b",
        t,
    )
    mfg_date = re.search(
        r"\b(\d{1,2}[\s./\-]+\d{1,2}[\s./\-]+(?:\d{2}|\d{4})\b|"
        r"\d{4}[\s./\-]+\d{1,2}[\s./\-]+\d{1,2}\b|"
        r"\d{1,2}[\s./\-]+(?:\d{2}|\d{4})\b|"
        r"(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)[A-Z]*[\s./\-]+(?:\d{2}|\d{4})\b|"
        r"\d{1,2}[\s./\-]*(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)[A-Z]*[\s./\-]*(?:\d{2}|\d{4})\b)",
        t,
    )

    if mfg_kw:
        if mfg_date:
            return (
                "Manufacture Date",
                "Passed",
                f"Passed: Valid Date of Manufacture/Packing found ({mfg_date.group(1)})",
            )
        else:
            return (
                "Manufacture Date",
                "Failed",
                "Failed: Manufacture keyword present but date is missing or unreadable",
            )

    # =====================================================================
    # 4. Net Weight / Net Quantity / Net Content
    # =====================================================================
    # Keywords: Net Weight, Net Wt, Net Wt., Net WT, Net Quantity, Net Qty,
    #   Net Qty., Net Content, Net Contents, Net Volume, Net Vol, Net Vol.,
    #   Net Mass, N.W., NW, N W, Nett Wt, Nett Weight, PKD WT, Gross Wt,
    #   Contents, Drained Weight, Drained Wt, Fill Weight, Fill Vol
    # Units: g, gm, gms, gram, grams, kg, kgs, kilogram, kilograms,
    #   ml, mL, millilitre, millilitres, l, L, ltr, litre, litres, liter,
    #   liters, cl, fl oz, oz, cc, pieces, pcs, nos, units, N, no, nos.,
    #   tablets, tabs, capsules, caps, sachets, bags, packs, pair, pairs,
    #   meters, m, cm, mm, sqm, sq ft
    # =====================================================================
    net_kw = re.search(
        r"\b(NET(?:T)?[\s.]*(?:WT\.?|WEIGHT|QTY\.?|QUANTITY|CONTENT(?:S)?|VOL(?:UME)?\.?|MASS)|"
        r"N[\s.]*W[\s.]*|PKD[\s.]*WT\.?|"
        r"DRAINED[\s.]*(?:WT\.?|WEIGHT)|FILL[\s.]*(?:WT\.?|WEIGHT|VOL(?:UME)?\.?))(?:\s*[:\-=])?\b",
        t,
    )
    unit = re.search(
        r"(\d+(?:[.,]\d+)?\s*(?:G(?:M|MS|RAM|RAMS)?|KG(?:S|\.)?|KILOGRAMS?|"
        r"ML|MILLILITRES?|MILLILITERS?|"
        r"L(?:TR)?|LITRES?|LITERS?|CL|FL\.?\s*OZ|OZ|CC|"
        r"PCS|PIECES?|NOS\.?|UNITS?|TABS?|TABLETS?|CAPS(?:ULES)?|"
        r"SACHETS?|BAGS?|PACKS?|PAIRS?|"
        r"MM|CM|M(?:ETERS?|ETRES?)?|SQ\.?\s*(?:M|FT)|"
        r"N(?:\b)))(?:\s|$|\b)",
        t,
    )

    if net_kw:
        if unit:
            return (
                "Net Weight",
                "Passed",
                f"Passed: Valid Net Quantity found ({unit.group(1).strip()})",
            )
        else:
            return (
                "Net Weight",
                "Failed",
                "Failed: Net Quantity keyword present but missing valid standard unit of measure",
            )
    elif unit and re.search(r"\b(WT\.?|WEIGHT|QTY\.?|QUANTITY|CONTENT(?:S)?)\b", t):
        return (
            "Net Weight",
            "Passed",
            f"Passed: Valid Net Quantity found ({unit.group(1).strip()})",
        )

    # =====================================================================
    # 5. Consumer Info / Customer Care / Contact
    # =====================================================================
    # Keywords: Consumer Care, Customer Care, Cust. Care, Cust Care,
    #   Consumer Helpline, Customer Service, Helpline, Help Line,
    #   Toll Free, Toll-Free, Tollfree, Feedback, Grievance,
    #   Care No, Care Number, Care Line, Contact, Contact Us,
    #   Contact No, Contact Number, For Queries, For Complaints,
    #   Queries, Complaints, Enquiry, Enquiries, Call Us, Write To Us,
    #   Marketed By, Manufactured By, Imported By, Packed By,
    #   Registered Office, Corporate Office, Address
    # Values: Phone (10-digit Indian mobile/landline), Toll-free (1800...),
    #   Email (xxx@xxx.com), Website (www.xxx.com)
    # =====================================================================
    care_kw = re.search(
        r"\b(CONSUMER\s*(?:CARE|HELPLINE)|CUSTOMER\s*(?:CARE|SERVICE)|CUST\.?\s*CARE|"
        r"HELP\s*LINE|HELPLINE|TOLL[\s\-]*FREE|"
        r"FEEDBACK|GRIEVANCE|CARE\s*(?:NO\.?|NUMBER|LINE)|"
        r"CONTACT(?:\s*(?:US|NO\.?|NUMBER|DETAILS))?|"
        r"(?:FOR\s*)?(?:QUERIES|COMPLAINTS?|ENQUIR(?:Y|IES))|"
        r"CALL\s*US|WRITE\s*TO\s*US)(?:\s*[:\-=])?\b",
        t,
    )
    phone = re.search(
        r"\b((?:\+?91[\s\-]?)?[6-9]\d{4}[\s\-]?\d{5}|"             # Indian mobile: 9876543210
        r"1800[\s\-]?\d{2,4}[\s\-]?\d{3,5}|"                        # Toll-free: 1800-xxx-xxxx
        r"(?:\+?91[\s\-]?)?[6-9]\d{9}|"                              # Mobile: +91 9876543210
        r"0\d{2,4}[\s\-]?\d{6,8}|"                                   # Landline: 022-12345678
        r"(?:\+?91[\s\-]?)?\d{3,5}[\s\-]?\d{6,8})\b",               # STD: 0124-1234567
        text,
    )
    email = re.search(
        r"([a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,})", text
    )
    website = re.search(
        r"((?:https?://)?(?:www\.)?[a-zA-Z0-9\-]+\.[a-zA-Z]{2,}(?:\.[a-zA-Z]{2,})?)", text
    )

    if care_kw or email or (phone and re.search(r"\b(TEL\.?|PH\.?|PHONE|CALL|CARE|CONTACT|MOBILE|MOB\.?)\b", t)):
        val_parts = []
        if phone:
            val_parts.append(phone.group(1).strip())
        if email:
            val_parts.append(email.group(1).strip())
        if not val_parts and website:
            val_parts.append(website.group(1).strip())

        if val_parts:
            return (
                "Consumer Info",
                "Passed",
                f"Passed: Valid Consumer Care contact details found ({' / '.join(val_parts)})",
            )
        elif care_kw:
            return (
                "Consumer Info",
                "Failed",
                "Failed: Consumer Care declared without valid telephone number or email address",
            )

    # Not one of the 5 target categories — ignore
    return None


def cleanup_old_uploads(upload_dir="uploads", max_age_seconds=7200):
    now = time.time()
    try:
        for fname in os.listdir(upload_dir):
            fpath = os.path.join(upload_dir, fname)
            if os.path.isfile(fpath) and (now - os.path.getmtime(fpath)) > max_age_seconds:
                try:
                    os.remove(fpath)
                except OSError:
                    pass
    except Exception as e:
        logger.error(f"Cleanup error: {e}")

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

@app.post("/api/v1/analyze")
async def analyze_package(
    request: Request, 
    background_tasks: BackgroundTasks, 
    file: UploadFile = File(...),
    package_height_cm: Optional[float] = Form(None)
):
    """
    Two-phase Legal Metrology compliance scanner:
    
    Phase 1 — SCAN EVERYTHING:
      Extract all text from the package image and reconstruct visual rows.
      
    Phase 2 — ASSIGN KEYWORDS TO VALUES:
      With the full picture of ALL text available, search for each of the 
      5 mandatory categories and assign the correct values intelligently.
    """
    background_tasks.add_task(cleanup_old_uploads)
    try:
        image_bytes = await file.read()
        if len(image_bytes) > MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail="File too large. Maximum allowed size is 10 MB.")
    finally:
        await file.close()

    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid or readable image.")

    original_height, original_width, _ = img.shape
    product_name = ""
    manufacturer = ""
    
    # Save the original image to disk to serve it to the frontend via static URL
    # This avoids storing massive Base64 strings in the browser's localStorage
    ext = file.filename.split('.')[-1] if file.filename and '.' in file.filename else 'jpg'
    saved_filename = f"{uuid.uuid4()}.{ext}"
    saved_filepath = os.path.join("uploads", saved_filename)
    with open(saved_filepath, "wb") as f:
        f.write(image_bytes)
    
    # Construct the accessible URL for the frontend
    base_url = os.getenv("BASE_URL", str(request.base_url).rstrip("/"))
    image_url = f"{base_url}/uploads/{saved_filename}"

    # =====================================================================
    # PHASE 1: SCAN EVERYTHING
    # =====================================================================
    rows = await asyncio.to_thread(extract_packaging_lines, img, package_height_cm)
    num_rows = len(rows)

    # Build the full text corpus (all text on the label)
    full_text = " ".join(row["text"] for row in rows)

    # =====================================================================
    # PHASE 2: KEYWORD-ANCHORED ASSIGNMENT
    # For each category:
    #   1. Find which row(s) contain the KEYWORD for that category
    #   2. Evaluate that row alone — if it passes, done
    #   3. If failed (keyword found but value missing), expand ONLY to
    #      immediately adjacent rows (i±1, i±2, i±3) — never far-away rows
    #   4. This prevents MRP from stealing Net Weight's "250 g" etc.
    # =====================================================================
    KEYWORD_PATTERNS = {
        "MRP": r"\b(M[\s.]*R[\s.]*P\.?|MAX(?:IMUM)?[\s.]*RET(?:AIL)?[\s.]*PRICE|RETAIL\s*(?:SELLING\s*)?PRICE|RSP)\b",
        "Expire Date": r"\b(EXP(?:IRY)?[\s.]*(?:DATE|DT\.?)?|EXPD\.?|DATE\s*(?:OF\s*)?EXP(?:IRY)?|BEST\s*BEFORE|BBE|B[\s.]*B[\s.]*|USE[\s.]*(?:BY|BEFORE)|CONSUME[\s.]*(?:BY|BEFORE)|VALID\s*(?:TILL|UNTIL|UPTO)|SHELF\s*LIFE)\b",
        "Manufacture Date": r"\b(M[EF][GCD][\s.]*(?:DATE|DT\.?)?|M[FE]D[\s.]*(?:DATE|DT\.?)?|DATE\s*(?:OF\s*)?M(?:FG|FD|ANUFACTUR)|MANUFACTUR\w*\s*(?:ON|DATE)|P[KR]D[\s.]*(?:DATE|DT\.?)?|PACK(?:ED|ING)?\s*(?:ON|DATE)|D[\s.]*O[\s.]*M|PROD(?:UCTION)?[\s.]*DATE)\b",
        "Net Weight": r"\b(NET(?:T)?[\s.]*(?:WT\.?|WEIGHT|QTY\.?|QUANTITY|CONTENT|VOL)|N[\s.]*W[\s.]*|DRAINED[\s.]*WT)\b",
        "Consumer Info": r"\b(CONSUMER\s*(?:CARE|HELPLINE)|CUSTOMER\s*(?:CARE|SERVICE)|CUST\.?\s*CARE|HELP\s*LINE|HELPLINE|TOLL[\s\-]*FREE|CONTACT(?:\s*(?:US|NO|DETAILS))?|MARKETED\s*BY|MANUFACTURED\s*BY|IMPORTED\s*BY)\b",
    }

    categories = ["MRP", "Expire Date", "Manufacture Date", "Net Weight", "Consumer Info"]
    detections = []
    used_rows = set()

    for category in categories:
        kw_pattern = KEYWORD_PATTERNS[category]
        best = None  # (match_tuple, text, row_indices)

        # --- Step 1: Find all rows that contain the KEYWORD for this category ---
        keyword_rows = []
        for i in range(num_rows):
            if re.search(kw_pattern, rows[i]["text"].upper()):
                keyword_rows.append(i)

        # --- Step 2: For each keyword row, try single-row match first ---
        for i in keyword_rows:
            match = evaluate_declaration(rows[i]["text"])
            if match and match[0] == category:
                if match[1] == "Passed":
                    best = (match, rows[i]["text"], {i})
                    break
                elif best is None or best[0][1] != "Passed":
                    best = (match, rows[i]["text"], {i})

        # --- Step 3: If keyword found but failed, expand to ADJACENT rows only ---
        if best and best[0][1] == "Failed":
            anchor_row = min(best[2])  # the row where the keyword lives

            # Try combining with 1, 2, then 3 adjacent rows
            for expansion in range(1, 4):
                if best[0][1] == "Passed":
                    break

                # Build candidate adjacent indices (above and below the anchor)
                adj_candidates = []
                for offset in range(1, expansion + 1):
                    for direction in [1, -1]:
                        j = anchor_row + direction * offset
                        if 0 <= j < num_rows and j not in used_rows:
                            adj_candidates.append(j)

                # Try each adjacent row individually with the anchor
                for j in adj_candidates:
                    indices = sorted({anchor_row, j})
                    combined = " ".join(rows[idx]["text"] for idx in indices)
                    match = evaluate_declaration(combined)
                    if match and match[0] == category:
                        if match[1] == "Passed":
                            best = (match, combined, set(indices))
                            break
                        elif match[2] != best[0][2]:
                            # Improved (e.g., found price or tax text)
                            best = (match, combined, set(indices))

                # If still failed after single additions, try pairs of adjacent rows
                if best[0][1] == "Failed" and len(adj_candidates) >= 2:
                    for j_idx, j in enumerate(adj_candidates):
                        for k in adj_candidates[j_idx + 1:]:
                            indices = sorted({anchor_row, j, k})
                            combined = " ".join(rows[idx]["text"] for idx in indices)
                            match = evaluate_declaration(combined)
                            if match and match[0] == category and match[1] == "Passed":
                                best = (match, combined, set(indices))
                                break
                        if best[0][1] == "Passed":
                            break

        # --- Step 4: If keyword not found at all, try each single row ---
        if best is None:
            for i in range(num_rows):
                if i in used_rows:
                    continue
                match = evaluate_declaration(rows[i]["text"])
                if match and match[0] == category:
                    if match[1] == "Passed":
                        best = (match, rows[i]["text"], {i})
                        break
                    elif best is None:
                        best = (match, rows[i]["text"], {i})

        # Record the detection
        if best:
            match_tuple, text, row_indices = best
            category_name, status, message = match_tuple

            min_x = min(rows[idx]["box"]["x"] for idx in row_indices)
            min_y = min(rows[idx]["box"]["y"] for idx in row_indices)
            max_x = max(rows[idx]["box"]["x"] + rows[idx]["box"]["width"] for idx in row_indices)
            max_y = max(rows[idx]["box"]["y"] + rows[idx]["box"]["height"] for idx in row_indices)

            detections.append({
                "label": f"[{category_name}] {text} ({message})",
                "status": status,
                "category": category_name,
                "box": {
                    "x": min_x,
                    "y": min_y,
                    "width": max_x - min_x,
                    "height": max_y - min_y,
                },
            })
            used_rows.update(row_indices)

    # =====================================================================
    # PHASE 3: AI VISION FALLBACK (GEMINI MULTIMODAL)
    # Triggers if: 
    # 1. OCR detects zero text
    # 2. Category is missing or failed
    # 3. Category was found but value is suspiciously long (e.g., > 35 chars)
    # =====================================================================
    needs_recovery = []
    
    for cat in categories:
        cat_detections = [d for d in detections if d["category"] == cat]
        if not cat_detections:
            needs_recovery.append(cat)
        else:
            best_det = cat_detections[0]
            if best_det["status"] != "Passed":
                needs_recovery.append(cat)
            else:
                # Extract just the matched text to check length
                match = re.match(r"^\[.*?\]\s*(.*?)\s*\(.*\)$", best_det["label"])
                val = match.group(1) if match else best_det["label"]
                
                # If a short field (like MRP, Date, Weight) grabbed too much junk text
                if cat in ["MRP", "Net Weight", "Expire Date", "Manufacture Date"] and len(val.strip()) > 35:
                    needs_recovery.append(cat)
    
    if gemini_client:
        row_context = json.dumps([{"id": idx, "text": r["text"]} for idx, r in enumerate(rows)], indent=2)
        
        prompt = f"""
You are an expert Legal Metrology compliance AI.
I am providing BOTH the raw product package image and the OCR extracted text rows.

1. MUST EXTRACT: 'product_name' and 'manufacturer' from the package visually. Give concise, clear names.
2. The standard system failed, found zero text, or returned junk/suspiciously long values for these compliance categories: {needs_recovery}

OCR Row Data (may be empty or incomplete if OCR failed):
{row_context}

For each category in {needs_recovery}, carefully analyze the IMAGE and the OCR rows. Find the exact, concise value.
Remember Legal Metrology rules:
- MRP: e.g., '₹1120' or '120/-'
- Net Weight: e.g., '250g' or '50 ml'
- Manufacture Date: e.g., '04/23' or 'PKD 12/24'

Respond in strict JSON with the following schema:
{{
  "product_name": "Extracted product name, e.g., 'Lays Classic'",
  "manufacturer": "Extracted manufacturer name, e.g., 'PepsiCo'",
  "recoveries": [
    {{
      "category": "Category Name",
      "extracted_value": "The exact value found on the label",
      "rule_message": "Brief explanation",
      "row_ids": [integer IDs from OCR data, or leave as empty array [] if text wasn't in OCR data]
    }}
  ]
}}
Only include categories from the {needs_recovery} list in the recoveries array.
"""
        try:
            mime = file.content_type if file.content_type else "image/jpeg"
            contents = [
                prompt,
                types.Part.from_bytes(data=image_bytes, mime_type=mime)
            ]
            
            # Run synchronous LLM call in a thread pool to avoid blocking the event loop
            response = await asyncio.to_thread(
                gemini_client.models.generate_content,
                model='gemini-2.5-flash',
                contents=contents,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.1
                )
            )
            
            raw_text = re.sub(r"^```(?:json)?\s*|\s*```$", "", response.text.strip(), flags=re.MULTILINE)
            try:
                llm_result = json.loads(raw_text)
            except json.JSONDecodeError as err:
                logger.warning("Gemini returned invalid JSON: %s", err)
                llm_result = {}
                
            product_name = llm_result.get("product_name", "")
            manufacturer = llm_result.get("manufacturer", "")
            
            for recovery in llm_result.get("recoveries", []):
                cat = recovery.get("category")
                if cat in needs_recovery:
                    # Remove the bad/failed/long detection
                    detections = [d for d in detections if d.get("category") != cat]
                    
                    row_ids = recovery.get("row_ids", [])
                    
                    # If AI matched it to OCR rows, use those exact bounding boxes
                    if row_ids and all(isinstance(idx, int) and 0 <= idx < num_rows for idx in row_ids):
                        min_x = min(rows[idx]["box"]["x"] for idx in row_ids)
                        min_y = min(rows[idx]["box"]["y"] for idx in row_ids)
                        max_x = max(rows[idx]["box"]["x"] + rows[idx]["box"]["width"] for idx in row_ids)
                        max_y = max(rows[idx]["box"]["y"] + rows[idx]["box"]["height"] for idx in row_ids)
                    else:
                        # Fallback bounding box if OCR missed the text completely
                        min_x = min(10, max(0, original_width // 4))
                        max_x = max(min_x + 10, original_width - 10)
                        min_y = 10
                        max_y = max(20, int(original_height * 0.1))

                    detections.append({
                        "label": f"[{cat}] {recovery.get('extracted_value')} ({recovery.get('rule_message')} [AI Recovered])",
                        "status": "Passed",
                        "category": cat,
                        "box": {
                            "x": min_x,
                            "y": min_y,
                            "width": max_x - min_x,
                            "height": max_y - min_y,
                        },
                    })
        except Exception as e:
            logger.error(f"Gemini Vision Fallback Error: {e}")

    return {
        "product_name": locals().get("product_name", ""), 
        "manufacturer": locals().get("manufacturer", ""),
        "filename": file.filename or "package_label.jpg",
        "original_width": original_width,
        "original_height": original_height,
        "detections": detections,
        "image_url": image_url,
    }