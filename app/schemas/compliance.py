from typing import List, Optional
from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    x: int = Field(..., description="Normalized x coordinate (0-1000 scale)")
    y: int = Field(..., description="Normalized y coordinate (0-1000 scale)")
    width: int = Field(..., description="Normalized width (0-1000 scale)")
    height: int = Field(..., description="Normalized height (0-1000 scale)")


class DetectionItem(BaseModel):
    category: str = Field(..., description="Statutory declaration category under Rule 6")
    label: str = Field(..., description="Verbatim text extracted from package")
    status: str = Field(..., description="'Passed' or 'Failed'")
    box_2d: Optional[List[int]] = Field(default=[0, 0, 0, 0], description="Raw [ymin, xmin, ymax, xmax] 0-1000")
    box: Optional[BoundingBox] = Field(default=None, description="Calculated UI bounding box")


class ComplianceAnalysisResponse(BaseModel):
    product_name: Optional[str] = Field(default="", description="Identified product name")
    manufacturer: Optional[str] = Field(default="", description="Identified manufacturer / packer details")
    detections: List[DetectionItem] = Field(default_factory=list, description="List of Rule 6 compliance declarations")
    filename: str = Field(..., description="Saved image filename")
    original_width: int = Field(default=1000)
    original_height: int = Field(default=1000)
    real_width: int = Field(default=1000)
    real_height: int = Field(default=1000)
    image_url: str = Field(..., description="Static accessible URL for uploaded image")


class HealthResponse(BaseModel):
    status: str = "ok"
    keys_configured: int = 0


class RootResponse(BaseModel):
    service: str = "Legal Metrology Vision API"
    status: str = "online"
    docs: str = "/docs"
    health: str = "/api/v1/health"
