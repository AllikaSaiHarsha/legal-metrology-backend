import os
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.core.config import settings
from app.api.v1.router import api_router
from app.schemas.compliance import RootResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_application() -> FastAPI:
    app = FastAPI(
        title=settings.PROJECT_NAME,
        version=settings.VERSION,
        description="Multimodal AI vision inspection microservice for Indian Legal Metrology (Rule 6) packaging compliance.",
        docs_url="/docs",
        redoc_url="/redoc"
    )

    # CORS Middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Static file serving for uploaded package scans
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    app.mount(f"/{settings.UPLOAD_DIR}", StaticFiles(directory=settings.UPLOAD_DIR), name=settings.UPLOAD_DIR)

    # Register API Routers
    app.include_router(api_router, prefix=settings.API_V1_PREFIX)

    @app.get("/", response_model=RootResponse, tags=["Root"])
    async def root():
        return RootResponse(
            service=settings.PROJECT_NAME,
            status="online",
            docs="/docs",
            health=f"{settings.API_V1_PREFIX}/health"
        )

    return app


app = create_application()
