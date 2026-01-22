from pydantic import BaseModel, Field, validator
from typing import Optional, List, Literal, Dict
from enum import Enum
from datetime import datetime

# --- OUTPUT SCHEMA ---

class LeadRecord(BaseModel):
    """Normalized lead output schema"""
    name: str = Field(..., description="Company or person name")
    phone: Optional[str] = Field(None, description="Phone number (normalized)")
    email: Optional[str] = Field(None, description="Email address (lowercase)")
    website: Optional[str] = Field(None, description="Website URL")
    address: Optional[str] = Field(None, description="Physical address")
    niche: Optional[str] = Field(None, description="Business category/niche")
    rating: Optional[float] = Field(None, ge=0, le=5, description="Review rating if available")
    reviews_count: Optional[int] = Field(None, ge=0, description="Number of reviews")
    source_url: str = Field(..., description="URL where lead was found")
    source_platform: str = Field(..., description="Platform: google_maps, yellow_pages, etc.")
    scraped_at: str = Field(..., description="ISO timestamp when scraped")
    
    class Config:
        json_schema_extra = {
            "example": {
                "name": "Dentista Silva",
                "phone": "11999999999",
                "email": "contato@dentista.com.br",
                "website": "https://dentista.com.br",
                "address": "Prainha, Rio de Janeiro, Brazil",
                "niche": "Dental Services",
                "rating": 4.8,
                "reviews_count": 125,
                "source_url": "https://maps.google.com/...",
                "source_platform": "google_maps",
                "scraped_at": "2026-01-22T16:40:00Z"
            }
        }

    @validator('phone', 'email', pre=True, always=True)
    def normalize_contact(cls, v):
        """Normalize phone and email fields"""
        if v:
            return str(v).strip().lower()
        return None

    @validator('website', pre=True)
    def normalize_url(cls, v):
        """Ensure URLs start with http"""
        if v and not v.startswith(('http://', 'https://')):
            return f"https://{v}"
        return v


# --- INPUT SCHEMAS ---

class ScrapeSource(str, Enum):
    GOOGLE_MAPS = "google_maps"
    YELLOW_PAGES = "yellow_pages"
    CUSTOM_SITE = "custom_site"

class FilterConfig(BaseModel):
    min_reviews: Optional[int] = None
    min_rating: Optional[float] = None
    has_website: bool = False
    has_phone: bool = False

class CustomSiteConfig(BaseModel):
    """For CUSTOM_SITE source"""
    base_url: str = Field(..., description="Starting URL to scrape")
    search_endpoint: str = Field(..., description="URL pattern for search")
    selectors: Dict[str, str] = Field(..., description="CSS selectors for extraction")
    pagination_selector: Optional[str] = None
    next_page_pattern: Optional[str] = None

class LeadScraperDirective(BaseModel):
    """Main directive for lead scraper engine"""
    source: ScrapeSource = Field(..., description="Data source platform")
    search_query: str = Field(..., min_length=1, description="Search term or business type")
    location: Optional[str] = Field("Brazil", description="Geographic location")
    max_pages: int = Field(default=3, ge=1, le=10, description="Max pages to scrape")
    max_results: int = Field(default=50, ge=1, le=500, description="Max leads to return")
    filters: Optional[FilterConfig] = None
    anti_bot: bool = Field(default=True, description="Enable headless browser + delays")
    timeout_seconds: int = Field(default=30, ge=5, le=120)
    custom_config: Optional[CustomSiteConfig] = None
    output_format: Literal["json", "csv"] = "json"
    
    @validator('location')
    def default_location(cls, v):
        return v or "Brazil"

    class Config:
        json_schema_extra = {
            "example": {
                "source": "google_maps",
                "search_query": "dentistas Prainha",
                "location": "Rio de Janeiro, Brazil",
                "max_pages": 2,
                "max_results": 20,
                "filters": {"min_reviews": 10, "min_rating": 4.0},
                "anti_bot": True
            }
        }
