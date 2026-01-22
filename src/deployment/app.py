import modal
import json
import asyncio
import logging
from pydantic import ValidationError
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("lead-scraper-modal")

# Define Modal app
app = modal.App("lead-scraper-doe")

# Define container image with dependencies and local source
image = (
    modal.Image.debian_slim()
    .apt_install("ca-certificates", "libssl-dev", "libnss3", "libatk1.0-0",
                 "libatk-bridge2.0-0", "libcups2", "libdrm2", "libxkbcommon0",
                 "libxcomposite1", "libxdamage1", "libxfixes3", "libxrandr2",
                 "libgbm1", "libasound2")  # Chromium dependencies
    .pip_install(
        "pydantic>=2.0.0",
        "aiohttp",
        "tenacity",
        "beautifulsoup4",
        "playwright",
        "fastapi[standard]"
    )
    .run_commands("playwright install chromium")  # Install Chromium for Playwright
    .add_local_dir("src", remote_path="/root/src")  # Mount local source
)


@app.function(image=image, timeout=300)
async def scrape_leads_endpoint(directive_json: str) -> str:
    """
    Main serverless entrypoint for lead scraper.

    Args:
        directive_json: JSON string containing LeadScraperDirective

    Returns:
        JSON string with List[LeadRecord] results
    """
    import sys
    sys.path.insert(0, "/root")

    from src.domain.schemas import LeadScraperDirective, LeadRecord
    from src.orchestration.maestro import LeadScraperOrchestrator
    from src.utils.export import export_leads

    try:
        # Parse JSON to directive
        directive_dict = json.loads(directive_json)
        directive = LeadScraperDirective(**directive_dict)

        logger.info(f"Starting lead scrape. Directive: {directive.model_dump()}")

        # Execute orchestrator
        async with LeadScraperOrchestrator(directive) as orchestrator:
            leads = await orchestrator.execute_scrape()

        # Export based on requested format
        if directive.output_format == "csv":
            # Return CSV directly
            return export_leads(leads, "csv")
        else:
            # Return JSON response with metadata
            response = {
                "success": True,
                "count": len(leads),
                "leads": [lead.model_dump() for lead in leads],
                "metrics": orchestrator.get_metrics()
            }
            return json.dumps(response, indent=2, default=str)

    except ValidationError as e:
        error_response = {
            "success": False,
            "error": "Invalid directive",
            "details": str(e)
        }
        return json.dumps(error_response, indent=2)
    except Exception as e:
        logger.error(f"Scraping failed: {e}", exc_info=True)
        error_response = {
            "success": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }
        return json.dumps(error_response, indent=2)


@app.function(image=image, timeout=300)
@modal.fastapi_endpoint(method="POST")
async def scrape_leads_api(request: dict) -> dict:
    """
    REST API endpoint for calling from n8n or other platforms.
    Expects body: { "directive": { ... } }
    """
    import sys
    sys.path.insert(0, "/root")

    from src.domain.schemas import LeadScraperDirective
    from src.orchestration.maestro import LeadScraperOrchestrator
    from src.utils.export import export_leads

    try:
        directive_data = request.get("directive", {})

        # Parse directive and execute
        directive = LeadScraperDirective(**directive_data)
        async with LeadScraperOrchestrator(directive) as orchestrator:
            leads = await orchestrator.execute_scrape()

        # Return based on requested format
        if directive.output_format == "csv":
            return {
                "success": True,
                "format": "csv",
                "data": export_leads(leads, "csv")
            }
        else:
            return {
                "success": True,
                "count": len(leads),
                "leads": [lead.model_dump() for lead in leads],
                "metrics": orchestrator.get_metrics()
            }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


# Local testing
if __name__ == "__main__":
    directive = {
        "source": "google_maps",
        "search_query": "dentistas",
        "location": "Rio de Janeiro, Brazil",
        "max_pages": 1,
        "max_results": 5
    }

    # We can run the async function locally
    result = asyncio.run(scrape_leads_endpoint(json.dumps(directive)))
    print(result)
