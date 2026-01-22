import asyncio
import json
import logging
import sys
import os

# Add project root to path so we can import src
sys.path.append(os.getcwd())

from src.domain.schemas import LeadScraperDirective, ScrapeSource
from src.orchestration.maestro import LeadScraperOrchestrator
from src.utils.export import export_leads

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("local_runner")

async def main():
    logger.info("Starting local MSPP test...")

    # Define a test directive
    test_directive = LeadScraperDirective(
        source=ScrapeSource.GOOGLE_MAPS,
        search_query="dentistas",
        location="Rio de Janeiro, Brazil",
        max_pages=1,
        max_results=3, # Small number for testing
        filters={
            "min_rating": 4.5
        },
        anti_bot=False # Disable delay for test
    )

    logger.info(f"Directive created: {test_directive.model_dump_json(indent=2)}")

    try:
        # Initialize Orchestrator
        async with LeadScraperOrchestrator(test_directive) as orchestrator:
            # Run execution
            results = await orchestrator.execute_scrape()
            
            # Output results
            print("\n" + "="*50)
            print(f"SUCCESS: Retrieved {len(results)} leads")
            print("="*50)
            for lead in results:
                print(f"- {lead.name} ({lead.rating}*) - {lead.phone} - {lead.source_platform}")
            print("="*50 + "\n")
            
            # Show metrics
            metrics = orchestrator.get_metrics()
            print(f"Metrics: {json.dumps(metrics, indent=2)}")

            # Demonstrate CSV export
            print("\n" + "="*50)
            print("CSV Export Preview:")
            print("="*50)
            csv_output = export_leads(results, "csv")
            # Show first few lines
            csv_lines = csv_output.split('\n')[:5]
            for line in csv_lines:
                print(line)
            if len(csv_output.split('\n')) > 5:
                print("...")
            print("="*50 + "\n")

    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)

if __name__ == "__main__":
    asyncio.run(main())
