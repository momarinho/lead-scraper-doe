import logging
import asyncio
from datetime import datetime
from typing import List
from pydantic import ValidationError
from tenacity import retry, stop_after_attempt, wait_exponential
import aiohttp

from src.domain.schemas import LeadScraperDirective, LeadRecord, ScrapeSource
from src.execution.executors import GoogleMapsExecutor, YellowPagesExecutor, CustomSiteExecutor, SourceExecutor

logger = logging.getLogger(__name__)

class LeadScraperOrchestrator:
    """
    Main orchestrator for lead scraping.
    
    Responsibilities:
    - Validate directive (Pydantic)
    - Plan execution pipeline based on source
    - Route to correct source executor
    - Normalize results to LeadRecord schema
    - Handle retries and errors
    - Log metrics
    """
    
    def __init__(self, directive: LeadScraperDirective):
        # Validation happens on instantiation via Pydantic type hinting for 'directive'
        # but the caller usually passes the object.
        self.directive = directive
        self.session = None
        self.start_time = None
        logger.info(
            f"Initialized orchestrator for {directive.source} - "
            f"Query: '{directive.search_query}' - "
            f"Location: {directive.location}"
        )

    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession()
        self.start_time = datetime.now()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
        if self.start_time:
            elapsed = (datetime.now() - self.start_time).total_seconds()
            logger.info(f"Orchestrator completed. Elapsed: {elapsed:.2f}s")

    def plan_pipeline(self) -> List[str]:
        """
        Determine execution pipeline based on source.
        
        Returns:
            List of execution step names
        """
        steps = ["search", "paginate", "extract"]
        
        if self.directive.source == ScrapeSource.CUSTOM_SITE:
            steps.append("custom_parse")
        
        if self.directive.filters:
            steps.append("filter")
        
        steps.append("normalize")
        
        logger.info(f"Planned pipeline: {' -> '.join(steps)}")
        return steps

    def get_executor(self) -> SourceExecutor:
        """
        Route to appropriate source executor.
        
        Returns:
            Executor instance for the configured source
        """
        if self.directive.source == ScrapeSource.GOOGLE_MAPS:
            return GoogleMapsExecutor(self.session, self.directive)
        elif self.directive.source == ScrapeSource.YELLOW_PAGES:
            return YellowPagesExecutor(self.session, self.directive)
        elif self.directive.source == ScrapeSource.CUSTOM_SITE:
            return CustomSiteExecutor(self.session, self.directive)
        else:
            raise ValueError(f"Unknown source: {self.directive.source}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=8)
    )
    async def execute_scrape(self) -> List[LeadRecord]:
        """
        Main scraping orchestration with retry logic.
        
        Returns:
            List of normalized LeadRecord objects
        """
        pipeline = self.plan_pipeline()
        executor = self.get_executor()
        
        try:
            # Execute scraping pipeline
            raw_leads = await executor.execute()
            
            # Apply filters if configured
            if self.directive.filters:
                raw_leads = self._apply_filters(raw_leads)
            
            # Normalize to LeadRecord schema
            normalized_leads = []
            for i, lead_dict in enumerate(raw_leads[:self.directive.max_results]):
                try:
                    lead = LeadRecord(**lead_dict)
                    normalized_leads.append(lead)
                except ValidationError as e:
                    logger.warning(f"Failed to normalize lead {i}: {e}")
                    continue
            
            logger.info(
                f"Extracted {len(normalized_leads)} normalized leads "
                f"from {len(raw_leads)} total"
            )
            return normalized_leads
            
        except Exception as e:
            logger.error(f"Scraping execution failed: {e}", exc_info=True)
            raise

    def _apply_filters(self, leads: List[dict]) -> List[dict]:
        """Apply FilterConfig to raw leads"""
        if not self.directive.filters:
            return leads
        
        filters = self.directive.filters
        filtered = []
        
        for lead in leads:
            reviews = lead.get('reviews_count') or 0
            rating = lead.get('rating') or 0.0
            
            if filters.min_reviews and reviews < filters.min_reviews:
                continue
            if filters.min_rating and rating < filters.min_rating:
                continue
            if filters.has_website and not lead.get('website'):
                continue
            if filters.has_phone and not lead.get('phone'):
                continue
            filtered.append(lead)
        
        logger.info(f"Applied filters: {len(leads)} -> {len(filtered)} leads")
        return filtered

    def get_metrics(self) -> dict:
        """Return execution metrics for logging/monitoring"""
        elapsed = 0
        if self.start_time:
            elapsed = (datetime.now() - self.start_time).total_seconds()
            
        return {
            "source": self.directive.source,
            "query": self.directive.search_query,
            "location": self.directive.location,
            "max_pages": self.directive.max_pages,
            "max_results": self.directive.max_results,
            "elapsed_seconds": elapsed
        }
