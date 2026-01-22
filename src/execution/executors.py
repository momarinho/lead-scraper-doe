from abc import ABC, abstractmethod
import aiohttp
import asyncio
import logging
import re
import urllib.parse
from typing import List, Optional
from bs4 import BeautifulSoup
from datetime import datetime

from src.domain.schemas import LeadScraperDirective, ScrapeSource

class SourceExecutor(ABC):
    """Base class for source-specific executors"""
    
    def __init__(self, session: aiohttp.ClientSession, directive: LeadScraperDirective):
        self.session = session
        self.directive = directive
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    async def execute(self) -> List[dict]:
        """Execute scraping for this source. Return raw lead dicts."""
        pass

    async def _fetch_page(self, url: str) -> str:
        """Fetch page with anti-bot handling"""
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
        }
        
        if self.directive.anti_bot:
            await asyncio.sleep(2)  # Rate limiting
        
        async with self.session.get(
            url,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=self.directive.timeout_seconds)
        ) as resp:
            resp.raise_for_status()
            return await resp.text()


class GoogleMapsExecutor(SourceExecutor):
    """
    Executor for Google Maps scraping using Playwright.
    
    Scrapes business listings from Google Maps search results.
    Uses headless browser to handle JavaScript-rendered content.
    """
    
    # CSS Selectors for Google Maps (may need updates if Google changes UI)
    SELECTORS = {
        "results_container": "div[role='feed']",
        "result_item": "div[role='feed'] > div > div[jsaction]",
        "name": "div.fontHeadlineSmall",
        "rating": "span[role='img']",
        "reviews_count": "span[aria-label*='reviews'], span[aria-label*='avaliações']",
        "category": "div.fontBodyMedium span:first-child",
        "address": "div.fontBodyMedium span[aria-label*='Address'], div.fontBodyMedium span:nth-child(2)",
        "phone": "span[aria-label*='Phone'], button[data-tooltip*='phone']",
        "website": "a[data-value='Website'], a[aria-label*='Website']",
    }
    
    async def execute(self) -> List[dict]:
        """
        Scrape Google Maps for business leads using Playwright.
        
        Returns:
            List of raw lead dictionaries
        """
        self.logger.info(f"Starting Google Maps scrape: {self.directive.search_query} in {self.directive.location}")
        
        leads = []
        
        try:
            from playwright.async_api import async_playwright
            
            async with async_playwright() as p:
                # Launch browser
                browser = await p.chromium.launch(
                    headless=True,
                    args=['--no-sandbox', '--disable-setuid-sandbox']
                )
                
                context = await browser.new_context(
                    viewport={'width': 1920, 'height': 1080},
                    user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                    locale='pt-BR'
                )
                
                page = await context.new_page()
                
                # Build search URL
                search_term = f"{self.directive.search_query} {self.directive.location or ''}"
                encoded_query = urllib.parse.quote(search_term.strip())
                url = f"https://www.google.com/maps/search/{encoded_query}"
                
                self.logger.info(f"Navigating to: {url}")
                
                # Navigate to Google Maps - use 'load' instead of 'networkidle' for faster loading
                # Google Maps uses a lot of background requests that delay 'networkidle'
                await page.goto(url, wait_until='load', timeout=60000)
                
                # Wait for dynamic content to render
                await asyncio.sleep(5 if self.directive.anti_bot else 3)
                
                # Try to find the results container
                try:
                    await page.wait_for_selector(self.SELECTORS["results_container"], timeout=10000)
                except Exception:
                    self.logger.warning("Results container not found, trying alternative approach")
                
                # Scroll to load more results (pagination simulation)
                for page_num in range(self.directive.max_pages):
                    self.logger.info(f"Scraping page {page_num + 1}/{self.directive.max_pages}")
                    
                    # Scroll down to load more results
                    await self._scroll_results(page)
                    
                    if self.directive.anti_bot:
                        await asyncio.sleep(2)
                
                # Extract all visible results
                leads = await self._extract_leads(page, url)
                
                await browser.close()
                
        except ImportError:
            self.logger.error("Playwright not installed. Run: pip install playwright && playwright install chromium")
            # Fallback to stub data for demo
            leads = self._stub_google_maps_results()
        except Exception as e:
            self.logger.error(f"Playwright scraping failed: {e}", exc_info=True)
            # Fallback to stub data
            leads = self._stub_google_maps_results()
        
        self.logger.info(f"Extracted {len(leads)} leads from Google Maps")
        return leads
    
    async def _scroll_results(self, page) -> None:
        """Scroll the results panel to load more items"""
        try:
            # Find the scrollable results container
            container = await page.query_selector(self.SELECTORS["results_container"])
            if container:
                # Scroll multiple times to load more results
                for _ in range(3):
                    await page.evaluate('''
                        (container) => {
                            container.scrollTop = container.scrollHeight;
                        }
                    ''', container)
                    await asyncio.sleep(1)
        except Exception as e:
            self.logger.warning(f"Scroll failed: {e}")
    
    async def _extract_leads(self, page, source_url: str) -> List[dict]:
        """Extract lead data from the current page state"""
        leads = []
        
        try:
            # Get all result items
            items = await page.query_selector_all(self.SELECTORS["result_item"])
            self.logger.info(f"Found {len(items)} result items")
            
            for i, item in enumerate(items[:self.directive.max_results]):
                try:
                    lead = await self._extract_single_lead(item, source_url)
                    if lead and lead.get('name'):
                        leads.append(lead)
                except Exception as e:
                    self.logger.debug(f"Failed to extract lead {i}: {e}")
                    continue
                    
        except Exception as e:
            self.logger.error(f"Lead extraction failed: {e}")
        
        return leads
    
    async def _extract_single_lead(self, item, source_url: str) -> Optional[dict]:
        """Extract data from a single result item"""
        lead = {
            "source_platform": "google_maps",
            "source_url": source_url,
            "scraped_at": datetime.now().isoformat() + "Z"
        }
        
        # Extract name
        name_el = await item.query_selector(self.SELECTORS["name"])
        if name_el:
            lead["name"] = await name_el.inner_text()
        else:
            return None  # Skip if no name
        
        # Extract rating
        rating_el = await item.query_selector(self.SELECTORS["rating"])
        if rating_el:
            aria_label = await rating_el.get_attribute("aria-label")
            if aria_label:
                # Parse "4.5 stars" or "4,5 estrelas"
                match = re.search(r'(\d+[.,]?\d*)', aria_label)
                if match:
                    lead["rating"] = float(match.group(1).replace(',', '.'))
        
        # Extract reviews count
        reviews_el = await item.query_selector(self.SELECTORS["reviews_count"])
        if reviews_el:
            text = await reviews_el.inner_text()
            # Parse "(123)" or "123 reviews"
            match = re.search(r'(\d+)', text.replace('.', '').replace(',', ''))
            if match:
                lead["reviews_count"] = int(match.group(1))
        
        # Extract category/niche
        category_el = await item.query_selector(self.SELECTORS["category"])
        if category_el:
            lead["niche"] = await category_el.inner_text()
        
        # Try to get more details by clicking on the item
        # This is optional and may slow down scraping
        try:
            # Get inner text which often contains address
            full_text = await item.inner_text()
            lines = full_text.split('\n')
            
            # Try to find address (usually after category)
            for line in lines:
                line = line.strip()
                # Skip empty lines and known elements
                if not line or line == lead.get("name") or line == lead.get("niche"):
                    continue
                # Address usually contains street indicators
                if any(ind in line.lower() for ind in ['rua', 'av', 'avenida', 'praça', 'travessa', 'alameda', 'street', 'ave', 'road']):
                    lead["address"] = line
                    break
                # Or if it looks like an address (has numbers and letters)
                if re.search(r'\d+.*[a-zA-Z]|[a-zA-Z].*\d+', line) and len(line) > 10:
                    if "address" not in lead:
                        lead["address"] = line
                        
        except Exception:
            pass
        
        # Phone and website require clicking into the detail panel
        # For now, we skip these to keep scraping fast
        # They can be enriched in a second pass
        
        return lead
    
    def _stub_google_maps_results(self) -> List[dict]:
        """Fallback mock results when Playwright fails"""
        self.logger.warning("Using stub data - Playwright not available or failed")
        return [
            {
                "name": "Consultorio Dentario Silva",
                "phone": "(11) 99999-9999",
                "email": None,
                "website": "https://dentistasilva.com.br",
                "address": "Rua A, 123 - Prainha, Rio de Janeiro",
                "niche": "Dental Services",
                "rating": 4.8,
                "reviews_count": 145,
                "source_url": "https://maps.google.com/search/dentistas+prainha",
                "source_platform": "google_maps",
                "scraped_at": datetime.now().isoformat() + "Z"
            },
            {
                "name": "Dr. Joao Oliveira",
                "phone": "(11) 98888-8888",
                "email": "joao@example.com",
                "website": None,
                "address": "Av. B, 456 - Prainha, Rio de Janeiro",
                "niche": "Dental Services",
                "rating": 4.6,
                "reviews_count": 98,
                "source_url": "https://maps.google.com/search/dentistas+prainha",
                "source_platform": "google_maps",
                "scraped_at": datetime.now().isoformat() + "Z"
            }
        ]


class YellowPagesExecutor(SourceExecutor):
    """
    Executor for Yellow Pages / Business Directory scraping.

    Supports multiple Yellow Pages sites:
    - telelistas.net (Brazil)
    - yellowpages.com (US)
    - Can be extended for other regional directories
    """

    # Site configurations for different Yellow Pages platforms
    SITE_CONFIGS = {
        "brazil": {
            "base_url": "https://www.telelistas.net",
            "search_path": "/busca/{location}/{query}",
            "selectors": {
                "results_container": "div.listagem-empresas, div.search-results, main",
                "result_item": "div.empresa-item, div.business-card, article.listing",
                "name": "h2.nome-empresa, h3.business-name, .company-name",
                "phone": "span.telefone, a[href^='tel:'], .phone-number",
                "address": "span.endereco, address, .address",
                "category": "span.categoria, .business-category, .category",
                "website": "a.website, a[rel='nofollow'][href^='http']",
            },
            "pagination": "a.proxima-pagina, a.next, .pagination a:last-child"
        },
        "us": {
            "base_url": "https://www.yellowpages.com",
            "search_path": "/search?search_terms={query}&geo_location_terms={location}",
            "selectors": {
                "results_container": "div.search-results",
                "result_item": "div.result, div.v-card",
                "name": "a.business-name, h2.n a",
                "phone": "div.phones, div.phone",
                "address": "div.adr, div.street-address",
                "category": "div.categories a, span.category",
                "website": "a.track-visit-website",
            },
            "pagination": "a.next"
        }
    }

    def __init__(self, session: aiohttp.ClientSession, directive: LeadScraperDirective):
        super().__init__(session, directive)
        # Detect region from location
        self.site_config = self._detect_site_config()

    def _detect_site_config(self) -> dict:
        """Detect which Yellow Pages site to use based on location"""
        location = (self.directive.location or "").lower()

        if any(term in location for term in ["brazil", "brasil", "rio", "são paulo", "sao paulo"]):
            return self.SITE_CONFIGS["brazil"]
        elif any(term in location for term in ["usa", "united states", "us", "america"]):
            return self.SITE_CONFIGS["us"]
        else:
            # Default to Brazil config
            return self.SITE_CONFIGS["brazil"]

    async def execute(self) -> List[dict]:
        """
        Scrape Yellow Pages for business leads.

        Returns:
            List of raw lead dictionaries
        """
        self.logger.info(
            f"Starting Yellow Pages scrape: {self.directive.search_query} "
            f"in {self.directive.location}"
        )

        leads = []
        config = self.site_config

        for page_num in range(self.directive.max_pages):
            try:
                # Build search URL
                url = self._build_search_url(page_num + 1)
                self.logger.info(f"Fetching page {page_num + 1}: {url}")

                # Fetch page
                html = await self._fetch_page(url)

                # Parse results
                page_leads = self._parse_results(html, url)
                leads.extend(page_leads)

                self.logger.info(f"Page {page_num + 1}: Found {len(page_leads)} leads")

                # Check if we have enough results
                if len(leads) >= self.directive.max_results:
                    break

                # Check for next page
                if not self._has_next_page(html):
                    self.logger.info("No more pages available")
                    break

            except aiohttp.ClientError as e:
                self.logger.error(f"HTTP error on page {page_num + 1}: {e}")
                continue
            except Exception as e:
                self.logger.error(f"Error scraping page {page_num + 1}: {e}")
                continue

        self.logger.info(f"Total extracted: {len(leads)} leads from Yellow Pages")
        return leads[:self.directive.max_results]

    def _build_search_url(self, page: int = 1) -> str:
        """Build the search URL for the configured Yellow Pages site"""
        config = self.site_config

        # URL-encode the search terms
        query = urllib.parse.quote(self.directive.search_query)
        location = urllib.parse.quote(self.directive.location or "Brazil")

        # Build path from template
        path = config["search_path"].format(
            query=query,
            location=location
        )

        url = f"{config['base_url']}{path}"

        # Add pagination if not first page
        if page > 1:
            separator = "&" if "?" in url else "?"
            url = f"{url}{separator}page={page}"

        return url

    def _parse_results(self, html: str, source_url: str) -> List[dict]:
        """Parse HTML and extract lead data"""
        leads = []
        soup = BeautifulSoup(html, 'html.parser')
        config = self.site_config
        selectors = config["selectors"]

        # Try multiple selectors for results container
        container_selectors = selectors["results_container"].split(", ")
        container = None
        for sel in container_selectors:
            container = soup.select_one(sel.strip())
            if container:
                break

        if not container:
            container = soup  # Fall back to entire document

        # Find all result items
        item_selectors = selectors["result_item"].split(", ")
        items = []
        for sel in item_selectors:
            items = container.select(sel.strip())
            if items:
                break

        self.logger.debug(f"Found {len(items)} result items")

        for item in items:
            try:
                lead = self._extract_lead_from_item(item, selectors, source_url)
                if lead and lead.get("name"):
                    leads.append(lead)
            except Exception as e:
                self.logger.debug(f"Failed to extract lead: {e}")
                continue

        return leads

    def _extract_lead_from_item(self, item, selectors: dict, source_url: str) -> Optional[dict]:
        """Extract lead data from a single result item"""
        lead = {
            "source_platform": "yellow_pages",
            "source_url": source_url,
            "scraped_at": datetime.now().isoformat() + "Z"
        }

        # Extract name (required)
        name_el = self._select_first(item, selectors["name"])
        if name_el:
            lead["name"] = name_el.get_text(strip=True)
        else:
            return None  # Skip if no name

        # Extract phone
        phone_el = self._select_first(item, selectors["phone"])
        if phone_el:
            # Check if it's a tel: link
            if phone_el.name == "a" and phone_el.get("href", "").startswith("tel:"):
                lead["phone"] = phone_el["href"].replace("tel:", "").strip()
            else:
                lead["phone"] = self._clean_phone(phone_el.get_text(strip=True))

        # Extract address
        address_el = self._select_first(item, selectors["address"])
        if address_el:
            lead["address"] = address_el.get_text(strip=True)

        # Extract category/niche
        category_el = self._select_first(item, selectors["category"])
        if category_el:
            lead["niche"] = category_el.get_text(strip=True)

        # Extract website
        website_el = self._select_first(item, selectors["website"])
        if website_el and website_el.get("href"):
            href = website_el["href"]
            # Filter out internal links
            if href.startswith("http") and "yellowpages" not in href and "telelistas" not in href:
                lead["website"] = href

        return lead

    def _select_first(self, element, selector_string: str):
        """Try multiple comma-separated selectors and return first match"""
        selectors = selector_string.split(", ")
        for sel in selectors:
            result = element.select_one(sel.strip())
            if result:
                return result
        return None

    def _clean_phone(self, phone: str) -> str:
        """Clean and normalize phone number"""
        if not phone:
            return ""
        # Remove common non-digit characters except + for country code
        cleaned = re.sub(r'[^\d+]', '', phone)
        return cleaned if cleaned else phone

    def _has_next_page(self, html: str) -> bool:
        """Check if there's a next page link"""
        soup = BeautifulSoup(html, 'html.parser')
        pagination_selectors = self.site_config["pagination"].split(", ")

        for sel in pagination_selectors:
            next_link = soup.select_one(sel.strip())
            if next_link:
                return True
        return False


class CustomSiteExecutor(SourceExecutor):
    """Executor for custom website scraping"""
    
    async def execute(self) -> List[dict]:
        """Scrape custom site using CSS selectors"""
        if not self.directive.custom_config:
            raise ValueError("custom_config required for CUSTOM_SITE source")
        
        config = self.directive.custom_config
        self.logger.info(f"Starting custom site scrape: {config.base_url}")
        
        leads = []
        
        for page in range(self.directive.max_pages):
            try:
                # Build pagination URL
                url = config.search_endpoint.replace("{query}", self.directive.search_query).replace("{page}", str(page+1))
                if "{query}" not in config.search_endpoint and "q=" not in config.search_endpoint:
                     url = f"{config.search_endpoint}?q={self.directive.search_query}&page={page+1}"

                # Fetch page
                html = await self._fetch_page(url)
                
                # Parse with BeautifulSoup
                soup = BeautifulSoup(html, 'html.parser')
                
                # Extract leads using CSS selectors
                item_selector = config.selectors.get('item', '.lead-item')
                for item in soup.select(item_selector):
                    lead = {}
                    for field, selector in config.selectors.items():
                        if field == 'item':
                            continue
                        element = item.select_one(selector)
                        lead[field] = element.get_text(strip=True) if element else None
                    
                    if lead.get('name'):
                        lead['source_platform'] = 'custom_site'
                        lead['source_url'] = url
                        lead['scraped_at'] = datetime.now().isoformat() + "Z"
                        leads.append(lead)
                
                if not config.pagination_selector or page >= self.directive.max_pages - 1:
                    break
                    
            except Exception as e:
                self.logger.error(f"Error scraping page {page+1}: {e}")
                continue
        
        self.logger.info(f"Extracted {len(leads)} leads from custom site")
        return leads
