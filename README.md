# Lead Scraper Engine (DOE Framework)

**Status:** Phase 1 (Foundation) Complete
**Architecture:** Directive-Orchestrator-Execution (DOE)
**Infrastructure:** Modal Serverless Python

## Overview

This project implements a production-grade **Lead Scraper Engine** using the **DOE Framework**. It demonstrates how to build deterministic, resilient, and scalable agentic workflows without relying on fragile "glue code" or unreliable no-code tools.

### The DOE Architecture

1.  **Directive (Strategy):** `src/domain/schemas.py`
    *   Defines the "Source of Truth" using Pydantic models.
    *   Input: `LeadScraperDirective` (Source, Query, Filters).
    *   Output: `LeadRecord` (Normalized data contract).

2.  **Orchestration (Decision Engine):** `src/orchestration/maestro.py`
    *   `LeadScraperOrchestrator` class.
    *   Plans the execution pipeline (Search -> Paginate -> Extract -> Filter -> Normalize).
    *   Handles resilience (retries, error logging) and metrics.

3.  **Execution (Machinery):** `src/execution/executors.py`
    *   Atomic, stateless executors for specific sources.
    *   Designed to run as serverless functions (e.g., on Modal).

## Features

- **Google Maps Scraper** - Uses Playwright for JS-rendered content
- **Yellow Pages Scraper** - Supports Brazil (telelistas.net) and US (yellowpages.com)
- **Custom Site Scraper** - CSS selector-based extraction for any website
- **n8n/Webhook Integration** - REST API endpoint for automation platforms
- **Self-Annealing** - Automatic retries with exponential backoff
- **Pydantic Validation** - Strict data contracts at every layer

## Project Structure

```
├── src
│   ├── deployment
│   │   └── app.py          # Modal Serverless Entrypoint + REST API
│   ├── domain
│   │   └── schemas.py      # Pydantic Data Contracts
│   ├── execution
│   │   └── executors.py    # Source-specific scrapers
│   └── orchestration
│       └── maestro.py      # Main Orchestrator Class
├── tests
│   └── local_run.py        # Local test runner (No Cloud required)
└── requirements.txt
```

## How to Run

### 1. Local Test (Logic Verification)
Run the local test harness to verify the orchestration logic without deploying to the cloud.

```bash
pip install -r requirements.txt
playwright install chromium
python tests/local_run.py
```

### 2. Cloud Deployment (Modal)
To deploy to Modal's serverless infrastructure:

1.  Setup Modal: `modal setup`
2.  Deploy:
    ```bash
    modal deploy src/deployment/app.py
    ```
3.  Test via REST API:
    ```bash
    curl -X POST https://your-app.modal.run/scrape_leads_api \
      -H "Content-Type: application/json" \
      -d '{"directive": {"source": "google_maps", "search_query": "dentistas", "location": "Rio de Janeiro", "max_results": 10}}'
    ```

## Supported Sources

| Source | Status | Method |
|--------|--------|--------|
| Google Maps | Complete | Playwright (headless browser) |
| Yellow Pages (Brazil) | Complete | HTTP + BeautifulSoup |
| Yellow Pages (US) | Complete | HTTP + BeautifulSoup |
| Custom Site | Complete | CSS selectors |

## Example Directive

```json
{
  "source": "google_maps",
  "search_query": "dentistas",
  "location": "Rio de Janeiro, Brazil",
  "max_pages": 2,
  "max_results": 20,
  "filters": {
    "min_reviews": 10,
    "min_rating": 4.0,
    "has_phone": true
  },
  "anti_bot": true
}
```

## Output Formats

The scraper supports both JSON and CSV output formats. Set `output_format` in your directive:

```json
{
  "source": "google_maps",
  "search_query": "dentistas",
  "location": "Rio de Janeiro",
  "output_format": "csv"
}
```

## Phase 2 Roadmap

- [x] CSV export format support
- [ ] Lead enrichment (email finder, social profiles)
- [ ] Monitoring dashboard
- [ ] Rate limiting configuration per source
