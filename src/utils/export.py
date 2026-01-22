import csv
import io
import json
from typing import List, Literal

from src.domain.schemas import LeadRecord


def export_leads(
    leads: List[LeadRecord],
    output_format: Literal["json", "csv"] = "json"
) -> str:
    """
    Export leads to the specified format.

    Args:
        leads: List of LeadRecord objects
        output_format: "json" or "csv"

    Returns:
        Formatted string (JSON or CSV)
    """
    if output_format == "csv":
        return _export_to_csv(leads)
    else:
        return _export_to_json(leads)


def _export_to_json(leads: List[LeadRecord]) -> str:
    """Export leads to JSON string"""
    return json.dumps(
        [lead.model_dump() for lead in leads],
        indent=2,
        default=str,
        ensure_ascii=False
    )


def _export_to_csv(leads: List[LeadRecord]) -> str:
    """Export leads to CSV string"""
    if not leads:
        return ""

    # Define CSV columns in desired order
    fieldnames = [
        "name",
        "phone",
        "email",
        "website",
        "address",
        "niche",
        "rating",
        "reviews_count",
        "source_platform",
        "source_url",
        "scraped_at"
    ]

    output = io.StringIO()
    writer = csv.DictWriter(
        output,
        fieldnames=fieldnames,
        extrasaction='ignore',
        quoting=csv.QUOTE_MINIMAL
    )

    writer.writeheader()

    for lead in leads:
        # Convert Pydantic model to dict
        row = lead.model_dump()
        writer.writerow(row)

    return output.getvalue()


def save_leads_to_file(
    leads: List[LeadRecord],
    filepath: str,
    output_format: Literal["json", "csv"] = "json"
) -> None:
    """
    Save leads to a file.

    Args:
        leads: List of LeadRecord objects
        filepath: Path to output file
        output_format: "json" or "csv"
    """
    content = export_leads(leads, output_format)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
