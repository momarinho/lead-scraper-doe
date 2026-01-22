"""
Streamlit Frontend for Lead Scraper Engine

A web interface to configure and run lead scrapes easily.

Run with: streamlit run src/frontend/app.py
"""
import streamlit as st
import requests
import pandas as pd
import json
from datetime import datetime

# Page config
st.set_page_config(
    page_title="Lead Scraper Engine",
    page_icon="🔍",
    layout="wide"
)

# API Configuration
API_URL = "https://momarinho--lead-scraper-doe-scrape-leads-api.modal.run"

# Custom CSS
st.markdown("""
<style>
    .stButton > button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    .stButton > button:hover {
        background-color: #45a049;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.title("🔍 Lead Scraper Engine")
st.markdown("**DOE Framework** - Directive / Orchestration / Execution")
st.divider()

# Sidebar - Configuration
with st.sidebar:
    st.header("⚙️ Configuration")

    # Source selection
    source = st.selectbox(
        "Data Source",
        options=["google_maps", "yellow_pages", "custom_site"],
        format_func=lambda x: {
            "google_maps": "🗺️ Google Maps",
            "yellow_pages": "📒 Yellow Pages",
            "custom_site": "🌐 Custom Site"
        }.get(x, x)
    )

    st.divider()

    # Search parameters
    st.subheader("🔎 Search")
    search_query = st.text_input(
        "Search Query",
        value="dentistas",
        placeholder="e.g., dentistas, advogados, restaurantes"
    )

    location = st.text_input(
        "Location",
        value="Rio de Janeiro",
        placeholder="e.g., Rio de Janeiro, Sao Paulo"
    )

    st.divider()

    # Limits
    st.subheader("📊 Limits")
    col1, col2 = st.columns(2)
    with col1:
        max_results = st.number_input(
            "Max Results",
            min_value=1,
            max_value=100,
            value=20
        )
    with col2:
        max_pages = st.number_input(
            "Max Pages",
            min_value=1,
            max_value=10,
            value=3
        )

    st.divider()

    # Filters (expandable)
    with st.expander("🎯 Filters (Optional)"):
        use_filters = st.checkbox("Enable Filters")

        if use_filters:
            min_rating = st.slider(
                "Minimum Rating",
                min_value=0.0,
                max_value=5.0,
                value=0.0,
                step=0.5
            )

            min_reviews = st.number_input(
                "Minimum Reviews",
                min_value=0,
                max_value=1000,
                value=0
            )

            has_phone = st.checkbox("Must have phone")
            has_website = st.checkbox("Must have website")
        else:
            min_rating = None
            min_reviews = None
            has_phone = False
            has_website = False

    st.divider()

    # Options
    st.subheader("⚡ Options")
    anti_bot = st.checkbox("Anti-bot delays", value=True)
    output_format = st.radio(
        "Output Format",
        options=["json", "csv"],
        horizontal=True
    )

# Main content area
col_main, col_info = st.columns([3, 1])

with col_info:
    st.info("""
    **How to use:**
    1. Select a data source
    2. Enter search query and location
    3. Adjust limits and filters
    4. Click **Run Scrape**
    5. Download results
    """)

    st.warning("""
    **Note:** Scraping may take 30-60 seconds depending on the number of results.
    """)

with col_main:
    # Build directive
    directive = {
        "source": source,
        "search_query": search_query,
        "location": location,
        "max_pages": max_pages,
        "max_results": max_results,
        "anti_bot": anti_bot,
        "output_format": output_format
    }

    # Add filters if enabled
    if use_filters and (min_rating or min_reviews or has_phone or has_website):
        directive["filters"] = {}
        if min_rating and min_rating > 0:
            directive["filters"]["min_rating"] = min_rating
        if min_reviews and min_reviews > 0:
            directive["filters"]["min_reviews"] = min_reviews
        if has_phone:
            directive["filters"]["has_phone"] = True
        if has_website:
            directive["filters"]["has_website"] = True

    # Show directive preview
    with st.expander("📋 Directive Preview (JSON)"):
        st.json(directive)

    # Run button
    if st.button("🚀 Run Scrape", type="primary", use_container_width=True):
        with st.spinner("Scraping in progress... This may take 30-60 seconds."):
            try:
                # Make API request
                response = requests.post(
                    API_URL,
                    json={"directive": directive},
                    timeout=300  # 5 minute timeout
                )

                result = response.json()

                if result.get("success"):
                    leads = result.get("leads", [])
                    metrics = result.get("metrics", {})

                    # Success message
                    st.success(f"✅ Found {len(leads)} leads!")

                    # Metrics row
                    metric_cols = st.columns(4)
                    with metric_cols[0]:
                        st.metric("Total Leads", len(leads))
                    with metric_cols[1]:
                        st.metric("Source", source.replace("_", " ").title())
                    with metric_cols[2]:
                        elapsed = metrics.get("elapsed_seconds", 0)
                        st.metric("Time", f"{elapsed:.1f}s")
                    with metric_cols[3]:
                        with_phone = sum(1 for l in leads if l.get("phone"))
                        st.metric("With Phone", with_phone)

                    st.divider()

                    # Results table
                    if leads:
                        # Convert to DataFrame
                        df = pd.DataFrame(leads)

                        # Reorder columns
                        priority_cols = ["name", "phone", "address", "rating", "website"]
                        other_cols = [c for c in df.columns if c not in priority_cols]
                        cols = [c for c in priority_cols if c in df.columns] + other_cols
                        df = df[cols]

                        # Display table
                        st.subheader("📊 Results")
                        st.dataframe(
                            df,
                            use_container_width=True,
                            hide_index=True
                        )

                        st.divider()

                        # Download buttons
                        st.subheader("📥 Download")
                        download_cols = st.columns(2)

                        with download_cols[0]:
                            csv_data = df.to_csv(index=False)
                            st.download_button(
                                label="📥 Download CSV",
                                data=csv_data,
                                file_name=f"leads_{search_query}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv",
                                use_container_width=True
                            )

                        with download_cols[1]:
                            json_data = json.dumps(leads, indent=2, ensure_ascii=False)
                            st.download_button(
                                label="📥 Download JSON",
                                data=json_data,
                                file_name=f"leads_{search_query}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                mime="application/json",
                                use_container_width=True
                            )

                        # Store in session state for later use
                        st.session_state["last_results"] = leads
                        st.session_state["last_metrics"] = metrics
                    else:
                        st.warning("No leads found matching your criteria. Try adjusting filters.")

                else:
                    # Error response
                    error_msg = result.get("error", "Unknown error")
                    st.error(f"❌ Scraping failed: {error_msg}")

                    if result.get("details"):
                        with st.expander("Error Details"):
                            st.code(result.get("details"))

            except requests.exceptions.Timeout:
                st.error("❌ Request timed out. The scrape is taking longer than expected.")
            except requests.exceptions.ConnectionError:
                st.error("❌ Could not connect to the API. Please check your internet connection.")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

# Footer
st.divider()
st.caption("""
**Lead Scraper Engine** - Built with DOE Framework
Sources: Google Maps, Yellow Pages (Brazil/US), Custom Sites
Powered by Modal Serverless
""")
