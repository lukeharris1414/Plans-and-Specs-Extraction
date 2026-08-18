import streamlit as st
import pymupdf
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
import pandas as pd
import tempfile
import os
import time

# --- Page Configuration ---
st.set_page_config(
    page_title="Landscape Bid Plan Extractor",
    page_icon="🌿",
    layout="wide"
)

st.title("🌿 Construction Bid Plan Analyzer")
st.markdown("Upload large plan sets or specification books. The app will automatically slice the landscape pages and compile a master project summary table.")

# --- API Key Setup ---
api_key = st.sidebar.text_input("Gemini API Key", type="password", help="Enter your Gemini API key")
if not api_key:
    api_key = os.environ.get("GEMINI_API_KEY", "")

# --- Schema Definition for Structured Output ---
class ProjectSummary(BaseModel):
    recommendation: str = Field(description="Output: Qualified, Review, or Reject")
    confidence: int = Field(description="Confidence percentage from 0-100")
    reason: str = Field(description="1-2 short sentences summarizing the finding")
    location: str = Field(description="City, State")
    bid_date: str = Field(description="Exact Date/Time or 'Not Found'")
    substantial_completion: str = Field(description="Date, Timeframe, or 'Not Found'")
    wages: str = Field(description="Prevailing or Non-Prevailing")
    trees: int = Field(description="Total tree quantity")
    shrubs: int = Field(description="Total shrub quantity")
    perennials_and_grasses: int = Field(description="Total perennials and grasses (including ground covers, plugs, and vines)")
    restoration: str = Field(description="Yes or No")
    seeding: str = Field(description="Yes or No")
    irrigation: str = Field(description="Yes or No")
    landscape_sheets: str = Field(description="Landscape sheet range, e.g., L101-L114, or 'None'")
    plant_schedule: str = Field(description="Found or Not Found")
    division_32: str = Field(description="Found or Not Found")
    review_required: str = Field(description="Yes or No")

# --- Helper Function: Smart Capped PDF Slicing ---
def slice_landscape_pages(input_pdf_path, output_pdf_path):
    doc = pymupdf.open(input_pdf_path)
    sliced_doc = pymupdf.Document()
    
    landscape_keywords = [
        "PLANT SCHEDULE", "DIVISION 32", "LANDSCAPE QUANTITIES", 
        "SUMMARY OF QUANTITIES", "L-", "PLANTING PLAN"
    ]
    
    admin_keywords = [
        "ADVERTISEMENT FOR BID", "AD FOR BID", "INVITATION TO BID",
        "SUMMARY OF WORK", "CONTRACT TIME", "SUBSTANTIAL COMPLETION", 
        "COMPLETED BY", "PREVAILING WAGE", "WAGE RATE", "DAVIS-BACON", 
        "WAGE DETERMINATION"
    ]

    pages_to_keep = set()
    admin_page_count = 0

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text("text").upper()
        
        # Always keep every landscape scope page
        if any(k in text for k in landscape_keywords):
            pages_to_keep.add(page_num)
            
        # Cap front-end specs at 25 pages to prevent header/footer runaway
        elif any(k in text for k in admin_keywords):
            if admin_page_count < 25:
                pages_to_keep.add(page_num)
                admin_page_count += 1

    # Fallback if nothing matches
    if len(pages_to_keep) == 0:
        pages_to_keep.add(0)

    for p in sorted(list(pages_to_keep)):
        sliced_doc.insert_pdf(doc, from_page=p, to_page=p)

    sliced_doc.save(output_pdf_path, garbage=4, deflate=True)
    return len(doc), len(sliced_doc), sorted(list(pages_to_keep))

# --- UI: Drag and Drop Area ---
uploaded_files = st.file_uploader("Drop Bid PDF(s) (Plans or Specs)", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    file_names_str = ", ".join([f"`{f.name}`" for f in uploaded_files])
    st.info(f"📁 **Files Uploaded:** {file_names_str}")

    if st.button("🚀 Analyze Batch & Build Master Table", type="primary"):
        if not api_key:
            st.error("Please enter a valid Gemini API Key in the sidebar.")
        else:
            results_list = []
            
            with st.spinner("Uploading large documents and processing sequentially..."):
                client = genai.Client(api_key=api_key)
                
                for uploaded_file in uploaded_files:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_in:
                        tmp_in.write(uploaded_file.getbuffer())
                        temp_input_path = tmp_in.name

                    temp_output_path = temp_input_path.replace(".pdf", "_sliced.pdf")

                    total_pgs, sliced_pgs, matched_list = slice_landscape_pages(temp_input_path, temp_output_path)
                    st.success(f"⚡ `{uploaded_file.name}`: Scanned {total_pgs} pages. Filtered down to **{sliced_pgs}** core pages.")

                    # Safely upload using the Files API
                    uploaded_doc = client.files.upload(file=temp_output_path)
                    
                    # Waiting room loop to ensure Google servers fully index the file
                    while True:
                        file_info = client.files.get(name=uploaded_doc.name)
                        if "ACTIVE" in str(file_info.state):
                            break
                        elif "FAILED" in str(file_info.state):
                            st.error(f"Google failed to process {uploaded_file.name}")
                            break
                        time.sleep(3)

                    prompt = f"""
                    Role: Expert Commercial Landscape Estimating AI.
                    Task: Analyze the attached document part ({uploaded_file.name}) for a construction project.
                    Rules:
                    1. Scope Detection: Extract exact quantities for Trees, Shrubs, and Perennials/Grasses found in this document.
                    2. Ground Covers, Plugs, and Vines must be grouped under Perennials/Grasses.
                    3. Front-End Specs: Thoroughly scan for Bid Dates, Substantial Completion parameters, and Wage Rates.
                    4. Wages: Look for 'Prevailing Wage', 'Davis-Bacon', 'Union'. If not found, output 'Non-Prevailing'.
                    5. Output strictly to the structured schema provided.
                    """

                    st.info(f"🧠 Extracting bidding data for `{uploaded_file.name}`...")

                    # --- AUTOMATED RETRY LOOP FOR SERVER TRAFFIC JAMS ---
                    max_retries = 3
                    for attempt in range(max_retries):
                        try:
                            response = client.models.generate_content(
                                model='gemini-3.7-flash',
                                contents=[uploaded_doc, prompt],
                                config={
                                    'response_mime_type': 'application/json',
                                    'response_schema': ProjectSummary,
                                }
                            )
                            result_data = response.parsed.model_dump()
                            result_data['Source File'] = uploaded_file.name
                            results_list.append(result_data)
                            break  # Success! Exit the retry loop.
                            
                        except Exception as e:
                            error_msg = str(e).upper()
                            if "503" in error_msg or "UNAVAILABLE" in error_msg:
                                if attempt < max_retries - 1:
                                    st.warning(f"Google servers are currently busy. Retrying in 10 seconds... (Attempt {attempt + 1} of {max_retries})")
                                    time.sleep(10)
                                else:
                                    st.error(f"Google servers are too busy to process `{uploaded_file.name}` right now. Please try again later.")
                            else:
                                st.error(f"Failed to generate summary for `{uploaded_file.name}`.")
                                st.write(e)
                                break  # Break loop if it's a different kind of error

                    # Clean up cloud file and local temp files
                    client.files.delete(name=uploaded_doc.name)
                    if os.path.exists(temp_input_path):
                        os.remove(temp_input_path)
                    if os.path.exists(temp_output_path):
                        os.remove(temp_output_path)

                st.session_state["batch_analysis_results"] = results_list

# --- Display Master Combined Table ---
if "batch_analysis_results" in st.session_state:
    st.divider()
    st.subheader("📊 Master Unified Project Summary Table")
    st.markdown("Each document's extracted metrics shown together for complete project visibility:")

    master_rows = []
    for data in st.session_state["batch_analysis_results"]:
        row_data = {
            "Source File": data.get("Source File"),
            "Recommendation": data.get("recommendation", "N/A"),
            "Confidence": f"{data.get('confidence', 0)}%",
            "Location": data.get("location", "N/A"),
            "Wages": data.get("wages", "N/A"),
            "Trees": data.get("trees", 0),
            "Shrubs": data.get("shrubs", 0),
            "Perennials & Grasses": data.get("perennials_and_grasses", 0),
            "Seeding": data.get("seeding", "No"),
            "Restoration": data.get("restoration", "No"),
            "Irrigation": data.get("irrigation", "No"),
            "Bid Date": data.get("bid_date", "Not Found"),
            "Substantial Completion": data.get("substantial_completion", "Not Found"),
            "Landscape Sheets": data.get("landscape_sheets", "None"),
            "Plant Schedule": data.get("plant_schedule", "Not Found"),
            "Division 32": data.get("division_32", "Not Found"),
            "Reasoning": data.get("reason", "")
        }
        master_rows.append(row_data)

    master_df = pd.DataFrame(master_rows)
    st.dataframe(master_df, use_container_width=True, hide_index=True)

    st.divider()
    col_a, col_b = st.columns(2)
    with col_a:
        csv_data = master_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Master Summary (CSV)",
            data=csv_data,
            file_name="master_batch_extraction_summary.csv",
            mime="text/csv",
            type="secondary"
        )
    with col_b:
        if st.button("✅ Approve & Log Master Batch to Tracker", type="primary"):
            st.success("Master batch table packaged and ready to sync directly to your tracking sheet!")
