import streamlit as st
import pymupdf
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from typing import List
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
st.markdown("Upload large plan sets or specification books. The app will automatically slice the landscape pages, cross-reference them, and build master project summaries and PM Bid plant schedules.")

# --- API Key Setup ---
api_key = st.sidebar.text_input("Gemini API Key", type="password", help="Enter your Gemini API key")
if not api_key:
    api_key = os.environ.get("GEMINI_API_KEY", "")

# --- Schema Definition for Structured Output ---
class PlantItem(BaseModel):
    size: str = Field(description="Size (Ex: 1.5\" CAL, #15, 4\", 18\", 24\" HT, 2\" CAL)")
    type: str = Field(description="Type or Method (Ex: BR, CONT, B&B, SP)")
    variety: str = Field(description="Variety or Common Name (Ex: Bur Oak, Froebel Spirea)")
    quantity: int = Field(description="Quantity")

class ProjectSummary(BaseModel):
    recommendation: str = Field(description="Output: Qualified, Review, or Reject")
    confidence: int = Field(description="Confidence percentage from 0-100")
    reason: str = Field(description="1-2 short sentences summarizing the finding")
    location: str = Field(description="City, State")
    bid_date: str = Field(description="Exact Date/Time or 'Not Found'")
    substantial_completion: str = Field(description="Date, Timeframe, or 'Not Found'")
    wages: str = Field(description="Prevailing or Non-Prevailing")
    trees: int = Field(description="Total tree quantity across all documents")
    shrubs: int = Field(description="Total shrub quantity across all documents")
    perennials_and_grasses: int = Field(description="Total perennials/grasses/plugs across all documents")
    restoration: str = Field(description="Yes or No")
    seeding: str = Field(description="Yes or No")
    irrigation: str = Field(description="Yes or No")
    landscape_sheets: str = Field(description="Landscape sheet range, e.g., L101-L114, or 'None'")
    plant_schedule: str = Field(description="Found or Not Found")
    division_32: str = Field(description="Found or Not Found")
    review_required: str = Field(description="Yes or No")
    plant_schedule_list: List[PlantItem] = Field(description="List of all plants extracted from the most up-to-date plant schedule.")

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
    st.info(f"📁 **Project Files Uploaded:** {file_names_str}")

    if st.button("🚀 Cross-Reference & Extract Master Project", type="primary"):
        if not api_key:
            st.error("Please enter a valid Gemini API Key in the sidebar.")
        else:
            with st.spinner("Slicing and uploading all documents to secure cloud storage..."):
                client = genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=600000))
                cloud_documents = []
                
                # STEP 1: Process and upload everything first
                for uploaded_file in uploaded_files:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_in:
                        tmp_in.write(uploaded_file.getbuffer())
                        temp_input_path = tmp_in.name

                    temp_output_path = temp_input_path.replace(".pdf", "_sliced.pdf")

                    total_pgs, sliced_pgs, matched_list = slice_landscape_pages(temp_input_path, temp_output_path)
                    st.success(f"⚡ `{uploaded_file.name}` sliced from {total_pgs} to **{sliced_pgs}** core pages.")

                    # Safely upload using the Files API
                    uploaded_doc = client.files.upload(file=temp_output_path)
                    
                    while True:
                        file_info = client.files.get(name=uploaded_doc.name)
                        if "ACTIVE" in str(file_info.state):
                            cloud_documents.append(uploaded_doc)
                            break
                        elif "FAILED" in str(file_info.state):
                            st.error(f"Google failed to process {uploaded_file.name}")
                            break
                        time.sleep(3)

                    # Cleanup local temp files
                    if os.path.exists(temp_input_path): os.remove(temp_input_path)
                    if os.path.exists(temp_output_path): os.remove(temp_output_path)

                # STEP 2: Send ONE massive request to the AI with all files combined
                st.info("🧠 All documents uploaded! Running cross-document AI analysis & building Plant Schedule...")

                prompt = f"""
                Role: Expert Commercial Landscape Estimating AI.
                Task: Analyze ALL attached documents together as ONE single unified construction project. Cross-reference the plans, specs, and addenda.
                Rules:
                1. Scope Detection: Extract exact cumulative quantities for Trees, Shrubs, and Perennials/Grasses across all documents.
                2. Plant Schedule Extraction: Locate the most up-to-date plant schedule. Extract EVERY plant line item into the 'plant_schedule_list'. Map 'Size' to size, 'Method/Root' to type, 'Common Name' to variety, and 'Quantity' to quantity. If there are revised sheets or addenda covering the schedule, use the revised quantities. If 'Method' or 'Common Name' is missing, output 'N/A'.
                3. Ground Covers, Plugs, and Vines must be grouped under Perennials/Grasses.
                4. Front-End Specs: Thoroughly scan for the Master Bid Date, Substantial Completion, and Wage Rates.
                5. Wages: Look for 'Prevailing Wage', 'Davis-Bacon', 'Union'. If not found, output 'Non-Prevailing'.
                6. Output strictly to the structured schema provided.
                """
                
                contents_payload = cloud_documents + [prompt]

                # --- AUTOMATED RETRY LOOP FOR SERVER TRAFFIC JAMS ---
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        response = client.models.generate_content(
                            model='gemini-3.7-flash',
                            contents=contents_payload,
                            config={
                                'response_mime_type': 'application/json',
                                'response_schema': ProjectSummary,
                            }
                        )
                        result_data = response.parsed.model_dump()
                        result_data['Source Files'] = file_names_str
                        st.session_state["master_project_result"] = result_data
                        break  # Success! Exit loop.
                        
                    except Exception as e:
                        error_msg = str(e).upper()
                        if "503" in error_msg or "UNAVAILABLE" in error_msg or "429" in error_msg:
                            if attempt < max_retries - 1:
                                st.warning(f"Google servers are currently busy. Retrying in 15 seconds... (Attempt {attempt + 1} of {max_retries})")
                                time.sleep(15)
                            else:
                                st.error("Google servers are too busy right now. Please try again later.")
                        else:
                            st.error("Failed to generate master summary.")
                            st.write(e)
                            break 

                # STEP 3: Clean up cloud storage
                for doc in cloud_documents:
                    try:
                        client.files.delete(name=doc.name)
                    except:
                        pass

# --- Display Master Unified Table & Plant Schedule ---
if "master_project_result" in st.session_state:
    data = st.session_state["master_project_result"]
    st.divider()
    
    # --- SECTION 1: MASTER SUMMARY ---
    st.subheader("📊 Master Unified Project Summary")
    st.markdown(f"**Source Documents Analyzed:** {data.get('Source Files')}")

    col1, col2, col3, col4 = st.columns(4)
    rec = data.get("recommendation", "N/A")
    rec_color = "green" if rec == "Qualified" else ("orange" if rec == "Review" else "red")
    
    col1.metric("Recommendation", f":{rec_color}[{rec}]")
    col2.metric("Confidence", f"{data.get('confidence', 0)}%")
    col3.metric("Wages", data.get("wages", "N/A"))
    col4.metric("Location", data.get("location", "N/A"))

    st.write(f"**Reasoning:** {data.get('reason', '')}")

    # FIX: Wrap all values in str() to prevent PyArrow Mixed Type Crashes
    table_rows = [
        {"Category / Scope": "Trees", "Value": str(data.get("trees", 0))},
        {"Category / Scope": "Shrubs", "Value": str(data.get("shrubs", 0))},
        {"Category / Scope": "Perennials & Grasses", "Value": str(data.get("perennials_and_grasses", 0))},
        {"Category / Scope": "Seeding", "Value": str(data.get("seeding", "No"))},
        {"Category / Scope": "Restoration", "Value": str(data.get("restoration", "No"))},
        {"Category / Scope": "Irrigation", "Value": str(data.get("irrigation", "No"))},
        {"Category / Scope": "Bid Date", "Value": str(data.get("bid_date", "Not Found"))},
        {"Category / Scope": "Substantial Completion", "Value": str(data.get("substantial_completion", "Not Found"))},
        {"Category / Scope": "Landscape Sheets", "Value": str(data.get("landscape_sheets", "None"))},
        {"Category / Scope": "Plant Schedule", "Value": str(data.get("plant_schedule", "Not Found"))},
        {"Category / Scope": "Division 32", "Value": str(data.get("division_32", "Not Found"))},
    ]

    df = pd.DataFrame(table_rows)
    # FIX: use width='stretch' instead of use_container_width=True
    st.dataframe(df, width='stretch', hide_index=True)
    
    # --- SECTION 2: PM BID PLANT SCHEDULE ---
    st.divider()
    st.subheader("🌱 PM Bid Plant Schedule Export")
    
    plant_list = data.get("plant_schedule_list", [])
    
    # Safety Check: Only process if the list actually has items
    if plant_list:
        plant_df = pd.DataFrame(plant_list)
        
        # Rename columns to match PM Bid format
        plant_df.rename(columns={
            "size": "Size", 
            "type": "Type", 
            "variety": "Variety", 
            "quantity": "Quantity"
        }, inplace=True)
        
        # Safety Check: Ensure all columns exist before trying to reorder them
        expected_cols = ["Size", "Type", "Variety", "Quantity"]
        available_cols = [col for col in expected_cols if col in plant_df.columns]
        plant_df = plant_df[available_cols]
        
        # FIX: use width='stretch' instead of use_container_width=True
        st.dataframe(plant_df, width='stretch', hide_index=True)
        
        # Build Download Buttons
        col_a, col_b = st.columns(2)
        with col_a:
            # Plant Schedule Download
            plant_csv = plant_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Plant Schedule (CSV)",
                data=plant_csv,
                file_name="pm_bid_plant_schedule.csv",
                mime="text/csv",
                type="primary"
            )
        with col_b:
            # Master Summary Download
            flat_data = {
                "Source Files": data.get("Source Files"), 
                "Recommendation": rec, 
                "Confidence": f"{data.get('confidence', 0)}%",
                "Location": data.get("location"),
                "Wages": data.get("wages")
            }
            for row in table_rows:
                flat_data[row["Category / Scope"]] = row["Value"]
            master_df = pd.DataFrame([flat_data])
            
            summary_csv = master_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Master Summary (CSV)",
                data=summary_csv,
                file_name="master_project_extraction.csv",
                mime="text/csv",
                type="secondary"
            )
    else:
        st.warning("No plant schedule items were found or successfully extracted from these documents.")
