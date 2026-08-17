import streamlit as st
import pymupdf
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
import pandas as pd
import tempfile
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="Landscape Bid Plan Extractor",
    page_icon="🌿",
    layout="wide"
)

st.title("🌿 Construction Bid Plan Analyzer")
st.markdown("Upload large plan sets or specification books. The app will automatically slice the landscape pages and extract bidding data into a single master project summary.")

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
    trees: int = Field(description="Total tree quantity across all documents")
    shrubs: int = Field(description="Total shrub quantity across all documents")
    perennials_and_grasses: int = Field(description="Total perennials and grasses (including ground covers, plugs, and vines)")
    restoration: str = Field(description="Yes or No")
    seeding: str = Field(description="Yes or No")
    irrigation: str = Field(description="Yes or No")
    landscape_sheets: str = Field(description="Landscape sheet range, e.g., L101-L114, or 'None'")
    plant_schedule: str = Field(description="Found or Not Found")
    division_32: str = Field(description="Found or Not Found")
    review_required: str = Field(description="Yes or No")

# --- Helper Function: Tightened PDF Slicing ---
def slice_landscape_pages(input_pdf_path, output_pdf_path):
    doc = pymupdf.open(input_pdf_path)
    sliced_doc = pymupdf.Document()
    matched_pages = []

    keywords = ["PLANT SCHEDULE", "DIVISION 32", "LANDSCAPE QUANTITIES", "SUMMARY OF QUANTITIES"]

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text("text").upper()
        if any(keyword in text for keyword in keywords):
            sliced_doc.insert_pdf(doc, from_page=page_num, to_page=page_num)
            matched_pages.append(page_num + 1)

    if len(sliced_doc) == 0:
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text("text").upper()
            if "L-" in text or "PLANTING PLAN" in text:
                sliced_doc.insert_pdf(doc, from_page=page_num, to_page=page_num)
                matched_pages.append(page_num + 1)

    sliced_doc.save(output_pdf_path, garbage=4, deflate=True)
    return len(doc), len(sliced_doc), matched_pages

# --- UI: Drag and Drop Area ---
uploaded_files = st.file_uploader("Drop Bid PDF(s) (Plans or Specs)", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    file_names_str = ", ".join([f"`{f.name}`" for f in uploaded_files])
    st.info(f"📁 **Files Uploaded for Single Project Analysis:** {file_names_str}")

    if st.button("🚀 Analyze & Extract Master Scope", type="primary"):
        if not api_key:
            st.error("Please enter a valid Gemini API Key in the sidebar.")
        else:
            with st.spinner("Processing documents and compiling master project scope..."):
                client = genai.Client(api_key=api_key)
                content_parts = []
                
                # Process and slice every file, collecting their byte payloads
                for uploaded_file in uploaded_files:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_in:
                        tmp_in.write(uploaded_file.getbuffer())
                        temp_input_path = tmp_in.name

                    temp_output_path = temp_input_path.replace(".pdf", "_sliced.pdf")

                    total_pgs, sliced_pgs, matched_list = slice_landscape_pages(temp_input_path, temp_output_path)
                    st.success(f"⚡ `{uploaded_file.name}`: Scanned **{total_pgs}** pages. Filtered down to **{sliced_pgs}** core schedule/scope pages.")

                    with open(temp_output_path, "rb") as f:
                        pdf_bytes = f.read()

                    # Attach each sliced document part into the master request contents
                    content_parts.append(
                        types.Part.from_bytes(
                            data=pdf_bytes,
                            mime_type='application/pdf',
                        )
                    )

                    if os.path.exists(temp_input_path):
                        os.remove(temp_input_path)
                    if os.path.exists(temp_output_path):
                        os.remove(temp_output_path)

                prompt = """
                Role: Expert Commercial Landscape Estimating AI.
                Task: Analyze the attached set of documents (which represent parts of a single construction project, including plans, specs, and addenda). Cross-reference them to build a unified project summary.
                Rules:
                1. Scope Detection: Extract cumulative/exact totals for Trees, Shrubs, and Perennials/Grasses across all attached documents.
                2. Chain of Thought: If no printed total row exists, add line item quantities line-by-line.
                3. Ground Covers, Plugs, and Vines must be grouped under Perennials/Grasses.
                4. Wages: Look for 'Prevailing Wage', 'Davis-Bacon', 'Union'. If not found, output 'Non-Prevailing'.
                5. Output strictly to the structured schema provided.
                """
                content_parts.append(prompt)

                st.info("🧠 Running cross-document master analysis...")
                
                response = client.models.generate_content(
                    model='gemini-3.7-flash',
                    contents=content_parts,
                    config={
                        'response_mime_type': 'application/json',
                        'response_schema': ProjectSummary,
                    }
                )

                result_data = response.parsed.model_dump()
                result_data['Source Files'] = ", ".join([f.name for f in uploaded_files])
                st.session_state["master_analysis_result"] = result_data

# --- Display Master Unified Results ---
if "master_analysis_result" in st.session_state:
    data = st.session_state["master_analysis_result"]
    st.divider()
    st.subheader("📊 Master Unified Project Summary")
    st.markdown(f"**Source Documents Analyzed Together:** `{data.get('Source Files')}`")

    col1, col2, col3, col4 = st.columns(4)
    rec = data.get("recommendation", "N/A")
    rec_color = "green" if rec == "Qualified" else ("orange" if rec == "Review" else "red")
    
    col1.metric("Recommendation", f":{rec_color}[{rec}]")
    col2.metric("Confidence", f"{data.get('confidence', 0)}%")
    col3.metric("Wages", data.get("wages", "N/A"))
    col4.metric("Location", data.get("location", "N/A"))

    st.write(f"**Reasoning:** {data.get('reason', '')}")

    table_rows = [
        {"Category / Scope": "Trees", "Value": data.get("trees", 0)},
        {"Category / Scope": "Shrubs", "Value": data.get("shrubs", 0)},
        {"Category / Scope": "Perennials & Grasses", "Value": data.get("perennials_and_grasses", 0)},
        {"Category / Scope": "Seeding", "Value": data.get("seeding", "No")},
        {"Category / Scope": "Restoration", "Value": data.get("restoration", "No")},
        {"Category / Scope": "Irrigation", "Value": data.get("irrigation", "No")},
        {"Category / Scope": "Bid Date", "Value": data.get("bid_date", "Not Found")},
        {"Category / Scope": "Substantial Completion", "Value": data.get("substantial_completion", "Not Found")},
        {"Category / Scope": "Landscape Sheets", "Value": data.get("landscape_sheets", "None")},
        {"Category / Scope": "Plant Schedule Found", "Value": data.get("plant_schedule", "Not Found")},
        {"Category / Scope": "Division 32 Found", "Value": data.get("division_32", "Not Found")},
    ]

    df = pd.DataFrame(table_rows)
    st.dataframe(df, use_container_width=True, hide_index=True)
    
    # Format flat master row for export
    flat_data = {
        "Source Files": data.get("Source Files"), 
        "Recommendation": rec, 
        "Confidence": f"{data.get('confidence', 0)}%",
        "Wages": data.get("wages"), 
        "Location": data.get("location")
    }
    for row in table_rows:
        flat_data[row["Category / Scope"]] = row["Value"]
    
    master_df = pd.DataFrame([flat_data])

    st.divider()
    col_a, col_b = st.columns(2)
    with col_a:
        csv_data = master_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Master Project Summary (CSV)",
            data=csv_data,
            file_name="master_project_extraction_summary.csv",
            mime="text/csv",
            type="secondary"
        )
    with col_b:
        if st.button("✅ Approve & Log Master Project to Tracker", type="primary"):
            st.success("Master project data packaged and ready to sync directly to your tracking sheet!")