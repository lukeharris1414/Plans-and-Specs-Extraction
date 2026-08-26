import streamlit as st
import pymupdf
import pytesseract
from PIL import Image
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from typing import List
import pandas as pd
import tempfile
import os
import time
import json
from datetime import datetime
from streamlit_gsheets import GSheetsConnection

# --- Page Configuration ---
st.set_page_config(
    page_title="Plans and Specs Extraction",
    page_icon="🌿",
    layout="wide"
)

st.title("🌿 Plans and Specs Extraction")
st.markdown("Ingest construction documents, extract landscape scopes, and manage active bids across the team.")

# --- API Key Setup ---
api_key = st.secrets["GEMINI_API_KEY"]  # <--- THIS IS THE CRITICAL LINE

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
        "SUMMARY OF QUANTITIES", "L-", "PLANTING PLAN", "MASTER PLANT SCHEDULE", "PLANT"
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
        
        # --- AGGRESSIVE OCR FALLBACK ---
        # If the page has less than 400 digital characters, it is likely a flattened image or scan.
        if len(text.strip()) < 400:
            try:
                pix = page.get_pixmap(dpi=150) 
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                # Append the OCR text to whatever digital text already existed
                text += " " + pytesseract.image_to_string(img).upper()
            except Exception as e:
                pass # If OCR fails, ignore and move on
        # -------------------------------
        
        if any(k in text for k in landscape_keywords):
            pages_to_keep.add(page_num)
        elif any(k in text for k in admin_keywords):
            if admin_page_count < 25:
                pages_to_keep.add(page_num)
                admin_page_count += 1

    if len(pages_to_keep) == 0:
        pages_to_keep.add(0)

    for p in sorted(list(pages_to_keep)):
        sliced_doc.insert_pdf(doc, from_page=p, to_page=p)

    sliced_doc.save(output_pdf_path, garbage=4, deflate=True)
    return len(doc), len(sliced_doc), sorted(list(pages_to_keep))

# --- Main UI: Two Tabs ---
tab1, tab2 = st.tabs(["🌿 Analyze New Project", "🗄️ Pending Bid Sets"])

# ==========================================
# TAB 1: INGESTION & ANALYSIS
# ==========================================
with tab1:
    uploaded_files = st.file_uploader("Drop Bid PDF(s) (Plans or Specs)", type=["pdf"], accept_multiple_files=True)

    if uploaded_files:
        file_names_str = ", ".join([f"`{f.name}`" for f in uploaded_files])
        st.info(f"📁 **Project Files Uploaded:** {file_names_str}")

        if st.button("🚀 Analyze & Auto-Log to Tracker", type="primary"):
            if not api_key:
                st.error("Please enter a valid Gemini API Key in the sidebar.")
            else:
                with st.spinner("Slicing and uploading all documents to secure cloud storage..."):
                    client = genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=600000))
                    cloud_documents = []
                    
                    for uploaded_file in uploaded_files:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_in:
                            tmp_in.write(uploaded_file.getbuffer())
                            temp_input_path = tmp_in.name

                        temp_output_path = temp_input_path.replace(".pdf", "_sliced.pdf")

                        total_pgs, sliced_pgs, matched_list = slice_landscape_pages(temp_input_path, temp_output_path)
                        st.success(f"⚡ `{uploaded_file.name}` sliced to **{sliced_pgs}** core pages.")

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

                        if os.path.exists(temp_input_path): os.remove(temp_input_path)
                        if os.path.exists(temp_output_path): os.remove(temp_output_path)

                    st.info("🧠 Running cross-document AI analysis & building Plant Schedule...")

                    prompt = """
                    Role: Expert Commercial Landscape Estimating AI.
                    Task: Analyze ALL attached documents together as ONE single unified construction project.
                    Rules:
                    1. Scope Detection: Extract exact cumulative quantities for Trees, Shrubs, and Perennials/Grasses across all documents.
                    2. Plant Schedule Extraction: Locate the most up-to-date plant schedule. Extract EVERY plant line item into the 'plant_schedule_list'. Map 'Size' to size, 'Method/Root' to type, 'Common Name' to variety, and 'Quantity' to quantity. If 'Method' or 'Common Name' is missing, output 'N/A'.
                    3. Ground Covers, Plugs, and Vines must be grouped under Perennials/Grasses.
                    4. Front-End Specs: Scan for the Master Bid Date, Substantial Completion, and Wage Rates. Output Bid Date in an easily readable format (e.g. 'Oct 15, 2026 2:00 PM').
                    5. Wages: Look for 'Prevailing Wage', 'Davis-Bacon', 'Union'. If not found, output 'Non-Prevailing'.
                    6. Output strictly to the structured schema provided.
                    """
                    
                    contents_payload = cloud_documents + [prompt]
                    result_data = None

                    # --- AUTOMATED RETRY LOOP ---
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
                            break 
                            
                        except Exception as e:
                            error_msg = str(e).upper()
                            if "503" in error_msg or "UNAVAILABLE" in error_msg or "429" in error_msg:
                                if attempt < max_retries - 1:
                                    st.warning(f"Google servers busy. Retrying in 15s... (Attempt {attempt + 1}/{max_retries})")
                                    time.sleep(15)
                                else:
                                    st.error("Google servers are too busy right now. Please try again later.")
                            else:
                                st.error("Failed to generate master summary.")
                                st.write(e)
                                break 

                    for doc in cloud_documents:
                        try: client.files.delete(name=doc.name)
                        except: pass

                # --- AUTOMATED DATABASE LOGGING ---
                if result_data:
                    with st.spinner("Connecting to Google Sheets and logging project..."):
                        try:
                            conn = st.connection("gsheets", type=GSheetsConnection)
                            
                            # Read existing data to append to it
                            try:
                                existing_data = conn.read(worksheet="Sheet1", ttl=0)
                            except:
                                existing_data = pd.DataFrame() # Fallback if sheet is totally empty

                            # Generate the Project Name: "Bid Date_City"
                            raw_date = str(result_data.get("bid_date", "Unknown Date")).replace(",", "")
                            raw_loc = str(result_data.get("location", "Unknown Location")).split(",")[0]
                            project_name = f"{raw_date}_{raw_loc}"

                            # Package the new row
                            new_row = {
                                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "Project_Name": project_name,
                                "Source_Files": file_names_str,
                                "Recommendation": str(result_data.get("recommendation", "N/A")),
                                "Confidence": str(result_data.get("confidence", "0")) + "%",
                                "Bid_Date": str(result_data.get("bid_date", "Not Found")),
                                "Location": str(result_data.get("location", "Not Found")),
                                "Wages": str(result_data.get("wages", "N/A")),
                                "Trees": str(result_data.get("trees", "0")),
                                "Shrubs": str(result_data.get("shrubs", "0")),
                                "Perennials": str(result_data.get("perennials_and_grasses", "0")),
                                "Seeding": str(result_data.get("seeding", "No")),
                                "Restoration": str(result_data.get("restoration", "No")),
                                "Irrigation": str(result_data.get("irrigation", "No")),
                                "Landscape_Sheets": str(result_data.get("landscape_sheets", "None")),
                                "Substantial_Completion": str(result_data.get("substantial_completion", "Not Found")),
                                "Reasoning": str(result_data.get("reason", "")),
                                "Plant_Schedule_JSON": json.dumps(result_data.get("plant_schedule_list", []))
                            }

                            # Append and Update
                            new_df = pd.DataFrame([new_row])
                            if existing_data.empty:
                                updated_data = new_df
                            else:
                                updated_data = pd.concat([existing_data, new_df], ignore_index=True)
                            
                            conn.update(worksheet="Sheet1", data=updated_data)
                            st.success(f"✅ **{project_name}** successfully processed and injected into the database! Head over to the 'Pending Bid Sets' tab to view it.")
                            st.balloons()
                            
                        except Exception as e:
                            st.error("AI Extracted the data, but failed to write to Google Sheets. Check your Secrets formatting and Share permissions!")
                            st.write(e)


# ==========================================
# TAB 2: ACTIVE BID BOARD
# ==========================================
with tab2:
    st.header("🗄️ Pending Bid Sets")
    st.markdown("Projects extracted by the estimating team, ready for Sales & Marketing review.")
    
    if st.button("🔄 Refresh Database", type="secondary"):
        st.cache_data.clear() # Clears cache to pull the most recent sheet data

    try:
        conn = st.connection("gsheets", type=GSheetsConnection)
        df = conn.read(worksheet="Sheet1", ttl=0) # ttl=0 forces live data pull
        
        if df.empty or "Project_Name" not in df.columns:
            st.info("No projects have been logged yet. Upload a bid set in the first tab to get started!")
        else:
            # Drop empty rows that Google Sheets sometimes creates
            df = df.dropna(subset=["Project_Name"])
            
            # --- Auto-Sorting Dates ---
            active_bids = []
            expired_bids = []
            
            for index, row in df.iterrows():
                bid_date_str = str(row["Bid_Date"])
                is_expired = False
                
                # Attempt to parse date to see if it has passed
                try:
                    # Very basic fuzzy parsing check for standard dates
                    parsed_date = pd.to_datetime(bid_date_str, fuzzy=True)
                    if parsed_date < datetime.now():
                        is_expired = True
                except:
                    # If AI returned text like "Next Tuesday" or "Not Found", keep it in Active
                    pass
                
                if is_expired:
                    expired_bids.append(row)
                else:
                    active_bids.append(row)

            # --- Render Active Bids ---
            st.subheader(f"🟢 Active Projects ({len(active_bids)})")
            for row in reversed(active_bids): # Show newest first
                with st.expander(f"🏗️ {row['Project_Name']} | Logged: {row.get('Timestamp', 'Unknown')}", expanded=False):
                    
                    # Top Metrics
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Recommendation", row.get("Recommendation", "N/A"))
                    c2.metric("Bid Date", row.get("Bid_Date", "N/A"))
                    c3.metric("Wages", row.get("Wages", "N/A"))
                    c4.metric("Location", row.get("Location", "N/A"))
                    
                    st.write(f"**Reasoning:** {row.get('Reasoning', '')}")
                    
                    # Project Quantities Table
                    st.markdown("**Project Scope Overview**")
                    scope_data = {
                        "Trees": row.get("Trees", "0"),
                        "Shrubs": row.get("Shrubs", "0"),
                        "Perennials": row.get("Perennials", "0"),
                        "Seeding": row.get("Seeding", "No"),
                        "Restoration": row.get("Restoration", "No"),
                        "Irrigation": row.get("Irrigation", "No"),
                        "Landscape Sheets": row.get("Landscape_Sheets", "None"),
                        "Completion": row.get("Substantial_Completion", "Not Found")
                    }
                    st.dataframe(pd.DataFrame([scope_data]), width='stretch', hide_index=True)
                    
                    # Extracted Plant Schedule
                    st.markdown("**PM Bid Plant Schedule**")
                    try:
                        raw_json = row.get("Plant_Schedule_JSON", "[]")
                        plant_list = json.loads(raw_json)
                        if plant_list:
                            plant_df = pd.DataFrame(plant_list)
                            plant_df.rename(columns={"size": "Size", "type": "Type", "variety": "Variety", "quantity": "Quantity"}, inplace=True)
                            expected_cols = ["Size", "Type", "Variety", "Quantity"]
                            available_cols = [col for col in expected_cols if col in plant_df.columns]
                            st.dataframe(plant_df[available_cols], width='stretch', hide_index=True)
                        else:
                            st.warning("No plant schedule items found for this project.")
                    except:
                        st.error("Could not load plant schedule data.")
            
            # --- Render Expired Bids ---
            st.write("---")
            with st.expander(f"🔴 Expired / Past Bids ({len(expired_bids)})"):
                for row in reversed(expired_bids):
                    st.write(f"**{row['Project_Name']}** (Bid Date: {row['Bid_Date']})")

    except Exception as e:
        st.warning("Could not connect to the database. Make sure your Streamlit Secrets and Google Sheet sharing permissions are correct.")
        st.write(e)
