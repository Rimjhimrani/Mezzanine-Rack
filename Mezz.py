import streamlit as st
import pandas as pd
import os
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Spacer, Paragraph, PageBreak, Image, KeepTogether
from reportlab.lib.units import cm
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from io import BytesIO
import subprocess
import sys
import re
import tempfile

# --- Installation of Missing Libraries ---
# This ensures that when deployed, the app installs any missing dependencies automatically.
def install_if_missing(package):
    try:
        __import__(package)
    except ImportError:
        st.info(f"Installing missing library: {package}. Please wait...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])

# Check for essential libraries
install_if_missing('reportlab')
install_if_missing('qrcode')
install_if_missing('Pillow')
install_if_missing('openpyxl') # Often needed for modern Excel files

# Now import them
import qrcode
from PIL import Image as PILImage

# --- Global Constants & Styles ---
STICKER_WIDTH = 18 * cm
STICKER_HEIGHT = 9 * cm
CONTENT_BOX_WIDTH = 18 * cm

# Define paragraph styles for consistent PDF text formatting
bold_style = ParagraphStyle(
    name='Bold', 
    fontName='Helvetica-Bold', 
    fontSize=38, 
    alignment=TA_CENTER, 
    leading=38,
    wordWrap='CJK'  # Enable word wrapping for long part numbers
)

def get_dynamic_desc_style(text):
    """
    Dynamically adjusts the font size of the description based on its length
    to ensure it fits within the allocated space.
    """
    length = len(str(text))
    
    if length <= 10: font_size = 30
    elif length <= 15: font_size = 28
    elif length <= 20: font_size = 26
    elif length <= 25: font_size = 24
    elif length <= 35: font_size = 22
    elif length <= 45: font_size = 20
    elif length <= 55: font_size = 18
    elif length <= 65: font_size = 16
    elif length <= 75: font_size = 14
    elif length <= 85: font_size = 12
    else: font_size = 10  # Minimum font size
    
    return ParagraphStyle(
        name='DescriptionDynamic',
        fontName='Helvetica',
        fontSize=font_size,
        alignment=TA_LEFT,
        leading=font_size + 2, # Line spacing
        wordWrap='CJK',
        splitLongWords=1,
    )

qty_style = ParagraphStyle(
    name='Quantity', 
    fontName='Helvetica', 
    fontSize=22, 
    alignment=TA_CENTER, 
    leading=22,
    wordWrap='CJK'
)

# --- Helper Functions ---

def clean_number_format(value):
    """
    Cleans numeric values to remove unnecessary decimals (e.g., 5.0 becomes "5").
    """
    if pd.isna(value) or value == '':
        return ''
    if isinstance(value, str):
        value = value.strip()
        try:
            num_value = float(value)
            if num_value.is_integer():
                return str(int(num_value))
            return str(num_value)
        except (ValueError, TypeError):
            return value
    if isinstance(value, (int, float)):
        if isinstance(value, float) and value.is_integer():
            return str(int(value))
        return str(value)
    return str(value)

def find_column(df_columns, keywords, default_col=None):
    """
    A flexible function to find a column in the DataFrame that matches a list of keywords.
    """
    # Create a mapping from uppercase column name to original name
    cols_map = {str(col).upper(): col for col in df_columns}
    upper_cols = list(cols_map.keys())
    
    for keyword in keywords:
        # Try exact match first
        if keyword in upper_cols:
            return cols_map[keyword]
        # Try partial match
        for col in upper_cols:
            if keyword in col:
                return cols_map[col]
    
    return default_col

def group_by_part_number(df, part_no_col, bus_model_col, qty_veh_col):
    """
    The core logic: Groups rows by part number and combines their model and quantity data.
    """
    grouped_data = []
    
    # Group the DataFrame by the part number column
    for part_no, group in df.groupby(part_no_col):
        # Use the first row as the base for description, max capacity, etc.
        base_row = group.iloc[0].copy()
        model_qty_map = {}
        
        # Iterate through all rows for this part number to collect models
        for _, row in group.iterrows():
            model = ""
            qty = ""
            
            if bus_model_col and pd.notna(row[bus_model_col]):
                model = str(row[bus_model_col]).strip().upper()
            
            if qty_veh_col and pd.notna(row[qty_veh_col]):
                qty = clean_number_format(row[qty_veh_col])

            # Logic to handle different data formats
            if model and qty:
                # If model is found in its column, associate the quantity
                try:
                    current_qty = int(model_qty_map.get(model, 0))
                    model_qty_map[model] = str(current_qty + int(qty))
                except ValueError:
                    model_qty_map[model] = qty # Fallback for non-numeric qty
            elif qty:
                # If no model in the model column, check for format "MODEL:QTY" in the qty column
                matches = re.findall(r'([A-Z0-9]+)\s*[:=]\s*(\d+)', str(qty).upper())
                if matches:
                    for m, q in matches:
                        try:
                            current_qty = int(model_qty_map.get(m, 0))
                            model_qty_map[m] = str(current_qty + int(q))
                        except ValueError:
                            model_qty_map[m] = q
        
        # Store the combined model-quantity information in a new field in the base row
        base_row['_combined_models'] = model_qty_map
        grouped_data.append(base_row)
    
    return grouped_data

def generate_qr_code(data_string):
    """Generates a QR code image from a given text string."""
    try:
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_M,
            box_size=10,
            border=4,
        )
        qr.add_data(data_string)
        qr.make(fit=True)
        qr_img = qr.make_image(fill_color="black", back_color="white")
        
        img_buffer = BytesIO()
        qr_img.save(img_buffer, format='PNG')
        img_buffer.seek(0)
        
        return Image(img_buffer, width=2.2*cm, height=2.2*cm)
    except Exception as e:
        st.error(f"Error generating QR code: {e}")
        return None

def extract_store_location_data(row_data):
    """Extracts up to 12 store location values dynamically."""
    values = []
    # Find all columns that match the 'STORE LOC' pattern
    for i in range(1, 13):
        col_name_space = f'STORE LOC {i}'
        col_name_under = f'STORE_LOC_{i}'
        
        val = None
        if col_name_space in row_data.index and pd.notna(row_data[col_name_space]):
            val = row_data[col_name_space]
        elif col_name_under in row_data.index and pd.notna(row_data[col_name_under]):
            val = row_data[col_name_under]
            
        if val is not None:
            cleaned_val = clean_number_format(val)
            if cleaned_val:
                values.append(cleaned_val)

    return values

# --- Sticker Creation ---

def create_single_sticker(row, part_no_col, desc_col, max_capacity_col, all_models):
    """Constructs the layout and content for one sticker using ReportLab."""
    # 1. Extract and clean data from the grouped row
    part_no = clean_number_format(row.get(part_no_col, ''))
    desc = str(row.get(desc_col, ''))
    max_capacity = clean_number_format(row.get(max_capacity_col, ''))
    
    store_loc_values = extract_store_location_data(row)
    full_store_location = " ".join(store_loc_values)
    
    # Get the combined model quantities stored from the grouping step
    mtm_quantities = row.get('_combined_models', {})

    # 2. Generate QR code with combined model information
    models_info = "; ".join([f"{model}: {qty}" for model, qty in mtm_quantities.items() if qty])
    
    qr_data = (
        f"Part No: {part_no}\n"
        f"Description: {desc}\n"
        f"Max Capacity: {max_capacity}\n"
        f"Store Location: {full_store_location}\n"
        f"Models & Qty: {models_info}"
    )
    qr_image = generate_qr_code(qr_data)
    
    # 3. Define the structure of the sticker using ReportLab Tables
    
    # Main info table (Part No, Description, Max Capacity)
    main_table_data = [
        ["Part No", Paragraph(f"{part_no}", bold_style)],
        ["Description", Paragraph(desc, get_dynamic_desc_style(desc))],
        ["Max capacity", Paragraph(str(max_capacity), qty_style)]
    ]
    main_table = Table(main_table_data,
                     colWidths=[CONTENT_BOX_WIDTH/3, CONTENT_BOX_WIDTH*2/3],
                     rowHeights=[2.0*cm, 2.3*cm, 1.3*cm])
    main_table.setStyle(TableStyle([
        ('GRID', (0, 0), (-1, -1), 1, colors.black), ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('ALIGN', (0, 0), (0, -1), 'CENTER'), ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (0, -1), 20), ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8), ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
    ]))

    # Store Location Table (with a nested table for flexible column count)
    if not store_loc_values: store_loc_values = [""] # Ensure table has at least one cell
    inner_table = Table([store_loc_values], colWidths=[(CONTENT_BOX_WIDTH*2/3)/len(store_loc_values)]*len(store_loc_values))
    inner_table.setStyle(TableStyle([
        ('GRID', (0, 0), (-1, -1), 1, colors.black), ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'), ('FONTSIZE', (0, 0), (-1, -1), 16),
    ]))
    
    store_loc_table = Table([["Store Location", inner_table]],
                            colWidths=[CONTENT_BOX_WIDTH/3, CONTENT_BOX_WIDTH*2/3],
                            rowHeights=[1.3*cm])
    store_loc_table.setStyle(TableStyle([
        ('GRID', (0, 0), (-1, -1), 1, colors.black), ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('ALIGN', (0, 0), (0, 0), 'CENTER'), ('FONTNAME', (0, 0), (0, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (0, 0), 20),
    ]))
    
    # Bottom section (Model boxes and QR code)
    if all_models:
        headers = all_models
        values = [mtm_quantities.get(model, "") for model in all_models]
        mtm_table = Table([headers, values], colWidths=[1.6 * cm] * len(all_models), rowHeights=[0.9*cm, 0.9*cm])
        mtm_table.setStyle(TableStyle([
            ('GRID', (0, 0), (-1, -1), 1, colors.black), ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'), ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTNAME', (0, 1), (-1, 1), 'Helvetica-Bold'), ('FONTSIZE', (0, 0), (-1, -1), 18),
        ]))
        
        total_mtm_width = len(all_models) * 1.6 * cm
        remaining_width = CONTENT_BOX_WIDTH - total_mtm_width - (2.5 * cm)
        left_spacer = remaining_width * 0.5
        right_spacer = remaining_width * 0.5

        bottom_row = Table([[mtm_table, Spacer(left_spacer, 1), qr_image, Spacer(right_spacer, 1)]],
                           colWidths=[total_mtm_width, left_spacer, 2.5*cm, right_spacer])
        bottom_row.setStyle(TableStyle([('VALIGN', (0, 0), (-1, -1), 'MIDDLE')]))
    else:
        # Fallback if no models are found: Just show the QR code
        bottom_row = Table([[qr_image]], colWidths=[CONTENT_BOX_WIDTH])
        bottom_row.setStyle(TableStyle([('ALIGN', (0, 0), (0, 0), 'RIGHT')]))

    # 4. Assemble all parts into the final sticker layout
    sticker_content = [main_table, store_loc_table, Spacer(1, 0.1*cm), bottom_row]
    
    # Use a final wrapper table to add the outer border
    sticker_table = Table([[sticker_content]], colWidths=[CONTENT_BOX_WIDTH], rowHeights=[STICKER_HEIGHT])
    sticker_table.setStyle(TableStyle([
        ('BOX', (0, 0), (-1, -1), 2, colors.black),
        ('LEFTPADDING', (0, 0), (-1, -1), 4), ('RIGHTPADDING', (0, 0), (-1, -1), 4),
        ('TOPPADDING', (0, 0), (-1, -1), 4), ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    
    return KeepTogether([sticker_table]) # Prevents a sticker from splitting across a page break

# --- Main PDF Generation Orchestrator ---

def generate_sticker_labels(excel_file_path, output_pdf_path, status_callback):
    """
    Main function to orchestrate reading the file, processing data, and building the PDF.
    """
    status_callback("Reading and processing file...")
    try:
        df = pd.read_excel(excel_file_path, engine='openpyxl') if excel_file_path.lower().endswith(('.xlsx', '.xls')) else pd.read_csv(excel_file_path)
        original_columns = df.columns.tolist()
        df.columns = [str(col).upper() for col in original_columns]
    except Exception as e:
        status_callback(f"Error reading file: {e}")
        return None

    # Identify relevant columns using the helper function
    part_no_col = find_column(df.columns, ['PART NO', 'PART_NO', 'PARTNUMBER'], default_col=df.columns[0])
    desc_col = find_column(df.columns, ['DESC'], default_col=df.columns[1])
    max_capacity_col = find_column(df.columns, ['MAX CAPACITY', 'MAX_CAPACITY', 'CAPACITY'])
    qty_veh_col = find_column(df.columns, ['QTY/VEH', 'QTY_VEH', 'QTY PER VEH'])
    bus_model_col = find_column(df.columns, ['BUS MODEL', 'BUS_MODEL', 'MODEL'])

    # Discover all unique bus models to use as headers on the stickers
    all_models = []
    if bus_model_col and bus_model_col in df.columns:
        all_models = sorted([str(m).strip().upper() for m in df[bus_model_col].dropna().unique() if str(m).strip()])[:5]
    
    # If no models found, you can define a default list
    if not all_models:
        all_models = ['D6', 'M', 'P', 'S', '55T'] # Default models if column is empty/missing
        
    status_callback("Grouping data by part number...")
    grouped_rows = group_by_part_number(df, part_no_col, bus_model_col, qty_veh_col)
    
    status_callback(f"Created {len(grouped_rows)} unique part labels from {len(df)} original rows.")

    # Create the PDF document
    doc = SimpleDocTemplate(output_pdf_path, pagesize=A4,
                          topMargin=1*cm, bottomMargin=1*cm,
                          leftMargin=1.5*cm, rightMargin=1.5*cm)
    all_elements = []
    
    # Loop through the grouped data (one item per unique part number)
    for i in range(0, len(grouped_rows), 2):
        status_callback(f"Creating stickers {i+1}-{min(i+2, len(grouped_rows))} of {len(grouped_rows)}...")
        
        sticker1 = create_single_sticker(grouped_rows[i], part_no_col, desc_col, max_capacity_col, all_models)
        all_elements.append(sticker1)
        all_elements.append(Spacer(1, 1.5*cm)) # Vertical space between stickers on a page
        
        if i + 1 < len(grouped_rows):
            sticker2 = create_single_sticker(grouped_rows[i+1], part_no_col, desc_col, max_capacity_col, all_models)
            all_elements.append(sticker2)
        
        if i + 2 < len(grouped_rows):
            all_elements.append(PageBreak())

    try:
        status_callback("Building final PDF document...")
        doc.build(all_elements)
        status_callback("PDF generated successfully!")
        return output_pdf_path
    except Exception as e:
        status_callback(f"Error building PDF: {e}")
        return None

# --- Streamlit User Interface ---

def main():
    """Defines the Streamlit application layout and user interaction."""
    st.set_page_config(page_title="Mezzanine Label Generator", page_icon="🏷️", layout="wide")
    
    st.title("🏷️ Mezzanine Label Generator")
    st.markdown(
        "<p style='font-size:18px; font-style:italic; margin-top:-10px; text-align:left;'>"
        "Designed and Developed by Agilomatrix</p>",
        unsafe_allow_html=True
    )
    st.markdown("---")
    
    st.header("📁 Step 1: Upload Your File")
    uploaded_file = st.file_uploader(
        "Choose an Excel or CSV file",
        type=['xlsx', 'xls', 'csv'],
        help="Upload the file containing your part information. The app will automatically group by part number."
    )
    
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            temp_input_path = tmp_file.name
        
        st.success(f"✅ File uploaded: **{uploaded_file.name}**")
        
        try:
            preview_df = pd.read_excel(temp_input_path, engine='openpyxl').head() if temp_input_path.lower().endswith(('.xlsx', '.xls')) else pd.read_csv(temp_input_path).head()
            st.subheader("📊 Data Preview (First 5 Rows)")
            st.dataframe(preview_df, use_container_width=True)
        except Exception as e:
            st.error(f"Error previewing file: {e}")
            return
        
        st.header("🚀 Step 2: Generate Labels")
        if st.button("🏷️ Generate PDF Labels", type="primary", use_container_width=True):
            status_container = st.empty()
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_output:
                temp_output_path = tmp_output.name
            
            def update_status(message):
                status_container.info(f"📊 {message}")
            
            try:
                result_path = generate_sticker_labels(temp_input_path, temp_output_path, update_status)
                
                if result_path:
                    with open(result_path, 'rb') as pdf_file:
                        pdf_data = pdf_file.read()
                    
                    status_container.success("✅ Labels generated successfully!")
                    
                    st.download_button(
                        label="📥 Download PDF Labels",
                        data=pdf_data,
                        file_name=f"mezzanine_labels_{os.path.splitext(uploaded_file.name)[0]}.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
                else:
                    status_container.error("❌ Failed to generate PDF labels. Please check the file format.")
            finally:
                # Clean up temporary files
                os.unlink(temp_input_path)
                if 'temp_output_path' in locals() and os.path.exists(temp_output_path):
                    os.unlink(temp_output_path)
    else:
        st.info("👆 Please upload an Excel or CSV file to get started.")
        
    st.markdown("---")
    st.markdown(
        "<p style='text-align: center; color: gray; font-size: 14px;'>"
        "© 2025 Agilomatrix - Mezzanine Label Generator (Part Number Grouping)</p>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
