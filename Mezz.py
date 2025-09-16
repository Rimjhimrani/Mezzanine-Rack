import streamlit as st
import pandas as pd
import os
import sys
import re
import tempfile
import subprocess
from io import BytesIO

# --- Library Installation ---
# Check for required libraries and install them if they are missing.
def install_package(package):
    """A helper function to install Python packages."""
    st.info(f"Installing required library: {package}. Please wait...")
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
        st.success(f"Successfully installed {package}.")
    except Exception as e:
        st.error(f"Error installing {package}: {e}")
        st.stop()

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Spacer, Paragraph, PageBreak, Image, KeepTogether
    from reportlab.lib.units import cm
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.enums import TA_LEFT, TA_CENTER
except ImportError:
    install_package('reportlab')
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Spacer, Paragraph, PageBreak, Image, KeepTogether
    from reportlab.lib.units import cm
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.enums import TA_LEFT, TA_CENTER

try:
    import qrcode
except ImportError:
    install_package('qrcode')
    import qrcode

try:
    from PIL import Image as PILImage
except ImportError:
    install_package('pillow')
    from PIL import Image as PILImage


# --- Global Constants & Styles ---
STICKER_WIDTH = 18 * cm
STICKER_HEIGHT = 9 * cm

# Define Paragraph Styles for consistent text formatting in the PDF
bold_style = ParagraphStyle(
    name='Bold',
    fontName='Helvetica-Bold',
    fontSize=38,
    alignment=TA_CENTER,
    leading=38,
    wordWrap='CJK'  # Enable word wrapping for long part numbers
)

qty_style = ParagraphStyle(
    name='Quantity',
    fontName='Helvetica',
    fontSize=22,
    alignment=TA_CENTER,
    leading=22
)

def get_dynamic_desc_style(text):
    """
    Adjusts the font size of the description based on its length to ensure it fits.
    """
    length = len(text)
    if length <= 20:
        font_size = 26
    elif length <= 40:
        font_size = 22
    elif length <= 60:
        font_size = 18
    elif length <= 80:
        font_size = 14
    else:
        font_size = 12  # Minimum font size

    return ParagraphStyle(
        name='DescriptionDynamic',
        fontName='Helvetica',
        fontSize=font_size,
        alignment=TA_LEFT,
        leading=font_size + 2, # Dynamic line spacing
        wordWrap='CJK',
        splitLongWords=1,
    )

# --- Helper Functions ---

def clean_number_format(value):
    """
    Cleans numeric values to remove unnecessary decimals (e.g., 5.0 -> "5").
    """
    if pd.isna(value):
        return ''
    # If the value is a float and is a whole number, convert to an integer string
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)

def find_column(df_columns, keywords, default_index=0):
    """
    A flexible function to find a column in the DataFrame that matches a list of keywords.
    """
    for col in df_columns:
        for keyword in keywords:
            if keyword in str(col).upper():
                return col
    # If no match is found, return a default column based on index
    if len(df_columns) > default_index:
        return df_columns[default_index]
    return None

def consolidate_duplicate_parts(df, part_no_col, bus_model_col, qty_veh_col):
    """
    Core logic: Consolidates rows with the same part number into a single entry.
    It combines the bus model and quantity information for each duplicate.
    """
    if bus_model_col is None or qty_veh_col is None:
        st.warning("Bus Model or QTY/VEH column not found. Cannot consolidate duplicates.")
        return df

    # Group the DataFrame by the part number column
    grouped = df.groupby(part_no_col)
    consolidated_rows = []

    for _, group in grouped:
        if len(group) == 1:
            # If there's only one entry for this part number, keep it as is
            consolidated_rows.append(group.iloc[0])
        else:
            # If there are duplicates, create a new consolidated row
            base_row = group.iloc[0].copy()
            combined_model_quantities = {}

            for _, row in group.iterrows():
                model = str(row[bus_model_col]).strip().upper() if pd.notna(row[bus_model_col]) else ""
                qty = clean_number_format(row[qty_veh_col]) if pd.notna(row[qty_veh_col]) else ""

                if model and qty:
                    # Store the quantity for each model, overwriting if model repeats
                    combined_model_quantities[model] = qty

            # Create a combined string like "MODEL1:QTY1 MODEL2:QTY2"
            if combined_model_quantities:
                combined_qty_str = " ".join([f"{model}:{qty}" for model, qty in combined_model_quantities.items()])
                base_row[qty_veh_col] = combined_qty_str

            consolidated_rows.append(base_row)

    # Return a new DataFrame with the consolidated data
    return pd.DataFrame(consolidated_rows).reset_index(drop=True)

def detect_bus_models_and_qty(row_data, qty_veh_col):
    """
    Parses the QTY/VEH field to extract model-quantity pairs.
    Handles both single entries and the consolidated "MODEL:QTY" format.
    """
    models = {}
    qty_veh_value = str(row_data.get(qty_veh_col, ''))

    # Regex to find patterns like "M:5", "D6:2", etc.
    matches = re.findall(r'([A-Z0-9]+):(\d+)', qty_veh_value.upper())
    if matches:
        for model, qty in matches:
            models[model] = qty
            
    return models

def generate_qr_code(data_string):
    """
    Creates a QR code image from a given text string.
    """
    try:
        qr = qrcode.QRCode(version=1, error_correction=qrcode.constants.ERROR_CORRECT_M, box_size=10, border=4)
        qr.add_data(data_string)
        qr.make(fit=True)
        img = qr.make_image(fill_color="black", back_color="white")

        # Save image to a memory buffer to be used by ReportLab
        buffer = BytesIO()
        img.save(buffer, format='PNG')
        buffer.seek(0)
        return Image(buffer, width=2.2*cm, height=2.2*cm)
    except Exception:
        return None

# --- Sticker Creation ---

def create_single_sticker(row, part_no_col, desc_col, max_cap_col, qty_veh_col, all_models):
    """
    Constructs the layout and content for one sticker using ReportLab Flowables.
    """
    # 1. Extract and clean data from the DataFrame row
    part_no = clean_number_format(row.get(part_no_col, ''))
    description = str(row.get(desc_col, ''))
    max_capacity = clean_number_format(row.get(max_cap_col, ''))
    
    # Extract store location columns (up to 12)
    store_loc_values = [clean_number_format(row.get(f'STORE LOC {i}', '')) for i in range(1, 13)]
    store_loc_values = [v for v in store_loc_values if v] # Remove empty values
    full_store_location = " ".join(store_loc_values)

    # 2. Generate QR Code
    qr_data = f"Part No: {part_no}\nDesc: {description}\nLocation: {full_store_location}"
    qr_image = generate_qr_code(qr_data)

    # 3. Build the sticker layout using tables
    
    # Main info table (Part No, Description, Max Capacity)
    main_table_data = [
        ["Part No", Paragraph(part_no, bold_style)],
        ["Description", Paragraph(description, get_dynamic_desc_style(description))],
        ["Max capacity", Paragraph(max_capacity, qty_style)]
    ]
    main_table = Table(main_table_data, colWidths=[STICKER_WIDTH/3, STICKER_WIDTH*2/3], rowHeights=[2.0*cm, 2.5*cm, 1.3*cm])
    main_table.setStyle(TableStyle([
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('ALIGN', (0, 0), (0, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (0, -1), 20),
        ('LEFTPADDING', (1, 1), (1, 1), 8) # Padding for description
    ]))

    # Store Location Table (with a nested table for the values)
    if not store_loc_values: store_loc_values = [""] # Ensure table has at least one cell
    loc_cols = len(store_loc_values)
    loc_inner_table = Table([store_loc_values], colWidths=[(STICKER_WIDTH*2/3)/loc_cols]*loc_cols, rowHeights=[1.3*cm])
    loc_inner_table.setStyle(TableStyle([
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTSIZE', (0, 0), (-1, -1), 16)
    ]))
    
    store_loc_table = Table([
        [Paragraph("Store Location", ParagraphStyle(name='SL', fontName='Helvetica-Bold', fontSize=20, alignment=TA_CENTER)), loc_inner_table]
    ], colWidths=[STICKER_WIDTH/3, STICKER_WIDTH*2/3], rowHeights=[1.3*cm])
    store_loc_table.setStyle(TableStyle([
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))

    # Bottom Section (Bus Model Quantities and QR Code)
    model_quantities = detect_bus_models_and_qty(row, qty_veh_col)
    
    # Create headers and values for the model table
    model_headers = all_models
    model_values = [model_quantities.get(model, "") for model in all_models]

    model_table = Table([model_headers, model_values], colWidths=[1.6*cm]*len(all_models), rowHeights=[0.9*cm, 0.9*cm])
    model_table.setStyle(TableStyle([
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'), # Bold headers
        ('FONTNAME', (0, 1), (-1, 1), 'Helvetica-Bold'), # Bold values
        ('FONTSIZE', (0, 0), (-1, -1), 18),
    ]))

    # Combine models and QR code in a final table for alignment
    bottom_table_width = len(all_models) * 1.6*cm + 3.5*cm # Width of model boxes + spacer + QR
    spacer_width = STICKER_WIDTH - bottom_table_width
    
    bottom_table = Table([[model_table, Spacer(spacer_width, 1), qr_image if qr_image else ""]], 
                         colWidths=[len(all_models) * 1.6*cm, spacer_width, 3.5*cm])
    bottom_table.setStyle(TableStyle([('VALIGN', (0, 0), (-1, -1), 'MIDDLE')]))

    # 4. Assemble the sticker content and wrap it
    sticker_content = [main_table, store_loc_table, Spacer(1, 0.2*cm), bottom_table]
    
    # Use a final wrapper table to add the outer border
    sticker_wrapper = Table([[sticker_content]], colWidths=[STICKER_WIDTH], rowHeights=[STICKER_HEIGHT])
    sticker_wrapper.setStyle(TableStyle([
        ('BOX', (0, 0), (-1, -1), 2, colors.black), # Thick outer border
        ('LEFTPADDING', (0, 0), (-1, -1), 4),
        ('RIGHTPADDING', (0, 0), (-1, -1), 4),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4)
    ]))
    
    return KeepTogether([sticker_wrapper]) # Keep sticker from breaking across pages

# --- Main PDF Generation Function ---

def generate_pdf(input_path, output_path, status_callback):
    """
    Main orchestrator function that reads data, processes it, and builds the PDF.
    """
    try:
        status_callback("Reading data file...")
        df = pd.read_excel(input_path) if input_path.endswith(('.xlsx', '.xls')) else pd.read_csv(input_path)
        
        # Keep original column names for data access, but use uppercase for matching
        original_cols = df.columns.tolist()
        upper_cols = [str(col).upper() for col in original_cols]
        df.columns = upper_cols

    except Exception as e:
        st.error(f"Error reading the file: {e}")
        return None

    # Identify essential columns
    part_no_col = find_column(upper_cols, ['PART NO', 'PARTNO', 'PART_NO'], 0)
    desc_col = find_column(upper_cols, ['DESC'], 1)
    max_cap_col = find_column(upper_cols, ['MAX CAP', 'MAX_CAP', 'MAXIMUM'], 2)
    qty_veh_col = find_column(upper_cols, ['QTY/VEH', 'QTY_VEH'], 3)
    bus_model_col = find_column(upper_cols, ['BUS MODEL', 'BUS_MODEL', 'MODEL'], 4)

    # Consolidate duplicate parts
    status_callback("Consolidating duplicate part numbers...")
    df = consolidate_duplicate_parts(df, part_no_col, bus_model_col, qty_veh_col)
    status_callback(f"Found {len(df)} unique parts to generate labels for.")

    # Determine the unique set of bus models for the sticker headers
    all_models = set()
    for _, row in df.iterrows():
        models_found = detect_bus_models_and_qty(row, qty_veh_col)
        all_models.update(models_found.keys())
    # Limit to max 5 models and sort for consistency
    sorted_models = sorted(list(all_models))[:5]

    # Set up the PDF document
    doc = SimpleDocTemplate(output_path, pagesize=A4, topMargin=1*cm, bottomMargin=1*cm, leftMargin=1.5*cm, rightMargin=1.5*cm)
    elements = []
    
    # Process rows and create stickers (2 per page)
    for i in range(0, len(df), 2):
        status_callback(f"Processing stickers {i+1}-{min(i+2, len(df))} of {len(df)}...")
        
        # Create the first sticker
        sticker1 = create_single_sticker(df.iloc[i], part_no_col, desc_col, max_cap_col, qty_veh_col, sorted_models)
        elements.append(sticker1)
        
        # Add a vertical spacer between the two stickers on a page
        elements.append(Spacer(1, 1.5*cm))
        
        # Create the second sticker if it exists in the pair
        if i + 1 < len(df):
            sticker2 = create_single_sticker(df.iloc[i+1], part_no_col, desc_col, max_cap_col, qty_veh_col, sorted_models)
            elements.append(sticker2)
        
        # Add a page break after every pair of stickers
        if i + 2 < len(df):
            elements.append(PageBreak())
    
    try:
        status_callback("Building the final PDF document...")
        doc.build(elements)
        status_callback("PDF generated successfully!")
        return output_path
    except Exception as e:
        st.error(f"Failed to build PDF: {e}")
        return None

# --- Streamlit User Interface ---

def main():
    """Defines the Streamlit application layout and logic."""
    st.set_page_config(page_title="Mezzanine Label Generator", page_icon="🏷️", layout="wide")
    
    st.title("🏷️ Mezzanine Label Generator")
    st.markdown("An intelligent tool to create professional, QR-coded labels with automatic duplicate part consolidation.")
    st.markdown("---")

    uploaded_file = st.file_uploader(
        "**Step 1: Upload your Parts File**",
        type=['xlsx', 'xls', 'csv'],
        help="Upload an Excel or CSV file with your parts data."
    )

    if uploaded_file:
        st.success(f"File '{uploaded_file.name}' uploaded successfully.")
        
        # Use a temporary file to store the upload
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            input_file_path = tmp_file.name

        # Show a preview of the data
        try:
            preview_df = pd.read_excel(input_file_path).head() if input_file_path.endswith(('.xlsx', '.xls')) else pd.read_csv(input_file_path).head()
            st.subheader("Data Preview (First 5 Rows)")
            st.dataframe(preview_df)
        except Exception as e:
            st.error(f"Could not preview the file. Error: {e}")

        st.markdown("---")
        st.subheader("**Step 2: Generate Your Labels**")
        
        if st.button("🚀 Generate PDF Labels", type="primary"):
            with st.spinner("Generating labels... This may take a moment."):
                status_container = st.empty()
                def update_status(message):
                    status_container.info(message)
                
                # Create a temporary path for the output PDF
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_output:
                    output_file_path = tmp_output.name

                result_path = generate_pdf(input_file_path, output_file_path, update_status)
                
                # Clean up the temporary input file
                os.unlink(input_file_path)

                if result_path:
                    with open(result_path, "rb") as pdf_file:
                        pdf_bytes = pdf_file.read()

                    st.success("✅ PDF generation complete!")
                    
                    # Provide the download button
                    st.download_button(
                        label="📥 Download PDF Labels",
                        data=pdf_bytes,
                        file_name=f"Mezzanine_Labels_{os.path.splitext(uploaded_file.name)[0]}.pdf",
                        mime="application/pdf"
                    )
                    # Clean up the temporary output file
                    os.unlink(result_path)
                else:
                    st.error("❌ Failed to generate the PDF. Please check the file and try again.")
    else:
        st.info("Please upload an Excel or CSV file to begin.")
        st.markdown("""
            ### Key Features:
            - **Automatic Duplicate Consolidation:** If the same part number appears multiple times (e.g., for different bus models), it will be combined into a *single sticker*.
            - **QR Code Integration:** Each label includes a QR code for easy scanning and inventory management.
            - **Dynamic Layout:** The design adjusts to fit your data, including variable text length.
            - **Professional Output:** Creates a clean, print-ready PDF with two stickers per page.
        """)

if __name__ == "__main__":
    main()
