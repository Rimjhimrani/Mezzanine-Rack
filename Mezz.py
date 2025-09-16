import streamlit as st
import pandas as pd
import os
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Spacer, Paragraph, PageBreak, Image, KeepTogether
from reportlab.lib.units import cm, inch
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.lib.utils import ImageReader
from io import BytesIO
import subprocess
import sys
import re
import tempfile

# Define sticker dimensions - Updated for 2 per page
STICKER_WIDTH = 18 * cm
STICKER_HEIGHT = 9 * cm
STICKER_PAGESIZE = A4

# Define content box dimensions (reduced to fit 2 per page)
CONTENT_BOX_WIDTH = 18 * cm
CONTENT_BOX_HEIGHT = 9 * cm

# Check for PIL and install if needed
try:
    from PIL import Image as PILImage
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    st.write("PIL not available. Installing...")
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'pillow'])
    from PIL import Image as PILImage
    PIL_AVAILABLE = True

# Check for QR code library and install if needed
try:
    import qrcode
    QR_AVAILABLE = True
except ImportError:
    QR_AVAILABLE = False
    st.write("qrcode not available. Installing...")
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'qrcode'])
    import qrcode
    QR_AVAILABLE = True

# Define paragraph styles - UPDATED FONT SIZES WITH WORD WRAPPING
bold_style = ParagraphStyle(
    name='Bold', 
    fontName='Helvetica-Bold', 
    fontSize=38, 
    alignment=TA_CENTER, 
    leading=38,
    wordWrap='CJK'  # Enable word wrapping
)

def get_dynamic_desc_style(text):
    """
    Improved dynamic font sizing with more granular steps.
    Font size decreases gradually from 30px to 10px based on text length.
    Box dimensions remain unchanged.
    """
    length = len(text)
    
    # More granular font size breakpoints for better scaling
    if length <= 10:
        font_size = 30
    elif length <= 15:
        font_size = 28
    elif length <= 20:
        font_size = 26
    elif length <= 25:
        font_size = 24
    elif length <= 35:
        font_size = 22
    elif length <= 45:
        font_size = 20
    elif length <= 55:
        font_size = 18
    elif length <= 65:
        font_size = 16
    elif length <= 75:
        font_size = 14
    elif length <= 85:
        font_size = 12
    else:
        font_size = 10  # Minimum font size
    
    # Calculate appropriate leading (line spacing)
    leading = font_size + 2
    
    return ParagraphStyle(
        name='DescriptionDynamic',
        fontName='Helvetica',
        fontSize=font_size,
        alignment=TA_LEFT,
        leading=leading,
        wordWrap='CJK',  # Enable word wrapping
        splitLongWords=1,  # Allow splitting of long words
        spaceBefore=3,
        spaceAfter=3,
        allowWidows=1,    # Allow single lines at end of paragraph
        allowOrphans=1,   # Allow single lines at start of paragraph
    )
qty_style = ParagraphStyle(
    name='Quantity', 
    fontName='Helvetica', 
    fontSize=22, 
    alignment=TA_CENTER, 
    leading=22,
    wordWrap='CJK'  # Enable word wrapping
)

def clean_number_format(value):
    """
    Clean number formatting to preserve integers and handle decimals properly.
    Returns string representation maintaining original number format.
    """
    if pd.isna(value) or value == '':
        return ''
    
    # Handle string values
    if isinstance(value, str):
        value = value.strip()
        if value == '':
            return ''
        # Try to convert to number to check if it's numeric
        try:
            num_value = float(value)
            # If it's a whole number, return as integer
            if num_value.is_integer():
                return str(int(num_value))
            else:
                return str(num_value)
        except:
            return value
    
    # Handle numeric values
    if isinstance(value, (int, float)):
        # If it's a float that represents a whole number
        if isinstance(value, float) and value.is_integer():
            return str(int(value))
        # If it's already an integer
        elif isinstance(value, int):
            return str(value)
        # If it's a decimal
        else:
            return str(value)
    
    return str(value)

def find_bus_model_column(df_columns):
    """Enhanced function to find the bus model column with better detection"""
    cols = [str(col).upper() for col in df_columns]
    
    patterns = [
        lambda col: col == 'BUS_MODEL',
        lambda col: col == 'BUSMODEL',
        lambda col: col == 'BUS MODEL',
        lambda col: col == 'MODEL',
        lambda col: col == 'BUS_TYPE',
        lambda col: col == 'BUSTYPE',
        lambda col: col == 'BUS TYPE',
        lambda col: col == 'VEHICLE_TYPE',
        lambda col: col == 'VEHICLETYPE',
        lambda col: col == 'VEHICLE TYPE',
        lambda col: 'BUS' in col and 'MODEL' in col,
        lambda col: 'BUS' in col and 'TYPE' in col,
        lambda col: 'VEHICLE' in col and 'MODEL' in col,
        lambda col: 'VEHICLE' in col and 'TYPE' in col,
        lambda col: 'MODEL' in col,
        lambda col: 'BUS' in col,
        lambda col: 'VEHICLE' in col,
    ]
    
    for pattern in patterns:
        for i, col in enumerate(cols):
            if pattern(col):
                return df_columns[i]
    
    return None

def group_by_part_number(df, part_no_col, bus_model_col, qty_veh_col, all_models):
    """
    Group rows by part number and combine their models and quantities
    """
    grouped_data = []
    
    # Group by part number
    for part_no, group in df.groupby(part_no_col):
        # Take the first row as base (for description, max capacity, store location, etc.)
        base_row = group.iloc[0].copy()
        
        # Collect all model-quantity pairs for this part number
        model_qty_map = {}
        
        for _, row in group.iterrows():
            if bus_model_col and bus_model_col in row and pd.notna(row[bus_model_col]):
                model = str(row[bus_model_col]).strip().upper()
                qty = clean_number_format(row[qty_veh_col]) if qty_veh_col and pd.notna(row[qty_veh_col]) else ""
                
                if model:
                    # If model already exists, combine quantities (you can modify this logic)
                    if model in model_qty_map:
                        existing_qty = model_qty_map[model]
                        if existing_qty and qty:
                            try:
                                combined_qty = int(existing_qty) + int(qty)
                                model_qty_map[model] = str(combined_qty)
                            except:
                                model_qty_map[model] = f"{existing_qty},{qty}"
                        elif qty:
                            model_qty_map[model] = qty
                    else:
                        model_qty_map[model] = qty
        
        # Store the combined model-quantity information in the base row
        base_row['_combined_models'] = model_qty_map
        grouped_data.append(base_row)
    
    return grouped_data

def detect_bus_model_and_qty_combined(row, all_models):
    """
    Extract combined bus models and quantities from grouped data
    """
    if '_combined_models' in row:
        return row['_combined_models']
    
    # Fallback to original logic if no combined data
    return {}

def generate_qr_code(data_string):
    """Generate a QR code from the given data string"""
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

def extract_store_location_data_from_excel(row_data, max_cells=12):
    """Extract up to 12 store location values dynamically"""
    values = []
    
    def get_clean_value(possible_names):
        for name in possible_names:
            if name in row_data:
                val = row_data[name]
                if pd.notna(val) and str(val).strip().lower() not in ['nan', 'none', 'null', '']:
                    return clean_number_format(val)
            for col in row_data.index:
                if isinstance(col, str) and col.upper() == name.upper():
                    val = row_data[col]
                    if pd.notna(val) and str(val).strip().lower() not in ['nan', 'none', 'null', '']:
                        return clean_number_format(val)
        return None

    # Loop through possible Store Loc 1 → Store Loc 12
    for i in range(1, max_cells + 1):
        val = get_clean_value([f'Store Loc {i}', f'STORE_LOC_{i}'])
        if val:
            values.append(val)

    return values

def create_single_sticker(row, part_no_col, desc_col, max_capacity_col, qty_veh_col, store_loc_col, bus_model_col, all_models):
    """Create a single sticker layout with border around the entire sticker"""
    # Extract data with proper number formatting
    part_no = clean_number_format(row[part_no_col]) if pd.notna(row[part_no_col]) else ""
    desc = str(row[desc_col]).strip() if pd.notna(row[desc_col]) else ""
    
    max_capacity = ""
    if max_capacity_col and max_capacity_col in row and pd.notna(row[max_capacity_col]):
        max_capacity = clean_number_format(row[max_capacity_col])
        
    qty_veh = ""
    if qty_veh_col and qty_veh_col in row and pd.notna(row[qty_veh_col]):
        qty_veh = clean_number_format(row[qty_veh_col])
    
    # Get all store location parts for table and QR code
    store_loc_values = extract_store_location_data_from_excel(row)
    full_store_location = " ".join([str(v) for v in store_loc_values if v])  # join non-empty values
    
    # Use enhanced bus model detection for combined data
    mtm_quantities = detect_bus_model_and_qty_combined(row, all_models)

    # ✅ Generate QR code with combined model information
    models_info = ""
    if mtm_quantities:
        models_list = [f"{model}: {qty}" for model, qty in mtm_quantities.items() if qty]
        models_info = "; ".join(models_list)
    
    qr_data = (
        f"Part No: {part_no}\n"
        f"Description: {desc}\n"
        f"Max Capacity: {max_capacity}\n"
        f"Store Location: {full_store_location}\n"
        f"Models & Qty: {models_info}"
    )
    
    qr_image = generate_qr_code(qr_data)
    
    sticker_content = []
    
    # Define row heights - ADJUSTED FOR BETTER FIT
    header_row_height = 2.0*cm  # Increased for better spacing
    desc_row_height = 1.8*cm    # Increased for better text wrapping
    max_capacity_row_height = 1.32*cm
    store_loc_row_height = 1.3*cm

    # Main table data with improved Part No styling and proper wrapping
    main_table_data = [
        ["Part No", Paragraph(f"{part_no}", bold_style)],
        ["Description", Paragraph(desc, get_dynamic_desc_style(desc))],
        ["Max capacity", Paragraph(str(max_capacity), qty_style)]
    ]

    # Create main table with IMPROVED PADDING and wrapping
    main_table = Table(main_table_data,
                     colWidths=[CONTENT_BOX_WIDTH/3, CONTENT_BOX_WIDTH*2/3],
                     rowHeights=[header_row_height, desc_row_height, max_capacity_row_height])

    # UPDATED TABLE STYLE WITH PROPER PADDING AND WRAPPING SUPPORT
    main_table.setStyle(TableStyle([
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (0, -1), 20),
        # ADDED PROPER PADDING FOR ALL CELLS
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        # Enable text wrapping in cells
        ('WORDWRAP', (0, 0), (-1, -1), True),
    ]))

    sticker_content.append(main_table)

    store_loc_label = Paragraph("Store Location", ParagraphStyle(
        name='StoreLoc', fontName='Helvetica-Bold', fontSize=20, alignment=TA_CENTER
    ))
    store_loc_values = [v for v in extract_store_location_data_from_excel(row) if v]  # keep only non-empty
    if not store_loc_values:
        store_loc_values = [""]

    inner_table_width = CONTENT_BOX_WIDTH * 2 / 3
    num_cols = len(store_loc_values)
    inner_col_widths = [inner_table_width / num_cols] * num_cols

    store_loc_inner_table = Table(
        [store_loc_values],
        colWidths=inner_col_widths,
        rowHeights=[store_loc_row_height]
    )

    store_loc_inner_table.setStyle(TableStyle([
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 16),
        ('LEFTPADDING', (0, 0), (-1, -1), 1),
        ('RIGHTPADDING', (0, 0), (-1, -1), 1),
        ('TOPPADDING', (0, 0), (-1, -1), 2),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
        ('WORDWRAP', (0, 0), (-1, -1), True),
    ]))

    store_loc_table = Table(
        [[store_loc_label, store_loc_inner_table]],
        colWidths=[CONTENT_BOX_WIDTH/3, inner_table_width],
        rowHeights=[store_loc_row_height]
    )

    store_loc_table.setStyle(TableStyle([
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
    ]))
    
    sticker_content.append(store_loc_table)

    # Add small spacer
    sticker_content.append(Spacer(1, 0.1*cm))

    # Bottom section - MTM boxes and QR code (UPDATED FOR COMBINED MODELS)
    max_models = 5
    mtm_box_width = 1.6 * cm
    mtm_row_height = 1.8 * cm

    headers = []
    values = []

    # Fill boxes with combined model data
    for i in range(max_models):
        if i < len(all_models):
            model_name = all_models[i]
            # Get quantity from combined data
            qty_val = mtm_quantities.get(model_name, "")
            headers.append(model_name)
            values.append(Paragraph(
                f"<b>{qty_val}</b>" if qty_val else "",
                ParagraphStyle(name=f"Qty_{model_name}", fontName='Helvetica-Bold', fontSize=16, alignment=TA_CENTER)
            ))
        else:
            headers.append("")
            values.append("")
            
    position_matrix_data = [headers, values]

    mtm_table = Table(
        position_matrix_data,
        colWidths=[mtm_box_width] * max_models,
        rowHeights=[mtm_row_height/2, mtm_row_height/2]
    )
    mtm_table.setStyle(TableStyle([
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 18),
    ]))

    # QR code table (IMPROVED CENTERING)
    qr_width = 2.5*cm
    qr_height = 2.5*cm

    if qr_image:
        qr_image.drawWidth = qr_width
        qr_image.drawHeight = qr_height
        qr_table = Table(
            [[qr_image]],
            colWidths=[qr_width],
            rowHeights=[qr_height]
        )
    else:
        qr_table = Table(
            [[Paragraph("QR", ParagraphStyle(
                name='QRPlaceholder', fontName='Helvetica-Bold', fontSize=14, alignment=TA_CENTER
            ))]],
            colWidths=[qr_width],
            rowHeights=[qr_height]
        )

    qr_table.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))

    # Calculate spacing for better QR code centering (UPDATED FOR 4 BOXES)
    total_mtm_width = max_models * mtm_box_width
    remaining_width = CONTENT_BOX_WIDTH - total_mtm_width - qr_width
    
    # Split the remaining space more evenly for better centering
    left_spacer_width = remaining_width * 0.3   # 30% on left
    right_spacer_width = remaining_width * 0.7  # 70% on right

    # Combine MTM boxes and QR code in one row with better spacing
    bottom_row = Table(
        [[mtm_table, "", qr_table, ""]],
        colWidths=[total_mtm_width, left_spacer_width, qr_width, right_spacer_width],
        rowHeights=[max(mtm_row_height, qr_height)]
    )

    bottom_row.setStyle(TableStyle([
        ('ALIGN', (0, 0), (0, 0), 'LEFT'),     # MTM boxes aligned left
        ('ALIGN', (2, 0), (2, 0), 'CENTER'),   # QR code centered
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))

    sticker_content.append(bottom_row)
    
    # Create outer table that wraps all sticker content with border
    sticker_table = Table(
        [[sticker_content]],
        colWidths=[CONTENT_BOX_WIDTH],
        rowHeights=[CONTENT_BOX_HEIGHT]
    )
    
    # Add border around the entire sticker with NO PADDING (to prevent content touching border)
    sticker_table.setStyle(TableStyle([
        ('BOX', (0, 0), (-1, -1), 2, colors.black),  # Outer border for entire sticker
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('LEFTPADDING', (0, 0), (-1, -1), 4),   # Reduced padding
        ('RIGHTPADDING', (0, 0), (-1, -1), 4),  # Reduced padding
        ('TOPPADDING', (0, 0), (-1, -1), 0),    # Reduced padding
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4), # Reduced padding
    ]))
    
    # Wrap in KeepTogether to prevent page breaks within a sticker
    return KeepTogether([sticker_table])

def generate_sticker_labels(excel_file_path, output_pdf_path, status_callback=None):
    """Generate sticker labels with QR code from Excel data - 2 per page, grouped by part number"""
    if status_callback:
        status_callback(f"Processing file: {excel_file_path}")

    # Load the Excel data
    try:
        if excel_file_path.lower().endswith('.csv'):
            df = pd.read_csv(excel_file_path)
        else:
            try:
                df = pd.read_excel(excel_file_path)
            except Exception as e:
                try:
                    df = pd.read_excel(excel_file_path, engine='openpyxl')
                except Exception as e2:
                    df = pd.read_csv(excel_file_path, encoding='latin1')

        if status_callback:
            status_callback(f"Successfully read file with {len(df)} rows")
    except Exception as e:
        error_msg = f"Error reading file: {e}"
        if status_callback:
            status_callback(error_msg)
        return None

    # Identify columns
    original_columns = df.columns.tolist()
    df.columns = [col.upper() if isinstance(col, str) else col for col in df.columns]
    cols = df.columns.tolist()

    # Find relevant columns
    part_no_col = next((col for col in cols if 'PART' in col and ('NO' in col or 'NUM' in col or '#' in col)),
                   next((col for col in cols if col in ['PARTNO', 'PART']), cols[0]))

    desc_col = next((col for col in cols if 'DESC' in col),
                   next((col for col in cols if 'NAME' in col), cols[1] if len(cols) > 1 else part_no_col))

    max_capacity_col = next((col for col in cols if 'MAX' in col and 'CAPACITY' in col), 
                           next((col for col in cols if 'CAPACITY' in col),
                           next((col for col in cols if 'QTY' in col), None)))
    
    store_loc_col = next((col for col in cols if 'STORE' in col and 'LOC' in col),
                      next((col for col in cols if 'STORELOCATION' in col), None))

    qty_veh_col = next((col for col in cols if any(term in col for term in ['QTY/VEH', 'QTY_VEH', 'QTY PER VEH', 'QTYVEH', 'QTYPERCAR', 'QTYCAR', 'QTY/CAR'])), None)

    bus_model_col = find_bus_model_column(original_columns)

    if bus_model_col:
        bus_model_col = bus_model_col.upper()

    if bus_model_col and bus_model_col in df.columns:
        all_models = df[bus_model_col].dropna().unique().tolist()
        all_models = [str(m).strip().upper() for m in all_models if str(m).strip() != ""]
        all_models = all_models[:5]  # only first 5 if more
    else:
        all_models = []

    if status_callback:
        status_callback(f"Grouping data by part number...")

    # Group data by part number
    grouped_rows = group_by_part_number(df, part_no_col, bus_model_col, qty_veh_col, all_models)
    
    if status_callback:
        status_callback(f"Created {len(grouped_rows)} unique part number groups from {len(df)} original rows")

    # Create document with custom margins for 2 stickers per page
    doc = SimpleDocTemplate(output_pdf_path, pagesize=A4,
                          topMargin=1*cm, bottomMargin=1*cm,
                          leftMargin=1.5*cm, rightMargin=1.5*cm)

    all_elements = []
    total_rows = len(grouped_rows)

    # Process grouped rows in pairs for 2 per page
    for i in range(0, total_rows, 2):
        if status_callback:
            status_callback(f"Creating stickers {i+1}-{min(i+2, total_rows)} of {total_rows}")
        
        # First sticker
        sticker1 = create_single_sticker(
            grouped_rows[i], part_no_col, desc_col, max_capacity_col, 
            qty_veh_col, store_loc_col, bus_model_col, all_models
        )
        
        all_elements.append(sticker1)
        
        # Add spacer between stickers
        all_elements.append(Spacer(1, 1.5*cm))
        
        # Second sticker (if exists)
        if i + 1 < total_rows:
            sticker2 = create_single_sticker(
                grouped_rows[i+1], part_no_col, desc_col, max_capacity_col,
                qty_veh_col, store_loc_col, bus_model_col, all_models
            )
            all_elements.append(sticker2)
        
        # Add page break after every pair (except the last pair)
        if i + 2 < total_rows:
            all_elements.append(PageBreak())

    # Build the document
    try:
        doc.build(all_elements)
        if status_callback:
            status_callback(f"PDF generated successfully: {output_pdf_path}")
        return output_pdf_path
    except Exception as e:
        error_msg = f"Error building PDF: {e}"
        if status_callback:
            status_callback(error_msg)
        return None
        
def main():
    """Main Streamlit application"""
    st.set_page_config(page_title="Mezzanine Label Generator", page_icon="🏷️", layout="wide")
    
    st.title("🏷️ Mezzanine Label Generator")
    st.markdown(
        "<p style='font-size:18px; font-style:italic; margin-top:-10px; text-align:left;'>"
        "Designed and Developed by Agilomatrix</p>",
        unsafe_allow_html=True
    )

    st.markdown("---")
    
    # File upload
    st.header("📁 File Upload")
    uploaded_file = st.file_uploader(
        "Choose an Excel or CSV file",
        type=['xlsx', 'xls', 'csv'],
        help="Upload your Excel or CSV file containing part information"
    )
    
    if uploaded_file is not None:
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            temp_input_path = tmp_file.name
        
        # Display file info
        st.success(f"✅ File uploaded: {uploaded_file.name}")
        
        # Preview data
        try:
            if uploaded_file.name.lower().endswith('.csv'):
                preview_df = pd.read_csv(temp_input_path).head(5)
            else:
                preview_df = pd.read_excel(temp_input_path).head(5)
            
            st.subheader("📊 Data Preview (First 5 rows)")
            st.dataframe(preview_df, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error previewing file: {e}")
            return
        
        # Generate labels section
        st.subheader("🚀 Generate Labels")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            if st.button("🏷️ Generate PDF Labels", type="primary", use_container_width=True):
                # Create progress container
                status_container = st.empty()
                
                # Create temporary output file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_output:
                    temp_output_path = tmp_output.name
                
                # Progress tracking
                def update_status(message):
                    status_container.info(f"📊 {message}")
                
                try:
                    # Generate the PDF
                    update_status("Starting optimized label generation with part number grouping...")
                    
                    result_path = generate_sticker_labels(
                        temp_input_path, 
                        temp_output_path,
                        status_callback=update_status
                    )
                    
                    if result_path:
                        # Success - provide download
                        with open(result_path, 'rb') as pdf_file:
                            pdf_data = pdf_file.read()
                        
                        status_container.success("✅ Downloaded")
                        
                        # Download button
                        st.download_button(
                            label="📥 Download PDF Labels",
                            data=pdf_data,
                            file_name=f"mezzanine_labels_{uploaded_file.name.split('.')[0]}.pdf",
                            mime="application/pdf",
                            use_container_width=True
                        )
                        
                    else:
                        status_container.error("❌ Failed to generate PDF labels")
                        
                except Exception as e:
                    status_container.error(f"❌ Error generating labels: {str(e)}")
                    st.error("Please check your file format and try again.")
                
                finally:
                    # Clean up temporary files
                    try:
                        os.unlink(temp_input_path)
                        if 'temp_output_path' in locals():
                            os.unlink(temp_output_path)
                    except:
                        pass
        
        with col2:
            st.info(
                "**📋 Requirements:**\n"
                "- Excel (.xlsx, .xls) or CSV file\n"
                "- Part Number column\n"
                "- Description column\n"
                "- Optional: Max Capacity, Store Location columns (1-11)\n"
                "- Optional: QTY/VEH column\n"
                "- Optional: Bus Model column (D6, M, P, 55T)\n\n"
                "**🔄 New Feature:**\n"
                "- Rows with same Part Number are grouped into single sticker\n"
                "- Multiple models and quantities combined automatically"
            )
    
    else:
        # Instructions when no file is uploaded
        st.info("👆 Please upload an Excel or CSV file to get started")
        
        # Feature highlights
        st.subheader("✨ Features")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            **🏷️ Professional Labels**
            - Clean, readable design
            - Optimized for printing
            - 2 labels per page
            - Part number grouping
            """)
        
        with col2:
            st.markdown("""
            **📱 QR Code Integration**
            - Automatic QR code generation
            - Contains all part information
            - Easy scanning and tracking
            - Combined model data
            """)
        
        with col3:
            st.markdown("""
            **🏪 Enhanced Store Location**
            - Support for 11 location cells
            - Flexible column mapping
            - Compact display format
            - Multi-model support
            """)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<p style='text-align: center; color: gray; font-size: 14px;'>"
        "© 2025 Agilomatrix - Mezzanine Label Generator v2.2 (Part Number Grouping)</p>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
