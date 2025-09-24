import streamlit as st
import pandas as pd
import os
import re
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Spacer, Paragraph, PageBreak, Image, KeepTogether
from reportlab.lib.units import cm
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from io import BytesIO
import subprocess
import sys
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
        val = get_clean_value([f'Store Loc {i}', f'STORE_LOC_{i}', f'STORE LOC {i}'])
        if val:
            values.append(val)

    return values

def create_single_sticker(row, part_no_col, desc_col, max_capacity_col, all_models):
    """
    Create a single sticker layout with border around the entire sticker.
    Uses pre-aggregated model data.
    """
    # Extract data with proper number formatting
    part_no = clean_number_format(row[part_no_col]) if pd.notna(row[part_no_col]) else ""
    desc = str(row[desc_col]).strip() if pd.notna(row[desc_col]) else ""

    max_capacity = ""
    if max_capacity_col and max_capacity_col in row and pd.notna(row[max_capacity_col]):
        max_capacity = clean_number_format(row[max_capacity_col])

    # Get all store location parts for table and QR code
    store_loc_values = extract_store_location_data_from_excel(row)
    full_store_location = " ".join([str(v) for v in store_loc_values if v])

    # Use the pre-aggregated model quantities from the grouped data
    mtm_quantities = row['aggregated_models']
    qty_veh_string = ", ".join([f"{model}:{qty}" for model, qty in sorted(mtm_quantities.items())])

    # Generate QR code with aggregated QTY/VEH data
    qr_data = (
        f"Part No: {part_no}\n"
        f"Description: {desc}\n"
        f"Max Capacity: {max_capacity}\n"
        f"Store Location: {full_store_location}\n"
        f"QTY/VEH: {qty_veh_string}"
    )

    qr_image = generate_qr_code(qr_data)

    sticker_content = []

    # Define row heights - ADJUSTED FOR BETTER FIT
    header_row_height = 2.0*cm
    desc_row_height = 1.8*cm
    max_capacity_row_height = 1.32*cm
    store_loc_row_height = 1.3*cm

    # Main table data
    main_table_data = [
        ["Part No", Paragraph(f"{part_no}", bold_style)],
        ["Description", Paragraph(desc, get_dynamic_desc_style(desc))],
        ["Max capacity", Paragraph(str(max_capacity), qty_style)]
    ]

    main_table = Table(main_table_data,
                     colWidths=[CONTENT_BOX_WIDTH/3, CONTENT_BOX_WIDTH*2/3],
                     rowHeights=[header_row_height, desc_row_height, max_capacity_row_height])

    main_table.setStyle(TableStyle([
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (0, -1), 20),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('WORDWRAP', (0, 0), (-1, -1), True),
    ]))

    sticker_content.append(main_table)

    store_loc_label = Paragraph("Store Location", ParagraphStyle(
        name='StoreLoc', fontName='Helvetica-Bold', fontSize=20, alignment=TA_CENTER
    ))
    store_loc_values = [v for v in extract_store_location_data_from_excel(row) if v]
    if not store_loc_values:
        store_loc_values = [""]

    inner_table_width = CONTENT_BOX_WIDTH * 2 / 3
    num_cols = len(store_loc_values) if len(store_loc_values) > 0 else 1
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
    sticker_content.append(Spacer(1, 0.1*cm))

    # Bottom section - MTM boxes and QR code
    max_models = 5
    mtm_box_width = 1.6 * cm
    mtm_row_height = 1.8 * cm

    headers = []
    values = []

    for i in range(max_models):
        if i < len(all_models):
            model_name = all_models[i]
            headers.append(model_name)
            # Get quantity from the aggregated dictionary
            qty_val = mtm_quantities.get(model_name, "")
            values.append(Paragraph(
                f"<b>{clean_number_format(qty_val)}</b>" if qty_val else "",
                ParagraphStyle(name=f"Qty_{model_name}", fontName='Helvetica-Bold', fontSize=16, alignment=TA_CENTER)
            ))
        else:
            headers.append("")
            values.append("")

    position_matrix_data = [headers, values]
    mtm_table = Table(position_matrix_data,
        colWidths=[mtm_box_width] * max_models,
        rowHeights=[mtm_row_height/2, mtm_row_height/2])
    mtm_table.setStyle(TableStyle([
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 18),
    ]))

    # QR code table
    qr_width = 2.5*cm
    qr_height = 2.5*cm

    if qr_image:
        qr_image.drawWidth = qr_width
        qr_image.drawHeight = qr_height
        qr_table = Table([[qr_image]], colWidths=[qr_width], rowHeights=[qr_height])
    else:
        qr_table = Table([[Paragraph("QR", ParagraphStyle(name='QRPlaceholder', fontName='Helvetica-Bold', fontSize=14, alignment=TA_CENTER))]],
                         colWidths=[qr_width], rowHeights=[qr_height])

    qr_table.setStyle(TableStyle([('ALIGN', (0, 0), (-1, -1), 'CENTER'), ('VALIGN', (0, 0), (-1, -1), 'MIDDLE')]))

    total_mtm_width = max_models * mtm_box_width
    remaining_width = CONTENT_BOX_WIDTH - total_mtm_width - qr_width
    left_spacer_width = remaining_width * 0.3
    right_spacer_width = remaining_width * 0.7

    bottom_row = Table(
        [[mtm_table, "", qr_table, ""]],
        colWidths=[total_mtm_width, left_spacer_width, qr_width, right_spacer_width],
        rowHeights=[max(mtm_row_height, qr_height)]
    )
    bottom_row.setStyle(TableStyle([('ALIGN', (0, 0), (0, 0), 'LEFT'), ('ALIGN', (2, 0), (2, 0), 'CENTER'), ('VALIGN', (0, 0), (-1, -1), 'MIDDLE')]))

    sticker_content.append(bottom_row)

    sticker_table = Table(
        [[sticker_content]],
        colWidths=[CONTENT_BOX_WIDTH],
        rowHeights=[CONTENT_BOX_HEIGHT]
    )

    sticker_table.setStyle(TableStyle([
        ('BOX', (0, 0), (-1, -1), 2, colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('LEFTPADDING', (0, 0), (-1, -1), 4),
        ('RIGHTPADDING', (0, 0), (-1, -1), 4),
        ('TOPPADDING', (0, 0), (-1, -1), 0),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))

    return KeepTogether([sticker_table])

def generate_sticker_labels(excel_file_path, output_pdf_path, status_callback=None):
    """
    Generate sticker labels. Groups by Part No, Description, Max Capacity,
    and Store Location to create one sticker per unique part.
    """
    if status_callback: status_callback(f"Processing file: {excel_file_path}")

    try:
        if excel_file_path.lower().endswith('.csv'):
            df = pd.read_csv(excel_file_path)
        else:
            df = pd.read_excel(excel_file_path, engine='openpyxl')
        if status_callback: status_callback(f"Successfully read file with {len(df)} rows")
    except Exception as e:
        error_msg = f"Error reading file: {e}"
        if status_callback: status_callback(error_msg)
        return None

    original_columns = df.columns.tolist()

    # --- START OF MODIFICATION ---
    # Restrict the search for the bus model column to columns C through G.
    # Column C is index 2, and G is index 6. The slice end is exclusive (7).
    # This also safely handles files with fewer than 7 columns.
    model_search_columns = original_columns[2:7] if len(original_columns) >= 7 else original_columns[2:]
    bus_model_col = find_bus_model_column(model_search_columns)
    # --- END OF MODIFICATION ---

    df.columns = [col.upper() if isinstance(col, str) else col for col in df.columns]
    cols = df.columns.tolist()

    part_no_col = next((c for c in cols if 'PART' in c and ('NO' in c or 'NUM' in c or '#' in c)), next((c for c in cols if c in ['PARTNO', 'PART']), cols[0]))
    desc_col = next((c for c in cols if 'DESC' in c), next((c for c in cols if 'NAME' in c), cols[1] if len(cols) > 1 else part_no_col))
    max_capacity_col = next((c for c in cols if 'MAX' in c and 'CAPACITY' in c), next((c for c in cols if 'CAPACITY' in c), next((c for c in cols if 'QTY' in c), None)))
    qty_veh_col = next((c for c in cols if any(t in c for t in ['QTY/VEH', 'QTY_VEH', 'QTY PER VEH', 'QTYVEH', 'QTYPERCAR', 'QTYCAR', 'QTY/CAR'])), None)
    
    # The bus_model_col is now found from the restricted list above
    if bus_model_col:
        bus_model_col = bus_model_col.upper()

    if not all([part_no_col, desc_col, qty_veh_col, bus_model_col]):
        error_msg = "Error: Could not find all required columns (Part No, Description, QTY/VEH, Bus Model in columns C-G)."
        if status_callback: status_callback(error_msg)
        return None

    # Get all unique models for consistent sticker layout
    all_models = df[bus_model_col].dropna().unique().tolist()
    all_models = sorted([str(m).strip().upper() for m in all_models if str(m).strip() != ""])
    all_models = all_models[:5]

    # --- UPDATED: Grouping and Aggregation Logic ---
    # 1. Create a composite key for store location for accurate grouping
    store_loc_cols = [col for col in df.columns if re.search(r'STORE.LOC', col, re.IGNORECASE)]
    df['store_location_key'] = df[store_loc_cols].fillna('').astype(str).agg(' '.join, axis=1)
    df['store_location_key'] = df['store_location_key'].str.strip()

    # 2. Define the exact columns to group by
    grouping_keys = [part_no_col, desc_col, 'store_location_key']
    if max_capacity_col and max_capacity_col in df.columns:
        grouping_keys.append(max_capacity_col)
        # Fill NaN to ensure 'None' doesn't get grouped with empty strings
        df[max_capacity_col] = df[max_capacity_col].fillna('')


    # 3. Perform grouping and aggregation
    grouped = df.groupby(grouping_keys)
    sticker_data_list = []

    for _, group in grouped:
        base_data = group.iloc[0].to_dict()
        model_quantities = {}
        for _, row in group.iterrows():
            model = str(row.get(bus_model_col, '')).strip().upper()
            qty = clean_number_format(row.get(qty_veh_col, ''))
            if model and qty:
                model_quantities[model] = qty
        base_data['aggregated_models'] = model_quantities
        sticker_data_list.append(base_data)

    processed_df = pd.DataFrame(sticker_data_list)
    # --- End of Updated Logic ---

    doc = SimpleDocTemplate(output_pdf_path, pagesize=A4, topMargin=1*cm, bottomMargin=1*cm, leftMargin=1.5*cm, rightMargin=1.5*cm)
    all_elements = []
    total_stickers = len(processed_df)

    # Process grouped data in pairs for 2 per page
    for i in range(0, total_stickers, 2):
        if status_callback: status_callback(f"Creating stickers {i+1}-{min(i+2, total_stickers)} of {total_stickers}")

        sticker1 = create_single_sticker(processed_df.iloc[i], part_no_col, desc_col, max_capacity_col, all_models)
        all_elements.append(sticker1)
        all_elements.append(Spacer(1, 1.5*cm))

        if i + 1 < total_stickers:
            sticker2 = create_single_sticker(processed_df.iloc[i+1], part_no_col, desc_col, max_capacity_col, all_models)
            all_elements.append(sticker2)

        if i + 2 < total_stickers:
            all_elements.append(PageBreak())

    try:
        doc.build(all_elements)
        if status_callback: status_callback(f"PDF generated successfully: {output_pdf_path}")
        return output_pdf_path
    except Exception as e:
        error_msg = f"Error building PDF: {e}"
        if status_callback: status_callback(error_msg)
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

    st.header("📁 File Upload")
    uploaded_file = st.file_uploader(
        "Choose an Excel or CSV file",
        type=['xlsx', 'xls', 'csv'],
        help="Upload your Excel or CSV file containing part information"
    )

    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            temp_input_path = tmp_file.name

        st.success(f"✅ File uploaded: {uploaded_file.name}")

        try:
            preview_df = pd.read_excel(temp_input_path).head(5) if not uploaded_file.name.lower().endswith('.csv') else pd.read_csv(temp_input_path).head(5)
            st.subheader("📊 Data Preview (First 5 rows)")
            st.dataframe(preview_df, use_container_width=True)
        except Exception as e:
            st.error(f"Error previewing file: {e}")
            return

        st.subheader("🚀 Generate Labels")
        col1, col2 = st.columns([1, 1])

        with col1:
            if st.button("🏷️ Generate PDF Labels", type="primary", use_container_width=True):
                status_container = st.empty()
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_output:
                    temp_output_path = tmp_output.name

                def update_status(message):
                    status_container.info(f"📊 {message}")

                try:
                    update_status("Starting optimized label generation...")
                    result_path = generate_sticker_labels(temp_input_path, temp_output_path, status_callback=update_status)

                    if result_path:
                        with open(result_path, 'rb') as pdf_file:
                            pdf_data = pdf_file.read()
                        status_container.success("✅ PDF Generation Complete!")
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
                "- Part Number & Description columns\n"
                "- **Required**: `QTY/VEH` column\n"
                "- **Required**: `Bus Model` column (in C-G)\n"
                "- Optional: `Max Capacity`, `Store Location` columns"
            )

    else:
        st.info("👆 Please upload an Excel or CSV file to get started")
        st.subheader("✨ Features")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(" **🏷️ Professional Labels** \n - Clean, readable design\n - Optimized for printing\n - 2 labels per page")
        with col2:
            st.markdown(" **📱 QR Code Integration** \n - Automatic QR code generation\n - Contains all part information\n - Easy scanning and tracking")
        with col3:
            st.markdown(" **🔄 Smart Grouping** \n - Combines models for identical parts onto one sticker\n - Reduces redundant labels\n - Aggregates QTY/VEH data")

    st.markdown("---")
    st.markdown(
        "<p style='text-align: center; color: gray; font-size: 14px;'>"
        "© 2025 Agilomatrix - Mezzanine Label Generator v2.4 (Fixed Column Search)</p>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
