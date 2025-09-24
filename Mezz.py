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
CONTENT_BOX_WIDTH = 17.8 * cm
CONTENT_BOX_HEIGHT = 8.8 * cm

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
    else: font_size = 10
    leading = font_size + 2
    return ParagraphStyle(
        name='DescriptionDynamic', fontName='Helvetica', fontSize=font_size,
        alignment=TA_LEFT, leading=leading, wordWrap='CJK', splitLongWords=1,
        spaceBefore=3, spaceAfter=3, allowWidows=1, allowOrphans=1,
    )

qty_style = ParagraphStyle(
    name='Quantity', fontName='Helvetica', fontSize=22,
    alignment=TA_CENTER, leading=22, wordWrap='CJK'
)

def clean_number_format(value):
    """
    Clean number formatting to preserve integers and handle decimals properly.
    Returns string representation maintaining original number format.
    """
    if pd.isna(value) or value == '': return ''
    if isinstance(value, str):
        value = value.strip()
        if value == '': return ''
        try:
            num_value = float(value)
            return str(int(num_value)) if num_value.is_integer() else str(num_value)
        except: return value
    if isinstance(value, (int, float)):
        if isinstance(value, float) and value.is_integer(): return str(int(value))
        return str(value)
    return str(value)

def generate_qr_code(data_string):
    """Generate a QR code from the given data string"""
    try:
        qr = qrcode.QRCode(version=1, error_correction=qrcode.constants.ERROR_CORRECT_M, box_size=10, border=4)
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
        upper_possible = [n.upper() for n in possible_names]
        col_map = {str(k).upper(): k for k in row_data.keys()}
        
        for name in upper_possible:
            if name in col_map:
                original_col = col_map[name]
                val = row_data[original_col]
                if pd.notna(val) and str(val).strip().lower() not in ['nan', 'none', 'null', '']:
                    return clean_number_format(val)
        return None
        
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
    part_no = clean_number_format(row.get(part_no_col, ""))
    desc = str(row.get(desc_col, "")).strip()
    max_capacity = clean_number_format(row.get(max_capacity_col, "")) if max_capacity_col else ""
    
    store_loc_values = extract_store_location_data_from_excel(row)
    full_store_location = " ".join([str(v) for v in store_loc_values if v])
    
    mtm_quantities = row.get('aggregated_models', {})
    qty_veh_string = ", ".join([f"{model}:{qty}" for model, qty in sorted(mtm_quantities.items()) if model])

    qr_data = (f"Part No: {part_no}\nDescription: {desc}\nMax Capacity: {max_capacity}\n"
               f"Store Location: {full_store_location}\nQTY/VEH: {qty_veh_string}")
    qr_image = generate_qr_code(qr_data)

    sticker_content = []
    header_row_height, desc_row_height, max_capacity_row_height, store_loc_row_height = 2.0*cm, 1.8*cm, 1.32*cm, 1.3*cm

    main_table_data = [
        ["Part No", Paragraph(f"{part_no}", bold_style)],
        ["Description", Paragraph(desc, get_dynamic_desc_style(desc))],
        ["Max capacity", Paragraph(str(max_capacity), qty_style)]
    ]
    main_table = Table(main_table_data, colWidths=[CONTENT_BOX_WIDTH/3, CONTENT_BOX_WIDTH*2/3], rowHeights=[header_row_height, desc_row_height, max_capacity_row_height])
    main_table.setStyle(TableStyle([
        ('GRID', (0, 0), (-1, -1), 1, colors.black), ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'), ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (0, -1), 20), ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8), ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6), ('WORDWRAP', (0, 0), (-1, -1), True),
    ]))
    sticker_content.append(main_table)

    store_loc_label = Paragraph("Store Location", ParagraphStyle(name='StoreLoc', fontName='Helvetica-Bold', fontSize=20, alignment=TA_CENTER))
    store_loc_values = [v for v in extract_store_location_data_from_excel(row) if v] or [""]
    inner_table_width = CONTENT_BOX_WIDTH * 2 / 3
    num_cols = len(store_loc_values)
    inner_col_widths = [inner_table_width / num_cols] * num_cols if num_cols > 0 else [inner_table_width]
    
    store_loc_inner_table = Table([store_loc_values], colWidths=inner_col_widths, rowHeights=[store_loc_row_height])
    store_loc_inner_table.setStyle(TableStyle([
        ('GRID', (0, 0), (-1, -1), 1, colors.black), ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'), ('FONTSIZE', (0, 0), (-1, -1), 16),
    ]))
    
    store_loc_table = Table([[store_loc_label, store_loc_inner_table]], colWidths=[CONTENT_BOX_WIDTH/3, inner_table_width], rowHeights=[store_loc_row_height])
    store_loc_table.setStyle(TableStyle([
        ('GRID', (0, 0), (-1, -1), 1, colors.black), ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    sticker_content.append(store_loc_table)
    sticker_content.append(Spacer(1, 0.1*cm))

    max_models, mtm_box_width, mtm_row_height = 5, 1.6 * cm, 1.8 * cm
    headers, values = [], []
    for model_name in all_models:
        headers.append(model_name)
        # Only try to get quantity if the model_name is not an empty string
        qty_val = mtm_quantities.get(model_name, "") if model_name else ""
        values.append(Paragraph(f"<b>{clean_number_format(qty_val)}</b>" if qty_val else "",
            ParagraphStyle(name=f"Qty_{model_name}", fontName='Helvetica-Bold', fontSize=16, alignment=TA_CENTER)))
    
    mtm_table = Table([headers, values], colWidths=[mtm_box_width] * max_models, rowHeights=[mtm_row_height/2, mtm_row_height/2])
    mtm_table.setStyle(TableStyle([
        ('GRID', (0, 0), (-1, -1), 1, colors.black), ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'), ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 18),
    ]))

    qr_width, qr_height = 2.5*cm, 2.5*cm
    qr_table = Table([[qr_image]], colWidths=[qr_width], rowHeights=[qr_height]) if qr_image else Table([["QR"]], colWidths=[qr_width], rowHeights=[qr_height])
    qr_table.setStyle(TableStyle([('ALIGN', (0, 0), (-1, -1), 'CENTER'), ('VALIGN', (0, 0), (-1, -1), 'MIDDLE')]))
    
    total_mtm_width = max_models * mtm_box_width
    remaining_width = CONTENT_BOX_WIDTH - total_mtm_width - qr_width
    left_spacer_width, right_spacer_width = remaining_width * 0.3, remaining_width * 0.7
    
    bottom_row = Table([[mtm_table, "", qr_table, ""]], colWidths=[total_mtm_width, left_spacer_width, qr_width, right_spacer_width], rowHeights=[max(mtm_row_height, qr_height)])
    bottom_row.setStyle(TableStyle([('ALIGN', (0, 0), (0, 0), 'LEFT'), ('ALIGN', (2, 0), (2, 0), 'CENTER'), ('VALIGN', (0, 0), (-1, -1), 'MIDDLE')]))
    sticker_content.append(bottom_row)

    sticker_table = Table([[sticker_content]], colWidths=[CONTENT_BOX_WIDTH], rowHeights=[CONTENT_BOX_HEIGHT])
    sticker_table.setStyle(TableStyle([('BOX', (0, 0), (-1, -1), 2, colors.black), ('LEFTPADDING', (0, 0), (-1, -1), 4),
        ('RIGHTPADDING', (0, 0), (-1, -1), 4), ('TOPPADDING', (0, 0), (-1, -1), 0), ('BOTTOMPADDING', (0, 0), (-1, -1), 4)]))
    
    return KeepTogether([sticker_table])

def generate_sticker_labels(excel_file_path, output_pdf_path, status_callback=None):
    """
    Generate sticker labels from a file with a 'wide' format, where
    bus models are columns C-G. Handles empty or 'Unnamed' model columns correctly.
    """
    if status_callback: status_callback(f"Processing file: {excel_file_path}")
    try:
        df = pd.read_csv(excel_file_path, keep_default_na=False) if excel_file_path.lower().endswith('.csv') else pd.read_excel(excel_file_path, keep_default_na=False, engine='openpyxl')
        if status_callback: status_callback(f"Successfully read file with {len(df)} rows")
    except Exception as e:
        if status_callback: status_callback(f"Error reading file: {e}")
        return None

    original_columns = df.columns.tolist()
    
    part_no_col = next((c for c in original_columns if 'PART' in str(c).upper() and ('NO' in str(c).upper() or 'NUM' in str(c).upper())), original_columns[0])
    desc_col = next((c for c in original_columns if 'DESC' in str(c).upper()), original_columns[1] if len(original_columns) > 1 else part_no_col)
    max_capacity_col = next((c for c in original_columns if 'MAX' in str(c).upper() and 'CAPACITY' in str(c).upper()), None)
    
    if len(original_columns) < 3:
        if status_callback: status_callback("Error: File must have at least 3 columns for model data (C-G).")
        return None
        
    # --- START: CORRECTED LOGIC FOR UNNAMED/EMPTY MODELS ---
    model_cols_original = original_columns[2:7] if len(original_columns) >= 7 else original_columns[2:]
    
    all_models = []
    for col in model_cols_original:
        col_str = str(col).strip()
        # Check for empty, NaN, or pandas default "Unnamed: X" names
        if pd.isna(col) or col_str == '' or col_str.lower().startswith('unnamed:'):
            all_models.append('')  # Treat as a blank model slot
        else:
            all_models.append(col_str.upper())
            
    model_mapping = list(zip(model_cols_original, all_models))

    def get_model_quantities(row, mapping):
        model_quantities = {}
        for original_col, cleaned_model_name in mapping:
            if not cleaned_model_name:
                continue
            
            if original_col in row and pd.notna(row[original_col]) and row[original_col] != '':
                qty = clean_number_format(row[original_col])
                if qty and str(qty) != '0':
                    model_quantities[cleaned_model_name] = qty
        return model_quantities

    df['aggregated_models'] = df.apply(lambda row: get_model_quantities(row, model_mapping), axis=1)
    # --- END: CORRECTED LOGIC ---

    processed_df = df

    doc = SimpleDocTemplate(output_pdf_path, pagesize=A4, topMargin=1*cm, bottomMargin=1*cm, leftMargin=1.5*cm, rightMargin=1.5*cm)
    all_elements = []
    total_stickers = len(processed_df)

    for i in range(0, total_stickers, 2):
        if status_callback: status_callback(f"Creating stickers {i+1}-{min(i+2, total_stickers)} of {total_stickers}")

        sticker1 = create_single_sticker(processed_df.iloc[i].to_dict(), part_no_col, desc_col, max_capacity_col, all_models)
        all_elements.append(sticker1)
        
        if i + 1 < total_stickers:
            all_elements.append(Spacer(1, 1.5*cm)) # Add space between stickers on the same page
            sticker2 = create_single_sticker(processed_df.iloc[i+1].to_dict(), part_no_col, desc_col, max_capacity_col, all_models)
            all_elements.append(sticker2)

        if i + 2 < total_stickers:
            all_elements.append(PageBreak())

    try:
        doc.build(all_elements)
        if status_callback: status_callback(f"PDF generated successfully: {output_pdf_path}")
        return output_pdf_path
    except Exception as e:
        if status_callback: status_callback(f"Error building PDF: {e}")
        return None

def main():
    """Main Streamlit application"""
    st.set_page_config(page_title="Mezzanine Label Generator", page_icon="🏷️", layout="wide")
    st.title("🏷️ Mezzanine Label Generator")
    st.markdown("<p style='font-size:18px; font-style:italic; margin-top:-10px; text-align:left;'>Designed and Developed by Agilomatrix</p>", unsafe_allow_html=True)
    st.markdown("---")

    st.header("📁 File Upload")
    uploaded_file = st.file_uploader("Choose an Excel or CSV file", type=['xlsx', 'xls', 'csv'], help="Upload your file with parts data")

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
                def update_status(message): status_container.info(f"📊 {message}")
                try:
                    update_status("Starting label generation...")
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_output:
                        result_path = generate_sticker_labels(temp_input_path, tmp_output.name, status_callback=update_status)
                    if result_path:
                        with open(result_path, 'rb') as pdf_file:
                            pdf_data = pdf_file.read()
                        status_container.success("✅ PDF Generation Complete!")
                        st.download_button(
                            label="📥 Download PDF Labels", data=pdf_data,
                            file_name=f"mezzanine_labels_{uploaded_file.name.split('.')[0]}.pdf",
                            mime="application/pdf", use_container_width=True)
                    else:
                        status_container.error("❌ Failed to generate PDF labels. Please check file format.")
                except Exception as e:
                    status_container.error(f"❌ An unexpected error occurred: {str(e)}")
                finally:
                    if os.path.exists(temp_input_path): os.unlink(temp_input_path)
                    if 'result_path' in locals() and result_path and os.path.exists(result_path): os.unlink(result_path)

        with col2:
            st.info(
                "**📋 File Format Requirements:**\n"
                "- Column A: Part Number\n"
                "- Column B: Part Description\n"
                "- **Columns C to G**: Bus Models (e.g., 'M', 'S', 'P') in the header.\n"
                "- *Blank or empty headers in C-G are handled correctly.*\n"
                "- Cells under C-G must contain the quantity for that model.\n"
                "- Optional: `Max Capacity`, `Store Loc...` columns."
            )
    else:
        st.info("👆 Please upload an Excel or CSV file to get started")
        st.subheader("✨ Features")
        col1, col2, col3 = st.columns(3)
        with col1: st.markdown(" **🏷️ Professional Labels** \n - Clean, readable design\n - Optimized for printing\n - 2 labels per page")
        with col2: st.markdown(" **📱 QR Code Integration** \n - Automatic QR code generation\n - Contains all part information\n - Easy scanning and tracking")
        with col3: st.markdown(" **🔄 Smart Data Handling** \n - Reads models directly from columns C-G\n - Ignores empty/unnamed columns\n - Aggregates data onto one sticker")

    st.markdown("---")
    st.markdown("<p style='text-align: center; color: gray; font-size: 14px;'>© 2025 Agilomatrix - Mezzanine Label Generator v3.2 (Final)</p>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
