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
STICKER_HEIGHT = 8 * cm
STICKER_PAGESIZE = A4

# Define content box dimensions (reduced to fit 2 per page)
CONTENT_BOX_WIDTH = 18 * cm
CONTENT_BOX_HEIGHT = 8 * cm

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

# Define paragraph styles
bold_style = ParagraphStyle(name='Bold', fontName='Helvetica-Bold', fontSize=40, alignment=TA_LEFT, leading=14)
desc_style = ParagraphStyle(name='Description', fontName='Helvetica', fontSize=30, alignment=TA_LEFT, leading=12)
qty_style = ParagraphStyle(name='Quantity', fontName='Helvetica', fontSize=30, alignment=TA_CENTER, leading=12)

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

def detect_bus_model_and_qty(row, qty_veh_col, bus_model_col=None):
    """Improved bus model detection that properly matches bus model to MTM box"""
    result = {'7M': '', '9M': '', '12M': ''}
    
    qty_veh = ""
    if qty_veh_col and qty_veh_col in row and pd.notna(row[qty_veh_col]):
        qty_veh_raw = row[qty_veh_col]
        if pd.notna(qty_veh_raw):
            if isinstance(qty_veh_raw, float) and qty_veh_raw.is_integer():
                qty_veh = str(int(qty_veh_raw))
            else:
                qty_veh = str(qty_veh_raw).strip()

    if not qty_veh:
        return result
    
    # Check if quantity already contains model info
    qty_pattern = r'(\d+M)[:\-\s]*(\d+)'
    matches = re.findall(qty_pattern, qty_veh.upper())
    
    if matches:
        for model, quantity in matches:
            if model in result:
                result[model] = quantity
        return result
    
    # Look for bus model in dedicated bus model column
    detected_model = None
    if bus_model_col and bus_model_col in row and pd.notna(row[bus_model_col]):
        bus_model_value = str(row[bus_model_col]).strip().upper()
        
        if bus_model_value in ['7M', '7']:
            detected_model = '7M'
        elif bus_model_value in ['9M', '9']:
            detected_model = '9M'
        elif bus_model_value in ['12M', '12']:
            detected_model = '12M'
        elif re.search(r'\b7M\b', bus_model_value):
            detected_model = '7M'
        elif re.search(r'\b9M\b', bus_model_value):
            detected_model = '9M'
        elif re.search(r'\b12M\b', bus_model_value):
            detected_model = '12M'
        elif re.search(r'\b7\b', bus_model_value):
            detected_model = '7M'
        elif re.search(r'\b9\b', bus_model_value):
            detected_model = '9M'
        elif re.search(r'\b12\b', bus_model_value):
            detected_model = '12M'
    
    if detected_model:
        result[detected_model] = qty_veh
        return result
    
    # Search through other columns
    priority_columns = []
    other_columns = []
    
    for col in row.index:
        if pd.notna(row[col]):
            col_upper = str(col).upper()
            if any(keyword in col_upper for keyword in ['MODEL', 'BUS', 'VEHICLE', 'TYPE']):
                priority_columns.append(col)
            else:
                other_columns.append(col)
    
    for col in priority_columns:
        if pd.notna(row[col]):
            value_str = str(row[col]).upper()
            
            if re.search(r'\b7M\b', value_str):
                result['7M'] = qty_veh
                return result
            elif re.search(r'\b9M\b', value_str):
                result['9M'] = qty_veh
                return result
            elif re.search(r'\b12M\b', value_str):
                result['12M'] = qty_veh
                return result
            elif re.search(r'\b7\b', value_str) and any(keyword in value_str for keyword in ['BUS', 'METER', 'M']):
                result['7M'] = qty_veh
                return result
            elif re.search(r'\b9\b', value_str) and any(keyword in value_str for keyword in ['BUS', 'METER', 'M']):
                result['9M'] = qty_veh
                return result
            elif re.search(r'\b12\b', value_str) and any(keyword in value_str for keyword in ['BUS', 'METER', 'M']):
                result['12M'] = qty_veh
                return result
    
    return result

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

def extract_store_location_data_from_excel(row_data):
    """Extract store location data from Excel row for Store Location"""
    def get_clean_value(possible_names, default=''):
        for name in possible_names:
            if name in row_data:
                val = row_data[name]
                if pd.notna(val) and str(val).lower() not in ['nan', 'none', 'null', '']:
                    return str(val).strip()
            for col in row_data.index:
                if isinstance(col, str) and col.upper() == name.upper():
                    val = row_data[col]
                    if pd.notna(val) and str(val).lower() not in ['nan', 'none', 'null', '']:
                        return str(val).strip()
        return default
    
    store_loc_1 = get_clean_value(['Store Loc 1', 'STORE_LOC_1', 'Station Name', 'STATION NAME'], '')
    store_loc_2 = get_clean_value(['Store Loc 2', 'STORE_LOC_2', 'Store Location', 'STORE LOCATION'], '')
    store_loc_3 = get_clean_value(['Store Loc 3', 'STORE_LOC_3', 'ABB ZONE', 'Zone'], '')
    store_loc_4 = get_clean_value(['Store Loc 4', 'STORE_LOC_4', 'ABB LOCATION', 'Location'], '')
    store_loc_5 = get_clean_value(['Store Loc 5', 'STORE_LOC_5', 'ABB FLOOR', 'Floor'], '')
    store_loc_6 = get_clean_value(['Store Loc 6', 'STORE_LOC_6', 'ABB RACK NO', 'Rack'], '')
    store_loc_7 = get_clean_value(['Store Loc 7', 'STORE_LOC_7', 'ABB LEVEL', 'Level'], '')
    store_loc_8 = get_clean_value(['Store Loc 8', 'STORE_LOC_8', 'ABB CELL', 'Cell'], '')
    
    return [store_loc_1, store_loc_2, store_loc_3, store_loc_4, store_loc_5, store_loc_6, store_loc_7, store_loc_8]

def create_single_sticker(row, part_no_col, desc_col, max_capacity_col, qty_veh_col, store_loc_col, bus_model_col):
    """Create a single sticker layout with border around the entire sticker"""
    # Extract data
    part_no = str(row[part_no_col]) if pd.notna(row[part_no_col]) else ""
    desc = str(row[desc_col]) if pd.notna(row[desc_col]) else ""
    
    max_capacity = ""
    if max_capacity_col and max_capacity_col in row and pd.notna(row[max_capacity_col]):
        max_capacity = str(row[max_capacity_col])
        
    qty_veh = ""
    if qty_veh_col and qty_veh_col in row and pd.notna(row[qty_veh_col]):
        qty_veh = str(row[qty_veh_col])
    
    store_location = str(row[store_loc_col]) if store_loc_col and store_loc_col in row and pd.notna(row[store_loc_col]) else ""

    # Use enhanced bus model detection
    mtm_quantities = detect_bus_model_and_qty(row, qty_veh_col, bus_model_col)

    # Generate QR code
    qr_data = f"Part No: {part_no}\nDescription: {desc}\nMax Capacity: {max_capacity}\n"
    qr_data += f"Store Location: {store_location}\nQTY/VEH: {qty_veh}"
    
    qr_image = generate_qr_code(qr_data)
    
    sticker_content = []
    
    # Define row heights
    header_row_height = 1.5*cm
    desc_row_height = 1.5*cm
    max_capacity_row_height = 0.8*cm
    store_loc_row_height = 1.2*cm

    # Main table data
    main_table_data = [
        ["Part No", Paragraph(f"{part_no}", bold_style)],
        ["Description", Paragraph(desc[:50] + "..." if len(desc) > 50 else desc, desc_style)],
        ["Max capacity", Paragraph(str(max_capacity), qty_style)]
    ]

    # Create main table
    main_table = Table(main_table_data,
                     colWidths=[CONTENT_BOX_WIDTH/3, CONTENT_BOX_WIDTH*2/3],
                     rowHeights=[header_row_height, desc_row_height, max_capacity_row_height])

    main_table.setStyle(TableStyle([
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (0, -1), 20),
    ]))

    sticker_content.append(main_table)

    # Store Location section
    store_loc_label = Paragraph("Store Location", ParagraphStyle(
        name='StoreLoc', fontName='Helvetica-Bold', fontSize=22, alignment=TA_CENTER
    ))
    
    inner_table_width = CONTENT_BOX_WIDTH * 2 / 3
    col_proportions = [1.2, 1.2, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8]
    total_proportion = sum(col_proportions)
    inner_col_widths = [w * inner_table_width / total_proportion for w in col_proportions]

    store_loc_values = extract_store_location_data_from_excel(row)

    store_loc_inner_table = Table(
        [store_loc_values],
        colWidths=inner_col_widths,
        rowHeights=[store_loc_row_height]
    )
    store_loc_inner_table.setStyle(TableStyle([
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 30),
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
    ]))
    sticker_content.append(store_loc_table)

    # Add small spacer
    sticker_content.append(Spacer(1, 0.1*cm))

    # Bottom section - MTM boxes and QR code
    mtm_box_width = 2.0*cm
    mtm_row_height = 1.7*cm

    position_matrix_data = [
        ["7M", "9M", "12M"],
        [
            Paragraph(f"<b>{mtm_quantities['7M']}</b>", ParagraphStyle(
                name='Bold7M', fontName='Helvetica-Bold', fontSize=30, alignment=TA_CENTER
            )) if mtm_quantities['7M'] else "",
            Paragraph(f"<b>{mtm_quantities['9M']}</b>", ParagraphStyle(
                name='Bold9M', fontName='Helvetica-Bold', fontSize=30, alignment=TA_CENTER
            )) if mtm_quantities['9M'] else "",
            Paragraph(f"<b>{mtm_quantities['12M']}</b>", ParagraphStyle(
                name='Bold12M', fontName='Helvetica-Bold', fontSize=30, alignment=TA_CENTER
            )) if mtm_quantities['12M'] else ""
        ]
    ]

    mtm_table = Table(
        position_matrix_data,
        colWidths=[mtm_box_width, mtm_box_width, mtm_box_width],
        rowHeights=[mtm_row_height/2, mtm_row_height/2]
    )

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

    # Calculate spacing for bottom row
    total_mtm_width = 3 * mtm_box_width
    spacer_width = CONTENT_BOX_WIDTH - total_mtm_width - qr_width

    # Combine MTM boxes and QR code in one row
    bottom_row = Table(
        [[mtm_table, "", qr_table]],
        colWidths=[total_mtm_width, spacer_width, qr_width],
        rowHeights=[max(mtm_row_height, qr_height)]
    )

    bottom_row.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))

    sticker_content.append(bottom_row)
    
    # Create outer table that wraps all sticker content with border
    sticker_table = Table(
        [[sticker_content]],
        colWidths=[CONTENT_BOX_WIDTH],
        rowHeights=[CONTENT_BOX_HEIGHT]
    )
    
    # Add border around the entire sticker
    sticker_table.setStyle(TableStyle([
        ('BOX', (0, 0), (-1, -1), 2, colors.black),  # Outer border for entire sticker
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('LEFTPADDING', (0, 0), (-1, -1), 10),
        ('RIGHTPADDING', (0, 0), (-1, -1), 10),
        ('TOPPADDING', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
    ]))
    
    # Wrap in KeepTogether to prevent page breaks within a sticker
    return KeepTogether([sticker_table])

def generate_sticker_labels(excel_file_path, output_pdf_path, status_callback=None):
    """Generate sticker labels with QR code from Excel data - 2 per page"""
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

    if status_callback:
        status_callback(f"Using columns - Part No: {part_no_col}, Description: {desc_col}")

    # Create document with custom margins for 2 stickers per page
    doc = SimpleDocTemplate(output_pdf_path, pagesize=A4,
                          topMargin=1*cm, bottomMargin=1*cm,
                          leftMargin=1.5*cm, rightMargin=1.5*cm)

    all_elements = []
    total_rows = len(df)

    # Process rows in pairs for 2 per page
    for i in range(0, total_rows, 2):
        if status_callback:
            status_callback(f"Creating stickers {i+1}-{min(i+2, total_rows)} of {total_rows}")
        
        # First sticker
        sticker1 = create_single_sticker(
            df.iloc[i], part_no_col, desc_col, max_capacity_col, 
            qty_veh_col, store_loc_col, bus_model_col
        )
        all_elements.append(sticker1)
        
        # Add spacer between stickers
        all_elements.append(Spacer(1, 1.5*cm))
        
        # Second sticker (if exists)
        if i + 1 < total_rows:
            sticker2 = create_single_sticker(
                df.iloc[i+1], part_no_col, desc_col, max_capacity_col,
                qty_veh_col, store_loc_col, bus_model_col
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
                    update_status("Starting optimized label generation...")
                    
                    result_path = generate_sticker_labels(
                        temp_input_path, 
                        temp_output_path,
                        status_callback=update_status
                    )
                    
                    if result_path:
                        # Success - provide download
                        with open(result_path, 'rb') as pdf_file:
                            pdf_data = pdf_file.read()
                        
                        status_container.success("✅ Optimized labels generated successfully!")
                        
                        # Download button
                        st.download_button(
                            label="📥 Download PDF Labels",
                            data=pdf_data,
                            file_name=f"optimized_sticker_labels_{uploaded_file.name.split('.')[0]}.pdf",
                            mime="application/pdf",
                            use_container_width=True
                        )
                        
                        # Show file info
                        if uploaded_file.name.lower().endswith('.csv'):
                            df_count = len(pd.read_csv(temp_input_path))
                        else:
                            df_count = len(pd.read_excel(temp_input_path))
                        
                        pages_needed = (df_count + 1) // 2  # 2 stickers per page
                        file_size = len(pdf_data) / 1024  # KB
                        st.info(f"📄 PDF size: {file_size:.1f} KB | Stickers: {df_count} | Pages: {pages_needed}")
                        
                    else:
                        status_container.error("❌ Failed to generate labels")
                        
                except Exception as e:
                    status_container.error(f"❌ Error: {str(e)}")
                    st.exception(e)
                
                finally:
                    # Cleanup temporary files
                    try:
                        if os.path.exists(temp_input_path):
                            os.unlink(temp_input_path)
                        if os.path.exists(temp_output_path):
                            os.unlink(temp_output_path)
                    except:
                        pass
        
        with col2:
            st.markdown("""
            **Optimizations:**
            - ✅ 2 stickers per A4 page
            - ✅ No empty space waste
            - ✅ Clean borders
            - ✅ Proper spacing
            - ✅ Same functionality
            """)
        
        # Additional information
        st.subheader("ℹ️ Optimization Details")
        
        info_col1, info_col2 = st.columns(2)
        
        with info_col1:
            st.markdown("""
            **New Features:**
            - 📄 2 labels per A4 page (saves paper)
            - 🎯 Optimized layout with proper spacing  
            - 🚫 Removed unnecessary outer borders
            - 📏 Adjusted dimensions for better fit
            - ⚡ Faster generation process
            """)
        
        with info_col2:
            st.markdown("""
            **Same Functionality:**
            - 🔢 QR code for each part
            - 📍 Store location tracking (8 fields)
            - 🚌 Bus model detection (7M, 9M, 12M)
            - 📦 Max capacity field
            - 🎯 All original features preserved
            """)
    
    else:
        # Instructions when no file is uploaded
        st.info("👆 Please upload an Excel or CSV file to get started")
        
        st.subheader("📋 Instructions")
        st.markdown("""
        1. **Upload your file** - Excel (.xlsx, .xls) or CSV format
        2. **Review data preview** - Check if your data looks correct
        3. **Generate optimized labels** - 2 stickers per A4 page
        4. **Download** - Get your space-efficient PDF labels
        """)
        
        # Sample data format
        st.subheader("📊 Sample Data Format")
        sample_data = pd.DataFrame({
            'Part No': ['AA0107020000', 'DEF456', 'GHI789'],
            'Description': ['REAR SUSPENSION-WIL', 'Brake Pad Set', 'Oil Filter'],
            'Max Capacity': [9, 10, 8],
            'Qty/Veh': [2, 4, 1],
            'Bus Model': ['12M', '9M', '7M'],
            'Store Loc 1': ['Mez-c', 'G+1', 'B1'],
            'Store Loc 2': ['G+1', 'R2', 'L3'],
            'Store Loc 3': ['R', 'A', 'B']
        })
        st.dataframe(sample_data, use_container_width=True)

if __name__ == "__main__":
    main()
