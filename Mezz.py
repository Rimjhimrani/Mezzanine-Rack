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

def detect_bus_model_and_qty(row, qty_veh_col, bus_model_col=None):
    """Improved bus model detection that properly matches bus model to MTM box"""
    result = {'7M': '', '9M': '', '12M': ''}
    
    qty_veh = ""
    if qty_veh_col and qty_veh_col in row and pd.notna(row[qty_veh_col]):
        qty_veh_raw = row[qty_veh_col]
        if pd.notna(qty_veh_raw):
            qty_veh = clean_number_format(qty_veh_raw)

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

def auto_detect_store_location_columns(df):
    """Auto-detect Store Location columns from DataFrame"""
    store_loc_columns = []
    
    # Look for columns that match store location patterns
    for col in df.columns:
        if isinstance(col, str):
            col_upper = col.upper()
            
            # Check for Store Loc pattern
            if 'STORE' in col_upper and ('LOC' in col_upper or 'LOCATION' in col_upper):
                store_loc_columns.append(col)
            # Check for numbered store columns
            elif any(f'STORE{i}' in col_upper.replace(' ', '').replace('_', '') for i in range(1, 9)):
                store_loc_columns.append(col)
            # Check for ABB specific columns
            elif col_upper in ['ABB ZONE', 'ABB LOCATION', 'ABB FLOOR', 'ABB RACK NO', 'ABB LEVEL', 'ABB CELL']:
                store_loc_columns.append(col)
            # Check for generic location terms
            elif col_upper in ['STATION NAME', 'ZONE', 'FLOOR', 'RACK', 'LEVEL', 'CELL']:
                store_loc_columns.append(col)
    
    # Sort by column name to maintain order
    store_loc_columns.sort()
    
    # Ensure we have exactly 8 columns, pad with empty strings if needed
    while len(store_loc_columns) < 8:
        store_loc_columns.append('')
    
    return store_loc_columns[:8]  # Take only first 8

def extract_store_location_data_from_excel(row_data, store_loc_columns=None):
    """Enhanced Store Location extraction with better column detection"""
    if store_loc_columns is None:
        # Use original detection method as fallback
        def get_clean_value(possible_names, default=''):
            # First try exact matches from the possible names list
            for name in possible_names:
                if name in row_data:
                    val = row_data[name]
                    if pd.notna(val) and str(val).lower() not in ['nan', 'none', 'null', '']:
                        return clean_number_format(val)
            
            # Then try case-insensitive matching
            for name in possible_names:
                for col in row_data.index:
                    if isinstance(col, str) and col.upper() == name.upper():
                        val = row_data[col]
                        if pd.notna(val) and str(val).lower() not in ['nan', 'none', 'null', '']:
                            return clean_number_format(val)
            
            # Finally try partial matching for more flexible detection
            for name in possible_names:
                name_upper = name.upper()
                for col in row_data.index:
                    if isinstance(col, str):
                        col_upper = col.upper()
                        # Check if the column name contains key parts of our search term
                        if any(part in col_upper for part in name_upper.split() if len(part) > 2):
                            val = row_data[col]
                            if pd.notna(val) and str(val).lower() not in ['nan', 'none', 'null', '']:
                                return clean_number_format(val)
            
            return default
        
        # Try to find Store Location columns with more comprehensive search patterns
        store_loc_1 = get_clean_value([
            'Store Loc 1', 'STORE_LOC_1', 'Store Location 1', 'STORE LOCATION 1',
            'Station Name', 'STATION NAME', 'Station', 'STATION',
            'Store_Loc_1', 'StoreLocation1', 'Store1'
        ], '')
        
        store_loc_2 = get_clean_value([
            'Store Loc 2', 'STORE_LOC_2', 'Store Location 2', 'STORE LOCATION 2',
            'Store Location', 'STORE LOCATION', 'Location', 'LOCATION',
            'Store_Loc_2', 'StoreLocation2', 'Store2'
        ], '')
        
        store_loc_3 = get_clean_value([
            'Store Loc 3', 'STORE_LOC_3', 'Store Location 3', 'STORE LOCATION 3',
            'ABB ZONE', 'Zone', 'ZONE', 'Abb Zone',
            'Store_Loc_3', 'StoreLocation3', 'Store3'
        ], '')
        
        store_loc_4 = get_clean_value([
            'Store Loc 4', 'STORE_LOC_4', 'Store Location 4', 'STORE LOCATION 4',
            'ABB LOCATION', 'Location', 'LOCATION', 'Abb Location',
            'Store_Loc_4', 'StoreLocation4', 'Store4'
        ], '')
        
        store_loc_5 = get_clean_value([
            'Store Loc 5', 'STORE_LOC_5', 'Store Location 5', 'STORE LOCATION 5',
            'ABB FLOOR', 'Floor', 'FLOOR', 'Abb Floor',
            'Store_Loc_5', 'StoreLocation5', 'Store5'
        ], '')
        
        store_loc_6 = get_clean_value([
            'Store Loc 6', 'STORE_LOC_6', 'Store Location 6', 'STORE LOCATION 6',
            'ABB RACK NO', 'Rack', 'RACK', 'Abb Rack', 'Rack No', 'RACK NO',
            'Store_Loc_6', 'StoreLocation6', 'Store6'
        ], '')
        
        store_loc_7 = get_clean_value([
            'Store Loc 7', 'STORE_LOC_7', 'Store Location 7', 'STORE LOCATION 7',
            'ABB LEVEL', 'Level', 'LEVEL', 'Abb Level',
            'Store_Loc_7', 'StoreLocation7', 'Store7'
        ], '')
        
        store_loc_8 = get_clean_value([
            'Store Loc 8', 'STORE_LOC_8', 'Store Location 8', 'STORE LOCATION 8',
            'ABB CELL', 'Cell', 'CELL', 'Abb Cell',
            'Store_Loc_8', 'StoreLocation8', 'Store8'
        ], '')
        
        return [store_loc_1, store_loc_2, store_loc_3, store_loc_4, store_loc_5, store_loc_6, store_loc_7, store_loc_8]
    
    else:
        # Use pre-detected columns
        store_loc_values = []
        
        for col in store_loc_columns:
            if col and col in row_data:
                val = row_data[col]
                if pd.notna(val) and str(val).lower() not in ['nan', 'none', 'null', '']:
                    store_loc_values.append(clean_number_format(val))
                else:
                    store_loc_values.append('')
            else:
                store_loc_values.append('')
        
        # Ensure we always return 8 values
        while len(store_loc_values) < 8:
            store_loc_values.append('')
        
        return store_loc_values[:8]

def create_single_sticker(row, part_no_col, desc_col, max_capacity_col, qty_veh_col, store_loc_col, bus_model_col, store_loc_columns=None):
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
    
    # Use enhanced store location extraction
    store_loc_values = extract_store_location_data_from_excel(row, store_loc_columns)

    # Use enhanced bus model detection
    mtm_quantities = detect_bus_model_and_qty(row, qty_veh_col, bus_model_col)

    # Generate QR code with ALL store location values
    store_location_text = ' | '.join([str(val) for val in store_loc_values if str(val).strip()])
    qr_data = f"Part No: {part_no}\nDescription: {desc}\nMax Capacity: {max_capacity}\n"
    qr_data += f"Store Location: {store_location_text}\nQTY/VEH: {qty_veh}"
    
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

    # In the create_single_sticker function, replace this section:
    # Store Location section with all 8 values
    store_loc_label = Paragraph("Store Location", ParagraphStyle(
        name='StoreLoc', fontName='Helvetica-Bold', fontSize=20, alignment=TA_CENTE
    ))

    inner_table_width = CONTENT_BOX_WIDTH * 2 / 3
    col_proportions = [1.4, 1.2, 0.6, 1.2, 0.6, 0.8, 0.6, 0.8]
    total_proportion = sum(col_proportions)
    inner_col_widths = [w * inner_table_width / total_proportion for w in col_proportions]

    # Convert store location values to Paragraph objects for better display
    store_loc_paragraphs = []
    for val in store_loc_values:
        if val:  # Only create paragraph if value exists
            store_loc_paragraphs.append(
                Paragraph(str(val), ParagraphStyle(
                    name='StoreLocValue', 
                    fontName='Helvetica', 
                    fontSize=16, 
                    alignment=TA_CENTER,
                    wordWrap='CJK'
                ))
            )
        else:
            store_loc_paragraphs.append("")  # Empty cell for missing values
    # Use the converted paragraphs instead of raw values
    store_loc_inner_table = Table(
        [store_loc_paragraphs],  # Changed from store_loc_values to store_loc_paragraphs
        colWidths=inner_col_widths,
        rowHeights=[store_loc_row_height]
    )

    store_loc_inner_table.setStyle(TableStyle([
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 16),  # Consistent font size
        ('LEFTPADDING', (0, 0), (-1, -1), 2),
        ('RIGHTPADDING', (0, 0), (-1, -1), 2),
        ('TOPPADDING', (0, 0), (-1, -1), 2),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
        ('WORDWRAP', (0, 0), (-1, -1), True),
    ]))

    # Also add debugging - insert this right after extracting store_loc_values:
    print(f"DEBUG: Store Location Values: {store_loc_values}")  # Add this line for debugg
    
    sticker_content.append(store_loc_table)

    # Add small spacer
    sticker_content.append(Spacer(1, 0.1*cm))

    # Bottom section - MTM boxes and QR code
    mtm_box_width = 1.8*cm
    mtm_row_height = 1.8*cm

    position_matrix_data = [
        ["7M", "9M", "12M"],
        [
            Paragraph(f"<b>{mtm_quantities['7M']}</b>", ParagraphStyle(
                name='Bold7M', fontName='Helvetica-Bold', fontSize=16, alignment=TA_CENTER
            )) if mtm_quantities['7M'] else "",
            Paragraph(f"<b>{mtm_quantities['9M']}</b>", ParagraphStyle(
                name='Bold9M', fontName='Helvetica-Bold', fontSize=16, alignment=TA_CENTER
            )) if mtm_quantities['9M'] else "",
            Paragraph(f"<b>{mtm_quantities['12M']}</b>", ParagraphStyle(
                name='Bold12M', fontName='Helvetica-Bold', fontSize=16, alignment=TA_CENTER
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

    # Calculate spacing for better QR code centering
    total_mtm_width = 3 * mtm_box_width
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
    """Generate sticker labels with QR code from Excel data - 2 per page with enhanced Store Location"""
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

    # Debug: Show detected Store Location columns
    if status_callback:
        status_callback("Detecting Store Location columns...")
    
    store_loc_columns = auto_detect_store_location_columns(df)
    
    if status_callback:
        detected_info = f"Detected Store Location columns: {[col if col else '(empty)' for col in store_loc_columns]}"
        status_callback(detected_info)

    # Identify other columns
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
        
        # First sticker - use enhanced function
        sticker1 = create_single_sticker(
            df.iloc[i], part_no_col, desc_col, max_capacity_col, 
            qty_veh_col, store_loc_col, bus_model_col, store_loc_columns
        )
        all_elements.append(sticker1)
        
        # Add spacer between stickers
        all_elements.append(Spacer(1, 1.5*cm))
        
        # Second sticker (if exists)
        if i + 1 < total_rows:
            sticker2 = create_single_sticker(
                df.iloc[i+1], part_no_col, desc_col, max_capacity_col,
                qty_veh_col, store_loc_col, bus_model_col, store_loc_columns
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

def debug_store_location_columns(df):
    """Debug function to show what Store Location columns are detected"""
    print("=== DEBUGGING STORE LOCATION COLUMNS ===")
    print("\nAll columns in DataFrame:")
    for i, col in enumerate(df.columns):
        print(f"{i+1:2d}. {col}")
    
    print("\nAuto-detected Store Location columns:")
    detected_cols = auto_detect_store_location_columns(df)
    for i, col in enumerate(detected_cols, 1):
        print(f"Store Loc {i}: {col if col else '(empty)'}")
    
    print("\nFirst row data for detected columns:")
    if len(df) > 0:
        for i, col in enumerate(detected_cols, 1):
            if col and col in df.columns:
                value = df[col].iloc[0] if pd.notna(df[col].iloc[0]) else '(empty)'
                print(f"Store Loc {i} ({col}): {value}")
    
    return detected_cols
        
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
            
            # Show detected Store Location columns
            st.subheader("🔍 Detected Store Location Columns")
            detected_cols = auto_detect_store_location_columns(preview_df)
            
            col1, col2 = st.columns(2)
            with col1:
                for i in range(4):
                    if i < len(detected_cols) and detected_cols[i]:
                        st.info(f"**Store Loc {i+1}:** {detected_cols[i]}")
                    else:
                        st.warning(f"**Store Loc {i+1}:** Not detected")
            
            with col2:
                for i in range(4, 8):
                    if i < len(detected_cols) and detected_cols[i]:
                        st.info(f"**Store Loc {i+1}:** {detected_cols[i]}")
                    else:
                        st.warning(f"**Store Loc {i+1}:** Not detected")
            
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
                    update_status("Starting enhanced label generation with Store Location detection...")
                    
                    result_path = generate_sticker_labels(
                        temp_input_path, 
                        temp_output_path,
                        status_callback=update_status
                    )
                    
                    if result_path:
                        # Success - provide download
                        with open(result_path, 'rb') as pdf_file:
                            pdf_data = pdf_file.read()
                        
                        status_container.success("✅ PDF Generated Successfully!")
                        
                        # Download button
                        st.download_button(
                            label="📥 Download PDF Labels",
                            data=pdf_data,
                            file_name=f"mezzanine_labels_{uploaded_file.name.split('.')[0]}.pdf",
                            mime="application/pdf",
                            use_container_width=True
                        )
                        
                        # Show success message
                        st.balloons()
                        st.success("🎉 Your Mezzanine labels are ready! The QR codes now include all Store Location data.")
                        
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
                "- Optional: Max Capacity, Store Location, QTY/VEH columns\n"
                "- Optional: Bus Model column (7M, 9M, 12M)\n\n"
                "**🔧 New Features:**\n"
                "- Auto-detects all 8 Store Location columns\n"
                "- QR codes include complete Store Location data\n"
                "- Enhanced column matching algorithm"
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
            """)
        
        with col2:
            st.markdown("""
            **📱 Enhanced QR Code**
            - Contains ALL Store Location data
            - Complete part information
            - Easy scanning and tracking
            - Fixed Store Location detection
            """)
        
        with col3:
            st.markdown("""
            **🚌 Bus Model Support**
            - Automatic 7M, 9M, 12M detection
            - Flexible column mapping
            - Smart quantity parsing
            """)
        
        # Technical improvements
        st.subheader("🔧 Technical Improvements")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **🎯 Enhanced Store Location Detection:**
            - Auto-detects all 8 Store Location columns
            - Flexible column name matching
            - Handles various naming conventions
            - Case-insensitive detection
            """)
        
        with col2:
            st.markdown("""
            **📊 Improved QR Code Data:**
            - Includes complete Store Location: `Mez-C | G+1 | B | HRR | G | R | 0 | 1`
            - All part information preserved
            - Properly formatted for scanning
            - Better data organization
            """)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<p style='text-align: center; color: gray; font-size: 14px;'>"
        "© 2025 Agilomatrix - Mezzanine Label Generator v2.1 - Enhanced Store Location Support</p>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
