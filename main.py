import cv2
import numpy as np
import os
import pandas as pd
import shutil
import json
# Removed: from google.colab import files, patches
# These modules are specific to the Google Colab environment and not needed for local execution.

# --- A. SETUP & PARAMETERS ---
# Minimum contour area to consider a detected block valid.
MIN_CONTOUR_AREA = 1000
# Width of the kernel used for morphological operations (e.g., dilation).
KERNEL_WIDTH = 17
# Height of the kernel used for morphological operations (e.g., dilation).
KERNEL_HEIGHT = 11
# Number of iterations for the dilation operation.
DILATE_ITERATIONS = 3
# Lower threshold for Canny edge detection.
CANNY_LOWER = 50
# Upper threshold for Canny edge detection.
CANNY_UPPER = 150
# Accumulator threshold parameter for Hough Line Transform.
HOUGH_THRESHOLD = 100
# Minimum length of line to be detected by Hough Line Transform.
MIN_LINE_LENGTH = 50
# Maximum allowed gap between line segments to treat them as single line.
MAX_LINE_GAP = 10

# Output directory for processed images and JSON results
output_directory_single = 'single_invoice_processed'

# Ensure the output directory exists. If it doesn't, create it.
# Removed shutil.rmtree(output_directory_single) to avoid deleting previous results on every run
# You can manually delete the folder if you want to clear old outputs.
if not os.path.exists(output_directory_single):
    os.makedirs(output_directory_single)

print("### INVOICE DOCUMENT UNDERSTANDING APPLICATION STARTING ###")

def process_single_invoice(uploaded_file_name, image_content_bytes):
    """
    Processes a single invoice image (from local file content) to extract features
    and predict block labels, ensuring unique labels.

    Args:
        uploaded_file_name (str): The name of the input file (e.g., "invoice.jpg").
                                  Used primarily for naming output files and logging.
        image_content_bytes (bytes): The binary content of the image file read from disk.

    Returns:
        pd.DataFrame: A DataFrame containing the predicted unique labels and bounding box info.
        np.array: The image with predicted labels drawn on it. Returns None if image decoding fails.
    """
    master_data = []
    print(f"--- Processing {uploaded_file_name}... ---")

    # Decode the image bytes into an OpenCV image (numpy array)
    np_img = np.frombuffer(image_content_bytes, np.uint8)
    image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    if image is None:
        print(f"--- Error: Could not decode {uploaded_file_name}. Check if it's a valid image file (e.g., JPG, PNG). ---")
        return pd.DataFrame(), None

    output_image = image.copy()
    page_height, page_width, _ = image.shape
    if page_height == 0 or page_width == 0:
        print(f"--- Error: Invalid dimensions for {uploaded_file_name}. Image might be corrupted or empty. ---")
        return pd.DataFrame(), None

    print("\n--- Original Image ---")
    # Changed from patches.cv2_imshow to cv2.imshow for local execution
    cv2.imshow("Original Invoice Image", image)
    cv2.waitKey(0) # Wait indefinitely for a key press to close the window
    cv2.destroyAllWindows() # Close all OpenCV windows

    # --- C. IMAGE PROCESSING & FEATURE EXTRACTION ---
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Adaptive thresholding is good for varying lighting
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 4)
    # Define a rectangular kernel for morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (KERNEL_WIDTH, KERNEL_HEIGHT))
    # Dilate to connect broken text segments into larger blocks
    dilated = cv2.dilate(thresh, kernel, iterations=DILATE_ITERATIONS)
    # Find contours (potential blocks) in the dilated image
    contours, _ = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print(f"No significant blocks found in {uploaded_file_name}.")
        return pd.DataFrame(), output_image

    # Extract bounding boxes for all contours
    bounding_boxes = [cv2.boundingRect(c) for c in contours]
    # Sort contours and their bounding boxes from top-left to bottom-right
    sorted_pairs = sorted(zip(contours, bounding_boxes), key=lambda b: (b[1][1], b[1][0]))
    sorted_contours, sorted_bounding_boxes = zip(*sorted_pairs) if sorted_pairs else ([], [])

    block_id_counter = 1
    for contour, (x, y, w, h) in zip(sorted_contours, sorted_bounding_boxes):
        # Skip very small or invalid contours
        if cv2.contourArea(contour) < MIN_CONTOUR_AREA or w == 0 or h == 0:
            continue

        block_id = block_id_counter

        # Initialize variables with default values to prevent NameError
        # These are used in scoring functions and are set to defaults if not computed in loop.
        has_label_above = False
        uniform_height_score = 0.0
        baseline_alignment = False
        numeric_pattern_score = 0.0
        num_text_lines = 0
        left_alignment_score = 0.0
        content_density = 0.0
        has_contact_pattern = False
        num_horizontal_lines = 0
        num_vertical_lines = 0
        block_density = 0
        has_sender_receiver_above = False
        is_top_row_wide = False
        grid_score = 0
        has_nested_text = False
        has_total_below = False
        total_below_distance = 1.0

        # --- Common Features for all blocks ---
        norm_x = x / page_width
        norm_y = y / page_height
        norm_x_center = (x + w / 2) / page_width
        norm_y_center = (y + h / 2) / page_height
        norm_width = w / page_width
        norm_height = h / page_height
        norm_area = (w * h) / (page_width * page_height)
        aspect_ratio = w / h if h != 0 else 0 # Avoid division by zero

        # Analyze colors within the block's region of interest (ROI)
        block_roi_color = image[y:y+h, x:x+w]
        pixels = block_roi_color.reshape(-1, 3)
        num_distinct_colors = len(np.unique(pixels, axis=0))

        # Analyze pixel density and internal contours (for text/graphics)
        block_roi_gray = gray[y:y+h, x:x+w]
        _, roi_thresh = cv2.threshold(block_roi_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        black_pixels = cv2.countNonZero(roi_thresh)
        black_pixel_density = black_pixels / (w * h) if w * h != 0 else 0
        internal_contours, _ = cv2.findContours(roi_thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        num_internal_contours = len(internal_contours)

        # --- Logo/Signature Specific Features ---
        edges = cv2.Canny(block_roi_gray, CANNY_LOWER, CANNY_UPPER)
        edge_density = cv2.countNonZero(edges) / (w * h) if w * h != 0 else 0
        color_gradient_score = np.var(block_roi_gray) if block_roi_gray.size > 0 else 0
        # Check for small, dense internal contours that might resemble trademark symbols
        has_trademark_symbol = any(cv2.contourArea(c) < 200 and cv2.boundingRect(c)[2] * cv2.boundingRect(c)[3] != 0 and cv2.countNonZero(roi_thresh[cv2.boundingRect(c)[1]:cv2.boundingRect(c)[1]+cv2.boundingRect(c)[3], cv2.boundingRect(c)[0]:cv2.boundingRect(c)[0]+cv2.boundingRect(c)[2]]) / (cv2.boundingRect(c)[2] * cv2.boundingRect(c)[3]) > 0.5 for c in internal_contours) if internal_contours else False

        # Compute Whitespace_Margin relative to other blocks
        margins = []
        for _, (x2, y2, w2, h2) in zip(sorted_contours, sorted_bounding_boxes):
            if (x2, y2, w2, h2) == (x, y, w, h): # Skip self-comparison
                continue
            # Calculate horizontal and vertical distances to other blocks
            if x > x2 + w2: margins.append((x - (x2 + w2)) / page_width)
            if x2 > x + w: margins.append((x2 - (x + w)) / page_width)
            if y > y2 + h2: margins.append((y - (y2 + h2)) / page_height)
            if y2 > y + h: margins.append((y2 - (y + h)) / page_height)
        whitespace_margin = min(margins) if margins else 0 # Smallest margin to nearest block

        # --- Invoice Number Specific Features ---
        text_line_y = sorted(set([cv2.boundingRect(c)[1] for c in internal_contours]))
        num_text_lines = len(text_line_y)
        contour_heights = [cv2.boundingRect(c)[3] for c in internal_contours]
        uniform_height_score = np.var(contour_heights) if contour_heights and len(contour_heights) > 1 else 0.0
        baseline_alignment = len(set([cv2.boundingRect(c)[1] for c in internal_contours])) <= 1 + 5 / page_height if internal_contours else False
        # Heuristic for numeric patterns (small aspect ratio contours often digits)
        numeric_pattern_score = sum(1 for c in internal_contours if cv2.arcLength(c, False) / (cv2.boundingRect(c)[2] + cv2.boundingRect(c)[3] + 1) < 0.5) / max(len(internal_contours), 1) if internal_contours else 0.0
        # Checks if there's a smaller block directly above (could be a label like "Invoice No.")
        has_label_above = any(y2 < y and abs((y - y2) / page_height) < 0.1 and (w2 * h2 != 0 and cv2.countNonZero(roi_thresh[y2:y2+h2, x2:x2+w2]) / (w2 * h2) > 0.1) and (w2 * h2 != 0 and len(np.unique(image[y2:y2+h2, x2:x2+w2].reshape(-1, 3), axis=0)) < 30) for _, (x2, y2, w2, h2) in zip(sorted_contours, sorted_bounding_boxes) if w2 * h2 != 0)


        # --- Sender/Receiver Specific Features ---
        left_alignment_score = np.var([cv2.boundingRect(c)[0] for c in internal_contours]) if internal_contours and len(internal_contours) > 1 else 0.0
        content_density = num_internal_contours / (w * h) if w * h != 0 else 0.0
        # Heuristic for contact patterns (presence of small, dense contours like dots, dashes, numbers)
        has_contact_pattern = any(cv2.contourArea(c) < 500 and (cv2.boundingRect(c)[2] * cv2.boundingRect(c)[3] != 0 and cv2.countNonZero(roi_thresh[cv2.boundingRect(c)[1]:cv2.boundingRect(c)[1]+cv2.boundingRect(c)[3], cv2.boundingRect(c)[0]:cv2.boundingRect(c)[0]+cv2.boundingRect(c)[2]]) / (cv2.boundingRect(c)[2] * cv2.boundingRect(c)[3]) > 0.5) for c in internal_contours) if internal_contours else False


        # --- Item Table Specific Features ---
        # Detect horizontal and vertical lines within the block's edge map
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=HOUGH_THRESHOLD, minLineLength=MIN_LINE_LENGTH, maxLineGap=MAX_LINE_GAP)
        num_horizontal_lines = 0
        num_vertical_lines = 0
        if lines is not None:
            for line in lines:
                x1_l, y1_l, x2_l, y2_l = line[0]
                angle = np.arctan2(y2_l - y1_l, x2_l - x1_l) * 180 / np.pi
                if abs(angle) < 10 or abs(angle) > 170: # Horizontal lines (close to 0 or 180 degrees)
                    num_horizontal_lines += 1
                elif 80 < abs(angle) < 100: # Vertical lines (close to 90 degrees)
                    num_vertical_lines += 1
        # Block density based on larger internal contours (like item rows)
        block_density = sum(1 for c in internal_contours if cv2.contourArea(c) >= 500)

        # Check for multiple 'address-like' blocks above (common for sender/receiver)
        has_sender_receiver_above = sum(1 for _, (x2_s, y2_s, w2_s, h2_s) in zip(sorted_contours, sorted_bounding_boxes)
                                       if y2_s < y and abs((y - y2_s) / page_height) < 0.3 and abs(y2_s - y) < 5 and (w2_s * h2_s) > 0.001 * page_width * page_height) >= 2

        # Check if this block is wide compared to blocks directly below it
        blocks_below = [(x2_bb, y2_bb, w2_bb, h2_bb) for _, (x2_bb, y2_bb, w2_bb, h2_bb) in zip(sorted_contours, sorted_bounding_boxes) if y2_bb > y + h and abs((y2_bb - (y + h)) / page_height) < 0.2 and (w2_bb * h2_bb) > 0]
        if blocks_below:
            norm_widths_below = [w_val / page_width for (_,_,w_val,_) in blocks_below]
            is_top_row_wide = norm_width >= np.percentile(norm_widths_below, 75)
        else:
            is_top_row_wide = False

        grid_score = len(set([cv2.boundingRect(c)[1] for c in internal_contours])) if internal_contours else 0
        # Checks for text content within shapes, implying complex content like tables
        has_nested_text = any(cv2.contourArea(c) < 500 and (cv2.boundingRect(c)[2] * cv2.boundingRect(c)[3] != 0 and cv2.countNonZero(roi_thresh[cv2.boundingRect(c)[1]:cv2.boundingRect(c)[1]+cv2.boundingRect(c)[3], cv2.boundingRect(c)[0]:cv2.boundingRect(c)[0]+cv2.boundingRect(c)[2]]) / (cv2.boundingRect(c)[2] * cv2.boundingRect(c)[3]) >= 0.1) for c in internal_contours) if internal_contours else False

        # Look for a "total" block below (common end-of-table indicator)
        total_below_candidates = [(x2_tb, y2_tb, w2_tb, h2_tb) for _, (x2_tb, y2_tb, w2_tb, h2_tb) in zip(sorted_contours, sorted_bounding_boxes)
                                  if y2_tb > y + h and abs((y2_tb - (y + h)) / page_height) < 0.2 and (w2_tb * h2_tb) / (page_width * page_height) <= 0.05 and x2_tb / page_width >= 0.5 and (w2_tb*h2_tb) > 0]
        has_total_below = len(total_below_candidates) > 0
        total_below_distance = min([(y2_tb - (y + h)) / page_height for (x2_tb, y2_tb, w2_tb, h2_tb) in total_below_candidates], default=1.0)


        # Append all extracted features for the current block to the master data list
        master_data.append({
            'Invoice_ID': uploaded_file_name,
            'Block_ID': block_id,
            'Top_Left_X': x,
            'Top_Left_Y': y,
            'Bottom_Right_X': x + w,
            'Bottom_Right_Y': y + h,
            'Normalized_X': round(norm_x, 4),
            'Normalized_Y': round(norm_y, 4),
            'Normalized_X_Center': round(norm_x_center, 4),
            'Normalized_Y_Center': round(norm_y_center, 4),
            'Normalized_Width': round(norm_width, 4),
            'Normalized_Height': round(norm_height, 4),
            'Normalized_Area': round(norm_area, 4),
            'Aspect_Ratio': round(aspect_ratio, 4),
            'Num_Distinct_Colors': num_distinct_colors,
            'Black_Pixel_Density': round(black_pixel_density, 4),
            'Num_Internal_Contours': num_internal_contours,
            'Edge_Density': round(edge_density, 4),
            'Color_Gradient_Score': round(color_gradient_score, 4),
            'Has_Trademark_Symbol': has_trademark_symbol,
            'Whitespace_Margin': round(whitespace_margin, 4),
            'Has_Label_Above': has_label_above,
            'Uniform_Height_Score': round(uniform_height_score, 4),
            'Baseline_Alignment': baseline_alignment,
            'Numeric_Pattern_Score': round(numeric_pattern_score, 4),
            'Num_Text_Lines': num_text_lines,
            'Left_Alignment_Score': round(left_alignment_score, 4),
            'Content_Density': round(content_density, 4),
            'Has_Contact_Pattern': has_contact_pattern,
            'Num_Horizontal_Lines': num_horizontal_lines,
            'Num_Vertical_Lines': num_vertical_lines,
            'Block_Density': block_density,
            'Has_Sender_Receiver_Above': has_sender_receiver_above,
            'Is_Top_Row_Wide': is_top_row_wide,
            'Grid_Score': grid_score,
            'Has_Nested_Text': has_nested_text,
            'Has_Total_Below': has_total_below,
            'Total_Below_Distance': round(total_below_distance, 4)
        })
        block_id_counter += 1

    df_single_invoice = pd.DataFrame(master_data)
    # Convert appropriate columns to numeric/int and handle NaNs
    for col in df_single_invoice.columns:
        if df_single_invoice[col].dtype == 'object':
            df_single_invoice[col] = pd.to_numeric(df_single_invoice[col], errors='coerce').fillna(0)
        elif df_single_invoice[col].dtype == 'bool':
            df_single_invoice[col] = df_single_invoice[col].astype(int)
        elif df_single_invoice[col].isnull().any():
            df_single_invoice[col].fillna(0, inplace=True)

    # --- D. SCORING FUNCTIONS FOR UNIQUE LABELS ---
    # These functions calculate a score indicating how likely a block is a specific type (e.g., logo, table).
    # Scores are based on a combination of features extracted earlier.

    def calculate_logo_score(row):
        score = 0
        # Core conditions: Top part of the page and somewhat squarish
        if row['Normalized_Y'] < 0.25: # Typically found in the top quarter of the page
            score += 10
        if row['Aspect_Ratio'] > 0.3 and row['Aspect_Ratio'] < 3.0: # Relatively squarish (not too wide or too tall)
            score += 10

        # Strong positive indicators
        if row['Has_Trademark_Symbol']: score += 20 # Very strong indicator if present
        if row['Num_Internal_Contours'] > 0 and row['Num_Internal_Contours'] < 25: # Has some internal structure, but not overly complex like dense text
            score += 10

        # Color/Gradient indicators (made less strict)
        if row['Num_Distinct_Colors'] > 50: score += 5 # Indicates some color, not just monochrome text
        if row['Color_Gradient_Score'] > 100: score += 5 # Indicates some visual variation (e.g., image gradients)

        # Penalties for looking like text blocks
        if row['Num_Text_Lines'] > 5: score -= 15 # Logos usually don't have many distinct text lines
        if row['Content_Density'] > 0.15: score -= 10 # High density often means dense text

        # Further refine position to prefer left/center top
        if row['Normalized_X_Center'] < 0.6: score += 5 # Prefer left or center horizontally
        if row['Normalized_Area'] < 0.15: score += 5 # Logos are usually not huge blocks

        return max(0, score) # Ensure score is not negative

    def calculate_signature_score(row):
        score = 0
        if row['Normalized_Y_Center'] > 0.80: # Typically found in the bottom part of the page
            score += 10
            if row['Aspect_Ratio'] > 2.0 and row['Edge_Density'] > 0.03: score += 10 # Wider, with some edge detail
            if row['Normalized_Y'] > 0.75 and row['Color_Gradient_Score'] > 1500 and row['Normalized_Area'] < 0.05: score += 15
            if row['Num_Internal_Contours'] > 20: score += 5 # More complex shapes/scribbles
            # Penalize if it's too wide or too textual
            if row['Normalized_Width'] > 0.7 or row['Num_Text_Lines'] > 10: score -= 15
        return max(0, score)

    def calculate_table_score(row):
        score = 0
        if row['Normalized_Area'] > 0.15 and row['Normalized_Y_Center'] > 0.3: # Large area and usually in the middle or bottom half
            score += 10
            if row['Grid_Score'] > 8 or row['Num_Horizontal_Lines'] > 4 or row['Num_Vertical_Lines'] > 2: score += 15 # Strong indicator: presence of lines and structured grid
            if row['Has_Total_Below'] and row['Normalized_Width'] > 0.7: score += 20 # Very strong indicator: presence of a "total" block directly below and wide
            if row['Num_Internal_Contours'] > 50: score += 10 # Many contours (items, numbers)
            if row['Block_Density'] > 10: score += 5
            # Penalize if too narrow or too high up
            if row['Normalized_Width'] < 0.4 or row['Normalized_Y'] < 0.2: score -= 10
        return max(0, score)

    def calculate_invoice_no_score(row):
        score = 0
        # Prefer top-right / isolated
        if row['Normalized_Y'] < 0.3 and row['Normalized_X'] > 0.5: score += 10 # Top-right quadrant
        if row['Whitespace_Margin'] > 0.05: score += 10 # Good isolation from other elements
        if row['Num_Text_Lines'] <= 2: score += 15 # Typically short, single or double line
        if row['Numeric_Pattern_Score'] > 0.5: score += 20 # High numeric content (e.g., "INV-2023-012345")
        if row['Uniform_Height_Score'] < 100: score += 5 # Uniform character height (like printed text)
        # Penalize if it has too many lines or is very wide (indicates it's not just a number)
        if row['Num_Text_Lines'] > 3 or row['Normalized_Width'] > 0.4: score -= 20
        return max(0, score)

    def calculate_sender_score(row):
        score = 0
        if row['Normalized_Y'] < 0.4: # Top half of the page
            score += 5
            if row['Normalized_X'] < 0.5: score += 5 # Often left-aligned
            if row['Num_Text_Lines'] >= 3 and row['Num_Text_Lines'] <= 10: score += 10 # Multiple lines of address/contact info
            if row['Content_Density'] > 0.05: score += 5
            if row['Left_Alignment_Score'] < 500: score += 10 # Text within the block is well left-aligned
            if row['Has_Contact_Pattern']: score += 15 # Presence of phone numbers, emails, etc.
            if not row['Has_Sender_Receiver_Above']: score += 5 # Less likely to be receiver if nothing address-like above it
            # Penalize if too wide, too low, or too sparse
            if row['Normalized_Width'] > 0.6 or row['Normalized_Y'] > 0.4: score -= 10
            if row['Num_Internal_Contours'] < 10: score -= 10
        return max(0, score)

    def calculate_receiver_score(row):
        score = 0
        if row['Normalized_Y'] < 0.5: # Upper half (often below sender)
            score += 5
            if row['Normalized_X'] < 0.5: score += 10 # Strongly left-aligned
            if row['Num_Text_lines'] >= 3 and row['Num_Text_Lines'] <= 10: score += 10
            if row['Content_Density'] > 0.05: score += 5
            if row['Left_Alignment_Score'] < 500: score += 10
            if row['Has_Contact_Pattern']: score += 15
            # Prefer to be below a potential sender block
            if row['Has_Sender_Receiver_Above']: score += 10 # Indicates another address block (sender) is likely above
            # Penalize if too wide or too low
            if row['Normalized_Width'] > 0.6 or row['Normalized_Y'] > 0.5: score -= 10
            if row['Num_Internal_Contours'] < 10: score -= 10
        return max(0, score)

    # Apply all scoring functions to the DataFrame
    classified_df = df_single_invoice.copy()
    classified_df['logo_score'] = classified_df.apply(calculate_logo_score, axis=1)
    classified_df['signature_score'] = classified_df.apply(calculate_signature_score, axis=1)
    classified_df['table_score'] = classified_df.apply(calculate_table_score, axis=1)
    classified_df['invoice_no_score'] = classified_df.apply(calculate_invoice_no_score, axis=1)
    classified_df['sender_score'] = classified_df.apply(calculate_sender_score, axis=1)
    classified_df['receiver_score'] = classified_df.apply(calculate_receiver_score, axis=1)

    # --- Debugging: Print logo scores for all blocks (for development/tuning) ---
    print("\n--- Debugging: Logo Scores for all blocks (Top 5 by score) ---")
    print(classified_df[['Block_ID', 'Top_Left_X', 'Top_Left_Y', 'Normalized_Y', 'Aspect_Ratio',
                         'Num_Distinct_Colors', 'Color_Gradient_Score', 'Has_Trademark_Symbol',
                         'Num_Text_Lines', 'Content_Density', 'Normalized_X_Center', 'Normalized_Area',
                         'logo_score']].sort_values(by='logo_score', ascending=False).head())
    print("---------------------------------------------")
    # End Debugging Print


    classified_df['predicted_label'] = 'other' # Default all blocks to 'other' initially
    assigned_block_ids = set() # Keep track of blocks that have already been assigned a unique label

    # --- E. UNIQUE LABEL ASSIGNMENT (PRIORITY BASED) ---
    # Define a priority for labels. Higher priority labels are assigned first to ensure uniqueness.
    # The score_threshold is a minimum score a block must achieve to be considered for that label.
    priority_labels = [
        ('table', 30),        # Tables are often large and distinct, high priority
        ('logo', 15),         # Logos are usually top, unique
        ('sender', 20),       # Sender/Receiver often appear close, order matters
        ('receiver', 20),
        ('invoice_no', 25),   # Invoice number is critical, often isolated
        ('signature', 20)     # Signature is usually at the bottom
    ]

    for label_type, score_threshold in priority_labels:
        # Find the best candidate block for the current label type that hasn't been assigned yet
        best_candidate = classified_df[
            (classified_df['Block_ID'].apply(lambda x: x not in assigned_block_ids)) & # Must not be already assigned
            (classified_df[f'{label_type}_score'] >= score_threshold)                   # Must meet score threshold
        ].sort_values(by=f'{label_type}_score', ascending=False).head(1) # Get the one with highest score

        if not best_candidate.empty:
            block_id_to_assign = best_candidate.iloc[0]['Block_ID']
            # Assign the predicted label to the chosen block
            classified_df.loc[classified_df['Block_ID'] == block_id_to_assign, 'predicted_label'] = label_type
            assigned_block_ids.add(block_id_to_assign) # Add to assigned set to prevent re-assignment
            print(f"Assigned '{label_type}' to Block_ID: {block_id_to_assign} with score: {best_candidate.iloc[0][f'{label_type}_score']:.2f}")

    # --- Draw predictions on image for visual output ---
    # Filter for only the uniquely predicted labels (exclude 'other')
    final_predictions_for_display = classified_df[classified_df['predicted_label'] != 'other'].copy()

    for index, row in final_predictions_for_display.iterrows():
        # Get bounding box coordinates and label
        x, y, w, h = int(row['Top_Left_X']), int(row['Top_Left_Y']), \
                     int(row['Bottom_Right_X'] - row['Top_Left_X']), int(row['Bottom_Right_Y'] - row['Top_Left_Y'])
        label = row['predicted_label'].upper()

        # Define colors for different labels
        color = (0, 0, 255) # Default Red (should not be seen if filtering 'other' correctly)
        if label == 'LOGO': color = (255, 0, 0) # Blue
        elif label == 'SENDER': color = (0, 255, 0) # Green
        elif label == 'RECEIVER': color = (255, 255, 0) # Cyan
        elif label == 'INVOICE_NO': color = (0, 255, 255) # Yellow
        elif label == 'TABLE': color = (255, 0, 255) # Magenta
        elif label == 'SIGNATURE': color = (0, 165, 255) # Orange

        # Draw the rectangle and put the text label on the output image
        cv2.rectangle(output_image, (x, y), (x + w, y + h), color, 2)
        # Adjust text position to be above the rectangle
        text_y = y - 10 if y - 10 > 10 else y + 20 # Ensure text is visible and not off-image
        cv2.putText(output_image, label, (x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    return final_predictions_for_display, output_image

# --- MAIN APPLICATION LOGIC FOR LOCAL EXECUTION ---
if __name__ == "__main__":
    print("\n### Invoice Document Understanding Application ###")
    print("Please ensure your invoice image file is in a known location relative to this script,")
    print("or provide its full path.")
    
    # Prompt user for the image file path
    # Example for Windows: 'C:\\Users\\YourName\\Documents\\invoice.png'
    # Example for macOS/Linux: '/home/YourName/Documents/invoice.jpg'
    # Example for same directory: 'my_invoice.png'
    image_file_path = input("Enter the full path or filename of your invoice image: ")

    # Validate if the file exists
    if not os.path.exists(image_file_path):
        print(f"--- Error: File not found at '{image_file_path}'. Please check the path and try again. ---")
        exit() # Exit the script if file not found

    try:
        # Read the image content as bytes from the local file system
        with open(image_file_path, 'rb') as f:
            image_content_bytes = f.read()
        
        # Extract just the filename from the full path (e.g., "invoice.jpg" from "C:\data\invoice.jpg")
        # This is used for internal logging and naming output files.
        uploaded_file_name = os.path.basename(image_file_path)

        # Call the main processing function
        predicted_blocks_df, labeled_image = process_single_invoice(uploaded_file_name, image_content_bytes)

        print("\n" + "="*60)
        print(f"### CLASSIFICATION RESULTS FOR {uploaded_file_name} ###")
        print("="*60)

        # Define the desired order of labels for the JSON output
        desired_labels_order = ['logo', 'sender', 'receiver', 'invoice_no', 'table', 'signature']
        
        # Prepare the data structure for the JSON output
        invoice_data = {
            "file_name": uploaded_file_name,
            "detected_fields": {} # This will store the coordinates for each detected field
        }

        # Populate the detected_fields dictionary
        for label_name in desired_labels_order:
            found_block = predicted_blocks_df[predicted_blocks_df['predicted_label'] == label_name]
            if not found_block.empty:
                x_coord = int(found_block.iloc[0]['Top_Left_X'])
                y_coord = int(found_block.iloc[0]['Top_Left_Y'])
                # Store coordinates as a nested dictionary (x, y) for clarity in JSON
                invoice_data["detected_fields"][label_name] = {
                    "top_left_x": x_coord,
                    "top_left_y": y_coord
                }
            else:
                # If a field is not detected, explicitly set its value to None in JSON
                invoice_data["detected_fields"][label_name] = None

        # Print results to the console for immediate feedback
        print("\n--- Detected Fields (Console Output) ---")
        for label, data in invoice_data["detected_fields"].items():
            if data: # If data exists (i.e., not None)
                print(f"{label.upper()} : coordinates of top left position ({data['top_left_x']}, {data['top_left_y']})")
            else:
                print(f"{label.upper()} : coordinates of top left position null")

        # --- Save results to JSON file ---
        # Get the filename without its extension (e.g., "invoice" from "invoice.jpg")
        base_filename_no_ext = os.path.splitext(uploaded_file_name)[0]
        json_output_filename = f"{base_filename_no_ext}_results.json"
        json_output_path = os.path.join(output_directory_single, json_output_filename)
        
        # Write the JSON data to a file, with indentation for readability
        with open(json_output_path, 'w') as f:
            json.dump(invoice_data, f, indent=4)

        print(f"\nDetection results saved to: '{json_output_path}'")

        # --- Display and Save Processed Image ---
        if labeled_image is not None:
            print("\n--- Processed Image with Predicted Labels ---")
            # Changed from patches.cv2_imshow to cv2.imshow for local execution
            cv2.imshow("Processed Invoice Image", labeled_image)
            cv2.waitKey(0) # Wait indefinitely for a key press to close the window
            cv2.destroyAllWindows() # Close all OpenCV windows

            output_image_filename = f"{base_filename_no_ext}_predicted.jpg"
            output_image_path = os.path.join(output_directory_single, output_image_filename)
            cv2.imwrite(output_image_path, labeled_image) # Save the image to the specified path
            print(f"Processed image saved to: '{output_image_path}'")
        else:
            print(f"--- No processed image was generated for '{uploaded_file_name}'. ---")

    except Exception as e:
        # Catch any unexpected errors during execution and print them
        print(f"An unexpected error occurred during processing: {e}")


print("\n### INVOICE DOCUMENT UNDERSTANDING APPLICATION FINISHED ###")