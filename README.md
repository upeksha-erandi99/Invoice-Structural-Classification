# Invoice-Structural-Classification
My Final Year Project: Automating invoice data extraction through computer vision. This solution identifies and provides JSON coordinates for critical invoice fields like logos, sender details, and item tables.

## Project Overview

This repository contains the code for my Final Year Project (FYP) focused on **Invoice Document Understanding**. The primary goal of this project is to automatically identify and extract the precise locations (bounding box coordinates) of key elements within an invoice image, such as the company logo, sender information, receiver details, invoice number, itemized table, and signature.

Invoices are a common type of semi-structured document. While they contain essential business data, their layouts vary widely. This project aims to automate the initial step of data extraction by accurately localizing these critical fields, serving as a foundational component for further OCR (Optical Character Recognition) and data processing.

## What Was Done

This project implements a robust image processing and rule-based classification pipeline to achieve accurate element localization. Here's a breakdown of the key components and methodologies used:

### 1. Image Preprocessing
The input invoice image undergoes several preprocessing steps to enhance its features and prepare it for contour detection:
* **Grayscaling:** Converts the image to grayscale for simpler intensity-based processing.
* **Gaussian Blurring:** Reduces noise and smooths the image.
* **Adaptive Thresholding:** Binarizes the image, making text and graphic elements stand out. This is crucial for handling varied lighting conditions across different invoices.
* **Dilation:** Expands the white regions (foreground), effectively connecting broken text lines and merging related contours into larger, more meaningful blocks (e.g., an address block instead of individual characters).
* **Canny Edge Detection:** Used to find prominent edges, particularly useful for detecting distinct shapes like logos or table lines.

### 2. Contour Detection & Feature Extraction
After preprocessing, the system identifies distinct "blocks" (regions of interest) on the invoice. For each detected block, a comprehensive set of features is extracted:
* **Geometric Features:** Top-left/bottom-right coordinates, normalized position (X, Y, Center X, Center Y), normalized width, height, and area, aspect ratio.
* **Content Features:** Number of distinct colors, black pixel density, number of internal contours (indicating complexity or text presence), edge density, color gradient score.
* **Structural Features:** Presence of trademark symbols, whitespace margins, number of text lines, uniformity of text height, baseline alignment, numeric pattern scores (for invoice numbers), content density, contact patterns.
* **Layout Features:** Number of horizontal/vertical lines (crucial for tables), block density, relative positioning to other blocks (e.g., `Has_Sender_Receiver_Above`, `Has_Total_Below`).

### 3. Rule-Based Scoring Functions
Each extracted block is evaluated against a set of custom-defined scoring functions, one for each target label (logo, sender, receiver, invoice number, table, signature). These functions assign a numerical score to a block based on how well its extracted features match the typical characteristics of that specific invoice element. For example:
* **Logo Score:** Favors blocks in the top-left area, with specific aspect ratios, certain color complexities, and presence of trademark symbols, while penalizing high text line counts.
* **Table Score:** Prioritizes large blocks in the middle of the page with many horizontal/vertical lines and high content density, often located above a "total" block.
* **Invoice Number Score:** Looks for blocks in the top-right, with high whitespace margins, few text lines, and a strong numeric pattern.

### 4. Priority-Based Unique Label Assignment
A critical aspect of this project is ensuring that each key element (e.g., there's usually only one logo, one invoice number) is uniquely identified. A priority list of labels is defined (e.g., `table` > `logo` > `sender` > `receiver` > `invoice_no` > `signature`). The system iterates through this priority list:
* For each label, it identifies the unassigned block with the highest score that meets a predefined score threshold.
* Once a block is assigned a label, it cannot be assigned another, thus guaranteeing unique detection for the most confident predictions.

### 5. Output Generation
The project provides two forms of output for easy interpretation and integration:
* **Visual Output:** The original invoice image is rendered with bounding boxes drawn around the detected key elements, each labeled with its predicted classification and colored for easy visual verification.
* **Structured JSON Output:** A JSON file is generated containing the file name and a dictionary of detected fields. For each detected field (e.g., "logo", "sender"), it provides the `top_left_x` and `top_left_y` coordinates. If a field is not detected, its value is `null`, making the output consistent and machine-readable for downstream processing.

## Technologies Used

* **Python 3.x:** The core programming language.
* **OpenCV (`cv2`):** For all image processing, contour detection, and drawing functionalities.
* **NumPy:** For efficient numerical operations, especially with image arrays.
* **Pandas:** For managing and analyzing extracted features in a tabular (DataFrame) format.
* **`json` module:** For generating the structured output.
* **Google Colab:** The development and execution environment, leveraging its features for file uploads (`files.upload()`) and image display (`patches.cv2_imshow()`).

## How to Use (Google Colab)

1.  **Open in Google Colab:** Click on "Open in Colab" if you're viewing this on GitHub, or simply create a new Colab notebook and copy the entire `invoice_processor.py` (or your main script) code into a cell.
2.  **Run the Code Cell:** Execute the cell containing the code.
3.  **Upload Your Invoice Image:** The script will prompt you to upload an invoice image file (e.g., `invoice.jpg`, `invoice.png`). Select your file from your local machine.
4.  **Observe the Output:**
    * You will see the original image.
    * Debugging information will be printed, including top logo scores.
    * Messages indicating which blocks were assigned which labels.
    * A table summarizing the predicted unique labels.
    * The processed image with colored bounding boxes and labels.
    * **Crucially, a JSON file (`[your_invoice_name]_results.json`) will be generated and automatically downloaded to your local machine, containing the coordinates of the detected fields.**
    * The processed image (`[your_invoice_name]_predicted.jpg`) will also be downloaded.

## Future Enhancements

* **Machine Learning Integration:** Replace rule-based scoring with trained machine learning models (e.g., CNNs for classification, object detection models like YOLO/Faster R-CNN) for improved accuracy and generalization.
* **OCR Integration:** Integrate OCR capabilities (e.g., Tesseract, Google Cloud Vision API) to extract the actual text content from the identified bounding boxes.
* **More Robust Table Parsing:** Develop more sophisticated algorithms for extracting individual cells and their content within the detected table block.
* **Handling Skewed/Rotated Documents:** Implement de-skewing and rotation correction to improve performance on imperfect scans.
* **Support for Multiple Invoice Types:** Expand the feature set and rules to accommodate a wider variety of invoice layouts and languages.
