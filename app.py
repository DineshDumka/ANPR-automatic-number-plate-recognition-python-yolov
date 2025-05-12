import streamlit as st
import cv2
import os
import numpy as np
from anpr import ANPR
from utils import (
    create_output_dir, save_processed_image,
    process_uploaded_file, is_valid_image, format_indian_plate
)

# Load custom CSS
def load_css():
    if os.path.exists("style.css"):
        with open("style.css") as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Set page config
st.set_page_config(
    page_title="Indian ANPR System",
    page_icon="🚗",
    layout="wide"
)

# Load CSS
load_css()

# Create output and test_inputs directories
create_output_dir()
os.makedirs('test_inputs', exist_ok=True)

# Initialize ANPR system
@st.cache_resource
def load_anpr():
    return ANPR()

try:
    anpr = load_anpr()
    initialization_error = None
except Exception as e:
    initialization_error = str(e)
    anpr = None

# Title and description
st.markdown('<h1 class="main-title">🚗 Indian Number Plate Recognition System</h1>', unsafe_allow_html=True)
# Add Indian flag accent
st.markdown('<div class="indian-flag-accent"></div>', unsafe_allow_html=True)

# Display initialization error if any
if initialization_error:
    st.error(f"Error initializing ANPR system: {initialization_error}")
    st.info("Please make sure you have installed all required dependencies. Check the README.md file for more information.")
    st.stop()

# Add information about Indian license plates
with st.expander("ℹ️ About Indian License Plates"):
    st.markdown("""
    ### Indian License Plate Format
    
    Indian license plates follow a specific format:
    
    **Standard Format**: `[State Code]-[District Code]-[Series]-[Number]`
    
    Examples:
    - DL-01-AB-1234 (Delhi)
    - MH-02-CD-5678 (Maharashtra)
    - KA-03-EF-9012 (Karnataka)
    
    This system is optimized to detect and recognize Indian license plates with high accuracy.
    """)

st.markdown("""
This application detects and recognizes Indian license plates from images.
""")

# Function to display plate detection results with improved visibility
def display_plate_results(plates, confidences=None):
    st.markdown('<div class="plate-section">', unsafe_allow_html=True)
    if plates:
        st.markdown('<h3 class="section-header">📋 Detected License Plates:</h3>', unsafe_allow_html=True)
        for i, plate in enumerate(plates):
            confidence_text = f" (Confidence: {confidences[i]:.2f})" if confidences and i < len(confidences) else ""
            
            # Try to format as Indian plate if not already formatted
            if '-' not in plate:
                formatted_plate = format_indian_plate(plate)
                if formatted_plate != plate:
                    st.markdown(f'<div class="license-plate-box">{formatted_plate}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="original-text">Original text: {plate}{confidence_text}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="license-plate-box">{plate}{confidence_text}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="license-plate-box">{plate}{confidence_text}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="no-plate-container">', unsafe_allow_html=True)
        st.markdown('<p class="no-plate">No license plates detected!</p>', unsafe_allow_html=True)
        st.markdown('<p class="no-plate-suggestion">Try adjusting the image angle or using a clearer image.</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Sidebar for input selection
st.sidebar.title("Input Options")
input_type = st.sidebar.radio(
    "Select input type:",
    ["Image Upload", "Demo Images"]
)

# Add configuration options
st.sidebar.title("Configuration")
confidence_threshold = st.sidebar.slider("Detection Confidence", 0.1, 0.9, 0.15, 0.05)
show_regions = st.sidebar.checkbox("Show Plate Regions", value=True)
preprocessing_level = st.sidebar.selectbox(
    "Preprocessing Level",
    ["Low", "Medium", "High"],
    index=1
)

# Advanced options
with st.sidebar.expander("Advanced Options"):
    use_tesseract = st.checkbox("Use Tesseract OCR (if available)", value=True)
    allow_hindi = st.checkbox("Allow Hindi Characters", value=True)

# Main content area
if input_type == "Image Upload":
    st.markdown('<h2 class="section-header">Image Processing</h2>', unsafe_allow_html=True)
    st.markdown('<div class="indian-flag-accent"></div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Process uploaded image
        image = process_uploaded_file(uploaded_file)
        if image is not None:
            # Process the image without drawing boxes
            with st.spinner("Processing image..."):
                try:
                    # Override confidence threshold if user changed it
                    anpr.model.conf = confidence_threshold
                    processed_image, plates, plate_regions = anpr.process_image(image, draw_boxes=False)
                    
                    # Get confidences from the results_df
                    confidences = []
                    if not anpr.results_df.empty:
                        recent_results = anpr.results_df[anpr.results_df['filename'] == 'uploaded_image']
                        if not recent_results.empty:
                            for plate in plates:
                                plate_results = recent_results[recent_results['plate_number'] == plate]
                                if not plate_results.empty:
                                    confidences.append(float(plate_results.iloc[0]['confidence']))
                                else:
                                    confidences.append(0.0)
                    
                    # Display original image
                    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Original Image", use_container_width=True)
                    
                    # Display detected plates as text with improved visibility
                    display_plate_results(plates, confidences)
                    
                    # If plates are detected and user wants to see regions, show the plate regions in the image
                    if plates and plate_regions and show_regions:
                        st.markdown('<h3 class="section-header">Plate Regions:</h3>', unsafe_allow_html=True)
                        # Create a copy of the image to draw on
                        highlight_image = image.copy()
                        for i, (x1, y1, x2, y2) in enumerate(plate_regions):
                            # Draw a green rectangle around the plate
                            cv2.rectangle(highlight_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            # Add plate number as text
                            if i < len(plates):
                                cv2.putText(highlight_image, plates[i], (x1, y1-10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                        
                        # Display the image with highlighted plates
                        st.image(cv2.cvtColor(highlight_image, cv2.COLOR_BGR2RGB), 
                                caption="Detected Plate Regions", use_container_width=True)
                    
                    # Save processed image
                    if plates:
                        output_path = save_processed_image(image, f"original_{uploaded_file.name}")
                        st.info(f"Original image saved to: {output_path}")
                except Exception as e:
                    st.error(f"Error processing image: {str(e)}")
                    st.info("Please try another image or adjust the configuration settings.")

else:  # Demo Images
    st.markdown('<h2 class="section-header">Demo Images</h2>', unsafe_allow_html=True)
    st.markdown('<div class="indian-flag-accent"></div>', unsafe_allow_html=True)
    
    # Create demo directory if it doesn't exist
    os.makedirs('test_inputs', exist_ok=True)
    
    # List available demo images
    demo_images = []
    if os.path.exists('test_inputs'):
        demo_images = [f for f in os.listdir('test_inputs') if is_valid_image(f)]
    
    if not demo_images:
        st.warning("No demo images available in the test_inputs folder. Please add some test images.")
        
        # Option to upload a demo image
        st.markdown("### Upload a Demo Image")
        demo_upload = st.file_uploader("Upload a demo image to the test_inputs folder:", type=["jpg", "jpeg", "png"])
        
        if demo_upload is not None:
            # Save the uploaded image to the test_inputs folder
            demo_path = os.path.join('test_inputs', demo_upload.name)
            with open(demo_path, "wb") as f:
                f.write(demo_upload.getvalue())
            st.success(f"Demo image '{demo_upload.name}' uploaded successfully!")
            st.experimental_rerun()
    else:
        # Select demo image
        selected_image = st.selectbox("Select a demo image:", demo_images)
        
        if selected_image:
            image_path = os.path.join('test_inputs', selected_image)
            
            # Process the image without drawing boxes
            with st.spinner("Processing image..."):
                try:
                    # Override confidence threshold if user changed it
                    anpr.model.conf = confidence_threshold
                    original_image = cv2.imread(image_path)
                    processed_image, plates, plate_regions = anpr.process_image(image_path, draw_boxes=False)
                    
                    # Get confidences from the results_df
                    confidences = []
                    if not anpr.results_df.empty:
                        recent_results = anpr.results_df[anpr.results_df['filename'] == selected_image]
                        if not recent_results.empty:
                            for plate in plates:
                                plate_results = recent_results[recent_results['plate_number'] == plate]
                                if not plate_results.empty:
                                    confidences.append(float(plate_results.iloc[0]['confidence']))
                                else:
                                    confidences.append(0.0)
                    
                    # Display results
                    st.image(cv2.imread(image_path), caption="Original Image", channels="BGR", use_container_width=True)
                    
                    # Display detected plates as text with improved visibility
                    display_plate_results(plates, confidences)
                    
                    # If plates are detected and user wants to see regions, show the plate regions in the image
                    if plates and plate_regions and show_regions:
                        st.markdown('<h3 class="section-header">Plate Regions:</h3>', unsafe_allow_html=True)
                        # Create a copy of the image to draw on
                        highlight_image = original_image.copy()
                        for i, (x1, y1, x2, y2) in enumerate(plate_regions):
                            # Draw a green rectangle around the plate
                            cv2.rectangle(highlight_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            # Add plate number as text
                            if i < len(plates):
                                cv2.putText(highlight_image, plates[i], (x1, y1-10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                        
                        # Display the image with highlighted plates
                        st.image(cv2.cvtColor(highlight_image, cv2.COLOR_BGR2RGB), 
                                caption="Detected Plate Regions", use_container_width=True)
                except Exception as e:
                    st.error(f"Error processing image: {str(e)}")
                    st.info("Please try another image or adjust the configuration settings.")

# Add footer with Indian flag accent
st.markdown('<div class="indian-flag-accent"></div>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #666; margin-top: 30px;">© 2023 Indian ANPR System | Developed by Dinesh Dumka</p>', unsafe_allow_html=True)

# Display detection history
st.sidebar.title("Detection History")
if os.path.exists('output/detection_results.csv'):
    try:
        history = anpr.results_df
        if not history.empty:
            st.sidebar.markdown('<div class="history-table">', unsafe_allow_html=True)
            st.sidebar.dataframe(history)
            st.sidebar.markdown('</div>', unsafe_allow_html=True)
            
            # Add download button for CSV
            csv = history.to_csv(index=False)
            st.sidebar.download_button(
                label="Download Detection History",
                data=csv,
                file_name="anpr_detection_history.csv",
                mime="text/csv",
            )
        else:
            st.sidebar.info("No detection history yet.")
    except Exception as e:
        st.sidebar.error(f"Error loading detection history: {e}")
else:
    st.sidebar.info("No detection history yet.") 