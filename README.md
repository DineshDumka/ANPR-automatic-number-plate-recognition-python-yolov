# Advanced Indian Automatic Number Plate Recognition (ANPR) System

![ANPR System](https://img.shields.io/badge/ANPR-v3.0-blue)
![Python](https://img.shields.io/badge/Python-3.8%2B-brightgreen)
![OpenCV](https://img.shields.io/badge/OpenCV-4.9.0-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.45.0-red)
![YOLOv5](https://img.shields.io/badge/YOLOv5-8.0.0-yellow)
![EasyOCR](https://img.shields.io/badge/EasyOCR-1.7.2-purple)

**🚀 [Live Demo: Try the app online!](https://dineshdumka-anpr-automatic-number-plate-recognition--app-cilo5z.streamlit.app/)**

An advanced Automatic Number Plate Recognition (ANPR) system specifically optimized for Indian license plates. This system uses state-of-the-art computer vision and OCR techniques to accurately detect and recognize Indian license plates from images.

## 🚀 Features

- **Specialized for Indian License Plates**
  - Support for all Indian state codes (DL, MH, KA, etc.)
  - Handles both English and Hindi text on plates
  - Optimized for Indian license plate formats and fonts
  - Automatic formatting of detected plates (e.g., DL-01-AB-1234)

- **Advanced Detection Techniques**
  - Uses YOLOv5/YOLOv8 with optimized parameters for Indian plates
  - Multiple image enhancement techniques for better detection
  - Fallback to edge detection when YOLO doesn't detect plates
  - Robust against different lighting conditions and angles

- **Multi-Engine OCR Approach**
  - Primary: EasyOCR with Hindi and English language support
  - Secondary: Tesseract OCR (when available) for verification
  - Advanced image preprocessing pipeline with 15+ techniques
  - Confidence scoring for detected text

- **User-Friendly Interface**
  - Clean Streamlit web interface with Indian-themed styling
  - High-contrast display of detected plates
  - Visualization of plate regions in images
  - Detection history tracking with download option
  - Advanced configuration options

## 📋 Requirements

- Python 3.8 or higher
- OpenCV
- NumPy
- EasyOCR
- PyTesseract (optional)
- YOLOv5/YOLOv8 (Ultralytics)
- Streamlit
- Pandas
- Pillow
- Other dependencies in requirements.txt

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/DineshDumka/ANPR-automatic-number-plate-recognition-python-yolov.git
cd ANPR-automatic-number-plate-recognition-python-yolov
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. (Optional) Install Tesseract OCR for improved recognition:
   - **Windows**: Download and install from [Tesseract GitHub](https://github.com/UB-Mannheim/tesseract/wiki)
   - **Linux**: `sudo apt install tesseract-ocr`
   - **Mac**: `brew install tesseract`

## 🏃‍♂️ Running the Application

### Online Demo
You can try the application online without any installation:
**[Live Demo](https://dineshdumka-anpr-automatic-number-plate-recognition--app-cilo5z.streamlit.app/)**

### Local Installation
1. Start the Streamlit app:
```bash
streamlit run app.py
```

2. Open your browser and navigate to `http://localhost:8501`

## 🧪 How It Works

### 1. License Plate Detection
The system uses a multi-stage approach to detect license plates:

1. **Primary Detection**: YOLOv5/YOLOv8 object detection with parameters optimized for Indian plates
2. **Secondary Detection**: Edge detection with contour analysis when YOLO fails
3. **Region Refinement**: Aspect ratio filtering and padding to ensure the entire plate is captured

### 2. Image Enhancement
Multiple image enhancement techniques are applied to improve OCR accuracy:

1. **Contrast Enhancement**: Improves visibility of text
2. **Sharpening**: Enhances edges and text boundaries
3. **Noise Reduction**: Bilateral filtering and median filtering
4. **Thresholding**: Adaptive and Otsu's thresholding for better text/background separation
5. **Morphological Operations**: Opening and closing to clean up the image

### 3. Text Recognition
A multi-engine OCR approach ensures maximum accuracy:

1. **EasyOCR**: Primary OCR engine with Hindi and English language support
2. **Tesseract OCR**: Secondary OCR engine (when available) for verification
3. **Confidence Scoring**: Results are ranked by confidence score
4. **Text Cleaning**: Removing noise and formatting the detected text

### 4. Indian License Plate Formatting
The system formats detected text according to Indian license plate standards:

1. **Pattern Recognition**: Identifies state code, district code, series, and number
2. **Format Validation**: Ensures the detected text matches Indian license plate patterns
3. **Standardization**: Formats the output as XX-00-XX-0000 (e.g., DL-01-AB-1234)

## 🔧 Advanced Usage

### Fine-tuning for Indian License Plates
You can fine-tune the YOLO model specifically for Indian license plates:

```bash
# Train the model with your own dataset
python indian_plate_model.py --train

# Test the model on a specific image
python indian_plate_model.py --test --img test_inputs/plate_1.jpg
```

### Configuration Options
The application provides several configuration options:

- **Detection Confidence**: Adjust the confidence threshold for license plate detection
- **Preprocessing Level**: Choose between low, medium, and high preprocessing intensity
- **Show Plate Regions**: Toggle visualization of detected plate regions
- **Use Tesseract OCR**: Enable/disable Tesseract OCR (when available)
- **Allow Hindi Characters**: Enable/disable Hindi character recognition

## 📝 Recent Improvements

- Added multi-engine OCR approach with EasyOCR and Tesseract
- Implemented advanced image enhancement pipeline with 15+ techniques
- Added confidence scoring for detected text
- Improved Indian license plate formatting with state code recognition
- Enhanced UI with confidence display and better error handling
- Added fine-tuning capability for Indian license plates
- Implemented robust error handling and fallback mechanisms

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👨‍💻 Author

- [Dinesh Dumka](https://github.com/DineshDumka)

## 🙏 Acknowledgements

- [Ultralytics](https://github.com/ultralytics/ultralytics) for YOLOv5/YOLOv8
- [EasyOCR](https://github.com/JaidedAI/EasyOCR) for OCR functionality
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) for additional OCR support
- [Streamlit](https://streamlit.io/) for the web interface 