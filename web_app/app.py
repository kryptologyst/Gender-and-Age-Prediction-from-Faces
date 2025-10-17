"""
Streamlit web interface for gender and age prediction.
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import logging
from pathlib import Path
from typing import Dict, Any
import tempfile
import os

# Import our modules
from src.models import ModernGenderAgePredictor, LegacyOpenCVPredictor, create_synthetic_dataset
from src.config import get_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Gender & Age Prediction",
    page_icon="👤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'predictor' not in st.session_state:
    st.session_state.predictor = None
if 'last_prediction' not in st.session_state:
    st.session_state.last_prediction = None


def initialize_predictor() -> ModernGenderAgePredictor:
    """Initialize the prediction model."""
    if st.session_state.predictor is None:
        with st.spinner("Loading prediction model..."):
            try:
                st.session_state.predictor = ModernGenderAgePredictor()
                st.success("Model loaded successfully!")
            except Exception as e:
                st.error(f"Failed to load model: {e}")
                return None
    return st.session_state.predictor


def display_prediction_result(result: Dict[str, Any], image: np.ndarray):
    """Display prediction results with visualization."""
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📸 Image with Predictions")
        
        # Draw bounding boxes and labels on the image
        display_image = image.copy()
        
        if result.get('faces'):
            for i, (x, y, w, h) in enumerate(result['faces']):
                # Draw bounding box
                cv2.rectangle(display_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Add label
                label = f"{result['gender']}, {result['age']}"
                cv2.putText(display_image, label, (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Convert BGR to RGB for display
        display_image_rgb = cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB)
        st.image(display_image_rgb, caption="Image with Face Detection", use_column_width=True)
    
    with col2:
        st.subheader("🎯 Prediction Results")
        
        # Display prediction details
        st.metric("Gender", result.get('gender', 'Unknown'))
        st.metric("Age Range", result.get('age', 'Unknown'))
        
        if 'gender_confidence' in result:
            st.metric("Gender Confidence", f"{result['gender_confidence']:.2%}")
        if 'age_confidence' in result:
            st.metric("Age Confidence", f"{result['age_confidence']:.2%}")
        if 'overall_confidence' in result:
            st.metric("Overall Confidence", f"{result['overall_confidence']:.2%}")
        
        # Display face count
        face_count = len(result.get('faces', []))
        st.metric("Faces Detected", face_count)
        
        # Confidence indicator
        confidence = result.get('overall_confidence', 0)
        if confidence > 0.8:
            st.success("High Confidence")
        elif confidence > 0.6:
            st.warning("Medium Confidence")
        else:
            st.error("Low Confidence")


def create_synthetic_data_interface():
    """Interface for creating synthetic data."""
    st.subheader("🔬 Synthetic Data Generation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        num_samples = st.number_input("Number of samples", min_value=1, max_value=1000, value=10)
        output_dir = st.text_input("Output directory", value="data/synthetic")
    
    with col2:
        if st.button("Generate Synthetic Data", type="primary"):
            with st.spinner("Generating synthetic faces..."):
                try:
                    create_synthetic_dataset(output_dir, num_samples)
                    st.success(f"Generated {num_samples} synthetic images in {output_dir}")
                    
                    # Show sample images
                    synthetic_path = Path(output_dir)
                    if synthetic_path.exists():
                        image_files = list(synthetic_path.glob("*.jpg"))[:5]
                        if image_files:
                            st.subheader("Sample Generated Images")
                            cols = st.columns(min(len(image_files), 5))
                            for i, img_path in enumerate(image_files):
                                with cols[i]:
                                    img = Image.open(img_path)
                                    st.image(img, caption=f"Sample {i+1}", use_column_width=True)
                except Exception as e:
                    st.error(f"Failed to generate synthetic data: {e}")


def main():
    """Main application interface."""
    st.title("👤 Gender & Age Prediction from Faces")
    st.markdown("---")
    
    # Sidebar configuration
    st.sidebar.title("⚙️ Configuration")
    
    # Model selection
    model_type = st.sidebar.selectbox(
        "Select Model Type",
        ["Modern (PyTorch)", "Legacy (OpenCV)"],
        help="Choose between modern PyTorch-based model or legacy OpenCV implementation"
    )
    
    # Confidence threshold
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.1,
        help="Minimum confidence required for predictions"
    )
    
    # Main content area
    tab1, tab2, tab3, tab4 = st.tabs(["📷 Upload Image", "📹 Camera", "🔬 Synthetic Data", "📊 Batch Processing"])
    
    with tab1:
        st.subheader("Upload an Image")
        
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg'],
            help="Upload a face image for gender and age prediction"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Initialize predictor
            predictor = initialize_predictor()
            
            if predictor and st.button("🔍 Predict Gender & Age", type="primary"):
                with st.spinner("Analyzing image..."):
                    try:
                        # Save uploaded file temporarily
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                            image.save(tmp_file.name)
                            
                            # Make prediction
                            result = predictor.predict(tmp_file.name)
                            
                            # Clean up temp file
                            os.unlink(tmp_file.name)
                            
                            # Store result in session state
                            st.session_state.last_prediction = result
                            
                            # Display results
                            display_prediction_result(result, cv2.imread(tmp_file.name))
                            
                    except Exception as e:
                        st.error(f"Prediction failed: {e}")
                        logger.error(f"Prediction error: {e}")
    
    with tab2:
        st.subheader("Camera Input")
        st.info("Camera functionality requires additional setup. Please use the upload option for now.")
        
        # Placeholder for camera functionality
        if st.button("📷 Open Camera (Coming Soon)"):
            st.warning("Camera functionality is not yet implemented.")
    
    with tab3:
        create_synthetic_data_interface()
    
    with tab4:
        st.subheader("Batch Processing")
        
        uploaded_files = st.file_uploader(
            "Choose multiple image files",
            type=['png', 'jpg', 'jpeg'],
            accept_multiple_files=True,
            help="Upload multiple images for batch processing"
        )
        
        if uploaded_files:
            st.write(f"Uploaded {len(uploaded_files)} files")
            
            predictor = initialize_predictor()
            
            if predictor and st.button("🔄 Process All Images", type="primary"):
                progress_bar = st.progress(0)
                results = []
                
                for i, uploaded_file in enumerate(uploaded_files):
                    try:
                        # Save file temporarily
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                            image = Image.open(uploaded_file)
                            image.save(tmp_file.name)
                            
                            # Make prediction
                            result = predictor.predict(tmp_file.name)
                            result['filename'] = uploaded_file.name
                            results.append(result)
                            
                            # Clean up
                            os.unlink(tmp_file.name)
                            
                    except Exception as e:
                        logger.error(f"Failed to process {uploaded_file.name}: {e}")
                        results.append({
                            'filename': uploaded_file.name,
                            'error': str(e)
                        })
                    
                    # Update progress
                    progress_bar.progress((i + 1) / len(uploaded_files))
                
                # Display results
                st.subheader("📊 Batch Results")
                
                for result in results:
                    if 'error' in result:
                        st.error(f"❌ {result['filename']}: {result['error']}")
                    else:
                        st.success(f"✅ {result['filename']}: {result['gender']}, {result['age']} "
                                 f"(Confidence: {result.get('overall_confidence', 0):.2%})")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: gray;'>
            <p>Gender & Age Prediction System | Built with Streamlit & PyTorch</p>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
