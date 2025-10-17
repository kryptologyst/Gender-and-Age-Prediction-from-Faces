"""
Core prediction models for gender and age detection.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import cv2
from transformers import AutoImageProcessor, AutoModelForImageClassification
import requests
from io import BytesIO

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FaceDetector:
    """Face detection using OpenCV's Haar Cascade classifier."""
    
    def __init__(self, cascade_path: Optional[str] = None):
        """
        Initialize face detector.
        
        Args:
            cascade_path: Path to Haar cascade XML file. If None, uses default.
        """
        if cascade_path is None:
            # Try to download the cascade file if not present
            cascade_path = self._get_cascade_path()
        
        try:
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            if self.face_cascade.empty():
                raise ValueError(f"Could not load cascade from {cascade_path}")
        except Exception as e:
            logger.error(f"Failed to initialize face detector: {e}")
            raise
    
    def _get_cascade_path(self) -> str:
        """Get path to Haar cascade file, downloading if necessary."""
        cascade_path = Path("models/haarcascade_frontalface_default.xml")
        cascade_path.parent.mkdir(parents=True, exist_ok=True)
        
        if not cascade_path.exists():
            logger.info("Downloading Haar cascade file...")
            url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
            try:
                response = requests.get(url)
                response.raise_for_status()
                cascade_path.write_text(response.text)
                logger.info(f"Downloaded cascade to {cascade_path}")
            except Exception as e:
                logger.error(f"Failed to download cascade: {e}")
                raise
        
        return str(cascade_path)
    
    def detect_faces(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in an image.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            List of face bounding boxes as (x, y, w, h) tuples
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(30, 30)
        )
        return faces.tolist()


class ModernGenderAgePredictor:
    """
    Modern gender and age prediction using Hugging Face transformers.
    """
    
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium"):
        """
        Initialize the predictor with a pre-trained model.
        
        Args:
            model_name: Hugging Face model name for image classification
        """
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Initialize face detector
        self.face_detector = FaceDetector()
        
        # For this demo, we'll use a simpler approach with a custom model
        # In practice, you'd use a specialized face analysis model
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the prediction model."""
        try:
            # For demonstration, we'll create a simple CNN model
            # In practice, you'd load a pre-trained model like:
            # - microsoft/DialoGPT-medium for general image classification
            # - facebook/deit-base-distilled-patch16-224 for vision tasks
            # - Or a specialized face analysis model
            
            self.model = self._create_simple_model()
            self.model.to(self.device)
            self.model.eval()
            
            # Define transforms for image preprocessing
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
            
            logger.info("Model initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            raise
    
    def _create_simple_model(self) -> nn.Module:
        """Create a simple CNN model for demonstration."""
        class SimpleGenderAgeModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = nn.Sequential(
                    nn.Conv2d(3, 32, 3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(32, 64, 3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(64, 128, 3, padding=1),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((7, 7))
                )
                
                self.gender_head = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(128 * 7 * 7, 256),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Linear(256, 2)  # Male, Female
                )
                
                self.age_head = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(128 * 7 * 7, 256),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Linear(256, 8)  # Age buckets
                )
            
            def forward(self, x):
                features = self.backbone(x)
                gender_logits = self.gender_head(features)
                age_logits = self.age_head(features)
                return gender_logits, age_logits
        
        return SimpleGenderAgeModel()
    
    def predict(self, image_path: Union[str, Path]) -> Dict[str, Union[str, float, List[Tuple[int, int, int, int]]]]:
        """
        Predict gender and age from an image.
        
        Args:
            image_path: Path to the input image
            
        Returns:
            Dictionary containing predictions and face locations
        """
        try:
            # Load and preprocess image
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Could not load image from {image_path}")
            
            # Detect faces
            faces = self.face_detector.detect_faces(image)
            
            if not faces:
                logger.warning("No faces detected in the image")
                return {
                    "gender": "Unknown",
                    "age": "Unknown",
                    "confidence": 0.0,
                    "faces": []
                }
            
            # Process the first detected face
            x, y, w, h = faces[0]
            face_roi = image[y:y+h, x:x+w]
            
            # Convert to PIL Image and apply transforms
            face_pil = Image.fromarray(cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB))
            face_tensor = self.transform(face_pil).unsqueeze(0).to(self.device)
            
            # Make predictions
            with torch.no_grad():
                gender_logits, age_logits = self.model(face_tensor)
                
                # Get predictions
                gender_probs = torch.softmax(gender_logits, dim=1)
                age_probs = torch.softmax(age_logits, dim=1)
                
                gender_pred = gender_probs.argmax(dim=1).item()
                age_pred = age_probs.argmax(dim=1).item()
                
                gender_confidence = gender_probs[0, gender_pred].item()
                age_confidence = age_probs[0, age_pred].item()
            
            # Map predictions to labels
            gender_labels = ["Male", "Female"]
            age_labels = ["0-2", "4-6", "8-12", "15-20", "25-32", "38-43", "48-53", "60-100"]
            
            result = {
                "gender": gender_labels[gender_pred],
                "age": age_labels[age_pred],
                "gender_confidence": gender_confidence,
                "age_confidence": age_confidence,
                "overall_confidence": (gender_confidence + age_confidence) / 2,
                "faces": faces
            }
            
            logger.info(f"Prediction completed: {result['gender']}, {result['age']}")
            return result
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise
    
    def predict_batch(self, image_paths: List[Union[str, Path]]) -> List[Dict]:
        """
        Predict gender and age for multiple images.
        
        Args:
            image_paths: List of image paths
            
        Returns:
            List of prediction dictionaries
        """
        results = []
        for path in image_paths:
            try:
                result = self.predict(path)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process {path}: {e}")
                results.append({
                    "gender": "Error",
                    "age": "Error",
                    "confidence": 0.0,
                    "faces": [],
                    "error": str(e)
                })
        return results


class LegacyOpenCVPredictor:
    """
    Legacy OpenCV DNN implementation for comparison.
    """
    
    def __init__(self):
        """Initialize the legacy predictor."""
        self.age_buckets = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', 
                           '(25-32)', '(38-43)', '(48-53)', '(60-100)']
        self.gender_list = ['Male', 'Female']
        
        # Note: In practice, you'd need to download these model files
        # For this demo, we'll create a mock implementation
        logger.warning("Legacy predictor requires model files that are not included")
    
    def predict(self, image_path: Union[str, Path]) -> Dict[str, Union[str, float]]:
        """
        Legacy prediction method (requires model files).
        
        Args:
            image_path: Path to the input image
            
        Returns:
            Dictionary containing predictions
        """
        # This is a mock implementation since we don't have the model files
        logger.warning("Using mock predictions - model files not available")
        
        return {
            "gender": "Male",  # Mock prediction
            "age": "(25-32)",  # Mock prediction
            "confidence": 0.75,  # Mock confidence
            "method": "legacy_opencv"
        }


def create_synthetic_dataset(output_dir: Union[str, Path], num_samples: int = 100) -> None:
    """
    Create a synthetic dataset for testing.
    
    Args:
        output_dir: Directory to save synthetic images
        num_samples: Number of synthetic images to generate
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Creating {num_samples} synthetic images in {output_dir}")
    
    # Create synthetic face-like images
    for i in range(num_samples):
        # Generate random face-like pattern
        img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
        # Add some structure to make it more face-like
        cv2.circle(img, (112, 80), 30, (255, 255, 255), -1)  # Face outline
        cv2.circle(img, (100, 70), 5, (0, 0, 0), -1)  # Left eye
        cv2.circle(img, (124, 70), 5, (0, 0, 0), -1)  # Right eye
        cv2.ellipse(img, (112, 90), (10, 5), 0, 0, 180, (0, 0, 0), 2)  # Mouth
        
        # Save image
        img_path = output_dir / f"synthetic_face_{i:03d}.jpg"
        cv2.imwrite(str(img_path), img)
    
    logger.info(f"Synthetic dataset created successfully")


if __name__ == "__main__":
    # Example usage
    predictor = ModernGenderAgePredictor()
    
    # Create synthetic data for testing
    create_synthetic_dataset("data/synthetic", 10)
    
    # Test prediction on synthetic data
    test_image = "data/synthetic/synthetic_face_000.jpg"
    if Path(test_image).exists():
        result = predictor.predict(test_image)
        print(f"Prediction result: {result}")
    else:
        print("No test image available")
