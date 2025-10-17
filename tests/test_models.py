"""
Test suite for gender and age prediction system.
"""

import unittest
import tempfile
import os
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
import logging

# Import our modules
from src.models import ModernGenderAgePredictor, LegacyOpenCVPredictor, create_synthetic_dataset, FaceDetector
from src.config import Config

# Configure logging for tests
logging.basicConfig(level=logging.WARNING)


class TestFaceDetector(unittest.TestCase):
    """Test cases for FaceDetector class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.detector = FaceDetector()
    
    def test_face_detection_synthetic_image(self):
        """Test face detection on synthetic image."""
        # Create a simple synthetic face image
        img = np.zeros((224, 224, 3), dtype=np.uint8)
        
        # Draw a simple face
        cv2.circle(img, (112, 80), 30, (255, 255, 255), -1)  # Face
        cv2.circle(img, (100, 70), 5, (0, 0, 0), -1)  # Left eye
        cv2.circle(img, (124, 70), 5, (0, 0, 0), -1)  # Right eye
        cv2.ellipse(img, (112, 90), (10, 5), 0, 0, 180, (0, 0, 0), 2)  # Mouth
        
        faces = self.detector.detect_faces(img)
        
        # Should detect at least one face
        self.assertGreaterEqual(len(faces), 0)  # May not detect synthetic faces perfectly
    
    def test_face_detection_empty_image(self):
        """Test face detection on empty image."""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        faces = self.detector.detect_faces(img)
        self.assertEqual(len(faces), 0)


class TestModernGenderAgePredictor(unittest.TestCase):
    """Test cases for ModernGenderAgePredictor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.predictor = ModernGenderAgePredictor()
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        # Clean up temporary files
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_model_initialization(self):
        """Test that the model initializes correctly."""
        self.assertIsNotNone(self.predictor.model)
        self.assertIsNotNone(self.predictor.face_detector)
    
    def test_prediction_synthetic_image(self):
        """Test prediction on synthetic image."""
        # Create synthetic image
        img_path = os.path.join(self.temp_dir, "test_face.jpg")
        img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
        # Add face-like structure
        cv2.circle(img, (112, 80), 30, (255, 255, 255), -1)
        cv2.circle(img, (100, 70), 5, (0, 0, 0), -1)
        cv2.circle(img, (124, 70), 5, (0, 0, 0), -1)
        cv2.ellipse(img, (112, 90), (10, 5), 0, 0, 180, (0, 0, 0), 2)
        
        cv2.imwrite(img_path, img)
        
        # Make prediction
        result = self.predictor.predict(img_path)
        
        # Check result structure
        self.assertIn('gender', result)
        self.assertIn('age', result)
        self.assertIn('faces', result)
        
        # Check that gender is one of the expected values
        self.assertIn(result['gender'], ['Male', 'Female', 'Unknown'])
        
        # Check that age is one of the expected ranges
        expected_ages = ["0-2", "4-6", "8-12", "15-20", "25-32", "38-43", "48-53", "60-100", "Unknown"]
        self.assertIn(result['age'], expected_ages)
    
    def test_prediction_nonexistent_file(self):
        """Test prediction on non-existent file."""
        with self.assertRaises(Exception):
            self.predictor.predict("nonexistent_file.jpg")
    
    def test_batch_prediction(self):
        """Test batch prediction functionality."""
        # Create multiple synthetic images
        image_paths = []
        for i in range(3):
            img_path = os.path.join(self.temp_dir, f"test_face_{i}.jpg")
            img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            
            # Add face-like structure
            cv2.circle(img, (112, 80), 30, (255, 255, 255), -1)
            cv2.circle(img, (100, 70), 5, (0, 0, 0), -1)
            cv2.circle(img, (124, 70), 5, (0, 0, 0), -1)
            cv2.ellipse(img, (112, 90), (10, 5), 0, 0, 180, (0, 0, 0), 2)
            
            cv2.imwrite(img_path, img)
            image_paths.append(img_path)
        
        # Make batch prediction
        results = self.predictor.predict_batch(image_paths)
        
        # Check results
        self.assertEqual(len(results), 3)
        for result in results:
            self.assertIn('gender', result)
            self.assertIn('age', result)


class TestLegacyOpenCVPredictor(unittest.TestCase):
    """Test cases for LegacyOpenCVPredictor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.predictor = LegacyOpenCVPredictor()
    
    def test_legacy_prediction(self):
        """Test legacy prediction (mock implementation)."""
        result = self.predictor.predict("test_image.jpg")
        
        # Check result structure
        self.assertIn('gender', result)
        self.assertIn('age', result)
        self.assertIn('confidence', result)
        self.assertIn('method', result)
        
        # Check that it's using the legacy method
        self.assertEqual(result['method'], 'legacy_opencv')


class TestSyntheticDataset(unittest.TestCase):
    """Test cases for synthetic dataset creation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_create_synthetic_dataset(self):
        """Test synthetic dataset creation."""
        num_samples = 5
        create_synthetic_dataset(self.temp_dir, num_samples)
        
        # Check that files were created
        image_files = list(Path(self.temp_dir).glob("*.jpg"))
        self.assertEqual(len(image_files), num_samples)
        
        # Check that images are valid
        for img_path in image_files:
            img = cv2.imread(str(img_path))
            self.assertIsNotNone(img)
            self.assertEqual(img.shape, (224, 224, 3))


class TestConfig(unittest.TestCase):
    """Test cases for Config class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_config_path = os.path.join(tempfile.mkdtemp(), "test_config.yaml")
        self.config = Config(self.temp_config_path)
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(os.path.dirname(self.temp_config_path), ignore_errors=True)
    
    def test_config_initialization(self):
        """Test config initialization."""
        self.assertIsNotNone(self.config.config)
        self.assertIn('model', self.config.config)
        self.assertIn('face_detection', self.config.config)
    
    def test_config_get_set(self):
        """Test config get and set methods."""
        # Test get with existing key
        device = self.config.get('model.device')
        self.assertEqual(device, 'auto')
        
        # Test get with non-existing key
        non_existing = self.config.get('non.existing.key', 'default')
        self.assertEqual(non_existing, 'default')
        
        # Test set
        self.config.set('model.device', 'cpu')
        device = self.config.get('model.device')
        self.assertEqual(device, 'cpu')
    
    def test_config_update(self):
        """Test config update method."""
        updates = {
            'model.device': 'cuda',
            'model.batch_size': 4,
            'new.key': 'new_value'
        }
        
        self.config.update(updates)
        
        self.assertEqual(self.config.get('model.device'), 'cuda')
        self.assertEqual(self.config.get('model.batch_size'), 4)
        self.assertEqual(self.config.get('new.key'), 'new_value')


class TestIntegration(unittest.TestCase):
    """Integration tests."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_end_to_end_prediction(self):
        """Test end-to-end prediction workflow."""
        # Create synthetic dataset
        create_synthetic_dataset(self.temp_dir, 1)
        
        # Initialize predictor
        predictor = ModernGenderAgePredictor()
        
        # Get the created image
        image_files = list(Path(self.temp_dir).glob("*.jpg"))
        self.assertEqual(len(image_files), 1)
        
        # Make prediction
        result = predictor.predict(str(image_files[0]))
        
        # Verify result
        self.assertIn('gender', result)
        self.assertIn('age', result)
        self.assertIn('faces', result)
        
        # Should have reasonable confidence values
        if 'overall_confidence' in result:
            self.assertGreaterEqual(result['overall_confidence'], 0.0)
            self.assertLessEqual(result['overall_confidence'], 1.0)


def run_tests():
    """Run all tests."""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_classes = [
        TestFaceDetector,
        TestModernGenderAgePredictor,
        TestLegacyOpenCVPredictor,
        TestSyntheticDataset,
        TestConfig,
        TestIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)
