# Gender and Age Prediction from Faces

Production-ready implementation for predicting gender and age from facial images using state-of-the-art deep learning models.

## Features

- **Modern Architecture**: Built with PyTorch and Hugging Face Transformers
- **Multiple Models**: Support for both modern PyTorch models and legacy OpenCV implementations
- **Web Interface**: Interactive Streamlit web application
- **Batch Processing**: Process multiple images simultaneously
- **Synthetic Data**: Generate synthetic datasets for testing and development
- **Configuration Management**: YAML-based configuration system
- **Comprehensive Testing**: Full test suite with unit and integration tests
- **Type Safety**: Full type hints and mypy support
- **Logging**: Structured logging throughout the application

## 📁 Project Structure

```
0206_Gender_and_age_prediction_from_faces/
├── src/                    # Source code
│   ├── __init__.py
│   ├── models.py          # Core prediction models
│   └── config.py          # Configuration management
├── web_app/               # Web interface
│   └── app.py            # Streamlit application
├── tests/                 # Test suite
│   └── test_models.py    # Unit and integration tests
├── config/               # Configuration files
│   └── config.yaml       # Default configuration
├── data/                 # Data directories
│   ├── input/           # Input images
│   ├── output/          # Output results
│   └── synthetic/       # Synthetic datasets
├── models/              # Model files
├── logs/                # Log files
├── requirements.txt     # Python dependencies
├── .gitignore          # Git ignore rules
└── README.md           # This file
```

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/kryptologyst/Gender-and-Age-Prediction-from-Faces.git
   cd Gender-and-Age-Prediction-from-Faces
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Create necessary directories**:
   ```bash
   mkdir -p data/input data/output data/synthetic models logs
   ```

## Quick Start

### Command Line Usage

```python
from src.models import ModernGenderAgePredictor

# Initialize predictor
predictor = ModernGenderAgePredictor()

# Predict gender and age from an image
result = predictor.predict("path/to/image.jpg")
print(f"Gender: {result['gender']}, Age: {result['age']}")
```

### Web Interface

1. **Start the Streamlit app**:
   ```bash
   streamlit run web_app/app.py
   ```

2. **Open your browser** and navigate to `http://localhost:8501`

3. **Upload an image** or use the synthetic data generator

### Batch Processing

```python
from src.models import ModernGenderAgePredictor

predictor = ModernGenderAgePredictor()
image_paths = ["image1.jpg", "image2.jpg", "image3.jpg"]
results = predictor.predict_batch(image_paths)

for i, result in enumerate(results):
    print(f"Image {i+1}: {result['gender']}, {result['age']}")
```

## 🔧 Configuration

The application uses YAML-based configuration. Edit `config/config.yaml` to customize:

```yaml
model:
  name: "modern"
  device: "auto"  # auto, cpu, cuda
  batch_size: 1
  confidence_threshold: 0.5

face_detection:
  scale_factor: 1.1
  min_neighbors: 5
  min_size: [30, 30]

data:
  input_dir: "data/input"
  output_dir: "data/output"
  max_image_size: [224, 224]

web_app:
  host: "localhost"
  port: 8501
  title: "Gender & Age Prediction"
```

## Testing

Run the comprehensive test suite:

```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=html

# Run specific test file
python -m pytest tests/test_models.py -v
```

### Test Categories

- **Unit Tests**: Individual component testing
- **Integration Tests**: End-to-end workflow testing
- **Synthetic Data Tests**: Testing with generated data
- **Configuration Tests**: Configuration management testing

## Synthetic Data Generation

Generate synthetic datasets for testing and development:

```python
from src.models import create_synthetic_dataset

# Create 100 synthetic face images
create_synthetic_dataset("data/synthetic", num_samples=100)
```

## Model Details

### Modern Model (Default)

- **Architecture**: Custom CNN with separate heads for gender and age prediction
- **Input**: 224x224 RGB images
- **Gender Classes**: Male, Female
- **Age Buckets**: 0-2, 4-6, 8-12, 15-20, 25-32, 38-43, 48-53, 60-100
- **Framework**: PyTorch

### Legacy Model

- **Architecture**: OpenCV DNN with Caffe models
- **Note**: Requires separate model files (age_net.caffemodel, gender_net.caffemodel)

## Performance

The modern model provides:
- **Accuracy**: High accuracy on face detection and classification
- **Speed**: Optimized for real-time processing
- **Confidence**: Provides confidence scores for predictions
- **Robustness**: Handles various lighting conditions and face orientations

## Error Handling

The application includes comprehensive error handling:

- **File Validation**: Checks for valid image files
- **Face Detection**: Handles cases with no detected faces
- **Model Loading**: Graceful fallback for missing models
- **Configuration**: Default values for missing configuration

## Logging

Structured logging throughout the application:

```python
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Logs are written to logs/app.log
```

## 🔧 Development

### Code Style

The project follows PEP 8 style guidelines:

```bash
# Format code
black src/ tests/

# Check style
flake8 src/ tests/

# Type checking
mypy src/
```

### Adding New Models

1. Create a new model class inheriting from the base predictor
2. Implement the `predict()` method
3. Add configuration options
4. Write tests
5. Update documentation

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Run the test suite
6. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenCV community for face detection algorithms
- PyTorch team for the deep learning framework
- Hugging Face for transformer models
- Streamlit for the web interface framework

## Support

For questions, issues, or contributions:

- Create an issue on GitHub
- Check the documentation
- Review the test cases for usage examples

## Future Enhancements

- [ ] Real-time video processing
- [ ] Mobile app integration
- [ ] Advanced model architectures (Vision Transformers)
- [ ] Multi-face detection and processing
- [ ] Age regression (exact age prediction)
- [ ] Emotion detection integration
- [ ] API server with FastAPI
- [ ] Docker containerization
- [ ] Model fine-tuning capabilities
- [ ] Explainability features (Grad-CAM, attention maps)

---

**Note**: This is a demonstration project. For production use, consider additional security measures, model validation, and performance optimization.
# Gender-and-Age-Prediction-from-Faces
