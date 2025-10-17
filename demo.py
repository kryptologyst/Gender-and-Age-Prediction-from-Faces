#!/usr/bin/env python3
"""
Demo script for the Gender and Age Prediction project.
"""

import sys
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.models import ModernGenderAgePredictor, create_synthetic_dataset
from src.visualization import PredictionVisualizer, create_prediction_report
from src.config import get_config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    """Run the demo."""
    print("🎭 Gender & Age Prediction Demo")
    print("=" * 40)
    
    # Initialize predictor
    print("🔄 Initializing prediction model...")
    try:
        predictor = ModernGenderAgePredictor()
        print("✅ Model initialized successfully!")
    except Exception as e:
        print(f"❌ Failed to initialize model: {e}")
        return
    
    # Create synthetic data
    print("\n🎨 Creating synthetic dataset...")
    synthetic_dir = "data/synthetic"
    create_synthetic_dataset(synthetic_dir, num_samples=5)
    print(f"✅ Created 5 synthetic images in {synthetic_dir}")
    
    # Test predictions
    print("\n🔍 Running predictions...")
    synthetic_path = Path(synthetic_dir)
    image_files = list(synthetic_path.glob("*.jpg"))
    
    results = []
    for i, image_file in enumerate(image_files[:3]):  # Test first 3 images
        print(f"   Processing image {i+1}: {image_file.name}")
        try:
            result = predictor.predict(str(image_file))
            result['image_path'] = str(image_file)
            results.append(result)
            
            print(f"      Gender: {result['gender']}")
            print(f"      Age: {result['age']}")
            print(f"      Confidence: {result.get('overall_confidence', 0):.2%}")
            
        except Exception as e:
            print(f"      ❌ Error: {e}")
    
    # Create visualizations
    if results:
        print("\n📊 Creating visualizations...")
        try:
            visualizer = PredictionVisualizer()
            
            # Create individual visualizations
            for i, result in enumerate(results):
                if 'image_path' in result:
                    import cv2
                    image = cv2.imread(result['image_path'])
                    if image is not None:
                        fig = visualizer.visualize_prediction(image, result)
                        save_path = f"demo_prediction_{i+1}.png"
                        fig.savefig(save_path, dpi=150, bbox_inches='tight')
                        print(f"   ✅ Saved visualization: {save_path}")
            
            # Create summary report
            report_path = create_prediction_report(results, "demo_output")
            print(f"   ✅ Created report: {report_path}")
            
        except Exception as e:
            print(f"   ⚠️  Visualization failed: {e}")
    
    # Summary
    print(f"\n🎉 Demo completed!")
    print(f"   Processed: {len(results)} images")
    print(f"   Successful: {len([r for r in results if 'error' not in r])}")
    print(f"   Failed: {len([r for r in results if 'error' in r])}")
    
    print("\n📋 Next steps:")
    print("   1. Run the web interface: streamlit run web_app/app.py")
    print("   2. Use CLI: python cli.py predict --help")
    print("   3. Check the generated visualizations and reports")


if __name__ == "__main__":
    main()
