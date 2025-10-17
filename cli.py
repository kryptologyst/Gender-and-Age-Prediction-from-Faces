#!/usr/bin/env python3
"""
Command-line interface for gender and age prediction.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List

from src.models import ModernGenderAgePredictor, LegacyOpenCVPredictor, create_synthetic_dataset
from src.config import get_config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Gender and Age Prediction from Faces",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli.py predict image.jpg
  python cli.py predict --model legacy image.jpg
  python cli.py batch images/
  python cli.py generate-synthetic --count 50 --output data/synthetic
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Predict gender and age from an image')
    predict_parser.add_argument('image_path', help='Path to the input image')
    predict_parser.add_argument('--model', choices=['modern', 'legacy'], default='modern',
                              help='Model to use for prediction')
    predict_parser.add_argument('--output', help='Output file for results (JSON format)')
    predict_parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    # Batch command
    batch_parser = subparsers.add_parser('batch', help='Process multiple images')
    batch_parser.add_argument('input_dir', help='Directory containing input images')
    batch_parser.add_argument('--output', help='Output file for results (JSON format)')
    batch_parser.add_argument('--model', choices=['modern', 'legacy'], default='modern',
                             help='Model to use for prediction')
    
    # Generate synthetic data command
    synthetic_parser = subparsers.add_parser('generate-synthetic', help='Generate synthetic dataset')
    synthetic_parser.add_argument('--count', type=int, default=100, help='Number of synthetic images')
    synthetic_parser.add_argument('--output', default='data/synthetic', help='Output directory')
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Run tests')
    test_parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Set logging level
    if args.verbose if hasattr(args, 'verbose') else False:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        if args.command == 'predict':
            predict_single_image(args)
        elif args.command == 'batch':
            predict_batch_images(args)
        elif args.command == 'generate-synthetic':
            generate_synthetic_data(args)
        elif args.command == 'test':
            run_tests(args)
    except Exception as e:
        logger.error(f"Command failed: {e}")
        sys.exit(1)


def predict_single_image(args):
    """Predict gender and age for a single image."""
    image_path = Path(args.image_path)
    
    if not image_path.exists():
        logger.error(f"Image file not found: {image_path}")
        return
    
    logger.info(f"Processing image: {image_path}")
    
    # Initialize predictor
    if args.model == 'modern':
        predictor = ModernGenderAgePredictor()
    else:
        predictor = LegacyOpenCVPredictor()
    
    # Make prediction
    result = predictor.predict(image_path)
    
    # Display results
    print(f"\n🎯 Prediction Results:")
    print(f"   Gender: {result['gender']}")
    print(f"   Age: {result['age']}")
    
    if 'gender_confidence' in result:
        print(f"   Gender Confidence: {result['gender_confidence']:.2%}")
    if 'age_confidence' in result:
        print(f"   Age Confidence: {result['age_confidence']:.2%}")
    if 'overall_confidence' in result:
        print(f"   Overall Confidence: {result['overall_confidence']:.2%}")
    
    faces_detected = len(result.get('faces', []))
    print(f"   Faces Detected: {faces_detected}")
    
    # Save results if output file specified
    if args.output:
        import json
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Add metadata
        result['image_path'] = str(image_path)
        result['model_used'] = args.model
        
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)
        
        logger.info(f"Results saved to: {output_path}")


def predict_batch_images(args):
    """Predict gender and age for multiple images."""
    input_dir = Path(args.input_dir)
    
    if not input_dir.exists():
        logger.error(f"Input directory not found: {input_dir}")
        return
    
    # Find image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    image_files = [f for f in input_dir.iterdir() 
                   if f.suffix.lower() in image_extensions]
    
    if not image_files:
        logger.error(f"No image files found in: {input_dir}")
        return
    
    logger.info(f"Processing {len(image_files)} images...")
    
    # Initialize predictor
    if args.model == 'modern':
        predictor = ModernGenderAgePredictor()
    else:
        predictor = LegacyOpenCVPredictor()
    
    # Process images
    results = []
    for i, image_file in enumerate(image_files):
        logger.info(f"Processing {i+1}/{len(image_files)}: {image_file.name}")
        
        try:
            result = predictor.predict(image_file)
            result['image_path'] = str(image_file)
            result['model_used'] = args.model
            results.append(result)
            
            print(f"   {image_file.name}: {result['gender']}, {result['age']}")
            
        except Exception as e:
            logger.error(f"Failed to process {image_file.name}: {e}")
            results.append({
                'image_path': str(image_file),
                'error': str(e),
                'model_used': args.model
            })
    
    # Save results
    if args.output:
        import json
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Batch results saved to: {output_path}")
    
    # Summary
    successful = len([r for r in results if 'error' not in r])
    print(f"\n📊 Batch Processing Summary:")
    print(f"   Total images: {len(image_files)}")
    print(f"   Successful: {successful}")
    print(f"   Failed: {len(image_files) - successful}")


def generate_synthetic_data(args):
    """Generate synthetic dataset."""
    output_dir = Path(args.output)
    
    logger.info(f"Generating {args.count} synthetic images in {output_dir}")
    
    create_synthetic_dataset(output_dir, args.count)
    
    print(f"\n✅ Generated {args.count} synthetic images in {output_dir}")
    
    # Show sample
    image_files = list(output_dir.glob("*.jpg"))[:5]
    if image_files:
        print(f"   Sample files: {[f.name for f in image_files]}")


def run_tests(args):
    """Run the test suite."""
    logger.info("Running test suite...")
    
    try:
        import subprocess
        result = subprocess.run([
            sys.executable, '-m', 'pytest', 'tests/', '-v'
        ], capture_output=True, text=True)
        
        print(result.stdout)
        if result.stderr:
            print("Errors:", result.stderr)
        
        if result.returncode == 0:
            print("✅ All tests passed!")
        else:
            print("❌ Some tests failed!")
            sys.exit(1)
            
    except ImportError:
        logger.error("pytest not installed. Install with: pip install pytest")
        sys.exit(1)


if __name__ == "__main__":
    main()
