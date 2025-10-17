#!/usr/bin/env python3
"""
Setup script for the Gender and Age Prediction project.
"""

import os
import sys
import subprocess
from pathlib import Path


def run_command(command, description):
    """Run a command and handle errors."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        if e.stdout:
            print(f"   stdout: {e.stdout}")
        if e.stderr:
            print(f"   stderr: {e.stderr}")
        return False


def main():
    """Main setup function."""
    print("🚀 Setting up Gender & Age Prediction Project")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        sys.exit(1)
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    
    # Create virtual environment
    if not Path("venv").exists():
        if not run_command("python -m venv venv", "Creating virtual environment"):
            sys.exit(1)
    
    # Determine activation command
    if os.name == 'nt':  # Windows
        activate_cmd = "venv\\Scripts\\activate"
        pip_cmd = "venv\\Scripts\\pip"
        python_cmd = "venv\\Scripts\\python"
    else:  # Unix/Linux/macOS
        activate_cmd = "source venv/bin/activate"
        pip_cmd = "venv/bin/pip"
        python_cmd = "venv/bin/python"
    
    # Install dependencies
    if not run_command(f"{pip_cmd} install -r requirements.txt", "Installing dependencies"):
        print("⚠️  Some dependencies may have failed to install")
    
    # Create necessary directories
    directories = [
        "data/input", "data/output", "data/synthetic",
        "models", "logs", "config"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {directory}")
    
    # Create placeholder files
    placeholder_files = [
        "data/input/.gitkeep",
        "data/output/.gitkeep", 
        "data/synthetic/.gitkeep",
        "models/.gitkeep"
    ]
    
    for file_path in placeholder_files:
        Path(file_path).touch()
        print(f"✅ Created placeholder: {file_path}")
    
    # Run basic tests
    print("\n🧪 Running basic tests...")
    if run_command(f"{python_cmd} -m pytest tests/ -v", "Running test suite"):
        print("✅ All tests passed!")
    else:
        print("⚠️  Some tests may have failed - check the output above")
    
    # Generate sample synthetic data
    print("\n🎨 Generating sample synthetic data...")
    if run_command(f"{python_cmd} -c \"from src.models import create_synthetic_dataset; create_synthetic_dataset('data/synthetic', 5)\"", "Creating sample data"):
        print("✅ Sample synthetic data created!")
    
    print("\n🎉 Setup completed successfully!")
    print("\n📋 Next steps:")
    print("   1. Activate virtual environment:")
    if os.name == 'nt':
        print("      venv\\Scripts\\activate")
    else:
        print("      source venv/bin/activate")
    print("   2. Run the web interface:")
    print("      streamlit run web_app/app.py")
    print("   3. Or use the command line:")
    print("      python cli.py predict --help")
    print("\n📖 See README.md for detailed usage instructions")


if __name__ == "__main__":
    main()
