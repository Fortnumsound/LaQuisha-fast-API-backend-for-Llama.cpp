#!/usr/bin/env python3
"""
Setup script for LaQuisha FastAPI backend.

This script helps you get LaQuisha up and running quickly.
"""

import subprocess
import sys
import os

def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"📦 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        if e.stdout:
            print(f"stdout: {e.stdout}")
        if e.stderr:
            print(f"stderr: {e.stderr}")
        return False

def main():
    """Setup LaQuisha."""
    print("🌟 LaQuisha Setup Script 🌟\n")
    
    # Install base requirements
    if not run_command("pip install -r requirements.txt", "Installing base requirements"):
        print("❌ Failed to install base requirements. Please check your Python/pip installation.")
        sys.exit(1)
    
    # Test basic import
    if not run_command("python -c 'import laquisha_backend; print(\"LaQuisha backend imported successfully\")'", "Testing backend import"):
        print("❌ Failed to import LaQuisha backend.")
        sys.exit(1)
    
    # Create models directory
    models_dir = "./models"
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        print(f"✅ Created {models_dir} directory")
    else:
        print(f"✅ {models_dir} directory already exists")
    
    print("\n🎉 LaQuisha setup completed successfully!")
    print("\n📚 Next steps:")
    print("1. Download a GGUF model file to the 'models' directory")
    print("2. Optional: Install llama-cpp-python for full AI functionality:")
    print("   pip install llama-cpp-python")
    print("3. Start the server:")
    print("   python laquisha_backend.py")
    print("4. Test the API:")
    print("   python test_laquisha.py")
    print("\n💡 LaQuisha will work without a model file (using fallback responses)")
    print("   but for full AI responses, you'll need llama-cpp-python and a GGUF model.")

if __name__ == "__main__":
    main()