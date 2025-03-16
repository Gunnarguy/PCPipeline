#!/usr/bin/env python3
import sys
import subprocess
import os
import importlib.util
import shutil
import site

def check_package_imported(package_name):
    """Check if a package can be imported."""
    try:
        importlib.util.find_spec(package_name)
        return True
    except ImportError:
        return False

def install_package(package, force=False, ignore_errors=False):
    """Install a Python package using pip."""
    print(f"Installing {package}...")
    cmd = [sys.executable, "-m", "pip", "install", "--upgrade"]
    if force:
        cmd.append("--force-reinstall")
    cmd.append(package)
    
    try:
        subprocess.check_call(cmd)
        print(f"{package} installed successfully")
        return True
    except subprocess.CalledProcessError:
        if ignore_errors:
            print(f"Warning: Failed to install {package}, but continuing...")
            return False
        raise

def fix_six_package():
    """Ensure six package is properly installed with six.moves available."""
    print("\nEnsuring six package is correctly installed...")
    
    # Check for existing six installations
    if check_package_imported("six"):
        try:
            # Remove all existing six packages
            subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", "six"])
            print("Removed existing six package")
        except subprocess.CalledProcessError:
            print("Failed to uninstall existing six package")
    
    # Install six with a specific version known to work
    install_package("six==1.16.0", force=True)
    
    # Verify that six.moves can be imported
    if not check_package_imported("six.moves"):
        print("WARNING: Failed to import six.moves after installation.")
        
        # Try an alternative installation method
        print("Trying alternative installation method...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--ignore-installed", "six"])
        except subprocess.CalledProcessError:
            print("Alternative installation also failed.")
    
    # Final verification
    if check_package_imported("six.moves"):
        print("✓ six.moves is now available")
        return True
    else:
        print("✗ six.moves is still not available")
        return False

def install_dependencies():
    """Install all required dependencies with proper versions."""
    print("Checking and installing required dependencies...")
    
    # Ensure the latest pip is installed
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
    except subprocess.CalledProcessError:
        print("Warning: Failed to upgrade pip, continuing with existing version...")
    
    # Fix the six package installation
    if not fix_six_package():
        print("\nWARNING: Could not properly install six.moves.")
        print("You may need to fix this manually:")
        print("1. pip uninstall -y six")
        print("2. pip install six==1.16.0")
    
    # Install specific version of pinecone-client
    install_package("pinecone-client==3.0.0", ignore_errors=True)  # Use a newer version
    
    # Install other dependencies
    required_packages = [
        "transformers", 
        "langchain", 
        "openai", 
        "tiktoken",
        "python-magic",
        "PyMuPDF",
        "pandas",
        "numpy",
        "matplotlib"
    ]
    
    # Handle textract separately since it has a problematic dependency
    try:
        install_package("textract==1.6.5", ignore_errors=True)
    except Exception as e:
        print(f"Warning: Error installing textract: {e}")
        print("You may need to install it manually.")
    
    for package in required_packages:
        install_package(package, ignore_errors=True)
    
    print("\nAll dependencies have been installed.")
    print("Please run the diagnostic script if you encounter issues:")
    print(f"python {os.path.join(os.path.dirname(__file__), 'diagnose_six.py')}")
    print("\nThen run your application.")

if __name__ == "__main__":
    print("Installing dependencies for PCPipeline...")
    install_dependencies()
