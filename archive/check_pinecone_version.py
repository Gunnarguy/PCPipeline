import sys
import subprocess
import importlib
import os

def check_pinecone_version():
    """Check the installed pinecone-client version and its API capabilities."""
    print("Checking pinecone-client installation...")
    
    # Try to get the version
    try:
        cmd = [sys.executable, "-m", "pip", "show", "pinecone-client"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            version_line = [line for line in result.stdout.split('\n') if line.startswith('Version:')]
            if version_line:
                version = version_line[0].split('Version:')[1].strip()
                print(f"Installed pinecone-client version: {version}")
            else:
                print("Could not determine pinecone-client version")
        else:
            print("Pinecone client might not be installed")
    except Exception as e:
        print(f"Error checking version: {e}")
    
    # Check API capabilities
    print("\nChecking pinecone API capabilities...")
    try:
        import pinecone
        
        print("Available attributes and methods in pinecone module:")
        for attr in dir(pinecone):
            if not attr.startswith('__'):
                print(f"  - {attr}")
        
        # Test different initialization styles
        print("\nTesting initialization styles:")
        if hasattr(pinecone, 'Pinecone'):
            print("✓ pinecone.Pinecone constructor is available (new API style)")
        else:
            print("✗ pinecone.Pinecone constructor is NOT available")
        
        if hasattr(pinecone, 'init'):
            print("✓ pinecone.init function is available (legacy API style)")
        else:
            print("✗ pinecone.init function is NOT available")
            
        # Try to initialize with available method
        print("\nTrying to initialize pinecone:")
        if 'PINECONE_API_KEY' not in os.environ:
            print("Warning: PINECONE_API_KEY environment variable is not set")
        
        api_key = os.environ.get('PINECONE_API_KEY', 'dummy_key_for_testing')
        
        if hasattr(pinecone, 'init'):
            # Legacy API
            try:
                print("Attempting legacy init (pinecone.init)...")
                pinecone.init(api_key=api_key, environment="us-east1-gcp")
                print("✓ Legacy init succeeded")
            except Exception as e:
                print(f"✗ Legacy init failed: {e}")
        
        if hasattr(pinecone, 'Pinecone'):
            # New API
            try:
                print("Attempting new API init (pinecone.Pinecone)...")
                pc = pinecone.Pinecone(api_key=api_key)
                print("✓ New API init succeeded")
            except Exception as e:
                print(f"✗ New API init failed: {e}")
        
        # Recommendations based on findings
        print("\nRECOMMENDATION:")
        if hasattr(pinecone, 'init') and not hasattr(pinecone, 'Pinecone'):
            print("You have the older pinecone-client (v1.x or v2.x)")
            print("To use the installed version, use pinecone.init() in your code")
            print("Or upgrade to a newer version: pip install --upgrade pinecone-client")
        elif hasattr(pinecone, 'Pinecone') and not hasattr(pinecone, 'init'):
            print("You have the newer pinecone-client (v3.x)")
            print("To use the installed version, use pinecone.Pinecone() in your code")
        elif hasattr(pinecone, 'Pinecone') and hasattr(pinecone, 'init'):
            print("Your pinecone-client version supports both API styles")
            print("For forward compatibility, prefer using pinecone.Pinecone()")
        else:
            print("Could not determine the appropriate initialization method")
            print("Try reinstalling: pip install --upgrade pinecone-client")
    
    except ImportError:
        print("Could not import pinecone module. Please install it with:")
        print("pip install pinecone-client")
    except Exception as e:
        print(f"Error during pinecone checks: {e}")

if __name__ == "__main__":
    print("Pinecone Client Diagnostic Tool")
    print("==============================\n")
    check_pinecone_version()
