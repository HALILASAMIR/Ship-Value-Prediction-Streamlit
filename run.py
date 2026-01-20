"""
Streamlit App Launcher
======================

Simple launcher with dependency checking
"""

import subprocess
import sys
from pathlib import Path

def main():
    print("\n" + "="*60)
    print("   Ship Value Prediction - Streamlit")
    print("="*60 + "\n")
    
    app_file = Path("app.py")
    if not app_file.exists():
        print("ERROR: app.py not found!")
        sys.exit(1)
    
    print("Launching Streamlit application...\n")
    print("Opening http://localhost:8501\n")
    
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])
    except KeyboardInterrupt:
        print("\n\nApplication closed.")
        sys.exit(0)
    except Exception as e:
        print(f"\nERROR: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
