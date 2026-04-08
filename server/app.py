import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from inference import app
import uvicorn

def main():
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
