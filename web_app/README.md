# Mini FastAPI inference UI

This small app lets you upload a PDF or an image and returns annotated page images using a YOLO model.

Files:
- `app.py` — FastAPI application
- `templates/index.html` — simple upload UI
- `requirements.txt` — Python dependencies

Usage:

1. Create and activate a virtual environment, then install requirements:

```bash
python -m venv venv
source venv/bin/activate
pip install -r web_app/requirements.txt
```

2. Ensure Poppler is installed (macOS homebrew):

```bash
brew install poppler
```

3. Run the server:

```bash
uvicorn web_app.app:app --reload --host 0.0.0.0 --port 7860
```

4. Open `http://localhost:7860` in your browser, upload a PDF or image, and view annotated pages.

Notes:
- The app will try to load `runs/detect/train/weights/best.pt` from the project root. If not found, it will use `yolov8n.pt` from the project root or let `ultralytics` resolve the model name (which may trigger a download).
- For large PDFs, rasterization may take time; adjust DPI in `app.py` if needed.
