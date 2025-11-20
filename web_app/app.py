from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path
from io import BytesIO
import base64
import tempfile
import shutil
import asyncio

from PIL import Image
from pdf2image import convert_from_bytes

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
POPPLER_PATH = "/opt/homebrew/bin"

app = FastAPI(title="Mini PDF/Image inference UI")

templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))


def load_model() -> object:
    """Load a YOLO model: prefer trained checkpoint in runs/detect, fallback to bundled weights."""
    if YOLO is None:
        raise RuntimeError("ultralytics package not installed. Install with `pip install ultralytics`")

    # Candidate paths
    candidate = PROJECT_ROOT / "runs" / "detect" / "train" / "weights" / "best.pt"
    if candidate.exists():
        model_path = str(candidate)
    else:
        fallback = PROJECT_ROOT / "yolov8n.pt"
        if fallback.exists():
            model_path = str(fallback)
        else:
            # Let ultralytics resolve model name (will download if needed)
            model_path = "yolov8n.pt"

    print(f"Loading YOLO model from: {model_path}")
    model = YOLO(model_path)
    return model


@app.on_event("startup")
async def startup_event():
    # load model once and attach to app state
    try:
        app.state.model = load_model()
    except Exception as exc:
        app.state.model = None
        print("Warning: failed to load model at startup:", exc)


def annotate_image_with_model(model, image_path: str, conf: float = 0.25):
    """Run model.predict on an image file and return PNG bytes of annotated image."""
    res = model.predict(source=str(image_path), conf=conf, device=None, verbose=False, save=False)[0]
    annotated = res.plot()  # ndarray HxWxC (BGR? notebook used [:,:,::-1] but res.plot returns RGB)
    pil = Image.fromarray(annotated)
    buf = BytesIO()
    pil.save(buf, format="PNG")
    buf.seek(0)
    return buf.read()


async def process_upload_file(model, upload: UploadFile) -> list[bytes]:
    """Return list of PNG bytes (one per page/image) with annotations."""
    content = await upload.read()
    lower = upload.filename.lower() if upload.filename else ""

    images_bytes: list[bytes] = []

    with tempfile.TemporaryDirectory() as td:
        tmpdir = Path(td)
        if lower.endswith('.pdf'):
            # rasterize PDF pages
            pages = convert_from_bytes(content, dpi=200, poppler_path=POPPLER_PATH)
            for idx, page in enumerate(pages, start=1):
                page_path = tmpdir / f"page_{idx:03d}.png"
                page.save(page_path)
                annotated = await asyncio.get_event_loop().run_in_executor(None, annotate_image_with_model, model, str(page_path))
                images_bytes.append(annotated)
        else:
            # treat as single image
            img = Image.open(BytesIO(content)).convert('RGB')
            img_path = tmpdir / "upload.png"
            img.save(img_path)
            annotated = await asyncio.get_event_loop().run_in_executor(None, annotate_image_with_model, model, str(img_path))
            images_bytes.append(annotated)

    return images_bytes


@app.get('/', response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse('index.html', {"request": request})


@app.post('/upload', response_class=HTMLResponse)
async def upload(request: Request, file: UploadFile = File(...)):
    model = getattr(app.state, 'model', None)
    if model is None:
        return HTMLResponse("<h3>Model not loaded on server. Check server logs.</h3>")

    try:
        images = await process_upload_file(model, file)
    except Exception as exc:
        return HTMLResponse(f"<h3>Processing failed: {exc}</h3>")

    # Encode images as base64 and render
    b64_images = [base64.b64encode(b).decode('ascii') for b in images]
    return templates.TemplateResponse('index.html', {"request": request, "images": b64_images, "filename": file.filename})
