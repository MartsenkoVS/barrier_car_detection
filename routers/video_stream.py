from __future__ import annotations

from pathlib import Path
import threading

from fastapi import (
    APIRouter, BackgroundTasks, Request,
    UploadFile, File, Form
)
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from src.config import MJPEG_BOUNDARY, TARGET_WIDTH
from services.streamer import mjpeg_generator, plates_live


# --- инициализации -----------------------------------------------------------
router    = APIRouter()
templates = Jinja2Templates(directory="templates")

VIDEO_DIR = Path("video")
VIDEO_DIR.mkdir(exist_ok=True)

# --- эндпоинты ---------------------------------------------------------------
@router.get("/", response_class=HTMLResponse)
async def index(request: Request, src: str | None = None):
    """
    Главная страница. Если ?src=... передали — шаблон покажет <img>,
    иначе — текстовую подсказку.
    """
    context = {
        "request": request,
        "src": src,
        "width": TARGET_WIDTH,
    }
    return templates.TemplateResponse("index.html", context)


@router.post("/", response_class=HTMLResponse)
async def start(
    vid_file: UploadFile | None = File(None),
    local: str = Form(""),
):
    """
    Принимает файл (multipart) либо строку локального пути,
    сохраняет/проверяет и возвращает JSON {redirect:"/…"}.
    """
    if vid_file and vid_file.filename:
        dest = VIDEO_DIR / vid_file.filename
        with open(dest, "wb") as f:
            while chunk := await vid_file.read(16 * 1024 * 1024):
                f.write(chunk)
        src_path = dest

    elif local:
        src_path = Path(local)
        if not src_path.exists():
            return HTMLResponse(f"<h3>Файл не найден: {src_path}</h3>", 400)
    else:
        return HTMLResponse("<h3>Нужно выбрать файл или путь!</h3>", 400)

    return JSONResponse({"redirect": f"/?src={src_path}"})


@router.get("/video_feed")
def video_feed(
    background_tasks: BackgroundTasks,
    src: str,
) -> StreamingResponse:
    """
    Отдаёт MJPEG-поток для выбранного src.
    """
    stop_event = threading.Event()
    background_tasks.add_task(stop_event.set)
    generator = mjpeg_generator(Path(src), stop_event)

    return StreamingResponse(
        generator,
        media_type=f"multipart/x-mixed-replace; boundary={MJPEG_BOUNDARY.lstrip('-')}",
    )


@router.get("/plates")
async def get_plates() -> JSONResponse:
    """
    Список распознанных номеров.
    """
    return JSONResponse(list(plates_live.keys()))