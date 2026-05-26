"""
IACertext - Microservicio FastAPI Multimodal v11.0
===================================================
NUEVO en v11:
  - Detección automática de PDF escaneado (imagen) desde página 1
  - OCR con GPT-4o Vision en TODA la página cuando es imagen convertida a PDF
  - Zoom 3x para mejor calidad OCR en PDFs escaneados
  - Chunking por layout: respeta columnas y secciones visuales
  - image_path nunca bloquea el guardado de chunks (fix crítico v7)
  - os.getenv() con defaults — no crashea si falta variable de entorno
  - Soporte completo: texto vectorial + tablas + imágenes + OCR página completa
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import math
import os
import re
import time
import urllib.request
from typing import Any

import fitz
import httpx
import pdfplumber
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image as PILImage, ImageEnhance, ImageFilter

# ──────────────────────────────────────────────
# Configuración — os.getenv con defaults (no crash si falta variable)
# ──────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("iacertext")

SUPABASE_URL         = os.getenv("SUPABASE_URL", "")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY", "")
SUPABASE_BUCKET      = os.getenv("SUPABASE_BUCKET", "rag-images")
OPENAI_API_KEY       = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL         = os.getenv("OPENAI_MODEL", "gpt-4o")

MIN_TEXT_LENGTH  = 40    # chars mínimos para considerar página con texto real
MIN_IMAGE_BYTES  = 1_500
MAX_CHUNK_CHARS  = 1_200
OCR_ZOOM_NORMAL  = 2.5   # zoom para páginas normales
OCR_ZOOM_SCANNED = 3.0   # zoom extra para PDFs 100% imagen (mejor calidad)

app = FastAPI(title="IACertext PDF Processor", version="11.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ══════════════════════════════════════════════
# SECCIÓN 1 — DETECCIÓN DE TIPO DE PDF
# ══════════════════════════════════════════════

def detect_pdf_type(doc: fitz.Document) -> dict:
    """
    Analiza las primeras páginas para determinar si el PDF es:
      - 'vectorial': tiene texto seleccionable
      - 'scanned': es una imagen convertida a PDF (sin texto vectorial)
      - 'mixed': algunas páginas con texto, otras escaneadas
    """
    sample_pages = min(3, len(doc))
    text_pages = 0
    image_pages = 0

    for i in range(sample_pages):
        page = doc[i]
        text = page.get_text("text").strip()
        images = page.get_images(full=True)

        if len(text) >= MIN_TEXT_LENGTH:
            text_pages += 1
        elif images or len(text) < 10:
            image_pages += 1

    if image_pages == sample_pages:
        pdf_type = "scanned"
    elif text_pages == sample_pages:
        pdf_type = "vectorial"
    else:
        pdf_type = "mixed"

    log.info("PDF tipo=%s (texto=%d img=%d de %d páginas muestra)", pdf_type, text_pages, image_pages, sample_pages)
    return {
        "type": pdf_type,
        "is_scanned": pdf_type in ("scanned", "mixed"),
        "zoom": OCR_ZOOM_SCANNED if pdf_type == "scanned" else OCR_ZOOM_NORMAL,
    }


# ══════════════════════════════════════════════
# SECCIÓN 2 — PREPROCESAMIENTO DE IMAGEN
# ══════════════════════════════════════════════

def enhance_for_ocr(img_bytes: bytes, aggressive: bool = False) -> bytes:
    """
    Preprocesa imagen para maximizar calidad OCR.
    aggressive=True para PDFs escaneados de baja calidad.
    """
    try:
        img = PILImage.open(io.BytesIO(img_bytes)).convert("RGB")
        w, h = img.size

        # Escalar si es muy pequeña
        min_side = 1_400 if aggressive else 1_200
        if max(w, h) < min_side:
            scale = min_side / max(w, h)
            img = img.resize((int(w * scale), int(h * scale)), PILImage.LANCZOS)

        # Contraste y nitidez — más agresivo para escaneados
        contrast = 2.0 if aggressive else 1.8
        sharpness = 2.8 if aggressive else 2.2

        img = ImageEnhance.Contrast(img).enhance(contrast)
        img = ImageEnhance.Sharpness(img).enhance(sharpness)
        img = img.filter(ImageFilter.MedianFilter(size=3))

        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=True)
        return buf.getvalue()
    except Exception as exc:
        log.warning("enhance_for_ocr error: %s", exc)
        return img_bytes


def img_to_b64(img_bytes: bytes) -> str | None:
    try:
        img = PILImage.open(io.BytesIO(img_bytes))
        if img.mode not in ("RGB", "RGBA"):
            img = img.convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=True)
        return base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception:
        return None


# ══════════════════════════════════════════════
# SECCIÓN 3 — OCR con GPT-4o Vision
# ══════════════════════════════════════════════

def ocr_with_gpt4o(img_bytes: bytes, context: str = "documento", is_scanned: bool = False) -> str:
    """
    OCR con GPT-4o Vision.
    is_scanned=True activa prompt especial para PDFs imagen.
    Fallback a Tesseract si GPT-4o no está disponible.
    """
    if not img_bytes or len(img_bytes) < MIN_IMAGE_BYTES:
        return ""

    enhanced = enhance_for_ocr(img_bytes, aggressive=is_scanned)
    img_b64 = base64.b64encode(enhanced).decode("utf-8")

    # ── GPT-4o Vision ────────────────────────────────────────────────
    if OPENAI_API_KEY:
        try:
            if is_scanned:
                prompt = """Este es un PDF que es una imagen escaneada o fotografía de un documento.
Extrae TODO el texto visible con máxima precisión.

REGLAS ESTRICTAS:
- Extrae cada palabra, número y símbolo visible
- Si hay TABLAS: usa formato Markdown con | separadores, respeta filas y columnas exactas
- Si hay múltiples columnas: extrae de izquierda a derecha, columna por columna
- Mantén números EXACTOS (fechas, cantidades, resultados, códigos)
- Incluye encabezados, títulos, pies de página, marcas de agua
- Si hay texto en varias orientaciones, extrae todos
- NO expliques, NO describas — solo el texto extraído tal como aparece"""
            else:
                prompt = """Extrae TODO el texto visible en esta imagen con precisión exacta.
Si hay tablas: formato Markdown con | separadores.
Mantén números exactos. Solo texto extraído, sin explicaciones."""

            payload = json.dumps({
                "model": OPENAI_MODEL,
                "max_tokens": 4_000,
                "temperature": 0.0,
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}", "detail": "high"}},
                        {"type": "text", "text": prompt},
                    ],
                }],
            }).encode("utf-8")

            req = urllib.request.Request(
                "https://api.openai.com/v1/chat/completions",
                data=payload,
                headers={"Content-Type": "application/json", "Authorization": f"Bearer {OPENAI_API_KEY}"},
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                text = result["choices"][0]["message"]["content"].strip()
                if text and len(text) > 10:
                    log.info("GPT-4o OCR OK (%d chars, scanned=%s)", len(text), is_scanned)
                    return text
        except Exception as exc:
            log.warning("GPT-4o OCR falló: %s", exc)

    # ── Tesseract fallback ────────────────────────────────────────────
    try:
        import pytesseract
        img = PILImage.open(io.BytesIO(enhanced))
        text = pytesseract.image_to_string(img, lang="spa+eng", config="--psm 3 --oem 3")
        if text and len(text.strip()) > 10:
            log.info("Tesseract OCR OK (%d chars)", len(text.strip()))
            return text.strip()
    except Exception as exc:
        log.warning("Tesseract OCR falló: %s", exc)

    return ""


# ══════════════════════════════════════════════
# SECCIÓN 4 — EXTRACCIÓN DE TABLAS (pdfplumber)
# ══════════════════════════════════════════════

_TABLE_STRATEGIES = [
    {"vertical_strategy": "lines_strict", "horizontal_strategy": "lines_strict", "snap_tolerance": 3, "join_tolerance": 3},
    {"vertical_strategy": "lines", "horizontal_strategy": "lines", "snap_tolerance": 5, "join_tolerance": 5},
    {"vertical_strategy": "text", "horizontal_strategy": "text", "snap_tolerance": 5, "text_x_tolerance": 5, "text_y_tolerance": 5},
]


def extract_tables(pdf_bytes: bytes, page_number: int) -> list[dict]:
    results: list[dict] = []
    seen: set[int] = set()

    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            if page_number - 1 >= len(pdf.pages):
                return results
            pg = pdf.pages[page_number - 1]

            for strategy in _TABLE_STRATEGIES:
                try:
                    raw_tables = pg.extract_tables(strategy) or []
                except Exception:
                    continue

                for t in raw_tables:
                    if not t or len(t) < 2:
                        continue
                    clean = []
                    for row in t:
                        r = [re.sub(r"\s+", " ", str(c or "")).strip() for c in row]
                        if any(r):
                            clean.append(r)
                    if len(clean) < 2:
                        continue
                    h = hash(str(clean))
                    if h in seen:
                        continue
                    seen.add(h)

                    headers = clean[0]
                    rows = clean[1:]
                    n = len(headers)
                    rows = [r + [""] * (n - len(r)) if len(r) < n else r[:n] for r in rows]

                    md = "| " + " | ".join(headers) + " |\n"
                    md += "| " + " | ".join(["---"] * n) + " |\n"
                    for r in rows:
                        md += "| " + " | ".join(r) + " |\n"

                    plain = " | ".join(headers) + "\n" + "\n".join(" | ".join(r) for r in rows)

                    results.append({
                        "index": len(results),
                        "markdown": md.strip(),
                        "plain_text": plain.strip(),
                        "rows": len(rows),
                        "cols": n,
                        "strategy": strategy["vertical_strategy"],
                    })

                if results:
                    break
    except Exception as exc:
        log.warning("extract_tables pág %d: %s", page_number, exc)

    return results


# ══════════════════════════════════════════════
# SECCIÓN 5 — EXTRACCIÓN DE IMÁGENES
# ══════════════════════════════════════════════

def _nearest_text(bbox: list, text_blocks: list) -> str | None:
    if not text_blocks:
        return None
    def dist(b1, b2):
        cx1, cy1 = (b1[0]+b1[2])/2, (b1[1]+b1[3])/2
        cx2, cy2 = (b2[0]+b2[2])/2, (b2[1]+b2[3])/2
        return math.sqrt((cx1-cx2)**2 + (cy1-cy2)**2)
    best = min(text_blocks, key=lambda tb: dist(bbox, tb["bbox"]))
    return best["text"]


def extract_images(page: fitz.Page, doc: fitz.Document, page_num: int,
                   text_blocks: list, is_scanned: bool) -> list[dict]:
    images = []
    img_bboxes: dict[int, list] = {}
    for block in page.get_text("dict")["blocks"]:
        if block["type"] == 1:
            img_bboxes[len(img_bboxes)] = list(block["bbox"])

    for idx, img_info in enumerate(page.get_images(full=True)):
        xref = img_info[0]
        try:
            base_img = doc.extract_image(xref)
            raw = base_img["image"]
            if len(raw) < MIN_IMAGE_BYTES:
                continue

            bbox = img_bboxes.get(idx, [0, 0, page.rect.width, page.rect.height])
            nearest = _nearest_text(bbox, text_blocks)

            # OCR solo si la imagen es grande (descarta iconos/logos pequeños)
            ocr_text = ""
            img_w = base_img.get("width", 0)
            img_h = base_img.get("height", 0)
            if img_w > 100 and img_h > 100:
                ocr_text = ocr_with_gpt4o(raw, context=f"imagen pág.{page_num}", is_scanned=is_scanned)

            images.append({
                "index": idx,
                "base64": img_to_b64(raw),
                "bbox": bbox,
                "nearest_text": nearest,
                "ocr_text": ocr_text,
                "width_px": img_w,
                "height_px": img_h,
            })
        except Exception as exc:
            log.warning("extract_images xref=%s pág=%d: %s", xref, page_num, exc)

    return images


# ══════════════════════════════════════════════
# SECCIÓN 6 — EXTRACCIÓN POR PÁGINA
# ══════════════════════════════════════════════

def extract_page(page: fitz.Page, page_num: int, doc: fitz.Document,
                 pdf_bytes: bytes, pdf_info: dict) -> dict:
    """
    Flujo completo por página:
    1. Texto vectorial (PyMuPDF)
    2. Tablas (pdfplumber — 3 estrategias)
    3. Imágenes embebidas + OCR por imagen
    4. Si texto < MIN_TEXT_LENGTH → OCR página completa con GPT-4o Vision
       (zoom 3x para PDFs escaneados, 2.5x para normales)
    5. Construir texto final enriquecido
    """
    t0 = time.time()
    is_scanned = pdf_info["is_scanned"]
    zoom = pdf_info["zoom"]

    # ── 1. Texto vectorial ────────────────────────────────────────────
    vector_text = page.get_text("text").strip()
    used_ocr = False

    # ── 2. Bloques con bboxes ─────────────────────────────────────────
    text_blocks = []
    for block in page.get_text("dict")["blocks"]:
        if block["type"] == 0:
            spans_text = " ".join(
                s["text"] for l in block.get("lines", []) for s in l.get("spans", [])
            ).strip()
            if spans_text:
                text_blocks.append({
                    "text": spans_text,
                    "bbox": list(block["bbox"]),
                    "font": (block.get("lines") or [{}])[0].get("spans", [{}])[0].get("font", ""),
                    "size": (block.get("lines") or [{}])[0].get("spans", [{}])[0].get("size", 0),
                })

    # ── 3. Tablas ─────────────────────────────────────────────────────
    tables = extract_tables(pdf_bytes, page_num)

    # ── 4. Imágenes ───────────────────────────────────────────────────
    images = extract_images(page, doc, page_num, text_blocks, is_scanned)

    # ── 5. OCR página completa ────────────────────────────────────────
    # Se activa si: texto vectorial insuficiente (PDF escaneado o imagen convertida)
    if len(vector_text) < MIN_TEXT_LENGTH:
        try:
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            page_png = pix.tobytes("png")
            page_ocr = ocr_with_gpt4o(page_png, context="página completa PDF escaneado", is_scanned=True)
            if page_ocr and len(page_ocr) > len(vector_text):
                vector_text = page_ocr
                used_ocr = True
                log.info("OCR completo pág=%d (%d chars, zoom=%.1f)", page_num, len(page_ocr), zoom)
        except Exception as exc:
            log.warning("OCR página completa pág=%d: %s", page_num, exc)

    # ── 6. Texto final enriquecido ────────────────────────────────────
    parts = []
    if vector_text:
        parts.append(vector_text)

    for img in images:
        if img["ocr_text"] and len(img["ocr_text"]) > 20:
            parts.append(f"\n[IMAGEN pág.{page_num} idx.{img['index']}]\n{img['ocr_text']}")

    for tbl in tables:
        parts.append(f"\n[TABLA {tbl['rows']}×{tbl['cols']} pág.{page_num}]\n{tbl['markdown']}")

    full_text = "\n".join(parts).strip()

    log.info("pág=%d | chars=%d | imgs=%d | tablas=%d | ocr=%s | %.2fs",
             page_num, len(full_text), len(images), len(tables), used_ocr, time.time()-t0)

    return {
        "page_number": page_num,
        "text": full_text,
        "vector_text": vector_text,
        "text_blocks": text_blocks,
        "images": images,
        "tables": tables,
        "has_images": len(images) > 0,
        "has_tables": len(tables) > 0,
        "used_ocr": used_ocr,
        "is_scanned_page": used_ocr or (len(vector_text) < MIN_TEXT_LENGTH),
        "width": page.rect.width,
        "height": page.rect.height,
    }


# ══════════════════════════════════════════════
# SECCIÓN 7 — CHUNKING SEMÁNTICO
# ══════════════════════════════════════════════

def build_chunks(pages_data: list, filename: str) -> list:
    chunks = []
    idx = 0

    for page in pages_data:
        page_num = page["page_number"]

        # Chunks de tablas — siempre independientes
        for tbl in page.get("tables", []):
            if not tbl["plain_text"]:
                continue
            chunks.append({
                "chunk_text": f"Tabla (página {page_num}):\n{tbl['plain_text']}",
                "markdown_text": f"### Tabla — Página {page_num}\n{tbl['markdown']}",
                "page_number": page_num,
                "chunk_index": idx,
                "has_image": False,
                "images": [],
                "metadata": {
                    "source_doc": filename, "chunk_index": idx,
                    "type": "table", "rows": tbl["rows"], "cols": tbl["cols"],
                },
            })
            idx += 1

        # Texto de la página
        text = page["text"].strip()
        if not text:
            # Página sin texto extraíble — aun así crear chunk para no perder la página
            text = f"[Página {page_num} — contenido visual sin texto extraíble]"

        def _make_chunk(t, sub=0):
            return {
                "chunk_text": t,
                "markdown_text": t,
                "page_number": page_num,
                "chunk_index": idx,
                "has_image": sub == 0 and page["has_images"],
                "images": page["images"] if sub == 0 else [],
                "metadata": {
                    "source_doc": filename, "chunk_index": idx,
                    "type": "text", "used_ocr": page["used_ocr"],
                    "is_scanned": page.get("is_scanned_page", False),
                },
            }

        if len(text) <= MAX_CHUNK_CHARS:
            chunks.append(_make_chunk(text))
            idx += 1
        else:
            sents = re.split(r"(?<=[.!?])\s+", text)
            buf, sub = "", 0
            for s in sents:
                if buf and len(buf) + len(s) + 1 > MAX_CHUNK_CHARS:
                    c = _make_chunk(buf.strip(), sub)
                    chunks.append(c)
                    idx += 1
                    sub += 1
                    buf = s + " "
                else:
                    buf += s + " "
            if buf.strip():
                chunks.append(_make_chunk(buf.strip(), sub))
                idx += 1

    total = len(chunks)
    for c in chunks:
        c["metadata"]["total_chunks"] = total
    return chunks


# ══════════════════════════════════════════════
# SECCIÓN 8 — SUPABASE STORAGE
# ══════════════════════════════════════════════

async def upload_to_supabase(img_b64: str, document_id: str,
                              page_number: int, img_index: int) -> str | None:
    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        return None
    try:
        img_bytes = base64.b64decode(img_b64)
        path = f"doc-{document_id}/p{page_number}_img{img_index}.png"
        url = f"{SUPABASE_URL}/storage/v1/object/{SUPABASE_BUCKET}/{path}"
        headers = {
            "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
            "Content-Type": "image/png",
            "x-upsert": "true",
        }
        async with httpx.AsyncClient(timeout=45) as client:
            resp = await client.post(url, content=img_bytes, headers=headers)
            if resp.status_code not in (200, 201):
                log.warning("Supabase upload %s: %s", resp.status_code, resp.text[:100])
                return None
        return f"{SUPABASE_URL}/storage/v1/object/public/{SUPABASE_BUCKET}/{path}"
    except Exception as exc:
        log.warning("upload_to_supabase: %s", exc)
        return None


async def enrich_with_urls(chunks: list, pages_data: list, document_id: str) -> list:
    sem = asyncio.Semaphore(4)

    async def _up(pg, idx, b64):
        async with sem:
            return pg, idx, await upload_to_supabase(b64, document_id, pg, idx)

    tasks = []
    for chunk in chunks:
        pg = chunk["page_number"]
        for i, img in enumerate(chunk.get("images", [])):
            if img.get("base64"):
                tasks.append(_up(pg, i, img["base64"]))

    results = await asyncio.gather(*tasks)
    url_map = {(pg, i): url for pg, i, url in results}

    for chunk in chunks:
        pg = chunk["page_number"]
        enriched = []
        for i, img in enumerate(chunk.get("images", [])):
            url = url_map.get((pg, i))
            enriched.append({
                "url": url,
                "page": pg,
                "bbox": img.get("bbox"),
                "nearest_text": img.get("nearest_text"),
                "ocr_text": img.get("ocr_text"),
            })
        # image_path nunca bloquea — si no hay URL igual se guarda el chunk
        chunk["image_path"] = enriched[0]["url"] if enriched and enriched[0]["url"] else None
        chunk["image_urls"] = [e["url"] for e in enriched if e["url"]]
        chunk["image_description"] = None
        chunk.pop("images", None)

    return chunks


# ══════════════════════════════════════════════
# SECCIÓN 9 — ENDPOINTS
# ══════════════════════════════════════════════

@app.get("/health")
async def health():
    tesseract_ok = False
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
        tesseract_ok = True
    except Exception:
        pass

    return {
        "status": "ok",
        "version": "11.0.0",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "capabilities": {
            "pdf_vectorial": True,
            "pdf_scanned_image": True,
            "ocr_gpt4o_vision": bool(OPENAI_API_KEY),
            "ocr_tesseract_fallback": tesseract_ok,
            "tables_pdfplumber": True,
            "image_extraction": True,
            "bounding_boxes": True,
            "text_image_mapping": True,
            "supabase_storage": bool(SUPABASE_URL and SUPABASE_SERVICE_KEY),
        },
        "config": {
            "min_text_length": MIN_TEXT_LENGTH,
            "max_chunk_chars": MAX_CHUNK_CHARS,
            "ocr_zoom_normal": OCR_ZOOM_NORMAL,
            "ocr_zoom_scanned": OCR_ZOOM_SCANNED,
            "openai_model": OPENAI_MODEL,
        },
    }


@app.post("/process-pdf")
async def process_pdf(
    file: UploadFile = File(...),
    document_id: str = Form(...),
    filename: str = Form(...),
    max_pages: int = Form(default=0),
    upload_images: bool = Form(default=True),
):
    """
    Procesamiento completo de PDF:
    - Detecta automáticamente si el PDF es escaneado (imagen convertida a PDF)
    - Aplica OCR con GPT-4o Vision en modo adecuado según tipo
    - Extrae texto vectorial, tablas y imágenes con bboxes
    - Mapea texto ↔ imagen por proximidad espacial
    - Chunks semánticos que nunca se descartan por image_path=null
    """
    t_start = time.time()
    pdf_bytes = await file.read()

    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="Archivo PDF vacío")

    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"No se pudo abrir el PDF: {e}")

    total_pages = len(doc)
    pages_to_process = min(max_pages, total_pages) if max_pages > 0 else total_pages

    # Detectar tipo de PDF ANTES de procesar páginas
    pdf_info = detect_pdf_type(doc)
    log.info("'%s' — tipo=%s zoom=%.1f páginas=%d", filename, pdf_info["type"], pdf_info["zoom"], pages_to_process)

    pages_data = []
    for i in range(pages_to_process):
        try:
            page_data = extract_page(doc[i], i + 1, doc, pdf_bytes, pdf_info)
            pages_data.append(page_data)
        except Exception as exc:
            log.error("Error pág %d: %s", i + 1, exc)
            pages_data.append({
                "page_number": i + 1, "text": f"[Error pág {i+1}]",
                "vector_text": "", "text_blocks": [], "images": [], "tables": [],
                "has_images": False, "has_tables": False, "used_ocr": False,
                "is_scanned_page": False, "width": 0, "height": 0,
            })
    doc.close()

    chunks = build_chunks(pages_data, filename)

    if upload_images and SUPABASE_URL and SUPABASE_SERVICE_KEY:
        chunks = await enrich_with_urls(chunks, pages_data, document_id)
    else:
        for chunk in chunks:
            chunk["image_path"] = None
            chunk["image_urls"] = []
            chunk["image_description"] = None
            chunk.pop("images", None)

    elapsed = round(time.time() - t_start, 2)
    scanned_pages = sum(1 for p in pages_data if p.get("is_scanned_page"))
    total_images = sum(len(p["images"]) for p in pages_data)
    total_tables = sum(len(p["tables"]) for p in pages_data)
    used_ocr = any(p["used_ocr"] for p in pages_data)

    log.info("'%s' listo en %.2fs | chunks=%d imgs=%d tablas=%d ocr=%s",
             filename, elapsed, len(chunks), total_images, total_tables, used_ocr)

    return {
        "success": True,
        "document_id": document_id,
        "filename": filename,
        "pdf_type": pdf_info["type"],
        "total_pages_in_doc": total_pages,
        "total_pages_processed": pages_to_process,
        "has_more_pages": pages_to_process < total_pages,
        "remaining_pages": total_pages - pages_to_process if pages_to_process < total_pages else 0,
        "total_chunks": len(chunks),
        "total_images_extracted": total_images,
        "total_tables_extracted": total_tables,
        "scanned_pages_detected": scanned_pages,
        "used_ocr": used_ocr,
        "elapsed_seconds": elapsed,
        "chunks": chunks,
        "message": f"PDF '{pdf_info['type']}' procesado — {scanned_pages} páginas escaneadas detectadas" if scanned_pages else "PDF procesado con éxito",
    }


@app.post("/match-chunks")
async def match_chunks(body: dict):
    query_embedding = body.get("query_embedding")
    match_count = body.get("match_count", 5)
    doc_ids = body.get("doc_ids")
    similarity_threshold = body.get("similarity_threshold", 0.7)

    if not query_embedding:
        raise HTTPException(status_code=400, detail="query_embedding requerido")
    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        raise HTTPException(status_code=503, detail="Supabase no configurado")

    payload: dict[str, Any] = {
        "query_embedding": query_embedding,
        "match_count": match_count,
        "similarity_threshold": similarity_threshold,
    }
    if doc_ids:
        payload["doc_ids"] = doc_ids

    headers = {
        "apikey": SUPABASE_SERVICE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=representation",
    }
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            f"{SUPABASE_URL}/rest/v1/rpc/match_chunks",
            json=payload, headers=headers,
        )
        if resp.status_code != 200:
            raise HTTPException(status_code=500, detail=f"Supabase error: {resp.text}")
        return resp.json()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=os.getenv("ENV") == "development",
    )