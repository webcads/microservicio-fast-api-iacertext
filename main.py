"""
IACertext - Microservicio FastAPI para procesamiento real de PDFs
v6 - PyMuPDF (Fitz) + pdfplumber + OCR en imágenes individuales
     Extrae texto de imágenes embebidas (PDFs escaneados, fotos, diagramas con texto)
"""

import os
import io
import base64
import fitz  # PyMuPDF
import pdfplumber
import httpx
from PIL import Image as PILImage
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="IACertext PDF Processor", version="5.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_SERVICE_KEY = os.environ["SUPABASE_SERVICE_KEY"]
SUPABASE_BUCKET = "rag-images"
MIN_IMAGE_SIZE = 1000
MIN_TEXT_LENGTH = 30


def convert_to_png_base64(img_bytes: bytes) -> str | None:
    try:
        img = PILImage.open(io.BytesIO(img_bytes))
        if img.mode not in ("RGB", "RGBA"):
            img = img.convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=True)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")
    except Exception:
        return None


def ocr_image_bytes(img_bytes: bytes) -> str:
    """
    Aplica OCR (Tesseract) a una imagen individual para extraer texto.
    Funciona con imágenes embebidas en PDFs: fotos, capturas, diagramas con texto.
    Retorna el texto extraído o string vacío si no hay texto legible.
    """
    try:
        import pytesseract
        img = PILImage.open(io.BytesIO(img_bytes))

        # Convertir a RGB para Tesseract
        if img.mode not in ("RGB", "RGBA", "L"):
            img = img.convert("RGB")

        # Mejorar imagen para OCR: aumentar resolución si es pequeña
        min_dim = 800
        w, h = img.size
        if w < min_dim or h < min_dim:
            scale = max(min_dim / w, min_dim / h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            img = img.resize((new_w, new_h), PILImage.LANCZOS)

        # OCR con español e inglés
        text = pytesseract.image_to_string(img, lang='spa+eng', config='--psm 3')
        return text.strip()
    except ImportError:
        return ""
    except Exception:
        return ""


def page_to_png_base64(page, zoom: float = 2.0) -> str | None:
    try:
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        return base64.b64encode(pix.tobytes("png")).decode("utf-8")
    except Exception:
        return None


def extract_tables_pdfplumber(pdf_bytes: bytes, page_number: int) -> list[dict]:
    """
    Usa pdfplumber para extraer tablas con alta precisión.
    pdfplumber es mejor que PyMuPDF para tablas con bordes finos
    y tablas sin bordes (detectadas por alineación de columnas).
    """
    tables = []
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            if page_number - 1 >= len(pdf.pages):
                return tables
            page = pdf.pages[page_number - 1]
            
            # Configuración mejorada para detectar más tipos de tablas
            table_settings = {
                "vertical_strategy": "lines_strict",
                "horizontal_strategy": "lines_strict",
                "snap_tolerance": 3,
                "join_tolerance": 3,
            }
            
            extracted_tables = page.extract_tables(table_settings)
            
            # Fallback: si no encuentra tablas con líneas, intenta por texto
            if not extracted_tables:
                table_settings2 = {
                    "vertical_strategy": "text",
                    "horizontal_strategy": "text",
                    "snap_tolerance": 5,
                }
                extracted_tables = page.extract_tables(table_settings2)

            for t_idx, table_data in enumerate(extracted_tables):
                if not table_data or len(table_data) < 2:
                    continue

                # Limpiar celdas None
                clean_data = []
                for row in table_data:
                    clean_row = [str(cell or "").strip() for cell in row]
                    if any(c for c in clean_row):
                        clean_data.append(clean_row)

                if len(clean_data) < 2:
                    continue

                headers = clean_data[0]
                rows = clean_data[1:]

                # Markdown
                md_lines = ["| " + " | ".join(headers) + " |"]
                md_lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
                for row in rows:
                    # Asegurar misma cantidad de columnas
                    while len(row) < len(headers):
                        row.append("")
                    md_lines.append("| " + " | ".join(row[:len(headers)]) + " |")

                plain = " | ".join(headers) + "\n"
                for row in rows:
                    plain += " | ".join(row[:len(headers)]) + "\n"

                tables.append({
                    "index": t_idx,
                    "markdown": "\n".join(md_lines),
                    "plain_text": plain.strip(),
                    "rows": len(rows),
                    "cols": len(headers),
                    "extractor": "pdfplumber",
                })
    except Exception:
        pass
    return tables


def extract_text_with_ocr_fallback(page) -> tuple[str, bool]:
    text = page.get_text("text").strip()
    if len(text) >= MIN_TEXT_LENGTH:
        return text, False
    try:
        import pytesseract
        mat = fitz.Matrix(2.0, 2.0)
        pix = page.get_pixmap(matrix=mat)
        img = PILImage.open(io.BytesIO(pix.tobytes("png")))
        ocr_text = pytesseract.image_to_string(img, lang='spa+eng').strip()
        if len(ocr_text) >= MIN_TEXT_LENGTH:
            return ocr_text, True
    except Exception:
        pass
    return text, False


def bbox_distance(bbox1, bbox2) -> float:
    cy1 = (bbox1[1] + bbox1[3]) / 2
    cy2 = (bbox2[1] + bbox2[3]) / 2
    return abs(cy1 - cy2)


def extract_page_data(page, page_num: int, doc, pdf_bytes: bytes) -> dict:
    page_text, used_ocr = extract_text_with_ocr_fallback(page)

    # Tablas con pdfplumber (más preciso que PyMuPDF para tablas)
    tables = extract_tables_pdfplumber(pdf_bytes, page_num)

    # Bloques de texto con bounding boxes (PyMuPDF)
    blocks = page.get_text("dict")["blocks"]
    text_blocks = []
    image_block_bboxes = []

    for block in blocks:
        if block["type"] == 0:
            text = " ".join(
                span["text"]
                for line in block.get("lines", [])
                for span in line.get("spans", [])
            ).strip()
            if text:
                text_blocks.append({"text": text, "bbox": list(block["bbox"])})
        elif block["type"] == 1:
            image_block_bboxes.append(list(block["bbox"]))

    # Imágenes reales con PyMuPDF
    images_on_page = []
    image_list = page.get_images(full=True)

    for img_index, img_info in enumerate(image_list):
        xref = img_info[0]
        try:
            base_image = doc.extract_image(xref)
            img_bytes_raw = base_image["image"]
            if len(img_bytes_raw) < MIN_IMAGE_SIZE:
                continue
            png_b64 = convert_to_png_base64(img_bytes_raw)
            if not png_b64:
                continue
            img_bbox = image_block_bboxes[img_index] if img_index < len(image_block_bboxes) else [0, 0, page.rect.width, page.rect.height]

            # Mapeo texto↔imagen por proximidad de bounding boxes
            nearest_text = None
            min_dist = float("inf")
            for tb in text_blocks:
                dist = bbox_distance(img_bbox, tb["bbox"])
                if dist < min_dist:
                    min_dist = dist
                    nearest_text = tb["text"]

            # OCR en la imagen individual para extraer texto embebido
            # Esto permite extraer texto de: fotos de documentos, capturas,
            # diagramas con etiquetas, PDFs que son imágenes escaneadas
            ocr_text_in_image = ocr_image_bytes(img_bytes_raw)

            images_on_page.append({
                "index": img_index,
                "ext": "png",
                "base64": png_b64,
                "bbox": img_bbox,
                "nearest_text": nearest_text,
                "proximity_px": round(min_dist, 2) if min_dist != float("inf") else 0,
                "ocr_text": ocr_text_in_image,  # texto extraído de la imagen
            })
        except Exception:
            continue

    # Captura de página completa si no hay texto ni imágenes
    if len(page_text) < MIN_TEXT_LENGTH and len(images_on_page) == 0:
        page_png = page_to_png_base64(page)
        if page_png:
            # OCR en página completa capturada
            page_png_bytes = base64.b64decode(page_png)
            page_ocr_text = ocr_image_bytes(page_png_bytes)
            images_on_page.append({
                "index": 0, "ext": "png", "base64": page_png,
                "bbox": [0, 0, page.rect.width, page.rect.height],
                "nearest_text": None, "is_full_page": True,
                "ocr_text": page_ocr_text,
            })

    # Texto completo = texto vectorial + OCR de imágenes + tablas
    full_text_parts = [page_text] if page_text else []

    # Agregar texto OCR extraído de cada imagen
    for img in images_on_page:
        ocr_txt = img.get("ocr_text", "")
        if ocr_txt and len(ocr_txt) > 20:
            full_text_parts.append(f"\n[TEXTO EN IMAGEN pág.{page_num}]\n{ocr_txt}\n")

    for table in tables:
        full_text_parts.append(f"\n[TABLA - {table['rows']} filas x {table['cols']} columnas]\n{table['markdown']}\n")

    return {
        "page_number": page_num,
        "text": "\n".join(full_text_parts).strip(),
        "plain_text": page_text,
        "tables": tables,
        "text_blocks": text_blocks,
        "images": images_on_page,
        "has_images": len(images_on_page) > 0,
        "has_tables": len(tables) > 0,
        "used_ocr": used_ocr,
        "width": page.rect.width,
        "height": page.rect.height,
    }


def chunk_pages(pages_data: list, filename: str) -> list:
    chunks = []
    chunk_index = 0

    for page in pages_data:
        # Chunk especial por tabla (pdfplumber)
        for table in page.get("tables", []):
            if table["plain_text"]:
                chunks.append({
                    "chunk_text": f"Tabla (página {page['page_number']}):\n{table['plain_text']}",
                    "markdown_text": f"### Tabla (Página {page['page_number']})\n{table['markdown']}",
                    "page_number": page["page_number"],
                    "chunk_index": chunk_index,
                    "has_image": False,
                    "images": [],
                    "metadata": {
                        "source_doc": filename,
                        "chunk_index": chunk_index,
                        "type": "table",
                        "extractor": "pdfplumber",
                        "rows": table["rows"],
                        "cols": table["cols"],
                    }
                })
                chunk_index += 1

        text = page["text"].strip() or f"Página {page['page_number']} — contenido visual"

        if len(text) > 1500:
            sentences = text.split(". ")
            current = ""
            sub = 0
            for sentence in sentences:
                if len(current) + len(sentence) > 1000 and current:
                    chunks.append({
                        "chunk_text": current.strip(),
                        "markdown_text": current.strip(),
                        "page_number": page["page_number"],
                        "chunk_index": chunk_index,
                        "has_image": sub == 0 and page["has_images"],
                        "images": page["images"] if sub == 0 else [],
                        "metadata": {"source_doc": filename, "chunk_index": chunk_index, "type": "text", "used_ocr": page.get("used_ocr", False)}
                    })
                    chunk_index += 1
                    sub += 1
                    current = sentence + ". "
                else:
                    current += sentence + ". "
            if current.strip():
                chunks.append({
                    "chunk_text": current.strip(),
                    "markdown_text": current.strip(),
                    "page_number": page["page_number"],
                    "chunk_index": chunk_index,
                    "has_image": sub == 0 and page["has_images"],
                    "images": page["images"] if sub == 0 else [],
                    "metadata": {"source_doc": filename, "chunk_index": chunk_index, "type": "text", "used_ocr": page.get("used_ocr", False)}
                })
                chunk_index += 1
        else:
            chunks.append({
                "chunk_text": text,
                "markdown_text": text,
                "page_number": page["page_number"],
                "chunk_index": chunk_index,
                "has_image": page["has_images"],
                "images": page["images"],
                "metadata": {"source_doc": filename, "chunk_index": chunk_index, "type": "text", "used_ocr": page.get("used_ocr", False)}
            })
            chunk_index += 1

    for chunk in chunks:
        chunk["metadata"]["total_chunks"] = len(chunks)
    return chunks


async def upload_image_to_supabase(img_b64: str, document_id: str, page_number: int, img_index: int) -> str:
    img_bytes = base64.b64decode(img_b64)
    path = f"doc-{document_id}/p{page_number}_img{img_index}.png"
    url = f"{SUPABASE_URL}/storage/v1/object/{SUPABASE_BUCKET}/{path}"
    headers = {"Authorization": f"Bearer {SUPABASE_SERVICE_KEY}", "Content-Type": "image/png", "x-upsert": "true"}
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(url, content=img_bytes, headers=headers)
        if resp.status_code not in (200, 201):
            raise Exception(f"Error: {resp.text}")
    return f"{SUPABASE_URL}/storage/v1/object/public/{SUPABASE_BUCKET}/{path}"


async def process_pages(pages_data, document_id, filename):
    chunks = chunk_pages(pages_data, filename)
    image_urls = {}
    for page in pages_data:
        for img_index, img in enumerate(page["images"]):
            key = f"{page['page_number']}_{img_index}"
            try:
                image_urls[key] = await upload_image_to_supabase(img["base64"], document_id, page["page_number"], img_index)
            except Exception:
                image_urls[key] = None

    for chunk in chunks:
        page_num = chunk["page_number"]
        chunk_images = []
        for img_index, img in enumerate(chunk.get("images", [])):
            url = image_urls.get(f"{page_num}_{img_index}")
            if url:
                chunk_images.append({"url": url, "page": page_num, "bbox": img.get("bbox"), "nearest_text": img.get("nearest_text")})
        chunk["image_path"] = chunk_images[0]["url"] if chunk_images else None
        chunk["image_description"] = None
        chunk.pop("images", None)

    return chunks


@app.get("/health")
async def health():
    ocr_ok = False
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
        ocr_ok = True
    except Exception:
        pass
    return {
        "status": "ok", "version": "6.0.0",
        "extractors": ["PyMuPDF (fitz)", "pdfplumber", "Tesseract OCR"],
        "features": [
            "text_vectorial",
            "tables_pdfplumber",
            "images_extraction",
            "ocr_full_page",
            "ocr_per_image",       # NUEVO: OCR en cada imagen individual
            "bounding_boxes",
            "text_image_mapping",
        ],
        "ocr_available": ocr_ok,
    }


@app.post("/process-pdf")
async def process_pdf(
    file: UploadFile = File(...),
    document_id: str = Form(...),
    filename: str = Form(...),
    max_pages: int = Form(default=0),  # 0 = todas las páginas
):
    pdf_bytes = await file.read()
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        total_doc_pages = len(doc)
        pages_to_process = min(max_pages, total_doc_pages) if max_pages > 0 else total_doc_pages
        
        pages_data = []
        for page_num in range(pages_to_process):
            try:
                page = doc[page_num]
                page_data = extract_page_data(page, page_num + 1, doc, pdf_bytes)
                pages_data.append(page_data)
            except Exception:
                continue  # Si una página falla, continúa con la siguiente
        doc.close()
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Error procesando PDF: {str(e)}")

    chunks = await process_pages(pages_data, document_id, filename)

    total_images = sum(1 for c in chunks if c["image_path"])
    total_tables = sum(1 for p in pages_data if p["has_tables"])
    used_ocr = any(p.get("used_ocr") for p in pages_data)
    has_more = pages_to_process < total_doc_pages if max_pages > 0 else False

    return {
        "success": True,
        "document_id": document_id,
        "filename": filename,
        "total_pages_in_doc": total_doc_pages,
        "total_pages": pages_to_process,
        "total_chunks": len(chunks),
        "total_images_extracted": total_images,
        "total_tables_extracted": total_tables,
        "used_ocr": used_ocr,
        "has_more_pages": has_more,
        "remaining_pages": total_doc_pages - pages_to_process if has_more else 0,
        "warning": f"Se procesaron {pages_to_process} de {total_doc_pages} páginas." if has_more else None,
        "chunks": chunks,
    }


@app.post("/match-chunks")
async def match_chunks(body: dict):
    query_embedding = body.get("query_embedding")
    match_count = body.get("match_count", 5)
    doc_ids = body.get("doc_ids")
    if not query_embedding:
        raise HTTPException(status_code=400, detail="query_embedding requerido")
    payload = {"query_embedding": query_embedding, "match_count": match_count}
    if doc_ids:
        payload["doc_ids"] = doc_ids
    url = f"{SUPABASE_URL}/rest/v1/rpc/match_chunks"
    headers = {
        "apikey": SUPABASE_SERVICE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=representation",
    }
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(url, json=payload, headers=headers)
        if resp.status_code != 200:
            raise HTTPException(status_code=500, detail=f"Error Supabase: {resp.text}")
        return resp.json()