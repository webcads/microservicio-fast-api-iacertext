"""
IACertext - Microservicio FastAPI para procesamiento real de PDFs
v4 - Extracción de tablas, OCR para PDFs escaneados, bounding boxes completos
"""

import os
import io
import base64
import fitz  # PyMuPDF
import httpx
from PIL import Image as PILImage
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="IACertext PDF Processor", version="4.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

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


def page_to_png_base64(page, zoom: float = 2.0) -> str | None:
    try:
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        return base64.b64encode(pix.tobytes("png")).decode("utf-8")
    except Exception:
        return None


def extract_tables_from_page(page) -> list[dict]:
    """
    Extrae tablas de una página PDF usando PyMuPDF.
    Retorna lista de tablas con su contenido en Markdown y bounding box.
    """
    tables = []
    try:
        page_tables = page.find_tables()
        for table_index, table in enumerate(page_tables):
            try:
                data = table.extract()
                if not data or len(data) < 2:
                    continue

                # Convertir tabla a Markdown
                headers = [str(cell or "").strip() for cell in data[0]]
                rows = data[1:]

                md_lines = ["| " + " | ".join(headers) + " |"]
                md_lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
                for row in rows:
                    cells = [str(cell or "").strip() for cell in row]
                    md_lines.append("| " + " | ".join(cells) + " |")

                markdown_table = "\n".join(md_lines)

                # También texto plano para embedding
                plain_text = " | ".join(headers) + "\n"
                for row in rows:
                    plain_text += " | ".join([str(c or "").strip() for c in row]) + "\n"

                tables.append({
                    "index": table_index,
                    "markdown": markdown_table,
                    "plain_text": plain_text.strip(),
                    "bbox": list(table.bbox),
                    "rows": len(rows),
                    "cols": len(headers),
                })
            except Exception:
                continue
    except Exception:
        pass
    return tables


def extract_text_with_ocr_fallback(page) -> tuple[str, bool]:
    """Extrae texto vectorial, con fallback a OCR si es necesario."""
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


def extract_page_data(page, page_num: int, doc) -> dict:
    """Extrae todo el contenido de una página: texto, tablas, imágenes con bbox."""

    # 1. Texto con OCR fallback
    page_text, used_ocr = extract_text_with_ocr_fallback(page)

    # 2. Tablas estructuradas
    tables = extract_tables_from_page(page)

    # 3. Bloques de texto con bounding boxes
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
                text_blocks.append({
                    "text": text,
                    "bbox": list(block["bbox"]),
                })
        elif block["type"] == 1:
            image_block_bboxes.append(list(block["bbox"]))

    # 4. Imágenes reales con bounding boxes
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

            # Encontrar texto más cercano (mapeo texto↔imagen)
            nearest_text = None
            min_dist = float("inf")
            for tb in text_blocks:
                dist = bbox_distance(img_bbox, tb["bbox"])
                if dist < min_dist:
                    min_dist = dist
                    nearest_text = tb["text"]

            images_on_page.append({
                "index": img_index,
                "ext": "png",
                "base64": png_b64,
                "bbox": img_bbox,
                "nearest_text": nearest_text,
                "proximity_distance": round(min_dist, 2),
            })
        except Exception:
            continue

    # 5. Si la página no tiene texto ni imágenes, capturarla completa
    if len(page_text) < MIN_TEXT_LENGTH and len(images_on_page) == 0:
        page_png = page_to_png_base64(page)
        if page_png:
            images_on_page.append({
                "index": 0,
                "ext": "png",
                "base64": page_png,
                "bbox": [0, 0, page.rect.width, page.rect.height],
                "nearest_text": None,
                "is_full_page": True,
            })

    # 6. Combinar texto con contenido de tablas
    full_text_parts = [page_text] if page_text else []
    for table in tables:
        full_text_parts.append(f"\n[TABLA]\n{table['markdown']}\n")

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


def chunk_pages(pages_data: list, filename: str, document_id: str) -> list:
    chunks = []
    chunk_index = 0

    for page in pages_data:
        text = page["text"].strip()

        if not text:
            text = f"Página {page['page_number']} — contenido visual sin texto extraible"

        # Crear chunk especial para cada tabla de la página
        for table in page.get("tables", []):
            if table["plain_text"]:
                chunks.append({
                    "chunk_text": f"Tabla página {page['page_number']}:\n{table['plain_text']}",
                    "markdown_text": f"### Tabla (Página {page['page_number']})\n{table['markdown']}",
                    "page_number": page["page_number"],
                    "chunk_index": chunk_index,
                    "has_image": False,
                    "images": [],
                    "bbox": table["bbox"],
                    "metadata": {
                        "source_doc": filename,
                        "chunk_index": chunk_index,
                        "type": "table",
                        "rows": table["rows"],
                        "cols": table["cols"],
                    }
                })
                chunk_index += 1

        # Chunks de texto normal
        if len(text) > 1500:
            sentences = text.split(". ")
            current_chunk = ""
            sub_index = 0
            for sentence in sentences:
                if len(current_chunk) + len(sentence) > 1000 and current_chunk:
                    chunks.append({
                        "chunk_text": current_chunk.strip(),
                        "markdown_text": current_chunk.strip(),
                        "page_number": page["page_number"],
                        "chunk_index": chunk_index,
                        "has_image": sub_index == 0 and page["has_images"],
                        "images": page["images"] if sub_index == 0 else [],
                        "bbox": None,
                        "metadata": {
                            "source_doc": filename,
                            "chunk_index": chunk_index,
                            "used_ocr": page.get("used_ocr", False),
                            "type": "text",
                        }
                    })
                    chunk_index += 1
                    sub_index += 1
                    current_chunk = sentence + ". "
                else:
                    current_chunk += sentence + ". "
            if current_chunk.strip():
                chunks.append({
                    "chunk_text": current_chunk.strip(),
                    "markdown_text": current_chunk.strip(),
                    "page_number": page["page_number"],
                    "chunk_index": chunk_index,
                    "has_image": sub_index == 0 and page["has_images"],
                    "images": page["images"] if sub_index == 0 else [],
                    "bbox": None,
                    "metadata": {
                        "source_doc": filename,
                        "chunk_index": chunk_index,
                        "used_ocr": page.get("used_ocr", False),
                        "type": "text",
                    }
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
                "bbox": None,
                "metadata": {
                    "source_doc": filename,
                    "chunk_index": chunk_index,
                    "used_ocr": page.get("used_ocr", False),
                    "type": "text",
                }
            })
            chunk_index += 1

    for chunk in chunks:
        chunk["metadata"]["total_chunks"] = len(chunks)

    return chunks


async def upload_image_to_supabase(img_b64: str, document_id: str, page_number: int, img_index: int) -> str:
    img_bytes = base64.b64decode(img_b64)
    path = f"doc-{document_id}/p{page_number}_img{img_index}.png"
    url = f"{SUPABASE_URL}/storage/v1/object/{SUPABASE_BUCKET}/{path}"
    headers = {
        "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
        "Content-Type": "image/png",
        "x-upsert": "true",
    }
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(url, content=img_bytes, headers=headers)
        if resp.status_code not in (200, 201):
            raise Exception(f"Error subiendo imagen: {resp.text}")
    return f"{SUPABASE_URL}/storage/v1/object/public/{SUPABASE_BUCKET}/{path}"


@app.get("/health")
async def health():
    ocr_available = False
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
        ocr_available = True
    except Exception:
        pass
    return {
        "status": "ok",
        "service": "IACertext PDF Processor",
        "version": "4.0.0",
        "features": ["text", "tables", "images", "bounding_boxes", "text_image_mapping"],
        "ocr_available": ocr_available,
    }


@app.post("/process-pdf")
async def process_pdf(
    file: UploadFile = File(...),
    document_id: str = Form(...),
    filename: str = Form(...),
):
    pdf_bytes = await file.read()

    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        pages_data = []
        for page_num, page in enumerate(doc, start=1):
            page_data = extract_page_data(page, page_num, doc)
            pages_data.append(page_data)
        doc.close()
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Error procesando PDF: {str(e)}")

    chunks = chunk_pages(pages_data, filename, document_id)

    # Subir imágenes a Supabase Storage
    image_urls = {}
    for page in pages_data:
        for img_index, img in enumerate(page["images"]):
            key = f"{page['page_number']}_{img_index}"
            try:
                public_url = await upload_image_to_supabase(
                    img_b64=img["base64"],
                    document_id=document_id,
                    page_number=page["page_number"],
                    img_index=img_index,
                )
                image_urls[key] = public_url
            except Exception:
                image_urls[key] = None

    # Enriquecer chunks con URLs de imágenes
    for chunk in chunks:
        page_num = chunk["page_number"]
        chunk_images = []
        for img_index, img in enumerate(chunk.get("images", [])):
            key = f"{page_num}_{img_index}"
            url = image_urls.get(key)
            if url:
                chunk_images.append({
                    "url": url,
                    "page": page_num,
                    "bbox": img.get("bbox"),
                    "nearest_text": img.get("nearest_text"),
                })
        chunk["image_path"] = chunk_images[0]["url"] if chunk_images else None
        chunk["image_description"] = None
        chunk.pop("images", None)

    total_images = sum(1 for c in chunks if c["image_path"])
    total_tables = sum(1 for p in pages_data if p["has_tables"])
    used_ocr = any(p.get("used_ocr") for p in pages_data)

    return {
        "success": True,
        "document_id": document_id,
        "filename": filename,
        "total_pages": len(pages_data),
        "total_chunks": len(chunks),
        "total_images_extracted": total_images,
        "total_tables_extracted": total_tables,
        "used_ocr": used_ocr,
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