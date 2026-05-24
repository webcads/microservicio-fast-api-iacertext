"""
IACertext - Microservicio FastAPI para procesamiento real de PDFs
Usa PyMuPDF (fitz) para extraer texto, imágenes y bounding boxes
v2 - Siempre convierte imágenes a PNG, maneja formatos no soportados
"""

import os
import io
import base64
import fitz  # PyMuPDF
import httpx
from PIL import Image as PILImage
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="IACertext PDF Processor", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_SERVICE_KEY = os.environ["SUPABASE_SERVICE_KEY"]
SUPABASE_BUCKET = "rag-images"

MIN_IMAGE_SIZE = 1000  # bytes mínimos para considerar imagen real


def convert_to_png_base64(img_bytes: bytes) -> str | None:
    """Convierte cualquier formato de imagen a PNG usando Pillow."""
    try:
        img = PILImage.open(io.BytesIO(img_bytes))
        # Convertir a RGB si es necesario (CMYK, palette, etc.)
        if img.mode not in ("RGB", "RGBA"):
            img = img.convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=True)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")
    except Exception:
        return None


def extract_text_and_images(pdf_bytes: bytes, document_id: str, filename: str):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages_data = []

    for page_num, page in enumerate(doc, start=1):
        blocks = page.get_text("dict")["blocks"]
        text_blocks = []
        image_blocks = []

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
                        "bbox": block["bbox"],
                    })
            elif block["type"] == 1:
                image_blocks.append({"bbox": block["bbox"]})

        # Extraer imágenes reales y convertir a PNG
        images_on_page = []
        image_list = page.get_images(full=True)

        for img_index, img_info in enumerate(image_list):
            xref = img_info[0]
            try:
                base_image = doc.extract_image(xref)
                img_bytes_raw = base_image["image"]

                # Ignorar imágenes demasiado pequeñas (iconos, píxeles de fondo)
                if len(img_bytes_raw) < MIN_IMAGE_SIZE:
                    continue

                # Convertir siempre a PNG para compatibilidad con GPT-4o
                png_b64 = convert_to_png_base64(img_bytes_raw)
                if not png_b64:
                    continue

                img_bbox = image_blocks[img_index]["bbox"] if img_index < len(image_blocks) else None

                images_on_page.append({
                    "index": img_index,
                    "ext": "png",  # siempre PNG
                    "base64": png_b64,
                    "bbox": img_bbox or [0, 0, page.rect.width, page.rect.height],
                    "xref": xref,
                })
            except Exception:
                continue

        pages_data.append({
            "page_number": page_num,
            "text_blocks": text_blocks,
            "images": images_on_page,
            "width": page.rect.width,
            "height": page.rect.height,
        })

    doc.close()
    return pages_data


def map_text_to_images(pages_data):
    def bbox_distance(bbox1, bbox2):
        cy1 = (bbox1[1] + bbox1[3]) / 2
        cy2 = (bbox2[1] + bbox2[3]) / 2
        return abs(cy1 - cy2)

    result = []
    for page in pages_data:
        page_images = page["images"]
        full_text_parts = [b["text"] for b in page["text_blocks"]]

        result.append({
            "page_number": page["page_number"],
            "full_text": "\n".join(full_text_parts),
            "images": page_images,
            "has_images": len(page_images) > 0,
        })

    return result


def chunk_by_pages(mapped_pages, filename: str, document_id: str):
    chunks = []
    chunk_index = 0

    for page in mapped_pages:
        text = page["full_text"].strip()
        if not text or len(text) < 20:
            continue

        if len(text) > 1500:
            sentences = text.split(". ")
            current_chunk = ""
            sub_index = 0
            for sentence in sentences:
                if len(current_chunk) + len(sentence) > 1000 and current_chunk:
                    chunks.append({
                        "chunk_text": current_chunk.strip(),
                        "page_number": page["page_number"],
                        "chunk_index": chunk_index,
                        "has_image": sub_index == 0 and page["has_images"],
                        "images": page["images"] if sub_index == 0 else [],
                        "metadata": {"source_doc": filename, "chunk_index": chunk_index, "sub_index": sub_index}
                    })
                    chunk_index += 1
                    sub_index += 1
                    current_chunk = sentence + ". "
                else:
                    current_chunk += sentence + ". "
            if current_chunk.strip():
                chunks.append({
                    "chunk_text": current_chunk.strip(),
                    "page_number": page["page_number"],
                    "chunk_index": chunk_index,
                    "has_image": sub_index == 0 and page["has_images"],
                    "images": page["images"] if sub_index == 0 else [],
                    "metadata": {"source_doc": filename, "chunk_index": chunk_index, "sub_index": sub_index}
                })
                chunk_index += 1
        else:
            chunks.append({
                "chunk_text": text,
                "page_number": page["page_number"],
                "chunk_index": chunk_index,
                "has_image": page["has_images"],
                "images": page["images"],
                "metadata": {"source_doc": filename, "chunk_index": chunk_index}
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
    return {"status": "ok", "service": "IACertext PDF Processor", "version": "2.0.0"}


@app.post("/process-pdf")
async def process_pdf(
    file: UploadFile = File(...),
    document_id: str = Form(...),
    filename: str = Form(...),
):
    pdf_bytes = await file.read()

    pages_data = extract_text_and_images(pdf_bytes, document_id, filename)
    mapped_pages = map_text_to_images(pages_data)
    chunks = chunk_by_pages(mapped_pages, filename, document_id)

    # Subir imágenes a Supabase Storage
    image_urls = {}
    for page in mapped_pages:
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

    # Enriquecer chunks con URLs
    for chunk in chunks:
        page_num = chunk["page_number"]
        chunk_images = []
        for img_index, img in enumerate(chunk.get("images", [])):
            key = f"{page_num}_{img_index}"
            url = image_urls.get(key)
            if url:
                chunk_images.append({"url": url, "page": page_num})

        chunk["image_path"] = chunk_images[0]["url"] if chunk_images else None
        chunk["image_description"] = None
        chunk.pop("images", None)

    total_images = sum(1 for c in chunks if c["image_path"])

    return {
        "success": True,
        "document_id": document_id,
        "filename": filename,
        "total_pages": len(pages_data),
        "total_chunks": len(chunks),
        "total_images_extracted": total_images,
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
            raise HTTPException(status_code=500, detail=f"Error Supabase RPC: {resp.text}")
        return resp.json()