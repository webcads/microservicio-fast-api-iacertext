"""
IACertext - Microservicio FastAPI para procesamiento real de PDFs
Usa PyMuPDF (fitz) para extraer texto, imágenes y bounding boxes
"""

import os
import io
import base64
import json
import fitz  # PyMuPDF
import httpx
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import math

app = FastAPI(title="IACertext PDF Processor", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Variables de entorno ────────────────────────────────────────────────────
SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_SERVICE_KEY = os.environ["SUPABASE_SERVICE_KEY"]
SUPABASE_BUCKET = "rag-images"


# ─── Helpers ─────────────────────────────────────────────────────────────────

def extract_text_and_images(pdf_bytes: bytes, document_id: str, filename: str):
    """
    Extrae texto estructurado e imágenes reales del PDF usando PyMuPDF.
    Mapea qué texto está cerca de qué imagen usando bounding boxes.
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages_data = []

    for page_num, page in enumerate(doc, start=1):
        # ── Extraer texto estructurado con bloques ──────────────────────────
        blocks = page.get_text("dict")["blocks"]
        text_blocks = []
        image_blocks = []

        for block in blocks:
            if block["type"] == 0:  # texto
                text = " ".join(
                    span["text"]
                    for line in block.get("lines", [])
                    for span in line.get("spans", [])
                ).strip()
                if text:
                    text_blocks.append({
                        "text": text,
                        "bbox": block["bbox"],  # [x0, y0, x1, y1]
                    })
            elif block["type"] == 1:  # imagen embebida
                image_blocks.append({
                    "bbox": block["bbox"],
                    "number": block.get("number", 0),
                })

        # ── Extraer imágenes reales ─────────────────────────────────────────
        images_on_page = []
        image_list = page.get_images(full=True)

        for img_index, img_info in enumerate(image_list):
            xref = img_info[0]
            try:
                base_image = doc.extract_image(xref)
                img_bytes = base_image["image"]
                img_ext = base_image["ext"]
                img_b64 = base64.b64encode(img_bytes).decode("utf-8")

                # Obtener bbox de la imagen en la página
                img_bbox = None
                for block in image_blocks:
                    if img_bbox is None:
                        img_bbox = block["bbox"]

                images_on_page.append({
                    "index": img_index,
                    "ext": img_ext,
                    "base64": img_b64,
                    "bbox": img_bbox or [0, 0, 100, 100],
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
    """
    Para cada bloque de texto, encuentra la imagen más cercana
    en la misma página usando distancia de bounding boxes.
    """
    PROXIMITY_THRESHOLD = 200  # píxeles

    def bbox_distance(bbox1, bbox2):
        """Distancia vertical entre dos bounding boxes."""
        # Centro vertical de cada bbox
        cy1 = (bbox1[1] + bbox1[3]) / 2
        cy2 = (bbox2[1] + bbox2[3]) / 2
        return abs(cy1 - cy2)

    result = []
    for page in pages_data:
        page_images = page["images"]
        full_text_parts = []

        for text_block in page["text_blocks"]:
            nearest_img = None
            min_dist = float("inf")

            for img in page_images:
                if img["bbox"]:
                    dist = bbox_distance(text_block["bbox"], img["bbox"])
                    if dist < min_dist:
                        min_dist = dist
                        nearest_img = img

            full_text_parts.append(text_block["text"])

        result.append({
            "page_number": page["page_number"],
            "full_text": "\n".join(full_text_parts),
            "images": page["images"],
            "has_images": len(page["images"]) > 0,
        })

    return result


def chunk_by_pages(mapped_pages, filename: str, document_id: str):
    """
    Crea chunks inteligentes agrupando páginas con contenido relacionado.
    Cada chunk mantiene referencia a las imágenes de su página.
    """
    chunks = []
    chunk_index = 0

    for page in mapped_pages:
        text = page["full_text"].strip()
        if not text or len(text) < 20:
            continue

        # Dividir texto largo en sub-chunks de ~1000 caracteres
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
                        "metadata": {
                            "source_doc": filename,
                            "chunk_index": chunk_index,
                            "sub_index": sub_index,
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
                    "page_number": page["page_number"],
                    "chunk_index": chunk_index,
                    "has_image": sub_index == 0 and page["has_images"],
                    "images": page["images"] if sub_index == 0 else [],
                    "metadata": {
                        "source_doc": filename,
                        "chunk_index": chunk_index,
                        "sub_index": sub_index,
                    }
                })
                chunk_index += 1
        else:
            chunks.append({
                "chunk_text": text,
                "page_number": page["page_number"],
                "chunk_index": chunk_index,
                "has_image": page["has_images"],
                "images": page["images"],
                "metadata": {
                    "source_doc": filename,
                    "chunk_index": chunk_index,
                }
            })
            chunk_index += 1

    # Actualizar total_chunks en metadata
    for chunk in chunks:
        chunk["metadata"]["total_chunks"] = len(chunks)

    return chunks


async def upload_image_to_supabase(
    img_b64: str,
    img_ext: str,
    document_id: str,
    page_number: int,
    img_index: int,
) -> str:
    """Sube una imagen a Supabase Storage y retorna la URL pública."""
    img_bytes = base64.b64decode(img_b64)
    content_type = f"image/{img_ext}" if img_ext != "jpeg" else "image/jpeg"
    path = f"doc-{document_id}/p{page_number}_img{img_index}.{img_ext}"

    url = f"{SUPABASE_URL}/storage/v1/object/{SUPABASE_BUCKET}/{path}"
    headers = {
        "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
        "Content-Type": content_type,
        "x-upsert": "true",
    }

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(url, content=img_bytes, headers=headers)
        if resp.status_code not in (200, 201):
            raise HTTPException(
                status_code=500,
                detail=f"Error subiendo imagen a Supabase: {resp.text}"
            )

    public_url = f"{SUPABASE_URL}/storage/v1/object/public/{SUPABASE_BUCKET}/{path}"
    return public_url


# ─── Endpoints ───────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "service": "IACertext PDF Processor"}


@app.post("/process-pdf")
async def process_pdf(
    file: UploadFile = File(...),
    document_id: str = Form(...),
    filename: str = Form(...),
):
    """
    Endpoint principal: recibe el PDF binario, extrae texto e imágenes,
    sube imágenes a Supabase Storage y retorna chunks estructurados.
    """
    if not file.filename.endswith(".pdf") and file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Solo se aceptan archivos PDF")

    pdf_bytes = await file.read()

    # 1. Extraer texto e imágenes con PyMuPDF
    pages_data = extract_text_and_images(pdf_bytes, document_id, filename)

    # 2. Mapear texto cerca de imágenes
    mapped_pages = map_text_to_images(pages_data)

    # 3. Crear chunks inteligentes
    chunks = chunk_by_pages(mapped_pages, filename, document_id)

    # 4. Subir imágenes a Supabase Storage y obtener URLs públicas
    image_urls = {}  # "page_imgindex" → URL pública

    for page in mapped_pages:
        for img_index, img in enumerate(page["images"]):
            key = f"{page['page_number']}_{img_index}"
            try:
                public_url = await upload_image_to_supabase(
                    img_b64=img["base64"],
                    img_ext=img.get("ext", "png"),
                    document_id=document_id,
                    page_number=page["page_number"],
                    img_index=img_index,
                )
                image_urls[key] = public_url
            except Exception as e:
                image_urls[key] = None

    # 5. Enriquecer chunks con URLs de imágenes
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
                })

        chunk["image_path"] = chunk_images[0]["url"] if chunk_images else None
        chunk["image_description"] = None  # Se describirá con GPT-4o en n8n
        chunk.pop("images", None)  # No enviar base64 en la respuesta

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
    """
    Proxy al RPC match_chunks de Supabase pgvector.
    Recibe el embedding de la pregunta y retorna chunks similares.
    """
    query_embedding = body.get("query_embedding")
    match_count = body.get("match_count", 5)
    doc_ids = body.get("doc_ids")

    if not query_embedding:
        raise HTTPException(status_code=400, detail="query_embedding requerido")

    payload = {
        "query_embedding": query_embedding,
        "match_count": match_count,
    }
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
            raise HTTPException(
                status_code=500,
                detail=f"Error en Supabase RPC: {resp.text}"
            )
        return resp.json()
