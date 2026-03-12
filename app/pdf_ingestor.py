import re
import io
import logging
from pypdf import PdfReader
from app.chunking import chunk_text_parent_child
from app.database import insert_document, insert_parent_chunk, create_source, finalize_source, source_already_ingested

logger = logging.getLogger(__name__)


def extract_text_from_pdf(file_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(file_bytes))
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)
    return "\n\n".join(pages)


def normalize_text(text: str) -> str:
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'^\s*-?\s*\d+\s*-?\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'-\n(\w)', r'\1', text)
    return text.strip()


def ingest_pdf(filepath: str) -> dict:
    import os
    filename = os.path.basename(filepath)
    title_base = filename.replace(".pdf", "").replace("_", " ").replace("-", " ").title()

    if source_already_ingested(filename):
        logger.info(f"Skipping already ingested: {filename}")
        return {"skipped": True, "filename": filename}

    source_id = create_source(filename, title_base)
    try:
        with open(filepath, "rb") as f:
            file_bytes = f.read()

        raw_text = extract_text_from_pdf(file_bytes)
        clean_text = normalize_text(raw_text)
        chunk_pairs = chunk_text_parent_child(clean_text)

        child_count = 0
        for i, pair in enumerate(chunk_pairs):
            parent_id = insert_parent_chunk(
                url=f"pdf://{filename}",
                title=f"{title_base} (section {i+1})",
                content=pair["parent"],
                source_id=source_id
            )
            for j, child in enumerate(pair["children"]):
                insert_document(
                    url=f"pdf://{filename}",
                    title=f"{title_base} (section {i+1}, part {j+1})",
                    content=child,
                    source_id=source_id,
                    parent_id=parent_id
                )
                child_count += 1

        finalize_source(source_id, child_count, "done")
        logger.info(f"Ingested '{title_base}' — {len(chunk_pairs)} sections, {child_count} child chunks")
        return {"title": title_base, "chunks_inserted": child_count}

    except Exception as e:
        finalize_source(source_id, 0, "error")
        logger.error(f"Failed to ingest {filename}: {e}")
        raise
