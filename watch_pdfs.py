import time
import os
import logging
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from app.pdf_ingestor import ingest_pdf

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

WATCH_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pdfs")


class PDFHandler(FileSystemEventHandler):
    def on_created(self, event):
        if event.is_directory:
            return
        if event.src_path.lower().endswith(".pdf"):
            logger.info(f"New PDF detected: {event.src_path}")
            time.sleep(1)  # wait for file to finish copying
            try:
                result = ingest_pdf(event.src_path)
                if result.get("skipped"):
                    logger.info(f"Already ingested, skipping.")
                else:
                    logger.info(f"Done: '{result['title']}' — {result['chunks_inserted']} chunks stored.")
            except Exception as e:
                logger.error(f"Ingestion failed: {e}")


if __name__ == "__main__":
    os.makedirs(WATCH_FOLDER, exist_ok=True)
    logger.info(f"Watching folder: {WATCH_FOLDER}")
    logger.info("Drop PDF files into the 'pdfs/' folder to ingest them. Press Ctrl+C to stop.")

    handler = PDFHandler()
    observer = Observer()
    observer.schedule(handler, WATCH_FOLDER, recursive=False)
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
