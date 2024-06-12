import logging

from dotenv import load_dotenv
import sys
import os

# Assuming your_script.py is located in the app/engine directory
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
sys.path.append(project_root)

from app.engine.constants import DATA_DIR, STORAGE_DIR
from app.engine.context import create_service_context
from app.engine.loader import get_documents

load_dotenv()

from llama_index import (
    SimpleDirectoryReader,
    VectorStoreIndex,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


def generate_datasource(service_context):
    logger.info("Creating new index")
    # load the documents and create the index
    documents = get_documents()
    VectorStoreIndex.from_documents(documents, service_context=service_context)
    # store it for later
    logger.info(f"Finished creating new index. Stored in AstraDB")


if __name__ == "__main__":
    service_context = create_service_context()
    generate_datasource(service_context)
