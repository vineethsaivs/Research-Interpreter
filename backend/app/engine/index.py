import os
import logging
from dotenv import load_dotenv
from llama_index.vector_stores import AstraDBVectorStore
from app.engine.loader import get_documents
import os
from llama_index import (
    StorageContext,
    SimpleDirectoryReader,
    VectorStoreIndex
)

from app.engine.constants import STORAGE_DIR
from app.engine.context import create_service_context

def get_chat_engine():
    service_context = create_service_context()
    # check if storage already exists
    if not os.path.exists(STORAGE_DIR):
        raise Exception(
            "StorageContext is empty - call 'python app/engine/generate.py' to generate the storage first"
        )
    logger = logging.getLogger("uvicorn")
    # load the existing index
    logger.info(f"Loading index from AstraDB...")
    # Create a vector store instance
    ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
    ASTRA_DB_API_ENDPOINT = os.getenv("ASTRA_DB_API_ENDPOINT")
    astra_db_store = AstraDBVectorStore(
        token=ASTRA_DB_APPLICATION_TOKEN,
        api_endpoint=ASTRA_DB_API_ENDPOINT,
        collection_name="base",
        embedding_dimension=1536,
    )

    storage_context = StorageContext.from_defaults(vector_store=astra_db_store)
    documents = SimpleDirectoryReader(r"data").load_data()
    index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
    logger.info(f"Finished loading index from AstraDB")
    return index.as_chat_engine(similarity_top_k=5, chat_mode="condense_plus_context")

