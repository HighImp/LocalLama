from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, StorageContext, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from pathlib import Path
import time
import os
from typing import Optional
import logging

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class DataLama:
    def __init__(self,
                 persistent_dir: Path,
                 path_to_data: Path,
                 model_name: str = "llama3.1",
                 request_timeout: float = 120.0):
        """
        Initializes the DataLama class. Either loads a saved index from persistent storage or creates a new one.

        :param persistent_dir: Path to the directory where the persistent index data is stored
        :param path_to_data: Path to the directory containing the data to load
        :param model_name: Name of the model to use (default: "llama3.1")
        :param request_timeout: Timeout for model requests in seconds (default: 120.0)
        """
        
        # Initialize embedding and LLM model
        Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
        
        Settings.llm = Ollama(model=model_name, request_timeout=request_timeout)
        # Settings.embed_model = OllamaEmbedding(model_name=model_name)

        # Measure time for loading or reindexing
        start_time = time.perf_counter()

        # Check if persistent data exists and reindexing is needed
        if not persistent_dir.exists() or self._needs_reindexing(persistent_dir, path_to_data):
            # Load documents from data directory and create a new index
            logging.info(f"Creating new index for data in {path_to_data}")
            documents = SimpleDirectoryReader(path_to_data).load_data(show_progress=True)
            index = VectorStoreIndex.from_documents(documents)
            # Persist the index for future use
            index.storage_context.persist(persistent_dir)
            logging.info(f"Index persisted in {persistent_dir}")
        else:
            # Load existing index from persistent storage
            logging.info(f"Loading existing index from {persistent_dir}")
            index = load_index_from_storage(StorageContext.from_defaults(persist_dir=persistent_dir))

        # Measure time taken for data loading and indexing
        self._load_duration = time.perf_counter() - start_time
        logging.info(f"Indexing/Loading completed in {self._load_duration:.2f} seconds")
        
        # Set up query engine with top 4 similar results
        self.query_engine = index.as_query_engine(similarity_top_k=4)

        # Store the durations of the query responses
        self._query_durations: list = []
        

    def get_load_duration(self) -> float:
        """
        Returns the duration of the data loading or indexing process.
        """
        return self._load_duration
    
    def get_query_durations(self) -> tuple:
        """
        Returns the list of query durations.
        """
        return tuple(self._query_durations)

    def query(self, msg: str) -> str:
        """
        Queries the index and returns a response.

        :param msg: The query message as a string
        :return: The response from the query engine
        """
        start_time = time.perf_counter()  # Measure time for query response
        logging.info(f"Processing query: {msg}")
        response = self.query_engine.query(msg)
        query_duration = round(time.perf_counter() - start_time, 1)
        self._query_durations.append(query_duration)
        logging.info(f"Query completed in {query_duration:.1f} seconds")
        return response

    def _needs_reindexing(self, persistent_dir: Path, data_dir: Path) -> bool:
        """
        Checks if reindexing is required by comparing modification times of data files 
        with the persistent storage modification time.

        :param persistent_dir: Path to the directory containing persistent index files
        :param data_dir: Path to the directory containing data files
        :return: True if reindexing is required, False otherwise
        """
        # Get the modification time of the persistent storage directory
        persistent_time = max(os.path.getmtime(root) for root, _, _ in os.walk(persistent_dir))
        
        # Get the modification times of all files in the data directory
        for root, _, files in os.walk(data_dir):
            for file in files:
                file_path = Path(root) / file
                if os.path.getmtime(file_path) > persistent_time:
                    logging.info(f"Reindexing required: {file_path} was modified after persistent data.")
                    return True
        return False


if __name__ == "__main__":
    # Measure the total time for loading and querying
    overall_start = time.perf_counter()

    persistent_dir = Path(__file__).parent.parent / "persistent"
    data_dir = Path(__file__).parent.parent / "data"

    # Initialize the chatbot with the necessary paths and model settings
    chat_bot = DataLama(persistent_dir=persistent_dir, path_to_data=data_dir, model_name="llama3.1", request_timeout=240.0)
    
    # Perform a query and measure its time
    response = chat_bot.query("Who own the one ring?")
    
    overall_duration = time.perf_counter() - overall_start
    logging.info(f"Total execution time: {overall_duration:.2f} seconds")
    
    print(response)