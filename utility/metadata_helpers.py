import faiss
import json
import os

class MetaDataHelpers:
    # FAISS index file and metadata file paths
    INDEX_FILE = "local_index.faiss"
    METADATA_FILE = "metadata.json"

    metadata = []  # In-memory metadata

    @staticmethod
    def save_metadata(metadata):
        with open(MetaDataHelpers.METADATA_FILE, "w") as f:
            json.dump(metadata, f)

    @staticmethod
    def load_metadata():
        if os.path.exists(MetaDataHelpers.METADATA_FILE):
            with open(MetaDataHelpers.METADATA_FILE, "r") as f:
                return json.load(f)
        else:
            return []

    @staticmethod
    def save_index(index):
        faiss.write_index(index, MetaDataHelpers.INDEX_FILE)

    @staticmethod
    def load_index(dimension):
        if os.path.exists(MetaDataHelpers.INDEX_FILE):
            return faiss.read_index(MetaDataHelpers.INDEX_FILE)
        else:
            # Create a new FAISS index with the specified dimension
            return faiss.IndexFlatL2(dimension)

    @staticmethod
    def reset_index_and_metadata():
        MetaDataHelpers.metadata = []  # Clear in-memory metadata
        if os.path.exists(MetaDataHelpers.INDEX_FILE):
            os.remove(MetaDataHelpers.INDEX_FILE)
        if os.path.exists(MetaDataHelpers.METADATA_FILE):
            os.remove(MetaDataHelpers.METADATA_FILE)
