import faiss
import numpy as np
import os
import pickle
from typing import List, Tuple

class VectorStore:
    def __init__(self, embedding_size: int = 512, index_path: str = "data/faiss_index.bin", mapping_path: str = "data/mapping.pkl"):
        self.embedding_size = embedding_size
        self.index_path = index_path
        self.mapping_path = mapping_path
        
        # Mapping from faiss internal integer ID to real User ID (string)
        self.id_to_user_id = {}
        self.user_id_to_id = {}
        
        self.index = self._load_or_create_index()
        self._load_mapping()

    def _load_or_create_index(self):
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        if os.path.exists(self.index_path):
            print(f"Loading FAISS index from {self.index_path}")
            return faiss.read_index(self.index_path)
        else:
            print("Creating new FAISS Index")
            # IndexFlatL2 for exact search (L2 distance), use Inner Product (Cosine) if normalized
            return faiss.IndexFlatL2(self.embedding_size)

    def _load_mapping(self):
        if os.path.exists(self.mapping_path):
            with open(self.mapping_path, 'rb') as f:
                data = pickle.load(f)
                self.id_to_user_id = data.get('id_to_user_id', {})
                self.user_id_to_id = data.get('user_id_to_id', {})

    def _save_mapping(self):
        with open(self.mapping_path, 'wb') as f:
            pickle.dump({
                'id_to_user_id': self.id_to_user_id,
                'user_id_to_id': self.user_id_to_id
            }, f)

    def _save_index(self):
        faiss.write_index(self.index, self.index_path)

    def add_user(self, user_id: str, embedding: List[float]):
        """Add a new user's face embedding to the database."""
        emb_array = np.array([embedding], dtype=np.float32)
        
        # faiss.IndexFlatL2 does not support add_with_ids directly, 
        # so we rely on the internal sequential ID given by index.ntotal
        internal_id = self.index.ntotal
        
        self.index.add(emb_array)
        
        self.id_to_user_id[internal_id] = user_id
        self.user_id_to_id[user_id] = internal_id
        
        self._save_index()
        self._save_mapping()
        
        return True

    def search(self, embedding: List[float], k: int = 1, threshold: float = 200.0) -> List[Tuple[str, float]]:
        """
        Search for the top k closest faces.
        Returns a list of (user_id, distance) tuples.
        """
        if self.index.ntotal == 0:
            return []
            
        emb_array = np.array([embedding], dtype=np.float32)
        distances, indices = self.index.search(emb_array, k)
        
        results = []
        for i, internal_id in enumerate(indices[0]):
            if internal_id != -1 and internal_id in self.id_to_user_id:
                dist = float(distances[0][i])
                if dist <= threshold:
                    results.append((self.id_to_user_id[internal_id], dist))
                    
        return results
