import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import json
import os
import logging
import hashlib

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logging.warning("sentence-transformers not available, using fallback text similarity")

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logging.warning("faiss not available, using simple similarity search")

from memory import ConversationMemory

class MultimodalMemoryEmbeddings:
    def __init__(self):
        self.embedding_dim = 384
        self.memory_store = {}
        self.embedding_file = "multimodal_embeddings.jsonl"
        
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.text_encoder = SentenceTransformer('all-MiniLM-L6-v2')
            except Exception as e:
                logging.error(f"Failed to load sentence transformer: {e}")
                self.text_encoder = None
        else:
            self.text_encoder = None
        
        if FAISS_AVAILABLE and self.text_encoder:
            self.text_index = faiss.IndexFlatIP(self.embedding_dim)
            self.image_index = faiss.IndexFlatIP(self.embedding_dim)
        else:
            self.text_index = None
            self.image_index = None
        
        self._load_embeddings()
        
    def store_with_embedding(self, user_id: str, content: str, content_type: str, metadata: Dict = None) -> str:
        """Store content with vector embedding for semantic search."""
        try:
            if metadata is None:
                metadata = {}
            
            embedding = self._generate_embedding(content, content_type, metadata)
            
            memory_id = f"{user_id}_{int(datetime.utcnow().timestamp())}_{len(self.memory_store)}"
            
            memory_entry = {
                'memory_id': memory_id,
                'user_id': user_id,
                'content': content,
                'content_type': content_type,
                'metadata': metadata,
                'timestamp': datetime.utcnow().isoformat(),
                'embedding': embedding.tolist() if embedding is not None else None
            }
            
            self.memory_store[memory_id] = memory_entry
            
            if self.text_index is not None and embedding is not None:
                if content_type in ['text', 'conversation']:
                    self.text_index.add(np.array([embedding]))
                elif content_type in ['image', 'multimodal']:
                    self.image_index.add(np.array([embedding]))
            
            self._save_embedding(memory_entry)
            
            logging.info(f"Stored multimodal memory: {memory_id} ({content_type})")
            return memory_id
            
        except Exception as e:
            logging.error(f"Error storing multimodal embedding: {e}")
            return ""
    
    def semantic_search(self, query: str, user_id: str, content_type: str = 'text', k: int = 5) -> List[Dict]:
        """Perform semantic search across stored memories."""
        try:
            if not self.text_encoder:
                return self._fallback_search(query, user_id, content_type, k)
            
            query_embedding = self._generate_embedding(query, 'text')
            if query_embedding is None:
                return self._fallback_search(query, user_id, content_type, k)
            
            results = []
            
            if self.text_index is not None:
                if content_type in ['text', 'conversation']:
                    scores, indices = self.text_index.search(np.array([query_embedding]), k)
                elif content_type in ['image', 'multimodal']:
                    scores, indices = self.image_index.search(np.array([query_embedding]), k)
                else:
                    text_scores, text_indices = self.text_index.search(np.array([query_embedding]), k//2)
                    image_scores, image_indices = self.image_index.search(np.array([query_embedding]), k//2)
                    scores = np.concatenate([text_scores[0], image_scores[0]])
                    indices = np.concatenate([text_indices[0], image_indices[0]])
                
                for score, idx in zip(scores, indices):
                    if idx != -1:  # Valid result
                        memory_entries = [entry for entry in self.memory_store.values() 
                                        if entry['user_id'] == user_id]
                        if idx < len(memory_entries):
                            result = memory_entries[idx].copy()
                            result['similarity_score'] = float(score)
                            results.append(result)
            else:
                results = self._manual_similarity_search(query_embedding, user_id, content_type, k)
            
            results.sort(key=lambda x: x.get('similarity_score', 0), reverse=True)
            return results[:k]
            
        except Exception as e:
            logging.error(f"Error in semantic search: {e}")
            return self._fallback_search(query, user_id, content_type, k)
    
    def get_cross_modal_context(self, user_id: str, current_input: str, max_results: int = 3) -> List[Dict]:
        """Get relevant context across different modalities."""
        try:
            text_results = self.semantic_search(current_input, user_id, 'text', max_results)
            image_results = self.semantic_search(current_input, user_id, 'image', max_results)
            
            all_results = text_results + image_results
            seen_ids = set()
            unique_results = []
            
            for result in all_results:
                memory_id = result.get('memory_id')
                if memory_id and memory_id not in seen_ids:
                    seen_ids.add(memory_id)
                    unique_results.append(result)
            
            unique_results.sort(key=lambda x: (
                x.get('similarity_score', 0) * 0.7 + 
                self._get_recency_score(x.get('timestamp', '')) * 0.3
            ), reverse=True)
            
            return unique_results[:max_results]
            
        except Exception as e:
            logging.error(f"Error getting cross-modal context: {e}")
            return []
    
    def _generate_embedding(self, content: str, content_type: str, metadata: Dict = None) -> Optional[np.ndarray]:
        """Generate embedding for content based on type."""
        try:
            if not self.text_encoder:
                return None
            
            if content_type == 'image':
                if metadata and 'description' in metadata:
                    text_content = metadata['description']
                elif metadata and 'summary' in metadata:
                    text_content = metadata['summary']
                else:
                    text_content = f"Image content: {content[:100]}"
            elif content_type == 'multimodal':
                text_parts = [content]
                if metadata:
                    if 'image_description' in metadata:
                        text_parts.append(f"Image: {metadata['image_description']}")
                    if 'context' in metadata:
                        text_parts.append(f"Context: {metadata['context']}")
                text_content = " ".join(text_parts)
            else:
                text_content = content
            
            embedding = self.text_encoder.encode([text_content])[0]
            return embedding
            
        except Exception as e:
            logging.error(f"Error generating embedding: {e}")
            return None
    
    def _manual_similarity_search(self, query_embedding: np.ndarray, user_id: str, content_type: str, k: int) -> List[Dict]:
        """Manual similarity search when FAISS is not available."""
        results = []
        
        for memory_entry in self.memory_store.values():
            if memory_entry['user_id'] != user_id:
                continue
            
            if content_type != 'all' and memory_entry['content_type'] != content_type:
                continue
            
            stored_embedding = memory_entry.get('embedding')
            if stored_embedding:
                stored_embedding = np.array(stored_embedding)
                similarity = np.dot(query_embedding, stored_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(stored_embedding)
                )
                
                result = memory_entry.copy()
                result['similarity_score'] = float(similarity)
                results.append(result)
        
        return results
    
    def _fallback_search(self, query: str, user_id: str, content_type: str, k: int) -> List[Dict]:
        """Fallback search using simple text matching."""
        results = []
        query_lower = query.lower()
        
        for memory_entry in self.memory_store.values():
            if memory_entry['user_id'] != user_id:
                continue
            
            if content_type != 'all' and memory_entry['content_type'] != content_type:
                continue
            
            content = memory_entry['content'].lower()
            query_words = set(query_lower.split())
            content_words = set(content.split())
            overlap = len(query_words & content_words)
            
            if overlap > 0:
                similarity = overlap / len(query_words | content_words)
                result = memory_entry.copy()
                result['similarity_score'] = similarity
                results.append(result)
        
        results.sort(key=lambda x: x['similarity_score'], reverse=True)
        return results[:k]
    
    def _get_recency_score(self, timestamp_str: str) -> float:
        """Calculate recency score (0-1, higher = more recent)."""
        try:
            timestamp = datetime.fromisoformat(timestamp_str)
            age_hours = (datetime.utcnow() - timestamp).total_seconds() / 3600
            return np.exp(-age_hours / 24)
        except:
            return 0.0
    
    def _save_embedding(self, memory_entry: Dict):
        """Save embedding to persistent storage."""
        try:
            with open(self.embedding_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(memory_entry, default=str) + '\n')
        except Exception as e:
            logging.error(f"Error saving embedding: {e}")
    
    def _load_embeddings(self):
        """Load existing embeddings from storage."""
        try:
            if os.path.exists(self.embedding_file):
                with open(self.embedding_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                entry = json.loads(line)
                                memory_id = entry.get('memory_id')
                                if memory_id:
                                    self.memory_store[memory_id] = entry
                                    
                                    if (self.text_index is not None and 
                                        entry.get('embedding') and 
                                        self.text_encoder):
                                        embedding = np.array(entry['embedding'])
                                        if entry['content_type'] in ['text', 'conversation']:
                                            self.text_index.add(np.array([embedding]))
                                        elif entry['content_type'] in ['image', 'multimodal']:
                                            self.image_index.add(np.array([embedding]))
                            except json.JSONDecodeError as e:
                                logging.warning(f"Skipping corrupted embedding line: {e}")
                                continue
                                
                logging.info(f"Loaded {len(self.memory_store)} multimodal embeddings")
        except Exception as e:
            logging.error(f"Error loading embeddings: {e}")
    
    def get_embedding_stats(self) -> Dict:
        """Get statistics about stored embeddings."""
        stats = {
            'total_embeddings': len(self.memory_store),
            'by_content_type': {},
            'by_user': {},
            'has_faiss': FAISS_AVAILABLE and self.text_index is not None,
            'has_sentence_transformers': SENTENCE_TRANSFORMERS_AVAILABLE and self.text_encoder is not None
        }
        
        for entry in self.memory_store.values():
            content_type = entry.get('content_type', 'unknown')
            user_id = entry.get('user_id', 'unknown')
            
            stats['by_content_type'][content_type] = stats['by_content_type'].get(content_type, 0) + 1
            stats['by_user'][user_id] = stats['by_user'].get(user_id, 0) + 1
        
        return stats
