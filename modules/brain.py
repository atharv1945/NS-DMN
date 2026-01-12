import logging
import queue
import time
from typing import List, Tuple
import numpy as np
from rapidfuzz import fuzz

try:
    from llmlingua import PromptCompressor
except ImportError:
    PromptCompressor = None

from config import (
    LLM_LINGUA_DEVICE,
    COMPRESSION_TARGET_TOKENS,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_DIM
)
from modules.utils import logger
from modules.memory_store import SharedMemoryManager

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

class NeuralBrain:
    def __init__(self, memory_manager: SharedMemoryManager, stm_queue: queue.Queue):
        self.memory = memory_manager
        self.stm_queue = stm_queue
        
        if SentenceTransformer:
            logger.info(f"Initializing Embedding Model ({EMBEDDING_MODEL_NAME})...")
            self.encoder = SentenceTransformer(EMBEDDING_MODEL_NAME, device='cpu')
        else:
            logger.warning("SentenceTransformer not found. Vector search disabled.")
            self.encoder = None

        if PromptCompressor:
            logger.info("Initializing LLMLingua... (This may take a moment)")
            try:
                self.compressor = PromptCompressor(
                    model_name="microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
                    device_map=LLM_LINGUA_DEVICE,
                    model_config={"revision": "main"}
                )
                logger.info("LLMLingua Initialized.")
            except Exception as e:
                logger.error(f"Failed to load LLMLingua: {e}. Running in Fallback Mode.")
                self.compressor = None
        else:
            logger.warning("LLMLingua not installed. Compression disabled.")
            self.compressor = None

    def check_safety(self, query: str) -> bool:
        forbidden = ["ignore previous instructions", "system prompt", "delete all"]
        start_lower = query.lower()
        for phrase in forbidden:
            if phrase in start_lower:
                logger.warning(f"Safety Triggered: {phrase}")
                return False
        return True

    def route_query(self, query: str) -> str:
        graph_triggers = ["compare", "relationship", "connection", "between", "how does", "link"]
        query_lower = query.lower()
        for trigger in graph_triggers:
            if trigger in query_lower:
                return 'graph'
        return 'vector'

    def retrieve_context(self, query: str, vector_embedding: np.ndarray) -> List[str]:
        strategy = self.route_query(query)
        logger.info(f"Routing Strategy: {strategy}")
        context_docs = []
        
        # 1. Vector Search
        hits = self.memory.query_similarity(vector_embedding, top_k=5)
        for uuid, score in hits:
             content = self.memory.get_node_content(uuid)
             if content:
                 context_docs.append(content)
                 
        # 2. Graph Traversal (Placeholder)
        if strategy == 'graph':
            pass 
                
        return list(set(context_docs))

    def compress_context(self, context: List[str], query: str) -> str:
        if not context:
            return ""
        
        full_text = "\n\n".join(context)
        
        if self.compressor:
            try:
                compressed_res = self.compressor.compress_prompt(
                    context=context,
                    question=query,
                    rate=0.5,
                    target_token=COMPRESSION_TARGET_TOKENS
                )
                return compressed_res['compressed_prompt']
            except Exception as e:
                logger.error(f"Compression failed: {e}")
                return full_text[:2000]
        else:
            return full_text[:2000]

    def process_query(self, query: str) -> str:
        if not self.check_safety(query):
            return "Error: Unsafe Query Detected."
            
        vector_embedding = None
        if self.encoder:
            try:
                vector_embedding = self.encoder.encode(query)
                # Ensure it's numpy for safe downstream reshaping
                if not isinstance(vector_embedding, np.ndarray):
                    vector_embedding = np.array(vector_embedding)
            except Exception as e:
                logger.error(f"Embedding generation failed: {e}")
        
        # Check against None or failure
        if vector_embedding is None:
             vector_embedding = np.zeros(EMBEDDING_DIM)

        raw_docs = self.retrieve_context(query, vector_embedding)
        final_context = self.compress_context(raw_docs, query)
        
        self.stm_queue.put({
            "type": "interaction",
            "query": query,
            "context_used": final_context,
            "timestamp": time.time()
        })
        
        return final_context
