import time
import queue
import threading
import uuid
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

from config import (
    ENTROPY_DECAY_RATE,
    PRUNING_THRESHOLD,
    SAVE_INTERVAL_SECONDS,
    DREAMER_BATCH_SIZE,
    EMBEDDING_MODEL_NAME
)
from modules.utils import logger
from modules.memory_store import SharedMemoryManager

class MemoryDreamer(threading.Thread):
    def __init__(self, memory_manager: SharedMemoryManager, stm_queue: queue.Queue):
        super().__init__(name="DreamerThread", daemon=True)
        self.memory = memory_manager
        self.stm_queue = stm_queue
        self.stop_event = threading.Event()
        self.last_save_time = time.time()
        
        if SentenceTransformer:
            # Use config value for consistency with Brain
            self.encoder = SentenceTransformer(EMBEDDING_MODEL_NAME, device='cpu') 
        else:
            self.encoder = None
            logger.warning("SentenceTransformer not found. Dreamer cannot vectorize new memories.")

    def run(self):
        logger.info("Dreamer Thread Started.")
        processed_count = 0
        
        while not self.stop_event.is_set():
            try:
                item = self.stm_queue.get(timeout=1.0) 
                
                if item == "POISON_PILL":
                    logger.info("Poison Pill received. Saving state and exiting.")
                    self.memory.save_snapshot()
                    break
                
                self.consolidate_memory(item)
                processed_count += 1
                
                if processed_count >= DREAMER_BATCH_SIZE:
                    self.run_maintenance()
                    processed_count = 0

            except queue.Empty:
                if time.time() - self.last_save_time > SAVE_INTERVAL_SECONDS:
                    logger.info("Auto-Saving Memory Snapshot...")
                    self.memory.save_snapshot()
                    self.last_save_time = time.time()
                continue
            except Exception as e:
                logger.error(f"Dreamer Error: {e}")
        
        logger.info("Dreamer Thread Exited.")

    def consolidate_memory(self, item: dict):
        if not self.encoder:
            return
            
        try:
            text = f"Query: {item.get('query')} | Context: {item.get('context_used')}"
            mem_uuid = str(uuid.uuid4())
            vector = self.encoder.encode(text)
            
            # memory_store.add_memory will handle numpy conversion if needed
            self.memory.add_memory(mem_uuid, text, vector)
            
        except Exception as e:
            logger.error(f"Consolidation Failed: {e}")

    def run_maintenance(self):
        nodes_to_check = []
        with self.memory.lock:
            nodes_to_check = list(self.memory.graph.nodes())
            
        if not nodes_to_check:
            return

        nodes_to_prune = []
        import random
        sample_size = min(len(nodes_to_check), 50)
        sample_nodes = random.sample(nodes_to_check, sample_size)
        
        current_time = time.time()
        
        for node_uuid in sample_nodes:
            # READ
            with self.memory.lock:
                if not self.memory.graph.has_node(node_uuid): continue
                attributes = self.memory.graph.nodes[node_uuid]
                energy = attributes.get('energy', 10.0)
                last_access = attributes.get('last_access', current_time)
            
            # LOGIC
            hours_elapsed = (current_time - last_access) / 3600.0
            decay = ENTROPY_DECAY_RATE * hours_elapsed
            new_energy = energy - decay
            
            # WRITE Update or Prune
            with self.memory.lock:
                if self.memory.graph.has_node(node_uuid):
                    if new_energy <= PRUNING_THRESHOLD:
                         nodes_to_prune.append(node_uuid)
                    else:
                        self.memory.graph.nodes[node_uuid]['energy'] = new_energy
        
        # Prune
        for uuid in nodes_to_prune:
            self.memory.remove_node(uuid)
            logger.info(f"Pruned Node {uuid} due to low energy.")
