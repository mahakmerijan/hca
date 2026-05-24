"""
Vector Memory Store
===================
Evaluates Pinecone vs Weaviate at runtime, picks the faster/more accurate one,
and wraps both behind a unified interface used by the Digital Twin engine.

Architecture decision from architecture.md:
  "Among them check the inference – which one is performing faster and after
   evaluation which one is giving more accurate result – based on that choose
   the proper vector DB"
"""

import json
import os
import time
import hashlib
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv

load_dotenv()

# ── Optional imports ──────────────────────────────────────────────
try:
    from pinecone import Pinecone
    PINECONE_AVAILABLE = True
except ImportError:
    PINECONE_AVAILABLE = False

try:
    import weaviate
    from weaviate.classes.init import Auth
    WEAVIATE_AVAILABLE = True
except ImportError:
    WEAVIATE_AVAILABLE = False

try:
    from google import genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False


def _embed_text(text: str) -> List[float]:
    """
    Generate a text embedding. Uses Gemini text-embedding-004 if available,
    falls back to a deterministic hash-based pseudo-embedding for local dev.
    """
    api_key = os.getenv("GOOGLE_API_KEY", "")
    if GENAI_AVAILABLE and api_key:
        try:
            client = genai.Client(api_key=api_key)
            result = client.models.embed_content(
                model="models/text-embedding-004",
                contents=text,
            )
            return result.embeddings[0].values
        except Exception as e:
            print(f"[VectorStore] Embedding error: {e}, using fallback")

    # Deterministic fallback: 768-dim pseudo-embedding from hash
    h = hashlib.sha256(text.encode()).hexdigest()
    vec = []
    for i in range(0, min(len(h), 128), 2):
        val = int(h[i:i+2], 16) / 255.0 - 0.5
        vec.append(val)
    # Pad/truncate to 768
    while len(vec) < 768:
        vec.extend(vec[:min(64, 768 - len(vec))])
    return vec[:768]


class PineconeStore:
    """Wrapper around Pinecone for twin memory storage."""

    def __init__(self):
        self.available = False
        api_key = os.getenv("PINECONE_API_KEY", "")
        self.index_name = os.getenv("PINECONE_INDEX", "digital-twin-memory")
        if not PINECONE_AVAILABLE or not api_key:
            return
        try:
            self.pc = Pinecone(api_key=api_key)
            existing = [i.name for i in self.pc.list_indexes()]
            if self.index_name not in existing:
                self.pc.create_index(
                    name=self.index_name,
                    dimension=768,
                    metric="cosine",
                    spec={"serverless": {"cloud": "aws", "region": "us-east-1"}},
                )
            self.index = self.pc.Index(self.index_name)
            self.available = True
            print(f"[PineconeStore] Connected — index={self.index_name}")
        except Exception as e:
            print(f"[PineconeStore] Init error: {e}")

    def upsert(self, doc_id: str, text: str, metadata: dict) -> bool:
        if not self.available:
            return False
        try:
            vec = _embed_text(text)
            self.index.upsert(vectors=[{
                "id": doc_id,
                "values": vec,
                "metadata": {**metadata, "text": text[:500]},
            }])
            return True
        except Exception as e:
            print(f"[PineconeStore] Upsert error: {e}")
            return False

    def query(self, text: str, top_k: int = 5) -> List[Dict]:
        if not self.available:
            return []
        try:
            vec = _embed_text(text)
            results = self.index.query(vector=vec, top_k=top_k, include_metadata=True)
            return [
                {"id": m["id"], "score": m["score"], "text": m["metadata"].get("text", "")}
                for m in results.get("matches", [])
            ]
        except Exception as e:
            print(f"[PineconeStore] Query error: {e}")
            return []

    def latency_test(self, text: str = "test query for latency benchmark") -> float:
        if not self.available:
            return float("inf")
        start = time.perf_counter()
        self.query(text, top_k=1)
        return time.perf_counter() - start


class WeaviateStore:
    """Wrapper around Weaviate for twin memory storage."""

    def __init__(self):
        self.available = False
        url = os.getenv("WEAVIATE_URL", "")
        api_key = os.getenv("WEAVIATE_API_KEY", "")
        self.collection_name = "TwinMemory"
        if not WEAVIATE_AVAILABLE or not url:
            return
        try:
            auth = Auth.api_key(api_key) if api_key else None
            self.client = weaviate.connect_to_weaviate_cloud(
                cluster_url=url,
                auth_credentials=auth,
            )
            # Ensure collection exists
            if not self.client.collections.exists(self.collection_name):
                self.client.collections.create(
                    name=self.collection_name,
                    properties=[
                        weaviate.classes.config.Property(name="doc_id", data_type=weaviate.classes.config.DataType.TEXT),
                        weaviate.classes.config.Property(name="text", data_type=weaviate.classes.config.DataType.TEXT),
                        weaviate.classes.config.Property(name="meta_json", data_type=weaviate.classes.config.DataType.TEXT),
                    ],
                )
            self.collection = self.client.collections.get(self.collection_name)
            self.available = True
            print(f"[WeaviateStore] Connected — collection={self.collection_name}")
        except Exception as e:
            print(f"[WeaviateStore] Init error: {e}")

    def upsert(self, doc_id: str, text: str, metadata: dict) -> bool:
        if not self.available:
            return False
        try:
            vec = _embed_text(text)
            self.collection.data.insert(
                properties={"doc_id": doc_id, "text": text[:500], "meta_json": json.dumps(metadata)},
                vector=vec,
            )
            return True
        except Exception as e:
            print(f"[WeaviateStore] Upsert error: {e}")
            return False

    def query(self, text: str, top_k: int = 5) -> List[Dict]:
        if not self.available:
            return []
        try:
            vec = _embed_text(text)
            results = self.collection.query.near_vector(near_vector=vec, limit=top_k)
            return [
                {"id": obj.properties.get("doc_id", ""), "score": 0.9, "text": obj.properties.get("text", "")}
                for obj in results.objects
            ]
        except Exception as e:
            print(f"[WeaviateStore] Query error: {e}")
            return []

    def latency_test(self, text: str = "test query for latency benchmark") -> float:
        if not self.available:
            return float("inf")
        start = time.perf_counter()
        self.query(text, top_k=1)
        return time.perf_counter() - start

    def close(self):
        if self.available:
            try:
                self.client.close()
            except Exception:
                pass


class VectorMemoryStore:
    """
    Unified Vector DB interface.
    Auto-selects Pinecone vs Weaviate based on latency + availability.
    Falls back to in-memory dict if neither is configured.
    """

    def __init__(self):
        self.pinecone = PineconeStore()
        self.weaviate = WeaviateStore()
        self._memory: Dict[str, Dict] = {}  # in-memory fallback
        self.active_backend: str = "in_memory"
        self._select_backend()

    def _select_backend(self):
        """Benchmark both backends and choose the fastest available one."""
        available = []

        if self.pinecone.available:
            lat = self.pinecone.latency_test()
            available.append(("pinecone", lat))
            print(f"[VectorMemoryStore] Pinecone latency: {lat*1000:.1f}ms")

        if self.weaviate.available:
            lat = self.weaviate.latency_test()
            available.append(("weaviate", lat))
            print(f"[VectorMemoryStore] Weaviate latency: {lat*1000:.1f}ms")

        if not available:
            print("[VectorMemoryStore] No vector DB configured — using in-memory fallback")
            self.active_backend = "in_memory"
            return

        # Pick the fastest
        available.sort(key=lambda x: x[1])
        self.active_backend = available[0][0]
        print(f"[VectorMemoryStore] Selected backend: {self.active_backend}")

    def store_persona(self, user_id: str, persona: dict) -> bool:
        """Store a twin persona in vector DB."""
        text = json.dumps(persona, default=str)
        meta = {"user_id": user_id, "type": "persona"}
        doc_id = f"persona_{user_id}"
        return self._upsert(doc_id, text, meta)

    def store_simulation_result(self, sim_id: str, result: dict) -> bool:
        """Store a simulation result for later retrieval and clustering."""
        text = json.dumps(result, default=str)
        meta = {"sim_id": sim_id, "type": "simulation_result"}
        doc_id = f"sim_{sim_id}"
        return self._upsert(doc_id, text, meta)

    def retrieve_similar_simulations(self, query: str, top_k: int = 10) -> List[Dict]:
        """Retrieve similar past simulation results by semantic query."""
        return self._query(query, top_k)

    def retrieve_persona(self, user_id: str) -> Optional[dict]:
        """Retrieve a stored persona by user_id."""
        doc_id = f"persona_{user_id}"
        if self.active_backend == "in_memory":
            entry = self._memory.get(doc_id)
            return entry.get("metadata") if entry else None
        results = self._query(f"user_id:{user_id} persona", top_k=1)
        return results[0] if results else None

    # ── Backend dispatch ──────────────────────────────────────────

    def _upsert(self, doc_id: str, text: str, meta: dict) -> bool:
        if self.active_backend == "pinecone":
            return self.pinecone.upsert(doc_id, text, meta)
        elif self.active_backend == "weaviate":
            return self.weaviate.upsert(doc_id, text, meta)
        else:
            self._memory[doc_id] = {"text": text, "metadata": meta}
            return True

    def _query(self, text: str, top_k: int) -> List[Dict]:
        if self.active_backend == "pinecone":
            return self.pinecone.query(text, top_k)
        elif self.active_backend == "weaviate":
            return self.weaviate.query(text, top_k)
        else:
            # Simple keyword match fallback
            results = []
            for doc_id, entry in self._memory.items():
                if any(w in entry["text"].lower() for w in text.lower().split()):
                    results.append({"id": doc_id, "score": 0.5, "text": entry["text"][:200]})
            return results[:top_k]

    def get_backend_info(self) -> dict:
        return {
            "active_backend": self.active_backend,
            "pinecone_available": self.pinecone.available,
            "weaviate_available": self.weaviate.available,
        }
