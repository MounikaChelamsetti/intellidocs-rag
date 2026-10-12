"""ChromaDB wrapper. Uses cosine distance; score = 1 - distance (higher is better).

Default is an in-memory (ephemeral) client with a unique collection per session, so data from
unrelated sessions never mixes. Pass persist_dir for an on-disk store (not guaranteed to
survive on hosts with ephemeral filesystems such as Streamlit Community Cloud).
"""
from __future__ import annotations

import uuid
import weakref
from typing import Dict, List, Optional, Sequence

import chromadb
from chromadb.config import Settings as ChromaSettings

from .models import Chunk, RetrievedChunk

_CLIENTS: Dict[str, "chromadb.api.ClientAPI"] = {}


def get_client(persist_dir: Optional[str] = None):
    key = persist_dir or ":memory:"
    if key not in _CLIENTS:
        cfg = ChromaSettings(anonymized_telemetry=False)
        _CLIENTS[key] = (
            chromadb.PersistentClient(path=persist_dir, settings=cfg)
            if persist_dir
            else chromadb.EphemeralClient(settings=cfg)
        )
    return _CLIENTS[key]


def _drop_quietly(client, name: str) -> None:
    try:
        client.delete_collection(name)
    except Exception:
        pass


class VectorStore:
    def __init__(self, collection_name: Optional[str] = None, persist_dir: Optional[str] = None):
        self.persist_dir = persist_dir
        self.collection_name = collection_name or f"session_{uuid.uuid4().hex}"
        self._client = get_client(persist_dir)
        self._collection = self._create()
        if not persist_dir:
            # Free session memory when the store is garbage-collected (session ended).
            weakref.finalize(self, _drop_quietly, self._client, self.collection_name)

    def _create(self):
        return self._client.get_or_create_collection(
            name=self.collection_name, metadata={"hnsw:space": "cosine"}
        )

    def count(self) -> int:
        return self._collection.count()

    def has_document(self, doc_id: str) -> bool:
        return bool(self._collection.get(where={"doc_id": doc_id}, limit=1, include=[])["ids"])

    def add_chunks(self, chunks: Sequence[Chunk], embeddings: Sequence[Sequence[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have equal length")
        for i in range(0, len(chunks), 500):
            part = chunks[i : i + 500]
            self._collection.upsert(
                ids=[c.chunk_id for c in part],
                embeddings=[list(e) for e in embeddings[i : i + 500]],
                documents=[c.text for c in part],
                metadatas=[
                    {
                        "doc_id": c.doc_id,
                        "filename": c.filename,
                        "page_number": c.page_number,
                        "chunk_index": c.chunk_index,
                        "word_count": c.word_count,
                    }
                    for c in part
                ],
            )

    def query(self, embedding: Sequence[float], n_results: int) -> List[RetrievedChunk]:
        total = self.count()
        if total == 0 or n_results < 1:
            return []
        res = self._collection.query(
            query_embeddings=[list(embedding)],
            n_results=min(n_results, total),
            include=["documents", "metadatas", "distances"],
        )
        out: List[RetrievedChunk] = []
        for cid, doc, meta, dist in zip(
            res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]
        ):
            out.append(
                RetrievedChunk(
                    chunk_id=cid,
                    doc_id=meta["doc_id"],
                    filename=meta["filename"],
                    page_number=int(meta["page_number"]),
                    chunk_index=int(meta["chunk_index"]),
                    text=doc,
                    score=1.0 - float(dist),
                    distance=float(dist),
                )
            )
        return out

    def list_documents(self) -> List[dict]:
        data = self._collection.get(include=["metadatas"])
        docs: Dict[str, dict] = {}
        for meta in data["metadatas"] or []:
            d = docs.setdefault(
                meta["doc_id"], {"doc_id": meta["doc_id"], "filename": meta["filename"], "pages": set(), "chunks": 0}
            )
            d["pages"].add(meta["page_number"])
            d["chunks"] += 1
        return [
            {**d, "pages_indexed": len(d["pages"]), "pages": sorted(d["pages"])}
            for d in sorted(docs.values(), key=lambda x: x["filename"].lower())
        ]

    def delete_document(self, doc_id: str) -> None:
        self._collection.delete(where={"doc_id": doc_id})

    def reset(self) -> None:
        _drop_quietly(self._client, self.collection_name)
        self._collection = self._create()
