"""Create and load isolated BM25 + FAISS indexes per chunking strategy."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path

from src.chunking.fixed_size import chunk_pages as fixed_chunks
from src.chunking.recursive import chunk_pages as recursive_chunks
from src.chunking.statistics import summarize_chunks
from src.config import settings
from src.embedding.embedder import SentenceTransformerEmbedder
from src.ingestion.ingestion_pipeline import load_page_records
from src.indexing.bm25_index import BM25Index
from src.indexing.faiss_index import FaissIndex
from src.models import ChunkRecord, DocumentMetadata

LOGGER = logging.getLogger(__name__)


@dataclass
class IndexBundle:
    strategy: str
    chunks: list[ChunkRecord]
    bm25: BM25Index
    faiss: FaissIndex
    model_name: str


def _metadata_map(processed_dir: Path | None = None) -> dict[str, DocumentMetadata]:
    path = (processed_dir or settings.processed_dir) / "documents.jsonl"
    if not path.exists():
        return {}
    return {
        item["document_id"]: DocumentMetadata.model_validate(item)
        for item in (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines())
        if item
    }


def _fingerprint(chunks: list[ChunkRecord], model_name: str) -> str:
    serialized = json.dumps(
        [chunk.model_dump(mode="json") for chunk in chunks],
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(f"{model_name}\n{serialized}".encode("utf-8")).hexdigest()


def build_strategy(
    strategy: str,
    pages_path: Path | None = None,
    force: bool = False,
    *,
    output_root: Path | None = None,
    processed_dir: Path | None = None,
) -> IndexBundle:
    if strategy not in {"fixed_size", "recursive"}:
        raise ValueError(f"Unsupported strategy: {strategy}")
    pages = load_page_records(pages_path)
    metadata = _metadata_map(processed_dir)
    chunk_fn = fixed_chunks if strategy == "fixed_size" else recursive_chunks
    chunks = chunk_fn(pages, metadata, settings.chunk_size, settings.chunk_overlap)
    if not chunks:
        raise ValueError("Không có chunk văn bản để lập chỉ mục")

    output = (output_root or settings.index_dir) / strategy
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    expected = _fingerprint(chunks, settings.embedding_model)
    stats_path = output / "chunk_statistics.json"
    stats_path.write_text(
        json.dumps(summarize_chunks(chunks), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    files_exist = all((output / name).exists() for name in (
        "chunks.jsonl", "bm25_tokens.json", "faiss.index",
        "vector_mapping.json", "bm25_mapping.json"
    ))
    if not force and files_exist and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("fingerprint") == expected:
            return load_strategy(strategy)

    LOGGER.info("Đang tạo embeddings %s (%d chunks)", strategy, len(chunks))
    vectors = SentenceTransformerEmbedder().encode(
        [chunk.search_text or chunk.text for chunk in chunks]
    )
    if vectors.shape[0] != len(chunks):
        raise ValueError("Số vector không khớp số chunk")
    bm25 = BM25Index(chunks)
    faiss_index = FaissIndex(vectors)
    with (output / "chunks.jsonl").open("w", encoding="utf-8") as file:
        for chunk in chunks:
            file.write(chunk.model_dump_json() + "\n")
    bm25.save_tokens(output / "bm25_tokens.json")
    faiss_index.save(output / "faiss.index")
    mapping = [
        {"index": index, "chunk_id": chunk.chunk_id}
        for index, chunk in enumerate(chunks)
    ]
    for name in ("vector_mapping.json", "bm25_mapping.json"):
        (output / name).write_text(
            json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    manifest = {
        "strategy": strategy,
        "fingerprint": expected,
        "embedding_model": settings.embedding_model,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "chunk_count": len(chunks),
        "faiss_vectors": len(chunks),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return IndexBundle(strategy, chunks, bm25, faiss_index, settings.embedding_model)


def load_strategy(strategy: str, index_root: Path | None = None) -> IndexBundle:
    folder = (index_root or settings.index_dir) / strategy
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    chunks = [
        ChunkRecord.model_validate_json(line)
        for line in (folder / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(chunks) != manifest.get("chunk_count"):
        raise ValueError("Manifest index không khớp số chunks")
    expected_mapping = [
        {"index": index, "chunk_id": chunk.chunk_id}
        for index, chunk in enumerate(chunks)
    ]
    for name in ("vector_mapping.json", "bm25_mapping.json"):
        mapping = json.loads((folder / name).read_text(encoding="utf-8"))
        if mapping != expected_mapping:
            raise ValueError(f"{name} không nhất quán với corpus")
    bm25_index = BM25Index.load(chunks, folder / "bm25_tokens.json")
    faiss_index = FaissIndex.load(folder / "faiss.index")
    if len(bm25_index.tokens) != len(chunks):
        raise ValueError("Số BM25 records không khớp số chunks")
    if faiss_index.index.ntotal != len(chunks):
        raise ValueError("Số FAISS vectors không khớp số chunks")
    if manifest.get("faiss_vectors") != faiss_index.index.ntotal:
        raise ValueError("Manifest index không khớp số FAISS vectors")
    return IndexBundle(
        strategy=strategy,
        chunks=chunks,
        bm25=bm25_index,
        faiss=faiss_index,
        model_name=manifest["embedding_model"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy", choices=["fixed_size", "recursive", "both"], default="both")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    for strategy in (["fixed_size", "recursive"] if args.strategy == "both" else [args.strategy]):
        bundle = build_strategy(strategy, force=args.force)
        LOGGER.info("%s: %d chunks", strategy, len(bundle.chunks))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
