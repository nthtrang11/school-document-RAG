import numpy as np

from src.indexing.faiss_index import FaissIndex


def test_faiss_normalizes_search_and_round_trips(tmp_path):
    vectors = np.array([[3.0, 0.0], [0.0, 5.0]], dtype=np.float32)
    index = FaissIndex(vectors)
    assert index.search(np.array([2.0, 0.0]), 1) == [(0, 1.0)]
    path = tmp_path / "vectors.index"
    index.save(path)
    loaded = FaissIndex.load(path)
    assert loaded.search(np.array([0.0, 1.0]), 1)[0][0] == 1
