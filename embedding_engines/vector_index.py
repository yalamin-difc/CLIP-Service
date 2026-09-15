"""Engine-scoped vector index abstraction (P13 section 11).

`VectorIndex` is the interface the V2 match/AB endpoints program against.
`NumpyVectorIndex` is the first, dependency-light implementation: a
contiguous float32 NumPy matrix per engine, searched by brute-force cosine
(vectors are expected pre-normalized, so cosine similarity reduces to a
plain dot product). Tenant and site isolation is enforced inside the index
itself -- a caller cannot accidentally retrieve another tenant's vectors
by forgetting a filter, because search() requires tenant_id/site_ids and
only ever matches rows recorded under them.

This is intentionally not a persistent, cached index: callers (see
v2_router.py) rebuild it from the repository's current candidate set on
each request, exactly like the legacy /match endpoint already does with
its own in-request ranking loop. That keeps correctness simple (no
invalidation bugs) while still giving future work a stable seam: a FAISS-
or MongoDB-Atlas-Vector-Search-backed VectorIndex can implement the same
four methods (build/add_or_update/remove/search) plus stats() without any
caller changes.
"""
from __future__ import annotations

import abc
import threading
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# (candidate_id, tenant_id, site_id, vector)
IndexRow = Tuple[str, str, str, Sequence[float]]


class VectorIndex(abc.ABC):
    @abc.abstractmethod
    def build(self, engine: str, rows: List[IndexRow]) -> None:
        """Fully replace the index for `engine` with `rows`."""

    def refresh(self, engine: str, rows: List[IndexRow]) -> None:
        self.build(engine, rows)

    @abc.abstractmethod
    def add_or_update(self, engine: str, candidate_id: str, tenant_id: str, site_id: str, vector: Sequence[float]) -> None:
        ...

    @abc.abstractmethod
    def remove(self, engine: str, tenant_id: str, candidate_id: str) -> bool:
        ...

    @abc.abstractmethod
    def search(
        self,
        engine: str,
        tenant_id: str,
        permitted_site_ids: Sequence[str],
        query_vector: Sequence[float],
        top_k: int = 5,
    ) -> List[Tuple[str, float]]:
        """Return up to top_k (candidate_id, cosine_score) pairs, highest
        score first, restricted to `tenant_id` and one of `permitted_site_ids`.
        Never returns a candidate belonging to another tenant."""

    @abc.abstractmethod
    def stats(self, engine: Optional[str] = None) -> Dict[str, Any]:
        ...


def _empty_state(dim: int = 0) -> Dict[str, Any]:
    return {
        "vectors": np.zeros((0, max(dim, 0)), dtype=np.float32),
        "ids": [],
        "tenants": [],
        "sites": [],
        "dim": dim,
    }


class NumpyVectorIndex(VectorIndex):
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._state: Dict[str, Dict[str, Any]] = {}

    def build(self, engine: str, rows: List[IndexRow]) -> None:
        if not rows:
            with self._lock:
                self._state[engine] = _empty_state()
            return
        dims = {len(vector) for _, _, _, vector in rows}
        if len(dims) > 1:
            raise ValueError(f"Cannot build vector index for engine '{engine}' from mixed-dimension vectors: {sorted(dims)}")
        dim = dims.pop()
        matrix = np.ascontiguousarray(np.asarray([row[3] for row in rows], dtype=np.float32))
        state = {
            "vectors": matrix,
            "ids": [row[0] for row in rows],
            "tenants": [row[1] for row in rows],
            "sites": [row[2] for row in rows],
            "dim": dim,
        }
        with self._lock:
            self._state[engine] = state

    def add_or_update(self, engine: str, candidate_id: str, tenant_id: str, site_id: str, vector: Sequence[float]) -> None:
        vector_array = np.asarray(vector, dtype=np.float32)
        with self._lock:
            state = self._state.setdefault(engine, _empty_state(vector_array.shape[0]))
            existing_index = next(
                (
                    i
                    for i, (cid, tid) in enumerate(zip(state["ids"], state["tenants"]))
                    if cid == candidate_id and tid == tenant_id
                ),
                None,
            )
            if existing_index is not None:
                state["vectors"][existing_index] = vector_array
                state["sites"][existing_index] = site_id
                return
            if state["vectors"].shape[0] == 0:
                state["vectors"] = vector_array.reshape(1, -1)
                state["dim"] = vector_array.shape[0]
            else:
                if vector_array.shape[0] != state["dim"]:
                    raise ValueError(
                        f"Vector dimension {vector_array.shape[0]} does not match index dimension {state['dim']} for engine '{engine}'"
                    )
                state["vectors"] = np.vstack([state["vectors"], vector_array.reshape(1, -1)])
            state["ids"].append(candidate_id)
            state["tenants"].append(tenant_id)
            state["sites"].append(site_id)

    def remove(self, engine: str, tenant_id: str, candidate_id: str) -> bool:
        with self._lock:
            state = self._state.get(engine)
            if not state:
                return False
            for i, (cid, tid) in enumerate(zip(state["ids"], state["tenants"])):
                if cid == candidate_id and tid == tenant_id:
                    state["vectors"] = np.delete(state["vectors"], i, axis=0)
                    del state["ids"][i]
                    del state["tenants"][i]
                    del state["sites"][i]
                    return True
            return False

    def search(
        self,
        engine: str,
        tenant_id: str,
        permitted_site_ids: Sequence[str],
        query_vector: Sequence[float],
        top_k: int = 5,
    ) -> List[Tuple[str, float]]:
        permitted = set(permitted_site_ids)
        with self._lock:
            state = self._state.get(engine)
            if not state or state["vectors"].shape[0] == 0:
                return []
            # Tenant/site isolation is enforced here, not left to the caller:
            # rows are only ever selected when both the tenant and the site
            # match this identity's own permitted scope.
            mask = [
                i
                for i in range(len(state["ids"]))
                if state["tenants"][i] == tenant_id and state["sites"][i] in permitted
            ]
            if not mask:
                return []
            vectors = state["vectors"][mask]
            ids = [state["ids"][i] for i in mask]
        query = np.asarray(query_vector, dtype=np.float32)
        if query.shape[0] != vectors.shape[1]:
            raise ValueError(f"Query dimension {query.shape[0]} does not match index dimension {vectors.shape[1]}")
        scores = vectors @ query
        order = np.argsort(-scores)[: max(0, top_k)]
        return [(ids[i], float(scores[i])) for i in order]

    def stats(self, engine: Optional[str] = None) -> Dict[str, Any]:
        with self._lock:
            if engine is not None:
                state = self._state.get(engine, _empty_state())
                return {"engine": engine, "vectors": int(state["vectors"].shape[0]), "dimension": state["dim"]}
            return {
                eng: {"vectors": int(state["vectors"].shape[0]), "dimension": state["dim"]}
                for eng, state in self._state.items()
            }
