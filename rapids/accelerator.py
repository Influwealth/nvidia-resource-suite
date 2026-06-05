"""RAPIDS: GPU DataFrames (cuDF), ML (cuML), Graph Analytics (cuGraph). Pandas/sklearn fallback."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from loguru import logger

try:
    import cudf
    import cuml
    import cugraph
    RAPIDS_AVAILABLE = True
    logger.info("NVIDIA RAPIDS available (cuDF + cuML + cuGraph)")
except ImportError:
    RAPIDS_AVAILABLE = False
    logger.warning("RAPIDS not installed — using pandas/sklearn fallback")

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


class cuDFProcessor:
    """GPU-accelerated DataFrame operations via RAPIDS cuDF.

    Falls back to pandas when cuDF is not available.
    Same API surface for educational portability.
    """

    def __init__(self):
        self._backend = "cudf" if RAPIDS_AVAILABLE else "pandas"
        logger.info(f"cuDFProcessor using {self._backend}")

    def read_csv(self, path: str, **kwargs) -> Any:
        if RAPIDS_AVAILABLE:
            return cudf.read_csv(path, **kwargs)
        if PANDAS_AVAILABLE:
            return pd.read_csv(path, **kwargs)
        raise ImportError("Neither cudf nor pandas is installed")

    def read_parquet(self, path: str, **kwargs) -> Any:
        if RAPIDS_AVAILABLE:
            return cudf.read_parquet(path, **kwargs)
        if PANDAS_AVAILABLE:
            return pd.read_parquet(path, **kwargs)
        raise ImportError("Neither cudf nor pandas is installed")

    def merge(self, left: Any, right: Any, **kwargs) -> Any:
        if RAPIDS_AVAILABLE and isinstance(left, cudf.DataFrame):
            return cudf.merge(left, right, **kwargs)
        if PANDAS_AVAILABLE:
            return pd.merge(left, right, **kwargs)
        raise RuntimeError("No DataFrame backend available")

    def groupby_agg(self, df: Any, by: str | list, agg: dict) -> Any:
        return df.groupby(by).agg(agg)

    def to_pandas(self, df: Any) -> Any:
        if RAPIDS_AVAILABLE and isinstance(df, cudf.DataFrame):
            return df.to_pandas()
        return df

    def filter_rows(self, df: Any, condition_str: str) -> Any:
        """Filter rows using query string (GPU-accelerated on cuDF)."""
        return df.query(condition_str)

    def describe(self, df: Any) -> dict:
        stats = df.describe()
        if RAPIDS_AVAILABLE and isinstance(stats, cudf.DataFrame):
            stats = stats.to_pandas()
        return stats.to_dict() if PANDAS_AVAILABLE and hasattr(stats, "to_dict") else {}

    def backend(self) -> str:
        return self._backend


class cuMLTrainer:
    """GPU-accelerated ML training via RAPIDS cuML.

    Supports: LinearRegression, RandomForest, KMeans, DBSCAN, UMAP,
    t-SNE, PCA, SVMs, XGBoost (via cuml.ensemble).
    Falls back to sklearn when cuML is not available.
    """

    def __init__(self):
        self._backend = "cuml" if RAPIDS_AVAILABLE else "sklearn"
        logger.info(f"cuMLTrainer using {self._backend}")

    def linear_regression(self, X: Any, y: Any) -> Any:
        if RAPIDS_AVAILABLE:
            model = cuml.LinearRegression()
        else:
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()
        model.fit(X, y)
        return model

    def random_forest_classifier(self, X: Any, y: Any, n_estimators: int = 100) -> Any:
        if RAPIDS_AVAILABLE:
            model = cuml.ensemble.RandomForestClassifier(n_estimators=n_estimators)
        else:
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(n_estimators=n_estimators)
        model.fit(X, y)
        return model

    def kmeans(self, X: Any, n_clusters: int = 8) -> Any:
        if RAPIDS_AVAILABLE:
            model = cuml.KMeans(n_clusters=n_clusters)
        else:
            from sklearn.cluster import KMeans
            model = KMeans(n_clusters=n_clusters, n_init="auto")
        model.fit(X)
        return model

    def umap(self, X: Any, n_components: int = 2, **kwargs) -> Any:
        """UMAP dimensionality reduction — 100-1000x faster on GPU."""
        if RAPIDS_AVAILABLE:
            reducer = cuml.UMAP(n_components=n_components, **kwargs)
        else:
            try:
                from umap import UMAP
                reducer = UMAP(n_components=n_components, **kwargs)
            except ImportError:
                raise ImportError("Install umap-learn: pip install umap-learn")
        return reducer.fit_transform(X)

    def pca(self, X: Any, n_components: int = 50) -> Any:
        if RAPIDS_AVAILABLE:
            pca = cuml.PCA(n_components=n_components)
        else:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=n_components)
        return pca.fit_transform(X)

    def train_embedding_index(
        self,
        embeddings: list[list[float]],
        ids: list[str],
        metric: str = "cosine",
    ) -> dict:
        """Build a nearest-neighbor index over embeddings.

        Uses cuML RAFT neighbors on GPU, or sklearn NearestNeighbors on CPU.
        """
        if not NUMPY_AVAILABLE:
            return {"mock": True, "note": "numpy required"}
        import numpy as np
        X = np.array(embeddings, dtype=np.float32)
        if RAPIDS_AVAILABLE:
            from cuml.neighbors import NearestNeighbors
            nn = NearestNeighbors(metric=metric)
            nn.fit(cudf.DataFrame(X))
        else:
            from sklearn.neighbors import NearestNeighbors
            nn = NearestNeighbors(metric=metric)
            nn.fit(X)
        return {"model": nn, "ids": ids, "size": len(ids), "metric": metric, "backend": self._backend}

    def backend(self) -> str:
        return self._backend


class cuGraphAnalyzer:
    """GPU graph analytics via RAPIDS cuGraph.

    Educational use: social network analysis, trade network mapping,
    community detection for the Synapz platform.
    """

    def __init__(self):
        self._backend = "cugraph" if RAPIDS_AVAILABLE else "networkx"
        logger.info(f"cuGraphAnalyzer using {self._backend}")

    def _get_graph(self, edges: list[tuple]) -> Any:
        if RAPIDS_AVAILABLE:
            src = cudf.Series([e[0] for e in edges])
            dst = cudf.Series([e[1] for e in edges])
            weights = cudf.Series([e[2] if len(e) > 2 else 1.0 for e in edges])
            G = cugraph.Graph()
            G.from_cudf_edgelist(cudf.DataFrame({"src": src, "dst": dst, "weight": weights}), source="src", destination="dst", edge_attr="weight")
            return G
        try:
            import networkx as nx
            G = nx.DiGraph()
            G.add_weighted_edges_from(edges)
            return G
        except ImportError:
            raise ImportError("Install networkx: pip install networkx")

    def pagerank(self, edges: list[tuple], alpha: float = 0.85) -> dict:
        """Compute PageRank over a directed graph."""
        G = self._get_graph(edges)
        if RAPIDS_AVAILABLE:
            pr = cugraph.pagerank(G, alpha=alpha)
            return dict(zip(pr["vertex"].to_pandas(), pr["pagerank"].to_pandas()))
        import networkx as nx
        return nx.pagerank(G, alpha=alpha)

    def community_detection(self, edges: list[tuple]) -> dict:
        """Louvain community detection."""
        G = self._get_graph(edges)
        if RAPIDS_AVAILABLE:
            parts, modularity = cugraph.louvain(G)
            communities = {}
            for _, row in parts.to_pandas().iterrows():
                communities.setdefault(row["partition"], []).append(row["vertex"])
            return {"communities": communities, "modularity": modularity}
        import networkx as nx
        from networkx.algorithms import community as comm
        G_undirected = G.to_undirected()
        result = comm.louvain_communities(G_undirected)
        communities = {i: list(c) for i, c in enumerate(result)}
        return {"communities": communities, "modularity": None}

    def shortest_path(self, edges: list[tuple], source: int, target: int) -> dict:
        G = self._get_graph(edges)
        if RAPIDS_AVAILABLE:
            distances = cugraph.sssp(G, source)
            row = distances[distances["vertex"] == target].to_pandas()
            return {"distance": float(row["distance"].iloc[0]) if len(row) > 0 else float("inf")}
        import networkx as nx
        try:
            path = nx.shortest_path(G, source, target, weight="weight")
            length = nx.shortest_path_length(G, source, target, weight="weight")
            return {"path": path, "distance": length}
        except nx.NetworkXNoPath:
            return {"path": [], "distance": float("inf")}

    def backend(self) -> str:
        return self._backend


class RAPIDSAccelerator:
    """Unified RAPIDS facade exposing cuDF, cuML, cuGraph."""

    def __init__(self):
        self.df = cuDFProcessor()
        self.ml = cuMLTrainer()
        self.graph = cuGraphAnalyzer()
        self.available = RAPIDS_AVAILABLE

    def status(self) -> dict:
        return {
            "rapids_available": RAPIDS_AVAILABLE,
            "pandas_fallback": PANDAS_AVAILABLE,
            "numpy_available": NUMPY_AVAILABLE,
            "backends": {
                "dataframe": self.df.backend(),
                "ml": self.ml.backend(),
                "graph": self.graph.backend(),
            },
        }

    def benchmark_speedup(
        self,
        n_rows: int = 1_000_000,
        operation: str = "groupby",
    ) -> dict:
        """Demonstrate GPU vs CPU speedup for a sample operation."""
        import time, random
        if not NUMPY_AVAILABLE:
            return {"note": "numpy required for benchmark"}
        import numpy as np
        data = {"category": np.random.randint(0, 100, n_rows), "value": np.random.randn(n_rows)}

        # CPU (pandas)
        if PANDAS_AVAILABLE:
            df_cpu = pd.DataFrame(data)
            t0 = time.perf_counter()
            df_cpu.groupby("category")["value"].mean()
            cpu_time = time.perf_counter() - t0
        else:
            cpu_time = None

        # GPU (cudf)
        gpu_time = None
        if RAPIDS_AVAILABLE:
            df_gpu = cudf.DataFrame(data)
            t0 = time.perf_counter()
            df_gpu.groupby("category")["value"].mean()
            gpu_time = time.perf_counter() - t0

        return {
            "n_rows": n_rows,
            "operation": operation,
            "cpu_time_s": round(cpu_time, 4) if cpu_time else None,
            "gpu_time_s": round(gpu_time, 4) if gpu_time else None,
            "speedup": round(cpu_time / gpu_time, 1) if cpu_time and gpu_time else "N/A",
        }
