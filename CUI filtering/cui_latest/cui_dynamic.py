import logging
import time
import threading
import subprocess
import requests
import numpy as np
from dataclasses import dataclass, asdict
from typing import List, Dict, Set, Tuple, Optional
from collections import defaultdict, deque

from google.cloud import bigquery
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_distances

# ----------------------------
# Configure logging
# ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ----------------------------
# Filter function
# ----------------------------
def filter_allowed_cuis(cuis: Set[str], project_id: str, dataset_id: str) -> List[str]:
    if not cuis:
        return []

    try:
        client = bigquery.Client(project=project_id)
        query = f"""
        SELECT DISTINCT CUI
        FROM `{project_id}.{dataset_id}.MRCONSO`
        WHERE CUI IN UNNEST(@cuis)
          AND SAB IN (
            'CCSR_ICD10CM','CCSR_ICD10PCS','DMDICD10','ICD10','ICD10AE',
            'ICD10AM','ICD10AMAE','ICD10CM','ICD10DUT','ICD10PCS',
            'ICD9CM','ICPC2ICD10DUT','ICPC2ICD10ENG',
            'SNOMEDCT_US','LOINC'
          )
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ArrayQueryParameter("cuis", "STRING", list(cuis))
            ]
        )
        df = client.query(query, job_config=job_config).result(timeout=60).to_dataframe()
        allowed = df["CUI"].tolist()
        logger.info(f"{len(allowed)} CUIs after SAB filtering")
        return allowed
    except Exception as e:
        logger.error(f"Failed to filter CUIs by SAB: {str(e)}")
        return []

# ----------------------------
# Reduction statistics
# ----------------------------
@dataclass
class ReductionStats:
    initial_count: int
    after_ic_rollup: int
    final_count: int
    ic_rollup_reduction_pct: float
    semantic_clustering_reduction_pct: float
    total_reduction_pct: float
    processing_time: float
    ic_threshold_used: float
    hierarchy_size: int = 0
    api_call_time: float = 0.0

    def to_dict(self):
        return asdict(self)

# ----------------------------
# API Client
# ----------------------------
class CUIAPIClient:
    _token_lock = threading.Lock()
    _cached_token = None
    _token_expiry = 0

    def __init__(self, api_base_url: str, timeout: int = 60, top_k: int = 3, max_retries: int = 3):
        self.api_base_url = api_base_url.rstrip("/")
        self.timeout = timeout
        self.top_k = top_k
        self.max_retries = max_retries

        self.session = requests.Session()
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["POST"]
        )
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=10,
            pool_maxsize=20
        )
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

        self._update_gcp_token()

    def _update_gcp_token(self, force: bool = False):
        with self._token_lock:
            now = time.time()
            if not force and self._cached_token and now < self._token_expiry:
                return self._cached_token

            result = subprocess.run(
                ["gcloud", "auth", "print-identity-token"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            if result.returncode != 0:
                raise RuntimeError(result.stderr)

            token = result.stdout.strip()
            self._cached_token = {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json"
            }
            self._token_expiry = now + 3300
            return self._cached_token

    def extract_cuis_batch(self, texts: List[str], retry_auth: bool = True) -> Set[str]:
        if not texts:
            return set()

        payload = {"query_texts": texts, "top_k": self.top_k}
        headers = self._update_gcp_token()

        try:
            resp = self.session.post(
                self.api_base_url,
                json=payload,
                headers=headers,
                timeout=self.timeout
            )

            if resp.status_code == 401 and retry_auth:
                self._update_gcp_token(force=True)
                return self.extract_cuis_batch(texts, retry_auth=False)

            resp.raise_for_status()
            data = resp.json()

            cuis = set()
            for t in texts:
                for c in data.get(t, []):
                    if c:
                        cuis.add(str(c))
            return cuis

        except Exception as e:
            logger.error(f"API call failed: {str(e)}")
            return set()

# ----------------------------
# Enhanced CUI Reducer
# ----------------------------
class EnhancedCUIReducer:

    def __init__(
        self,
        project_id: str,
        dataset_id: str,
        mrrel_table: str = "MRREL",
        cui_description_table: str = "cui_descriptions",
        cui_embeddings_table: str = "cui_embeddings",
        cui_narrower_table: str = "cui_narrower_concepts",
        max_hierarchy_depth: int = 1,
        query_timeout: int = 300
    ):
        self.client = bigquery.Client(project=project_id)
        self.project_id = project_id
        self.dataset_id = dataset_id
        self.mrrel_table = mrrel_table
        self.cui_description_table = cui_description_table
        self.cui_embeddings_table = cui_embeddings_table
        self.cui_narrower_table = cui_narrower_table
        self.max_hierarchy_depth = max_hierarchy_depth
        self.query_timeout = query_timeout

        self._hierarchy_cache = None
        self._ic_scores_cache = None
        self._description_cache = {}

    # ----------------------------
    # Reduction
    # ----------------------------
    def reduce(
        self,
        input_cuis: List[str],
        target_reduction: float = 0.85,
        ic_threshold: Optional[float] = None,
        ic_percentile: float = 50.0,        # kept, ignored
        semantic_threshold: float = 0.88,
        use_semantic_clustering: bool = True,
        adaptive_threshold: bool = False    # kept, ignored
    ) -> Tuple[List[str], ReductionStats]:

        start_time = time.time()
        input_cuis = list(set(input_cuis))
        initial_count = len(input_cuis)

        hierarchy = self._build_hierarchy_safe(input_cuis)
        ic_scores = self._compute_ic_scores_safe(hierarchy)

        # MEDIAN IC THRESHOLD ONLY
        if ic_threshold is None:
            ic_threshold = float(np.median(list(ic_scores.values())))
            logger.info(f"Using MEDIAN IC threshold: {ic_threshold:.3f}")

        rolled_up = self._semantic_rollup_with_ic_safe(
            input_cuis, hierarchy, ic_scores, ic_threshold
        )
        after_rollup = len(rolled_up)

        final_cuis = rolled_up
        if use_semantic_clustering:
            final_cuis = self._semantic_clustering_safe(rolled_up, ic_scores)

        final_count = len(final_cuis)

        stats = ReductionStats(
            initial_count=initial_count,
            after_ic_rollup=after_rollup,
            final_count=final_count,
            ic_rollup_reduction_pct=self._safe_percentage(
                initial_count - after_rollup, initial_count
            ),
            semantic_clustering_reduction_pct=self._safe_percentage(
                after_rollup - final_count, initial_count
            ),
            total_reduction_pct=self._safe_percentage(
                initial_count - final_count, initial_count
            ),
            processing_time=time.time() - start_time,
            ic_threshold_used=ic_threshold,
            hierarchy_size=len(hierarchy.get("all_cuis", []))
        )

        return final_cuis, stats

    # ----------------------------
    # Hierarchy
    # ----------------------------
    def _build_hierarchy_safe(self, relevant_cuis: List[str]) -> Dict:
        if self._hierarchy_cache is not None:
            return self._hierarchy_cache

        child_to_parents = defaultdict(list)
        parent_to_children = defaultdict(list)
        all_cuis = set(relevant_cuis)

        visited = set()
        frontier = set(relevant_cuis)

        for depth in range(self.max_hierarchy_depth):
            if not frontier:
                break

            query = f"""
            SELECT DISTINCT cui1, cui2, rel
            FROM `{self.project_id}.{self.dataset_id}.{self.mrrel_table}`
            WHERE (cui1 IN UNNEST(@frontier) OR cui2 IN UNNEST(@frontier))
              AND rel IN ('PAR','CHD')
            """
            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ArrayQueryParameter("frontier", "STRING", list(frontier))
                ]
            )

            df = self.client.query(query, job_config=job_config).result(
                timeout=self.query_timeout
            ).to_dataframe()

            next_frontier = set()
            for _, row in df.iterrows():
                c1, c2, rel = row["cui1"], row["cui2"], row["rel"]
                if rel == "PAR":
                    parent_to_children[c1].append(c2)
                    child_to_parents[c2].append(c1)
                else:
                    parent_to_children[c2].append(c1)
                    child_to_parents[c1].append(c2)

                all_cuis.update([c1, c2])
                if c1 not in visited:
                    next_frontier.add(c1)
                if c2 not in visited:
                    next_frontier.add(c2)

            visited.update(frontier)
            frontier = next_frontier - visited

        self._hierarchy_cache = {
            "child_to_parents": dict(child_to_parents),
            "parent_to_children": dict(parent_to_children),
            "all_cuis": all_cuis
        }
        return self._hierarchy_cache

    # ----------------------------
    # IC computation
    # ----------------------------
    def _compute_ic_scores_safe(self, hierarchy: Dict) -> Dict[str, float]:
        if self._ic_scores_cache is not None:
            return self._ic_scores_cache

        parent_to_children = hierarchy.get("parent_to_children", {})
        all_cuis = hierarchy.get("all_cuis", set())
        total = len(all_cuis)

        descendant_counts = {}

        def count_desc(cui, visited=None):
            if visited is None:
                visited = set()
            if cui in visited:
                return 0
            visited.add(cui)
            count = 0
            for ch in parent_to_children.get(cui, []):
                count += 1 + count_desc(ch, visited)
            return count

        ic_scores = {}
        for cui in all_cuis:
            dc = count_desc(cui)
            ic_scores[cui] = max(0.0, -np.log((dc + 1) / total))

        self._ic_scores_cache = ic_scores
        return ic_scores

    # ----------------------------
    # Rollup
    # ----------------------------
    def _semantic_rollup_with_ic_safe(
        self,
        cui_list: List[str],
        hierarchy: Dict,
        ic_scores: Dict[str, float],
        ic_threshold: float
    ) -> List[str]:

        child_to_parents = hierarchy.get("child_to_parents", {})
        rolled = {}

        for cui in cui_list:
            ancestors = []
            queue = deque([cui])
            visited = set()

            while queue:
                cur = queue.popleft()
                if cur in visited:
                    continue
                visited.add(cur)
                for p in child_to_parents.get(cur, []):
                    ancestors.append(p)
                    queue.append(p)

            candidates = [cui] + ancestors
            valid = [c for c in candidates if ic_scores.get(c, 0) >= ic_threshold]

            rolled[cui] = min(valid, key=lambda x: ic_scores.get(x, float("inf"))) if valid else cui

        return list(set(rolled.values()))

    # ----------------------------
    # Clustering
    # ----------------------------
    def _semantic_clustering_safe(
        self,
        cui_list: List[str],
        ic_scores: Dict[str, float],
        base_similarity_percentile: float = 95.0,
        base_distance_percentile: float = 25.0
    ) -> List[str]:

        if len(cui_list) <= 1:
            return cui_list

        query = f"""
        SELECT REF_CUI as cui, REF_Embedding as embedding
        FROM `{self.project_id}.{self.dataset_id}.{self.cui_embeddings_table}`
        WHERE REF_CUI IN UNNEST(@cuis)
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ArrayQueryParameter("cuis", "STRING", cui_list)
            ]
        )

        df = self.client.query(query, job_config=job_config).result(timeout=60).to_dataframe()
        if df.empty:
            return cui_list

        embeddings = np.vstack(df["embedding"].values)
        cuis = np.array(df["cui"].values)

        sim = 1 - cosine_distances(embeddings)
        tri = sim[np.triu_indices_from(sim, k=1)]
        sim_thr = np.percentile(tri, base_similarity_percentile)
        dist_thr = 1 - sim_thr

        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=dist_thr,
            metric="cosine",
            linkage="average"
        )
        labels = clustering.fit_predict(embeddings)

        final = []
        for cid in np.unique(labels):
            idx = np.where(labels == cid)[0]
            cluster_cuis = cuis[idx]
            cluster_emb = embeddings[idx]

            if len(cluster_cuis) == 1:
                final.append(cluster_cuis[0])
                continue

            dist = cosine_distances(cluster_emb)
            tri_d = dist[np.triu_indices_from(dist, k=1)]
            prune_thr = np.percentile(tri_d, base_distance_percentile)

            order = sorted(
                range(len(cluster_cuis)),
                key=lambda i: ic_scores.get(cluster_cuis[i], 0),
                reverse=True
            )

            kept = []
            for i in order:
                if all(dist[i, j] >= prune_thr for j in kept):
                    kept.append(i)

            final.extend(cluster_cuis[kept])

        return list(set(final))

    # ----------------------------
    # Misc
    # ----------------------------
    def get_cui_descriptions(self, cui_list: List[str]) -> Dict[str, str]:
        if not cui_list:
            return {}

        uncached = [c for c in cui_list if c not in self._description_cache]
        if uncached:
            query = f"""
            SELECT CUI as cui, Definition as description
            FROM `{self.project_id}.{self.dataset_id}.{self.cui_description_table}`
            WHERE CUI IN UNNEST(@cuis)
            """
            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ArrayQueryParameter("cuis", "STRING", uncached)
                ]
            )
            df = self.client.query(query, job_config=job_config).result(timeout=30).to_dataframe()
            self._description_cache.update(dict(zip(df["cui"], df["description"])))

        return {c: self._description_cache.get(c, "N/A") for c in cui_list}

    def get_ic_scores(self, cui_list: List[str]) -> Dict[str, float]:
        return {c: self._ic_scores_cache.get(c, 0.0) for c in cui_list}

    @staticmethod
    def _safe_percentage(n, d):
        return (n / d * 100) if d else 0.0

# ----------------------------
# Main runner
# ----------------------------
def run_cui_reduction(
    texts: List[str],
    project_id: str,
    dataset_id: str,
    api_url: str,
    **kwargs
):
    api = CUIAPIClient(api_url)
    reducer = EnhancedCUIReducer(project_id, dataset_id)

    cuis = api.extract_cuis_batch(texts)
    cuis = filter_allowed_cuis(cuis, project_id, dataset_id)

    reduced, stats = reducer.reduce(list(cuis))
    descriptions = reducer.get_cui_descriptions(reduced)

    return cuis, reduced, descriptions, stats

# ----------------------------
# Example usage
# ----------------------------
if __name__ == "__main__":
    texts = ["grams"]

    initial_cuis, reduced_cuis, descriptions, stats = run_cui_reduction(
        texts=texts,
        project_id=project_id,
        dataset_id=dataset,
        api_url=url
    )

    if stats:
        print(
            f"Reduction complete: "
            f"{stats.initial_count} → {stats.final_count} "
            f"({stats.total_reduction_pct:.1f}%)"
        )
