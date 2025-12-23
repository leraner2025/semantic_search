import asyncio
import aiohttp
import subprocess
import re
import logging
from pathlib import Path
import json
from typing import List, Dict, Set, Optional, Tuple
from collections import defaultdict, deque
import time
from dataclasses import dataclass, asdict
import hashlib
import numpy as np
from google.cloud import bigquery
import nest_asyncio
import requests
from requests.adapters import HTTPAdapter, Retry
import threading

nest_asyncio.apply()

# ----------------------
# Logging
# ----------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ----------------------
# Data Classes
# ----------------------
@dataclass
class ReductionStats:
    initial_count: int
    after_ic_rollup: int
    final_count: int
    ic_rollup_reduction_pct: float
    total_reduction_pct: float
    processing_time: float
    ic_threshold_used: float
    hierarchy_size: int = 0
    api_call_time: float = 0.0

    def to_dict(self):
        return asdict(self)

# ----------------------
# Filter CUIs by SAB
# ----------------------
def filter_allowed_cuis(cuis: Set[str], project_id: str, dataset_id: str) -> List[str]:
    if not cuis:
        return []
    try:
        logger.info(f"Filtering {len(cuis)} CUIs to retain only ICD, SNOMED, and LOINC...")
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
            query_parameters=[bigquery.ArrayQueryParameter("cuis", "STRING", list(cuis))]
        )
        df = client.query(query, job_config=job_config).result(timeout=60).to_dataframe()
        allowed_cuis = df['CUI'].tolist()
        logger.info(f"{len(allowed_cuis)} CUIs after filter")
        return allowed_cuis
    except Exception as e:
        logger.error(f"Failed to filter CUIs by SAB: {str(e)}")
        return []

# ----------------------
# CUI API Client
# ----------------------
class CUIAPIClient:
    _token_lock = threading.Lock()
    _cached_token = None
    _token_expiry = 0

    def __init__(self, api_base_url: str, timeout: int = 60, top_k: int = 3, max_retries: int = 3):
        self.api_base_url = api_base_url.rstrip('/')
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
                ['gcloud', 'auth', 'print-identity-token'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                timeout=10,
                check=False
            )
            if result.returncode != 0:
                raise Exception(f"gcloud auth failed: {result.stderr}")

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
        logger.info(f"Extracting CUIs from {len(texts)} text(s)...")

        try:
            response = self.session.post(
                self.api_base_url,
                json=payload,
                headers=headers,
                timeout=self.timeout
            )
            if response.status_code == 401 and retry_auth:
                logger.warning("Token expired, refreshing...")
                self._update_gcp_token(force=True)
                return self.extract_cuis_batch(texts, retry_auth=False)

            response.raise_for_status()
            data = response.json()

            all_cuis = set()
            for text in texts:
                cuis = data.get(text, [])
                if isinstance(cuis, list):
                    all_cuis.update(str(c) for c in cuis if c)

            logger.info(f"Extracted {len(all_cuis)} unique CUIs")
            return all_cuis

        except Exception as e:
            logger.error(f"API call failed: {str(e)}")
            return set()

# ----------------------
# Enhanced Reducer (NO CLUSTERING, NO EMBEDDINGS)
# ----------------------
class EnhancedCUIReducer:
    def __init__(
        self,
        project_id: str,
        dataset_id: str,
        subnet_api_url: str,
        cui_description_table: str,
        cui_narrower_table: str,
        max_hierarchy_depth: int = 5,
        query_timeout: int = 300
    ):
        self.client = bigquery.Client(project=project_id)
        self.project_id = project_id
        self.dataset_id = dataset_id
        self.subnet_api_url = subnet_api_url
        self.cui_description_table = cui_description_table
        self.cui_narrower_table = cui_narrower_table
        self.max_hierarchy_depth = max_hierarchy_depth
        self.query_timeout = query_timeout

        self._hierarchy_cache = None
        self._ic_scores_cache = None
        self._description_cache = {}
        self._auth_token = None
        self._auth_expiry = 0
        self._auth_lock = threading.Lock()

    # --------------------------
    # Public API
    # --------------------------
    def reduce(
        self,
        input_cuis: List[str],
        target_reduction: float = 0.85,
        ic_threshold: Optional[float] = None
    ) -> Tuple[List[str], ReductionStats]:

        start_time = time.time()
        input_cuis = list(set(input_cuis))
        initial_count = len(input_cuis)

        hierarchy_start = time.time()
        hierarchy = self._build_hierarchy_depthwise(input_cuis)
        hierarchy_time = time.time() - hierarchy_start

        ic_scores = self._compute_ic_scores(hierarchy)
        if ic_threshold is None:
            ic_threshold = float(np.median(list(ic_scores.values())))
            logger.info(f"Using MEDIAN IC threshold: {ic_threshold:.3f}")

        rolled_up = self._semantic_rollup(
            input_cuis, hierarchy, ic_scores, ic_threshold
        )

        final_cuis = rolled_up
        final_count = len(final_cuis)

        self._fetch_descriptions(final_cuis)

        stats = ReductionStats(
            initial_count=initial_count,
            after_ic_rollup=len(rolled_up),
            final_count=final_count,
            ic_rollup_reduction_pct=self._safe_percentage(
                initial_count - len(rolled_up), initial_count
            ),
            total_reduction_pct=self._safe_percentage(
                initial_count - final_count, initial_count
            ),
            processing_time=time.time() - start_time,
            ic_threshold_used=ic_threshold,
            hierarchy_size=len(hierarchy.get("all_cuis", [])),
            api_call_time=hierarchy_time
        )

        return final_cuis, stats

    # --------------------------
    # Hierarchy depth-wise fetch
    # --------------------------
    def _build_hierarchy_depthwise(self, cuis: List[str]) -> Dict:
        if self._hierarchy_cache:
            return self._hierarchy_cache

        logger.info(f"Fetching hierarchy depth 1: {len(cuis)} CUIs")
        hierarchy = asyncio.run(self._fetch_hierarchy(cuis))
        self._hierarchy_cache = hierarchy
        logger.info(f"Built hierarchy: {len(hierarchy.get('all_cuis', []))} CUIs total")
        return hierarchy

    async def _fetch_hierarchy(self, cuis: List[str]) -> Dict:
        child_to_parents = defaultdict(list)
        parent_to_children = defaultdict(list)
        all_cuis = set(cuis)

        headers = self._get_auth_headers()
        timeout = aiohttp.ClientTimeout(total=self.query_timeout)

        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                f"{self.subnet_api_url}/subnet/",
                json={"cuis": cuis, "cross_context": False},
                headers=headers
            ) as resp:
                if resp.status != 200:
                    logger.error(f"Hierarchy fetch failed: {resp.status}")
                    return {}

                data = await resp.json()

        nodes, edges = data.get("output", ([], []))
        for p, c in edges:
            p, c = self._normalize_cui(p), self._normalize_cui(c)
            if p and c:
                parent_to_children[p].append(c)
                child_to_parents[c].append(p)
                all_cuis.update([p, c])

        return {
            "child_to_parents": dict(child_to_parents),
            "parent_to_children": dict(parent_to_children),
            "all_cuis": all_cuis
        }

    # --------------------------
    # Auth headers
    # --------------------------
    def _get_auth_headers(self) -> Dict[str, str]:
        with self._auth_lock:
            now = time.time()
            if self._auth_token and now < self._auth_expiry:
                return self._auth_token

            proc = subprocess.run(
                ["gcloud", "auth", "print-identity-token"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            if proc.returncode != 0:
                raise RuntimeError(proc.stderr)

            self._auth_token = {
                "Authorization": f"Bearer {proc.stdout.strip()}",
                "Content-Type": "application/json"
            }
            self._auth_expiry = now + 3300
            return self._auth_token

    # --------------------------
    # IC computation
    # --------------------------
    def _compute_ic_scores(self, hierarchy: Dict) -> Dict[str, float]:
        if self._ic_scores_cache:
            return self._ic_scores_cache

        parent_to_children = hierarchy.get("parent_to_children", {})
        all_cuis = hierarchy.get("all_cuis", set())
        total = len(all_cuis)

        descendant_counts = {}

        def count_descendants(cui, visited=None):
            if visited is None:
                visited = set()
            if cui in visited:
                return 0
            visited.add(cui)
            count = 0
            for child in parent_to_children.get(cui, []):
                count += 1 + count_descendants(child, visited)
            return count

        for cui in all_cuis:
            descendant_counts[cui] = count_descendants(cui)

        ic_scores = {
            cui: max(0.0, -np.log((descendant_counts[cui] + 1) / total))
            for cui in all_cuis
        }

        self._ic_scores_cache = ic_scores
        return ic_scores

    # --------------------------
    # Semantic rollup (UNCHANGED)
    # --------------------------
    def _semantic_rollup(self, cui_list, hierarchy, ic_scores, ic_threshold):
        child_to_parents = hierarchy.get("child_to_parents", {})
        rolled_up = {}

        for cui in cui_list:
            visited = set()
            queue = deque([cui])
            ancestors = []

            while queue and len(visited) < 100:
                curr = queue.popleft()
                if curr in visited:
                    continue
                visited.add(curr)
                for p in child_to_parents.get(curr, []):
                    if p not in visited:
                        ancestors.append(p)
                        queue.append(p)

            candidates = [cui] + ancestors
            valid = [c for c in candidates if ic_scores.get(c, 0) >= ic_threshold]
            rolled_up[cui] = min(valid, key=lambda c: ic_scores[c]) if valid else cui

        return list(set(rolled_up.values()))

    # --------------------------
    # Fetch descriptions
    # --------------------------
    def _fetch_descriptions(self, cuis: List[str]):
        to_fetch = [c for c in cuis if c not in self._description_cache]
        if not to_fetch:
            return

        logger.info(f"Fetching descriptions for {len(to_fetch)} CUIs...")
        query = f"""
        SELECT CUI as cui, Definition as description
        FROM `{self.project_id}.{self.dataset_id}.{self.cui_description_table}`
        WHERE CUI IN UNNEST(@cuis)
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[bigquery.ArrayQueryParameter("cuis", "STRING", to_fetch)]
        )
        df = self.client.query(query, job_config=job_config).result(timeout=60).to_dataframe()

        for _, row in df.iterrows():
            self._description_cache[row['cui']] = row['description']

    # --------------------------
    # Utilities
    # --------------------------
    @staticmethod
    def _normalize_cui(cui: str) -> Optional[str]:
        m = re.match(r"^(C\d{7})(?:-\d+)?$", str(cui))
        return m.group(1) if m else None

    @staticmethod
    def _safe_percentage(numerator: float, denominator: float) -> float:
        return (numerator / denominator * 100) if denominator else 0.0

# --------------------------
# Main
# --------------------------
if __name__ == "__main__":

    texts = ["pharmacotherapy depression type 2 diabetes"]

    api_client = CUIAPIClient(api_base_url=api_url)
    extracted_cuis = api_client.extract_cuis_batch(texts)

    filtered_cuis = filter_allowed_cuis(
        extracted_cuis, project_id, dataset_id
    )

    reducer = EnhancedCUIReducer(
        project_id,
        dataset_id,
        subnet_api_url=subnet_api_url,
        cui_description_table=cui_desc_table,
        cui_narrower_table=descendants_table
    )

    final_cuis, stats = reducer.reduce(filtered_cuis)

    logger.info(f"Final CUIs ({len(final_cuis)}): {final_cuis}")
    logger.info(f"Reduction Stats: {stats.to_dict()}")
