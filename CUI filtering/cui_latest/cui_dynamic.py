import logging
import time
import threading
import subprocess
import requests
import numpy as np

from typing import List, Dict, Set, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict, deque

from google.cloud import bigquery
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_distances


# --------------------------------------------------
# Configure logging
# --------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# --------------------------------------------------
# Filter function
# --------------------------------------------------
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
            query_parameters=[bigquery.ArrayQueryParameter("cuis", "STRING", list(cuis))]
        )
        df = client.query(query, job_config=job_config).result(timeout=60).to_dataframe()
        allowed = df["CUI"].tolist()
        logger.info(f"{len(allowed)} CUIs after filter")
        return allowed

    except Exception as e:
        logger.error(f"Failed to filter CUIs by SAB: {str(e)}")
        return []


# --------------------------------------------------
# Stats container
# --------------------------------------------------
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


# --------------------------------------------------
# API Client
# --------------------------------------------------
class CUIAPIClient:

    _token_lock = threading.Lock()
    _cached_token = None
    _token_expiry = 0

    def __init__(self, api_base_url: str, timeout: int = 60, top_k: int = 3, max_retries: int = 3):
        self.api_base_url = api_base_url.rstrip("/")
        self.timeout = timeout
        self.top_k = top_k

        self.session = requests.Session()
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["POST"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy, pool_connections=10, pool_maxsize=20)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

        self._update_gcp_token()

    def _update_gcp_token(self, force: bool = False):
        with self._token_lock:
            now = time.time()
            if not force and self._cached_token and now < self._token_expiry:
                return self._cached_token

            try:
                result = subprocess.run(
                    ["gcloud", "auth", "print-identity-token"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    universal_newlines=True,
                    timeout=10,
                    check=False,
                )

                if result.returncode != 0:
                    raise Exception(result.stderr)

                token = result.stdout.strip()
                if not token:
                    raise Exception("Empty token")

                self._cached_token = {
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                }
                self._token_expiry = now + 3300
                return self._cached_token

            except Exception as e:
                raise Exception(f"GCP auth failed: {str(e)}")

    def extract_cuis_batch(self, texts: List[str], retry_auth: bool = True) -> Set[str]:
        if not texts:
            return set()

        payload = {"query_texts": texts, "top_k": self.top_k}
        headers = self._update_gcp_token()

        try:
            response = self.session.post(
                self.api_base_url, json=payload, headers=headers, timeout=self.timeout
            )

            if response.status_code == 401 and retry_auth:
                logger.warning("Token expired, refreshing...")
                self._update_gcp_token(force=True)
                return self.extract_cuis_batch(texts, retry_auth=False)

            response.raise_for_status()
            data = response.json()

            cuis = set()
            for text in texts:
                for c in data.get(text, []):
                    if c:
                        cuis.add(str(c))

            logger.info(f"Extracted {len(cuis)} unique CUIs")
            return cuis

        except Exception as e:
            logger.error(f"API call failed: {str(e)}")
            return set()


# --------------------------------------------------
# Enhanced CUI Reducer (MEDIAN IC ONLY)
# --------------------------------------------------
class EnhancedCUIReducer:

    def __init__(
        self,
        project_id: str,
        dataset_id: str,
        mrrel_table: str = "MRREL",
        cui_description_table: str = "cui_descriptions",
        cui_embeddings_table: str = "cui_embeddings",
        max_hierarchy_depth: int = 1,
        query_timeout: int = 300,
    ):
        self.client = bigquery.Client(project=project_id)
        self.project_id = project_id
        self.dataset_id = dataset_id
        self.mrrel_table = mrrel_table
        self.cui_description_table = cui_description_table
        self.cui_embeddings_table = cui_embeddings_table
        self.max_hierarchy_depth = max_hierarchy_depth
        self.query_timeout = query_timeout

        self._hierarchy_cache = None
        self._ic_scores_cache = None
        self._description_cache = {}


    # --------------------------------------------------
    # Reduce (MEDIAN IC THRESHOLD)
    # --------------------------------------------------
    def reduce(
        self,
        input_cuis: List[str],
        target_reduction: float = 0.85,
        ic_threshold: Optional[float] = None,
        semantic_threshold: float = 0.88,
        use_semantic_clustering: bool = True,
    ) -> Tuple[List[str], ReductionStats]:

        start_time = time.time()
        input_cuis = list(set(c for c in input_cuis if c))
        initial_count = len(input_cuis)

        if initial_count == 0:
            logger.warning("Empty input CUI list")
            return [], self._create_empty_stats(start_time)

        try:
            hierarchy = self._build_hierarchy_safe(input_cuis)
            ic_scores = self._compute_ic_scores_safe(hierarchy)

            ic_threshold = self._determine_median_ic_threshold(ic_scores, ic_threshold)

            rolled_up = self._semantic_rollup_with_ic_safe(
                input_cuis, hierarchy, ic_scores, ic_threshold
            )

            after_rollup = len(rolled_up)
            rollup_reduction = self._safe_percentage(
                initial_count - after_rollup, initial_count
            )

            if use_semantic_clustering:
                final_cuis = self._semantic_clustering_safe(
                    rolled_up, ic_scores, semantic_threshold
                )
            else:
                final_cuis = rolled_up

            final_count = len(final_cuis)
            total_reduction = self._safe_percentage(
                initial_count - final_count, initial_count
            )

            stats = ReductionStats(
                initial_count=initial_count,
                after_ic_rollup=after_rollup,
                final_count=final_count,
                ic_rollup_reduction_pct=rollup_reduction,
                semantic_clustering_reduction_pct=total_reduction - rollup_reduction,
                total_reduction_pct=total_reduction,
                processing_time=time.time() - start_time,
                ic_threshold_used=ic_threshold,
                hierarchy_size=len(hierarchy.get("all_cuis", [])),
            )

            logger.info(
                f"Reduction complete: {initial_count} → {final_count} "
                f"({total_reduction:.1f}%) using median IC={ic_threshold:.3f}"
            )

            return final_cuis, stats

        except Exception as e:
            logger.error(f"Critical error in reduction pipeline: {str(e)}")
            return input_cuis, self._create_error_stats(initial_count, start_time, str(e))


    # --------------------------------------------------
    # Median IC threshold
    # --------------------------------------------------
    def _determine_median_ic_threshold(
        self,
        ic_scores: Dict[str, float],
        explicit_threshold: Optional[float],
    ) -> float:
        try:
            if explicit_threshold is not None:
                logger.info(f"Using explicit IC threshold: {explicit_threshold}")
                return float(explicit_threshold)

            values = list(ic_scores.values())
            if not values:
                logger.warning("No IC values found, using fallback 5.0")
                return 5.0

            median_ic = float(np.median(values))
            logger.info(f"Using median IC threshold: {median_ic:.3f}")
            return median_ic

        except Exception as e:
            logger.warning(f"Median IC computation failed: {str(e)}")
            return 5.0


    # --------------------------------------------------
    # Remaining methods (UNCHANGED from your code)
    # --------------------------------------------------
    # _build_hierarchy_safe
    # _compute_ic_scores_safe
    # _semantic_rollup_with_ic_safe
    # _semantic_clustering_safe
    # get_cui_descriptions
    # helpers (_safe_percentage, _create_empty_stats, _create_error_stats)
