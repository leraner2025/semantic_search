import os
import time
import numpy as np
import logging
import subprocess
import json
from typing import List, Set, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from google.cloud import bigquery
from sklearn.cluster import AgglomerativeClustering
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import threading

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================
# CONFIGURATION
# ============================================
class Config:
    """Configuration for CUI Reduction System"""
    # GCP Configuration
    PROJECT_ID = os.getenv("GCP_PROJECT_ID", "my-gcp-project")
    DATASET_ID = os.getenv("BIGQUERY_DATASET_ID", "my_dataset")
    # API Configuration
    NER_API_URL = os.getenv("NER_API_URL", "https://api.example.com/ner")
    API_TIMEOUT = int(os.getenv("API_TIMEOUT", "60"))
    API_TOP_K = int(os.getenv("API_TOP_K", "3"))
    # BigQuery Table Names
    MRREL_TABLE = os.getenv("MRREL_TABLE", "MRREL")
    CUI_DESCRIPTION_TABLE = os.getenv("CUI_DESCRIPTION_TABLE", "cui_descriptions")
    CUI_NARROWER_TABLE = os.getenv("CUI_NARROWER_TABLE", "cui_narrower_concepts")
    # Performance tuning
    MAX_HIERARCHY_DEPTH = int(os.getenv("MAX_HIERARCHY_DEPTH", "3"))
    BQ_QUERY_TIMEOUT = int(os.getenv("BQ_QUERY_TIMEOUT", "300"))  # 5 minutes
    MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
    # Reduction Parameters
    USE_SEMANTIC_CLUSTERING = os.getenv("USE_SEMANTIC_CLUSTERING", "True").lower() == "true"

@dataclass
class ReductionStats:
    """Statistics for the reduction process"""
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
        """Convert to dictionary for JSON serialization"""
        return asdict(self)

# ============================================
# API Client
# ============================================
class CUIAPIClient:
    """Thread-safe client for GCP-based CUI extraction API"""

    _token_lock = threading.Lock()
    _cached_token = None
    _token_expiry = 0

    def __init__(self, api_base_url: str, timeout: int = 60, top_k: int = 3):
        self.api_base_url = api_base_url.rstrip('/')
        self.timeout = timeout
        self.top_k = top_k

        # Configure session with retry and pooling
        self.session = requests.Session()
        retry_strategy = Retry(
            total=Config.MAX_RETRIES,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["POST"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy, pool_connections=10, pool_maxsize=20)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

        # Get initial token
        self._update_gcp_token()

    def _update_gcp_token(self, force: bool = False):
        """Get GCP identity token with caching"""
        with self._token_lock:
            current_time = time.time()
            if not force and self._cached_token and current_time < self._token_expiry:
                return self._cached_token
            try:
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
                identity_token = result.stdout.strip()
                if not identity_token:
                    raise Exception("Empty token received from gcloud")
                self._cached_token = {
                    "Authorization": f"Bearer {identity_token}",
                    "Content-Type": "application/json"
                }
                self._token_expiry = current_time + 3300  # 55 minutes
                logger.info("GCP authentication token updated")
                return self._cached_token
            except Exception as e:
                raise Exception(f"Failed to get GCP token: {str(e)}")

    def extract_cuis_batch(self, texts: List[str], retry_auth: bool = True) -> Set[str]:
        if not texts:
            return set()
        payload = {"query_texts": texts, "top_k": self.top_k}
        headers = self._update_gcp_token()
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
            logger.info(f"Extracted {len(all_cuis)} unique CUIs from {len(texts)} texts")
            return all_cuis
        except Exception as e:
            logger.error(f"API extraction failed: {str(e)}")
            return set()

    def extract_cuis_parallel(self, texts: List[str], batch_size: int = 50, max_workers: int = 5) -> Set[str]:
        if not texts:
            return set()
        batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
        all_cuis = set()
        failed_batches = 0
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_batch = {executor.submit(self.extract_cuis_batch, batch): i for i, batch in enumerate(batches)}
            for future in as_completed(future_to_batch):
                batch_idx = future_to_batch[future]
                try:
                    cuis = future.result(timeout=self.timeout + 10)
                    all_cuis.update(cuis)
                except Exception as e:
                    failed_batches += 1
                    logger.error(f"Batch {batch_idx + 1}/{len(batches)} failed: {str(e)}")
        if failed_batches > 0:
            logger.warning(f"{failed_batches}/{len(batches)} batches failed")
        logger.info(f"Extracted {len(all_cuis)} total unique CUIs")
        return all_cuis

# ============================================
# Enhanced Reducer
# ============================================
class EnhancedCUIReducer:
    """Production-grade CUI reducer"""

    def __init__(
        self,
        project_id: str,
        dataset_id: str,
        mrrel_table: str = "MRREL",
        cui_description_table: str = "cui_descriptions",
        cui_narrower_table: str = "cui_narrower_concepts",
        max_hierarchy_depth: int = 3
    ):
        try:
            self.client = bigquery.Client(project=project_id)
        except Exception as e:
            raise Exception(f"Failed to initialize BigQuery client: {str(e)}")
        self.project_id = project_id
        self.dataset_id = dataset_id
        self.mrrel_table = mrrel_table
        self.cui_description_table = cui_description_table
        self.cui_narrower_table = cui_narrower_table
        self.max_hierarchy_depth = max_hierarchy_depth

        # Caches
        self._hierarchy_cache = None
        self._ic_scores_cache = None
        self._description_cache = {}
        logger.info("EnhancedCUIReducer initialized")

    def reduce(
        self,
        input_cuis: List[str],
        use_semantic_clustering: bool = True
    ) -> Tuple[List[str], ReductionStats]:
        start_time = time.time()
        initial_count = len(input_cuis)
        if initial_count == 0:
            logger.warning("Empty input CUI list")
            return [], self._create_empty_stats(start_time)

        # Clean input
        input_cuis = list(set(str(c).strip() for c in input_cuis if c and str(c).strip()))
        initial_count = len(input_cuis)
        logger.info(f"Starting reduction: {initial_count} CUIs")

        try:
            hierarchy = self._build_hierarchy_safe(input_cuis)
            ic_scores = self._compute_ic_scores_safe(hierarchy)
            # Median-based IC threshold
            ic_values = list(ic_scores.values())
            ic_threshold = float(np.median(ic_values)) if ic_values else 5.0
            logger.info(f"Using median-based IC threshold: {ic_threshold:.3f}")

            # Stage 1: IC-based rollup
            rolled_up_cuis = self._semantic_rollup_with_ic_safe(input_cuis, hierarchy, ic_scores, ic_threshold)
            after_rollup = len(rolled_up_cuis)
            rollup_reduction = self._safe_percentage(initial_count - after_rollup, initial_count)
            logger.info(f"Stage 1 complete: {after_rollup} CUIs ({rollup_reduction:.1f}% reduction)")

            # Stage 2: Semantic clustering (optional)
            if use_semantic_clustering:
                final_cuis = self._semantic_clustering_safe(rolled_up_cuis, ic_scores)
            else:
                final_cuis = rolled_up_cuis
            final_count = len(final_cuis)
            clustering_reduction = self._safe_percentage(after_rollup - final_count, initial_count)

            # Stats
            total_reduction = self._safe_percentage(initial_count - final_count, initial_count)
            processing_time = time.time() - start_time
            stats = ReductionStats(
                initial_count=initial_count,
                after_ic_rollup=after_rollup,
                final_count=final_count,
                ic_rollup_reduction_pct=rollup_reduction,
                semantic_clustering_reduction_pct=clustering_reduction,
                total_reduction_pct=total_reduction,
                processing_time=processing_time,
                ic_threshold_used=ic_threshold,
                hierarchy_size=len(hierarchy.get('all_cuis', set()))
            )
            logger.info(f"Reduction complete: {initial_count} → {final_count} ({total_reduction:.1f}%)")
            return final_cuis, stats

        except Exception as e:
            logger.error(f"Critical error in reduction pipeline: {str(e)}")
            return input_cuis, self._create_error_stats(initial_count, start_time, str(e))

    # ====== Helper Methods ======
    def _build_hierarchy_safe(self, relevant_cuis: List[str]) -> Dict:
        if self._hierarchy_cache is not None:
            return self._hierarchy_cache
        child_to_parents = defaultdict(list)
        parent_to_children = defaultdict(list)
        all_cuis = set(relevant_cuis)
        try:
            visited = set()
            frontier = set(relevant_cuis)
            for depth in range(self.max_hierarchy_depth):
                if not frontier:
                    break
                query = f"""
                SELECT DISTINCT cui1, cui2, rel
                FROM `{self.project_id}.{self.dataset_id}.{self.mrrel_table}`
                WHERE (cui1 IN UNNEST(@frontier) OR cui2 IN UNNEST(@frontier))
                  AND rel IN ('PAR', 'CHD')
                """
                job_config = bigquery.QueryJobConfig(
                    query_parameters=[bigquery.ArrayQueryParameter("frontier", "STRING", list(frontier))],
                    timeout_ms=Config.BQ_QUERY_TIMEOUT * 1000)
                df = self.client.query(query, job_config=job_config).result(timeout=Config.BQ_QUERY_TIMEOUT).to_dataframe()
                if df.empty:
                    break
                next_frontier = set()
                for _, row in df.iterrows():
                    cui1, cui2, rel = str(row['cui1']), str(row['cui2']), str(row['rel'])
                    if rel == 'PAR':
                        parent_to_children[cui1].append(cui2)
                        child_to_parents[cui2].append(cui1)
                    elif rel == 'CHD':
                        parent_to_children[cui2].append(cui1)
                        child_to_parents[cui1].append(cui2)
                    all_cuis.update([cui1, cui2])
                    next_frontier.update([cui1, cui2])
                visited.update(frontier)
                frontier = next_frontier - visited
            self._add_narrower_concepts(relevant_cuis, parent_to_children, child_to_parents, all_cuis)
        except Exception as e:
            logger.error(f"Failed to build hierarchy: {str(e)}")
        hierarchy = {
            'child_to_parents': dict(child_to_parents),
            'parent_to_children': dict(parent_to_children),
            'all_cuis': all_cuis
        }
        self._hierarchy_cache = hierarchy
        return hierarchy

    def _add_narrower_concepts(self, relevant_cuis, parent_to_children, child_to_parents, all_cuis):
        query = f"""
        WITH narrower_raw AS (
          SELECT CUI as parent_cui, NarrowerCUI as narrower_list
          FROM `{self.project_id}.{self.dataset_id}.{self.cui_narrower_table}`
          WHERE CUI IN UNNEST(@cuis)
            AND NarrowerCUI IS NOT NULL
            AND LENGTH(NarrowerCUI) > 0
        )
        SELECT parent_cui, TRIM(child_cui) as child_cui
        FROM narrower_raw, UNNEST(SPLIT(narrower_list, ',')) as child_cui
        WHERE LENGTH(TRIM(child_cui)) > 0
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[bigquery.ArrayQueryParameter("cuis", "STRING", relevant_cuis[:1000])],
            timeout_ms=30000)
        df = self.client.query(query, job_config=job_config).result(timeout=30).to_dataframe()
        for _, row in df.iterrows():
            parent, child = str(row['parent_cui']), str(row['child_cui'])
            if parent and child:
                parent_to_children[parent].append(child)
                child_to_parents[child].append(parent)
                all_cuis.update([parent, child])

    def _compute_ic_scores_safe(self, hierarchy: Dict) -> Dict[str, float]:
        if self._ic_scores_cache is not None:
            return self._ic_scores_cache
        parent_to_children = hierarchy.get('parent_to_children', {})
        all_cuis = hierarchy.get('all_cuis', set())
        total = len(all_cuis)
        if total == 0:
            return {}
        descendant_counts = {}
        def count_descendants(cui, visited=None):
            if visited is None:
                visited = set()
            if cui in visited or cui in descendant_counts:
                return descendant_counts.get(cui, 0)
            visited.add(cui)
            children = parent_to_children.get(cui, [])
            count = len(children)
            for child in children:
                count += count_descendants(child, visited)
            descendant_counts[cui] = count
            return count
        for cui in all_cuis:
            if cui not in descendant_counts:
                try:
                    count_descendants(cui)
                except RecursionError:
                    descendant_counts[cui] = 0
        ic_scores = {cui: max(0.0, -np.log((descendant_counts.get(cui, 0)+1)/total)) for cui in all_cuis}
        self._ic_scores_cache = ic_scores
        return ic_scores

    def _semantic_rollup_with_ic_safe(self, cui_list, hierarchy, ic_scores, ic_threshold):
        try:
            child_to_parents = hierarchy.get('child_to_parents', {})
            rolled_up = {}
            for cui in cui_list:
                ancestors = []
                visited = set()
                queue = deque([cui])
                while queue and len(visited) < 100:
                    current = queue.popleft()
                    if current in visited:
                        continue
                    visited.add(current)
                    for parent in child_to_parents.get(current, []):
                        if parent not in visited:
                            ancestors.append(parent)
                            queue.append(parent)
                candidates = [cui] + ancestors
                valid = [c for c in candidates if ic_scores.get(c, 0) >= ic_threshold]
                rolled_up[cui] = min(valid, key=lambda c: ic_scores.get(c, float('inf'))) if valid else cui
            return list(set(rolled_up.values()))
        except Exception as e:
            logger.error(f"Semantic rollup failed: {str(e)}")
            return cui_list

    def _semantic_clustering_safe(self, cui_list, ic_scores):
        if len(cui_list) <= 1:
            return cui_list
        try:
            child_to_parents = self._hierarchy_cache.get('child_to_parents', {}) if self._hierarchy_cache else {}
            all_cuis_set = set(cui_list)
            visited = set()
            final_cuis = []
            for cui in cui_list:
                if cui in visited:
                    continue
                ancestors_cui = set(child_to_parents.get(cui, []))
                cluster = [cui]
                for other_cui in all_cuis_set - {cui}:
                    ancestors_other = set(child_to_parents.get(other_cui, []))
                    if ancestors_cui.intersection(ancestors_other):
                        cluster.append(other_cui)
                rep = max(cluster, key=lambda x: ic_scores.get(x, 0))
                final_cuis.append(rep)
                visited.update(cluster)
            return list(set(final_cuis))
        except Exception as e:
            logger.error(f"Hierarchy-based clustering failed: {str(e)}")
            return cui_list
            
    def get_cui_descriptions(self, cui_list: List[str]) -> Dict[str, str]:
        """Retrieve descriptions with caching"""
        if not cui_list:
            return {}
        # Check cache first
        uncached = [c for c in cui_list if c not in self._description_cache]
        if uncached:
            try:
                query = f"""
                SELECT CUI as cui, Definition as description
                FROM `{self.project_id}.{self.dataset_id}.{self.cui_description_table}`
                WHERE CUI IN UNNEST(@cuis)
                LIMIT 50000
                """
                job_config = bigquery.QueryJobConfig(
                    query_parameters=[bigquery.ArrayQueryParameter("cuis", "STRING", uncached)]
                )
                df = self.client.query(query, job_config=job_config, timeout=30).result().to_dataframe()
                new_descriptions = dict(zip(df['cui'], df['description']))
                self._description_cache.update(new_descriptions)
            except Exception as e:
                logger.error(f"Failed to fetch descriptions: {str(e)}")
        return {cui: self._description_cache.get(cui, "N/A") for cui in cui_list}

    def get_ic_scores(self, cui_list: List[str]) -> Dict[str, float]:
        """Get IC scores for CUIs"""
        if self._ic_scores_cache is None:
            return {cui: 0.0 for cui in cui_list}
        return {cui: self._ic_scores_cache.get(cui, 0.0) for cui in cui_list}

    @staticmethod
    def _safe_percentage(numerator: float, denominator: float) -> float:
        """Safe percentage calculation"""
        return (numerator / denominator * 100) if denominator > 0 else 0.0

    @staticmethod
    def _create_empty_stats(start_time: float) -> ReductionStats:
        """Create stats for empty input"""
        return ReductionStats(
            initial_count=0,
            after_ic_rollup=0,
            final_count=0,
            ic_rollup_reduction_pct=0.0,
            semantic_clustering_reduction_pct=0.0,
            total_reduction_pct=0.0,
            processing_time=time.time() - start_time,
            ic_threshold_used=0.0
        )

    @staticmethod
    def _create_error_stats(initial_count: int, start_time: float, error: str) -> ReductionStats:
        """Create stats for error scenario"""
        logger.error(f"Returning original CUIs due to error: {error}")
        return ReductionStats(
            initial_count=initial_count,
            after_ic_rollup=initial_count,
            final_count=initial_count,
            ic_rollup_reduction_pct=0.0,
            semantic_clustering_reduction_pct=0.0,
            total_reduction_pct=0.0,
            processing_time=time.time() - start_time,
            ic_threshold_used=0.0
        )


class CUIReductionPipeline:
    """End-to-end pipeline with comprehensive error handling"""

    def __init__(self, api_client, cui_reducer: EnhancedCUIReducer):
        self.api_client = api_client
        self.cui_reducer = cui_reducer

    def process_texts(
        self,
        texts: List[str],
        use_semantic_clustering: bool = True,
        batch_size: int = 50,
        max_workers: int = 5
    ) -> Tuple[List[str], Dict[str, str], Optional[ReductionStats]]:
        """
        Complete pipeline with guaranteed non-error return
        Always returns valid data, even on partial failures
        """
        if not texts:
            logger.warning("Empty text list provided")
            return [], {}, None

        start_time = time.time()
        try:
            # Extract CUIs
            logger.info(f"Extracting CUIs from {len(texts)} texts...")
            if len(texts) > batch_size:
                initial_cuis = self.api_client.extract_cuis_parallel(
                    texts, batch_size=batch_size, max_workers=max_workers
                )
            else:
                initial_cuis = self.api_client.extract_cuis_batch(texts)

            if not initial_cuis:
                logger.warning("No CUIs extracted from texts")
                return [], {}, None

            # Reduce CUIs
            reduced_cuis, stats = self.cui_reducer.reduce(
                list(initial_cuis),
                use_semantic_clustering=use_semantic_clustering
            )

            # Add API call time to stats
            stats.api_call_time = time.time() - start_time - stats.processing_time

            # Get descriptions
            logger.info("Fetching descriptions...")
            descriptions = self.cui_reducer.get_cui_descriptions(reduced_cuis)

            return reduced_cuis, descriptions, stats

        except Exception as e:
            logger.error(f"Pipeline error: {str(e)}")
            return [], {}, None


def main():
    """Example usage with comprehensive error handling"""
    try:
        # Initialize
        logger.info("Initializing CUI Reduction System...")
        api_client = CUIAPIClient(
            api_base_url=Config.NER_API_URL,
            timeout=Config.API_TIMEOUT,
            top_k=Config.API_TOP_K
        )

        cui_reducer = EnhancedCUIReducer(
            project_id=Config.PROJECT_ID,
            dataset_id=Config.DATASET_ID,
            mrrel_table=Config.MRREL_TABLE,
            cui_description_table=Config.CUI_DESCRIPTION_TABLE,
            cui_narrower_table=Config.CUI_NARROWER_TABLE,
            max_hierarchy_depth=Config.MAX_HIERARCHY_DEPTH
        )

        pipeline = CUIReductionPipeline(api_client, cui_reducer)

        # Example texts
        texts = ["History of Type 2 Diabetes Mellitus and hypertension."]

        # Process
        logger.info("Processing texts...")
        reduced_cuis, descriptions, stats = pipeline.process_texts(
            texts,
            use_semantic_clustering=Config.USE_SEMANTIC_CLUSTERING
        )

        # Display results
        print("\n" + "=" * 80)
        print("ENHANCED CUI REDUCTION RESULTS")
        print("=" * 80)

        if stats is None:
            print("\n No CUIs were extracted. Check:")
            print("  • API endpoint configuration")
            print("  • Authentication token")
            print("  • Input text contains medical concepts")
        else:
            print(f"\n Initial CUIs: {stats.initial_count}")
            print(f" After IC Rollup: {stats.after_ic_rollup} ({stats.ic_rollup_reduction_pct:.1f}%)")
            print(f" Final CUIs: {stats.final_count} ({stats.total_reduction_pct:.1f}% total)")
            print(f" IC Threshold: {stats.ic_threshold_used:.3f}")
            print(f" Hierarchy Size: {stats.hierarchy_size} CUIs")
            print(f" API Time: {stats.api_call_time:.2f}s")
            print(f" Reduction Time: {stats.processing_time:.2f}s")
            print(f" Total Time: {stats.api_call_time + stats.processing_time:.2f}s")

            if reduced_cuis:
                print("\n" + "=" * 80)
                print("SAMPLE RESULTS (first 10)")
                print("=" * 80)
                ic_scores = cui_reducer.get_ic_scores(reduced_cuis[:10])
                for i, cui in enumerate(reduced_cuis[:10], 1):
                    desc = descriptions.get(cui, "No description")
                    ic = ic_scores.get(cui, 0.0)
                    print(f"{i:2}. {cui} (IC={ic:.2f}): {desc}")
                if len(reduced_cuis) > 10:
                    print(f"\n... and {len(reduced_cuis) - 10} more CUIs")

            # Export stats as JSON
            print("\n" + "=" * 80)
            print("Stats JSON:")
            print(json.dumps(stats.to_dict(), indent=2))

    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}", exc_info=True)
        raise
if __name__ == "__main__":
    main()


