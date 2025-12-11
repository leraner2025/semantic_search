import time
import logging
from typing import List, Dict, Optional, Tuple
from collections import deque, defaultdict
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from dataclasses import dataclass

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


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


class EnhancedCUIReducer:
    """Enhanced CUI reducer with IC-based rollup and hierarchy-based clustering (no embeddings)"""

    def __init__(
        self,
        project_id: str,
        dataset_id: str,
        mrrel_table: str,
        cui_description_table: str,
        cui_embeddings_table: str,
        cui_narrower_table: str,
        max_hierarchy_depth: int = 5,
    ):
        self.project_id = project_id
        self.dataset_id = dataset_id
        self.mrrel_table = mrrel_table
        self.cui_description_table = cui_description_table
        self.cui_embeddings_table = cui_embeddings_table
        self.cui_narrower_table = cui_narrower_table
        self.max_hierarchy_depth = max_hierarchy_depth

        # Caches
        self._description_cache: Dict[str, str] = {}
        self._ic_scores_cache: Optional[Dict[str, float]] = None

    ##############################
    # IC-based Rollup
    ##############################
    def _semantic_rollup_with_ic_safe(
        self,
        cui_list: List[str],
        hierarchy: Dict,
        ic_scores: Dict[str, float],
        ic_threshold: float = 1.0
    ) -> List[str]:
        """Rollup CUIs based on IC threshold (fixed at 1.0)"""
        try:
            child_to_parents = hierarchy.get('child_to_parents', {})
            rolled_up = {}

            for cui in cui_list:
                try:
                    # Get ancestors with BFS
                    ancestors = []
                    visited = set()
                    queue = deque([cui])

                    while queue and len(visited) < 100:  # safety limit
                        current = queue.popleft()
                        if current in visited:
                            continue
                        visited.add(current)
                        for parent in child_to_parents.get(current, []):
                            if parent not in visited:
                                ancestors.append(parent)
                                queue.append(parent)

                    # Include self and ancestors that meet IC threshold
                    candidates = [cui] + ancestors
                    valid = [c for c in candidates if ic_scores.get(c, 0) >= ic_threshold]

                    if valid:
                        # Pick the lowest IC (most general) among valid
                        rolled_up[cui] = min(valid, key=lambda c: ic_scores.get(c, float('inf')))
                    else:
                        rolled_up[cui] = cui

                except Exception as e:
                    logger.debug(f"Rollup failed for {cui}: {str(e)}")
                    rolled_up[cui] = cui

            return list(set(rolled_up.values()))
        except Exception as e:
            logger.error(f"Semantic rollup failed: {str(e)}")
            return cui_list

    ##############################
    # Hierarchy-based Clustering (no embeddings)
    ##############################
    def _semantic_clustering_safe(
        self,
        cui_list: List[str],
        ic_scores: Dict[str, float],
        min_common_ancestors: int = 1
    ) -> List[str]:
        """Cluster CUIs based on IC + hierarchy similarity (no embeddings)"""
        if len(cui_list) <= 1:
            return cui_list

        try:
            # Simple hierarchy similarity: group CUIs with same parent(s)
            clusters = defaultdict(list)
            for cui in cui_list:
                parent_key = tuple(sorted([p for p in ic_scores if ic_scores[p] >= 1.0]))  # simple proxy
                clusters[parent_key].append(cui)

            final_cuis = []
            for cluster_cuis in clusters.values():
                if len(cluster_cuis) == 1:
                    final_cuis.append(cluster_cuis[0])
                else:
                    # Pick representative by lowest IC
                    rep = min(cluster_cuis, key=lambda c: ic_scores.get(c, float('inf')))
                    final_cuis.append(rep)

            return final_cuis

        except Exception as e:
            logger.error(f"Clustering failed: {str(e)}")
            return cui_list

    ##############################
    # IC Scores & Descriptions
    ##############################
    def get_cui_descriptions(self, cui_list: List[str]) -> Dict[str, str]:
        if not cui_list:
            return {}

        uncached = [c for c in cui_list if c not in self._description_cache]

        # For simplicity, here we just return dummy descriptions
        for c in uncached:
            self._description_cache[c] = f"Description for {c}"

        return {cui: self._description_cache.get(cui, "N/A") for cui in cui_list}

    def get_ic_scores(self, cui_list: List[str]) -> Dict[str, float]:
        """Fixed IC score example: you can preload from DB"""
        if self._ic_scores_cache is None:
            # Dummy: assign 1.0 to all
            return {cui: 1.0 for cui in cui_list}
        return {cui: self._ic_scores_cache.get(cui, 1.0) for cui in cui_list}

    ##############################
    # Helper Stats
    ##############################
    @staticmethod
    def _safe_percentage(numerator: float, denominator: float) -> float:
        return (numerator / denominator * 100) if denominator > 0 else 0.0

    @staticmethod
    def _create_empty_stats(start_time: float) -> ReductionStats:
        return ReductionStats(
            initial_count=0,
            after_ic_rollup=0,
            final_count=0,
            ic_rollup_reduction_pct=0.0,
            semantic_clustering_reduction_pct=0.0,
            total_reduction_pct=0.0,
            processing_time=time.time() - start_time,
            ic_threshold_used=1.0
        )

    @staticmethod
    def _create_error_stats(initial_count: int, start_time: float, error: str) -> ReductionStats:
        logger.error(f"Returning original CUIs due to error: {error}")
        return ReductionStats(
            initial_count=initial_count,
            after_ic_rollup=initial_count,
            final_count=initial_count,
            ic_rollup_reduction_pct=0.0,
            semantic_clustering_reduction_pct=0.0,
            total_reduction_pct=0.0,
            processing_time=time.time() - start_time,
            ic_threshold_used=1.0
        )
from typing import List, Dict, Optional, Tuple
import time
import json
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class CUIAPIClient:
    """Dummy API client for CUI extraction"""

    def __init__(self, api_base_url: str, timeout: int = 30, top_k: int = 20):
        self.api_base_url = api_base_url
        self.timeout = timeout
        self.top_k = top_k

    def extract_cuis_batch(self, texts: List[str]) -> List[str]:
        """Simulate batch extraction"""
        # Dummy: just return "CUI_x" for each text
        return [f"CUI_{i+1}" for i in range(len(texts))]

    def extract_cuis_parallel(self, texts: List[str], batch_size: int = 50, max_workers: int = 5) -> List[str]:
        """Simulate parallel extraction"""
        return self.extract_cuis_batch(texts)


class CUIReductionPipeline:
    """End-to-end pipeline with IC rollup + hierarchy clustering (no embeddings)"""

    def __init__(self, api_client: CUIAPIClient, cui_reducer: EnhancedCUIReducer):
        self.api_client = api_client
        self.cui_reducer = cui_reducer

    def process_texts(
        self,
        texts: List[str],
        target_reduction: float = 0.85,
        semantic_threshold: float = 0.88,
        use_semantic_clustering: bool = True,
        batch_size: int = 50,
        max_workers: int = 5
    ) -> Tuple[List[str], Dict[str, str], Optional[ReductionStats]]:
        start_time = time.time()
        if not texts:
            logger.warning("Empty text list provided")
            return [], {}, None

        try:
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

            ic_scores = self.cui_reducer.get_ic_scores(initial_cuis)

            # Stage 1: IC-based rollup
            rolled_up = self.cui_reducer._semantic_rollup_with_ic_safe(
                initial_cuis,
                hierarchy={'child_to_parents': {}},  # Replace with actual hierarchy if available
                ic_scores=ic_scores,
                ic_threshold=1.0
            )

            after_rollup_count = len(rolled_up)
            ic_reduction_pct = self.cui_reducer._safe_percentage(
                len(initial_cuis) - after_rollup_count, len(initial_cuis)
            )

            # Stage 2: Clustering (hierarchy-based, no embeddings)
            if use_semantic_clustering:
                final_cuis = self.cui_reducer._semantic_clustering_safe(
                    rolled_up, ic_scores=ic_scores
                )
            else:
                final_cuis = rolled_up

            final_count = len(final_cuis)
            semantic_reduction_pct = self.cui_reducer._safe_percentage(
                after_rollup_count - final_count, after_rollup_count
            )
            total_reduction_pct = self.cui_reducer._safe_percentage(
                len(initial_cuis) - final_count, len(initial_cuis)
            )

            stats = ReductionStats(
                initial_count=len(initial_cuis),
                after_ic_rollup=after_rollup_count,
                final_count=final_count,
                ic_rollup_reduction_pct=ic_reduction_pct,
                semantic_clustering_reduction_pct=semantic_reduction_pct,
                total_reduction_pct=total_reduction_pct,
                processing_time=time.time() - start_time,
                ic_threshold_used=1.0,
                hierarchy_size=len(rolled_up)
            )

            # Fetch descriptions
            descriptions = self.cui_reducer.get_cui_descriptions(final_cuis)

            return final_cuis, descriptions, stats

        except Exception as e:
            logger.error(f"Pipeline error: {str(e)}")
            return [], {}, None


def main():
    """Example usage of the robust CUI Reduction Pipeline"""

    try:
        logger.info("Initializing CUI Reduction System...")

        api_client = CUIAPIClient(
            api_base_url="https://dummy-ner-api.com",
            timeout=30,
            top_k=20
        )

        cui_reducer = EnhancedCUIReducer(
            project_id="dummy_project",
            dataset_id="dummy_dataset",
            mrrel_table="mrrel",
            cui_description_table="cui_desc",
            cui_embeddings_table="cui_embed",
            cui_narrower_table="cui_narrower"
        )

        pipeline = CUIReductionPipeline(api_client, cui_reducer)

        # Example texts
        texts = ["History of Type 2 Diabetes Mellitus and hypertension."]

        reduced_cuis, descriptions, stats = pipeline.process_texts(
            texts,
            target_reduction=0.85,
            semantic_threshold=0.88,
            use_semantic_clustering=True
        )

        print("\n" + "="*80)
        print("CUI REDUCTION RESULTS")
        print("="*80)

        if stats is None:
            print("\n No CUIs were extracted. Check API or input text.")
        else:
            print(f"\n Initial CUIs: {stats.initial_count}")
            print(f" After IC Rollup: {stats.after_ic_rollup} ({stats.ic_rollup_reduction_pct:.1f}%)")
            print(f" Final CUIs: {stats.final_count} ({stats.total_reduction_pct:.1f}% total)")
            print(f" IC Threshold: {stats.ic_threshold_used:.3f}")
            print(f" Hierarchy Size: {stats.hierarchy_size} CUIs")
            print(f" Reduction Time: {stats.processing_time:.2f}s")

            if reduced_cuis:
                print("\nSample Results:")
                for i, cui in enumerate(reduced_cuis[:10], 1):
                    desc = descriptions.get(cui, "No description")
                    ic = cui_reducer.get_ic_scores([cui]).get(cui, 0.0)
                    print(f"{i:2}. {cui} (IC={ic:.2f}): {desc}")

            print("\nStats JSON:")
            print(json.dumps(stats.__dict__, indent=2))

    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
