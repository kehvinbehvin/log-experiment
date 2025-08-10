#!/usr/bin/env python3
"""
Stable Cluster Registry

Manages persistent cluster signatures and stable ID mapping to ensure
consistent cluster IDs across different datasets.
"""

import json
import hashlib
import pickle
from pathlib import Path
from collections import Counter
from typing import Dict, List, Tuple, Any
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_distances
from scipy.spatial.distance import cdist


class StableClusterRegistry:
    """Manages stable cluster ID assignments using semantic signatures"""
    
    def __init__(self):
        # Legacy signature-based matching (kept for backwards compatibility)
        self.signature_to_stable_id = {}  # signature_hash -> stable_id
        self.stable_id_to_info = {}       # stable_id -> cluster_info
        self.next_stable_id = 0
        
        # New distance-based matching data
        self.centroids = {}               # stable_id -> centroid_embedding
        self.sample_embeddings = {}       # stable_id -> [sample_embeddings]
        self.cluster_stats = {}           # stable_id -> {size, intra_distance, etc}
        
        # Distance thresholds (can be tuned)
        self.primary_distance_threshold = 0.25    # Direct assignment threshold
        self.secondary_distance_threshold = 0.4   # Verification threshold  
        self.knn_k = 5                           # K for K-NN verification
        self.knn_confidence_threshold = 0.6      # Minimum confidence for assignment
        
    def create_cluster_signatures(self, results_df: pd.DataFrame) -> Dict[int, Dict]:
        """Create semantic signatures for each cluster based on preprocessed patterns"""
        cluster_signatures = {}
        
        for cluster_id in results_df['cluster_id'].unique():
            if cluster_id == -1:  # Skip outliers
                continue
                
            cluster_logs = results_df[results_df['cluster_id'] == cluster_id]
            
            # Get most common preprocessed patterns in this cluster
            pattern_counts = Counter(cluster_logs['preprocessed_log'])
            most_common_patterns = [pattern for pattern, count in pattern_counts.most_common(3)]
            
            # Create a signature hash
            signature_text = "|".join(sorted(most_common_patterns))
            signature_hash = hashlib.md5(signature_text.encode()).hexdigest()[:12]
            
            cluster_signatures[cluster_id] = {
                'signature_hash': signature_hash,
                'patterns': most_common_patterns,
                'size': len(cluster_logs),
                'sample_raw_logs': cluster_logs['raw_log'].head(2).tolist()
            }
        
        return cluster_signatures
    
    def build_stable_registry(self, cluster_signatures: Dict[int, Dict], embeddings: np.ndarray = None, cluster_labels: np.ndarray = None):
        """Build initial stable registry from training cluster signatures and embeddings"""
        print(f"Building stable registry from {len(cluster_signatures)} clusters...")
        
        for temp_cluster_id, info in cluster_signatures.items():
            signature_hash = info['signature_hash']
            
            # Assign stable ID
            stable_id = self.next_stable_id
            self.next_stable_id += 1
            
            # Store legacy signature mappings
            self.signature_to_stable_id[signature_hash] = stable_id
            self.stable_id_to_info[stable_id] = {
                'signature_hash': signature_hash,
                'patterns': info['patterns'],
                'training_size': info['size'],
                'sample_logs': info['sample_raw_logs']
            }
            
            # Store distance-based data if embeddings provided
            if embeddings is not None and cluster_labels is not None:
                cluster_embeddings = embeddings[cluster_labels == temp_cluster_id]
                
                if len(cluster_embeddings) > 0:
                    # Calculate centroid
                    centroid = np.mean(cluster_embeddings, axis=0)
                    self.centroids[stable_id] = centroid
                    
                    # Store sample embeddings (up to 10 samples)
                    sample_indices = np.random.choice(len(cluster_embeddings), 
                                                     min(10, len(cluster_embeddings)), 
                                                     replace=False)
                    self.sample_embeddings[stable_id] = cluster_embeddings[sample_indices]
                    
                    # Calculate cluster statistics
                    if len(cluster_embeddings) > 1:
                        intra_distances = cosine_distances(cluster_embeddings)
                        avg_intra_distance = np.mean(intra_distances[np.triu_indices_from(intra_distances, k=1)])
                    else:
                        avg_intra_distance = 0.0
                    
                    self.cluster_stats[stable_id] = {
                        'size': int(len(cluster_embeddings)),
                        'avg_intra_distance': float(avg_intra_distance),
                        'centroid_norm': float(np.linalg.norm(centroid))
                    }
        
        print(f"Registry built with {len(self.stable_id_to_info)} stable IDs")
        if embeddings is not None:
            print(f"Stored centroids and sample embeddings for distance-based matching")
    
    def map_new_clusters_to_stable_ids(self, new_cluster_signatures: Dict[int, Dict]) -> Dict[int, int]:
        """Map new dataset cluster IDs to stable IDs based on signature matching"""
        mapping = {}  # new_cluster_id -> stable_id
        new_patterns_found = 0
        matched_patterns = 0
        
        for new_cluster_id, info in new_cluster_signatures.items():
            signature_hash = info['signature_hash']
            
            if signature_hash in self.signature_to_stable_id:
                # Existing pattern - use stable ID
                stable_id = self.signature_to_stable_id[signature_hash]
                mapping[new_cluster_id] = stable_id
                matched_patterns += 1
                print(f"✅ Cluster {new_cluster_id} → Stable ID {stable_id} (known pattern)")
            else:
                # New pattern - assign new stable ID and extend registry
                stable_id = self.next_stable_id
                self.next_stable_id += 1
                
                # Add to registry
                self.signature_to_stable_id[signature_hash] = stable_id
                self.stable_id_to_info[stable_id] = {
                    'signature_hash': signature_hash,
                    'patterns': info['patterns'],
                    'training_size': 0,  # Not from training
                    'inference_size': info['size'],
                    'sample_logs': info['sample_raw_logs'],
                    'is_new_pattern': True
                }
                
                mapping[new_cluster_id] = stable_id
                new_patterns_found += 1
                print(f"🆕 Cluster {new_cluster_id} → Stable ID {stable_id} (new pattern)")
        
        print(f"\nMapping Summary:")
        print(f"Matched existing patterns: {matched_patterns}")
        print(f"New patterns discovered: {new_patterns_found}")
        
        return mapping
    
    def apply_stable_mapping(self, results_df: pd.DataFrame, cluster_mapping: Dict[int, int]) -> pd.DataFrame:
        """Apply stable ID mapping to results dataframe"""
        results_df = results_df.copy()
        results_df['stable_cluster_id'] = results_df['cluster_id'].map(cluster_mapping)
        
        # Handle outliers (-1) by keeping them as -1
        results_df.loc[results_df['cluster_id'] == -1, 'stable_cluster_id'] = -1
        
        return results_df
    
    def save_registry(self, model_dir: Path):
        """Save stable registry to disk"""
        model_dir = Path(model_dir)
        registry_path = model_dir / "stable_cluster_registry.json"
        
        # Convert to JSON-serializable format
        registry_data = {
            'signature_to_stable_id': self.signature_to_stable_id,
            'stable_id_to_info': self.stable_id_to_info,
            'next_stable_id': self.next_stable_id,
            'cluster_stats': self.cluster_stats,
            'distance_thresholds': {
                'primary_distance_threshold': self.primary_distance_threshold,
                'secondary_distance_threshold': self.secondary_distance_threshold,
                'knn_k': self.knn_k,
                'knn_confidence_threshold': self.knn_confidence_threshold
            }
        }
        
        with open(registry_path, 'w') as f:
            json.dump(registry_data, f, indent=2)
        
        print(f"Stable registry saved to: {registry_path}")
        
        # Save centroids and sample embeddings separately (binary format for efficiency)
        if self.centroids:
            centroids_path = model_dir / "centroids.pkl"
            with open(centroids_path, 'wb') as f:
                pickle.dump(self.centroids, f)
            print(f"Centroids saved to: {centroids_path}")
        
        if self.sample_embeddings:
            samples_path = model_dir / "sample_embeddings.pkl"
            with open(samples_path, 'wb') as f:
                pickle.dump(self.sample_embeddings, f)
            print(f"Sample embeddings saved to: {samples_path}")
    
    def load_registry(self, model_dir: Path):
        """Load stable registry from disk"""
        model_dir = Path(model_dir)
        registry_path = model_dir / "stable_cluster_registry.json"
        
        if not registry_path.exists():
            raise FileNotFoundError(f"Registry not found: {registry_path}")
        
        with open(registry_path, 'r') as f:
            registry_data = json.load(f)
        
        self.signature_to_stable_id = registry_data['signature_to_stable_id']
        self.stable_id_to_info = registry_data['stable_id_to_info']
        self.next_stable_id = registry_data['next_stable_id']
        
        # Load new distance-based data if available
        if 'cluster_stats' in registry_data:
            self.cluster_stats = registry_data['cluster_stats']
        
        if 'distance_thresholds' in registry_data:
            thresholds = registry_data['distance_thresholds']
            self.primary_distance_threshold = thresholds['primary_distance_threshold']
            self.secondary_distance_threshold = thresholds['secondary_distance_threshold']
            self.knn_k = thresholds['knn_k']
            self.knn_confidence_threshold = thresholds['knn_confidence_threshold']
        
        # Load centroids and sample embeddings
        centroids_path = model_dir / "centroids.pkl"
        if centroids_path.exists():
            with open(centroids_path, 'rb') as f:
                self.centroids = pickle.load(f)
            print(f"Loaded centroids for {len(self.centroids)} clusters")
        
        samples_path = model_dir / "sample_embeddings.pkl"
        if samples_path.exists():
            with open(samples_path, 'rb') as f:
                self.sample_embeddings = pickle.load(f)
            print(f"Loaded sample embeddings for {len(self.sample_embeddings)} clusters")
        
        print(f"Stable registry loaded from: {registry_path}")
        print(f"Registry contains {len(self.stable_id_to_info)} stable cluster definitions")
        print(f"Distance-based matching: {'Enabled' if self.centroids else 'Disabled (using signature matching)'}")
    
    def get_registry_summary(self) -> Dict[str, Any]:
        """Get summary statistics about the registry"""
        training_clusters = sum(1 for info in self.stable_id_to_info.values() 
                               if 'is_new_pattern' not in info)
        new_clusters = sum(1 for info in self.stable_id_to_info.values() 
                          if info.get('is_new_pattern', False))
        
        return {
            'total_stable_ids': len(self.stable_id_to_info),
            'training_clusters': training_clusters,
            'new_patterns_discovered': new_clusters,
            'next_available_id': self.next_stable_id,
            'has_centroids': len(self.centroids) > 0,
            'distance_based_enabled': len(self.centroids) > 0
        }
    
    def assign_embeddings_to_stable_clusters(self, embeddings: np.ndarray, raw_logs: List[str], preprocessed_logs: List[str]) -> Tuple[np.ndarray, Dict]:
        """Assign individual embeddings to stable clusters using distance-based matching"""
        print(f"\n=== Distance-Based Stable Cluster Assignment ===")
        print(f"Assigning {len(embeddings)} embeddings to stable clusters...")
        
        if not self.centroids:
            print("Warning: No centroids available, falling back to signature matching")
            return self._fallback_to_clustering(embeddings, raw_logs, preprocessed_logs)
        
        stable_assignments = np.full(len(embeddings), -1, dtype=int)  # Initialize as outliers
        assignment_stats = {
            'direct_assignments': 0,
            'verified_assignments': 0,
            'new_clusters': 0,
            'outliers': 0
        }
        
        # Prepare centroid matrix for efficient distance computation
        stable_ids = list(self.centroids.keys())
        centroid_matrix = np.vstack([self.centroids[sid] for sid in stable_ids])
        
        # Assign each embedding individually
        for i, embedding in enumerate(embeddings):
            stable_id = self._assign_single_embedding(embedding, stable_ids, centroid_matrix)
            stable_assignments[i] = stable_id
            
            # Update statistics
            if stable_id != -1:
                if stable_id in stable_ids:  # Existing cluster
                    assignment_stats['direct_assignments'] += 1
                else:
                    assignment_stats['verified_assignments'] += 1
            else:
                assignment_stats['outliers'] += 1
        
        print(f"Assignment Results:")
        print(f"  Direct assignments: {assignment_stats['direct_assignments']}")
        print(f"  Verified assignments: {assignment_stats['verified_assignments']}")  
        print(f"  Outliers: {assignment_stats['outliers']}")
        
        return stable_assignments, assignment_stats
    
    def _assign_single_embedding(self, embedding: np.ndarray, stable_ids: List[int], centroid_matrix: np.ndarray) -> int:
        """Assign a single embedding to the best matching stable cluster"""
        # Compute distances to all centroids
        embedding_reshaped = embedding.reshape(1, -1)
        distances = cosine_distances(embedding_reshaped, centroid_matrix)[0]
        
        # Find closest centroid
        closest_idx = np.argmin(distances)
        closest_distance = distances[closest_idx]
        closest_stable_id = stable_ids[closest_idx]
        
        # Primary threshold: Direct assignment
        if closest_distance < self.primary_distance_threshold:
            return closest_stable_id
        
        # Secondary threshold: Verification with K-NN
        if closest_distance < self.secondary_distance_threshold:
            confidence = self._knn_verification(embedding, closest_stable_id)
            if confidence >= self.knn_confidence_threshold:
                return closest_stable_id
        
        # No good match found - mark as outlier
        return -1
    
    def _knn_verification(self, embedding: np.ndarray, candidate_stable_id: int) -> float:
        """Verify cluster assignment using K-NN voting from sample embeddings"""
        if candidate_stable_id not in self.sample_embeddings:
            return 0.0
        
        # Collect sample embeddings from all clusters
        all_samples = []
        all_labels = []
        
        for stable_id, samples in self.sample_embeddings.items():
            all_samples.extend(samples)
            all_labels.extend([stable_id] * len(samples))
        
        if len(all_samples) < self.knn_k:
            return 0.0
        
        # Find K nearest neighbors
        sample_matrix = np.vstack(all_samples)
        embedding_reshaped = embedding.reshape(1, -1)
        distances = cosine_distances(embedding_reshaped, sample_matrix)[0]
        
        # Get K nearest indices
        k_nearest_indices = np.argsort(distances)[:self.knn_k]
        k_nearest_labels = [all_labels[i] for i in k_nearest_indices]
        
        # Calculate confidence as percentage of votes for candidate cluster
        votes_for_candidate = sum(1 for label in k_nearest_labels if label == candidate_stable_id)
        confidence = votes_for_candidate / self.knn_k
        
        return confidence
    
    def _fallback_to_clustering(self, embeddings: np.ndarray, raw_logs: List[str], preprocessed_logs: List[str]) -> Tuple[np.ndarray, Dict]:
        """Fallback to signature-based clustering when centroids not available"""
        print("Falling back to signature-based assignment...")
        
        # This would require implementing HDBSCAN clustering here
        # For now, mark all as new/outliers
        stable_assignments = np.full(len(embeddings), -1, dtype=int)
        assignment_stats = {
            'direct_assignments': 0,
            'verified_assignments': 0,
            'new_clusters': 0,
            'outliers': len(embeddings)
        }
        
        return stable_assignments, assignment_stats