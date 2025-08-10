#!/usr/bin/env python3
"""
HDBSCAN Log Training Script for log-experiment-v5

Trains HDBSCAN clusterer on log data using semantic embeddings with distance-based
stable cluster registry (centroids + KNN) approach.

Usage:
    python train.py
"""

import os
import sys
import json
import pickle
import pandas as pd
import numpy as np
from collections import Counter
from pathlib import Path
from sentence_transformers import SentenceTransformer
import hdbscan
import torch

from stable_cluster_registry import StableClusterRegistry
from log_preprocessing import preprocess_log

class UniformDataClusterer:
    """Mock clusterer for extremely uniform data - creates single cluster"""
    def __init__(self, num_samples):
        self.labels_ = np.zeros(num_samples, dtype=int)
        self.num_samples = num_samples
        
    def approximate_predict(self, new_embeddings):
        return np.zeros(len(new_embeddings), dtype=int)
        
    def fit_predict(self, embeddings):
        return np.zeros(len(embeddings), dtype=int)


def setup_device():
    """Setup optimal device for sentence transformers"""
    if torch.cuda.is_available():
        device = 'cuda'
        print(f"Using GPU: {torch.cuda.get_device_name()}")
    elif torch.backends.mps.is_available():
        device = 'mps'
        print("Using Apple Silicon MPS")
    else:
        device = 'cpu'
        print("Using CPU")
    return device

def load_training_data(file_path):
    """Load training logs from data/train/data.log"""
    print(f"Loading training data from: {file_path}")
    
    if not os.path.exists(file_path):
        print(f"Error: Training file not found: {file_path}")
        sys.exit(1)
    
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        logs = [line.strip() for line in f if line.strip()]
    
    print(f"Loaded {len(logs)} raw logs")
    if logs:
        print(f"Sample raw log: {logs[0][:100]}...")
    return logs

def preprocess_and_save_logs(raw_logs, preprocessed_path):
    """Preprocess logs and save to data/preprocessed/"""
    print(f"Preprocessing {len(raw_logs)} logs...")
    
    preprocessed_logs = []
    for log in raw_logs:
        processed = preprocess_log(log)
        preprocessed_logs.append(processed)
    
    # Save preprocessed logs
    os.makedirs(os.path.dirname(preprocessed_path), exist_ok=True)
    with open(preprocessed_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(preprocessed_logs))
    
    print(f"Saved preprocessed logs to: {preprocessed_path}")
    print(f"Sample preprocessed log: {preprocessed_logs[0] if preprocessed_logs else 'None'}")
    
    return preprocessed_logs

def generate_embeddings(logs, device='cuda', batch_size=64):
    """Generate semantic embeddings using SentenceTransformers"""
    print(f"Generating embeddings using device: {device}")
    
    # Check for identical data problem
    unique_logs = set(logs)
    uniqueness_ratio = len(unique_logs) / len(logs)
    print(f"Data uniqueness: {len(unique_logs)} unique patterns out of {len(logs)} logs ({uniqueness_ratio:.1%})")
    
    if uniqueness_ratio < 0.05:
        print("WARNING: Data has very low variation - this may cause clustering issues")
        print(f"   Unique patterns: {list(unique_logs)[:3]}...")
    
    # Load pre-trained sentence transformer model
    model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    
    # Generate embeddings with progress bar
    embeddings = model.encode(
        logs, 
        batch_size=batch_size, 
        show_progress_bar=True,
        convert_to_numpy=True
    )
    
    print(f"Generated embeddings shape: {embeddings.shape}")
    return embeddings, model

def train_hdbscan_clusterer(embeddings, logs_for_analysis=None, min_cluster_size=25, min_samples=15, 
                           cluster_selection_epsilon=0.3, aggression_level='high'):
    """Train HDBSCAN clusterer with special handling for uniform data"""
    
    # Check for uniform data problem
    if logs_for_analysis:
        unique_logs = set(logs_for_analysis)
        uniqueness_ratio = len(unique_logs) / len(logs_for_analysis)
        
        if uniqueness_ratio < 0.01:
            print("SPECIAL CASE: Data is extremely uniform - creating single cluster manually")
            cluster_labels = np.zeros(len(logs_for_analysis), dtype=int)
            clusterer = UniformDataClusterer(len(logs_for_analysis))
            return clusterer, cluster_labels
    
    # Configuration presets for different data types
    presets = {
        'ultra_sensitive': {
            'min_cluster_size': 2,
            'min_samples': 1,
            'cluster_selection_epsilon': 0.05,
            'description': 'Maximum sensitivity for very similar preprocessed logs'
        },
        'preprocessed': {
            'min_cluster_size': 5,
            'min_samples': 3,
            'cluster_selection_epsilon': 0.1,
            'description': 'Less aggressive for highly similar preprocessed logs'
        },
        'moderate': {
            'min_cluster_size': 15,
            'min_samples': 10,
            'cluster_selection_epsilon': 0.2,
            'description': 'Balanced clustering for raw logs'
        },
        'high': {
            'min_cluster_size': 25,
            'min_samples': 15,
            'cluster_selection_epsilon': 0.3,
            'description': 'Aggressive clustering for diverse raw logs'
        }
    }
    
    # Apply preset if specified
    if aggression_level in presets:
        config = presets[aggression_level]
        min_cluster_size = config['min_cluster_size']
        min_samples = config['min_samples']
        cluster_selection_epsilon = config['cluster_selection_epsilon']
        print(f"Using '{aggression_level}' preset: {config['description']}")
    
    print(f"Training HDBSCAN with parameters:")
    print(f"  min_cluster_size={min_cluster_size}")
    print(f"  min_samples={min_samples}")
    print(f"  cluster_selection_epsilon={cluster_selection_epsilon}")
    
    # Initialize HDBSCAN clusterer
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_method='eom',
        cluster_selection_epsilon=cluster_selection_epsilon,
        prediction_data=True
    )
    
    # Fit clusterer and get cluster labels
    cluster_labels = clusterer.fit_predict(embeddings)
    
    print(f"Clustering complete. Labels shape: {cluster_labels.shape}")
    return clusterer, cluster_labels

def analyze_clusters(logs, cluster_labels, data_type='raw', aggression_level='high'):
    """Analyze cluster distribution and generate statistics"""
    print("Analyzing cluster distribution...")
    
    unique_clusters = set(cluster_labels)
    num_clusters = len([c for c in unique_clusters if c != -1])
    num_outliers = sum(1 for c in cluster_labels if c == -1)
    
    # Count logs per cluster
    cluster_counts = Counter(cluster_labels)
    
    # Generate detailed statistics
    stats = {
        'total_logs': len(logs),
        'num_clusters': num_clusters,
        'num_outliers': num_outliers,
        'outlier_percentage': (num_outliers / len(logs)) * 100,
        'cluster_sizes': {str(k): v for k, v in cluster_counts.items()},
        'data_type': data_type,
        'aggression_level': aggression_level,
        'clustering_approach': 'less aggressive' if aggression_level == 'preprocessed' else 'aggressive',
    }
    
    # Print summary
    print(f"Total logs: {stats['total_logs']}")
    print(f"Number of clusters: {stats['num_clusters']}")
    print(f"Number of outliers: {stats['num_outliers']} ({stats['outlier_percentage']:.1f}%)")
    
    # Show cluster size distribution
    print("\nCluster size distribution:")
    for cluster_id, count in sorted(cluster_counts.items()):
        if cluster_id == -1:
            print(f"  Outliers: {count} logs")
        else:
            print(f"  Cluster {cluster_id}: {count} logs")
    
    # Show sample logs from each cluster
    print("\nSample logs from each cluster:")
    for cluster_id in sorted(unique_clusters):
        cluster_logs = [logs[i] for i in range(len(logs)) if cluster_labels[i] == cluster_id]
        
        if cluster_id == -1:
            print(f"\n=== OUTLIERS ({len(cluster_logs)} logs) ===")
        else:
            print(f"\n=== CLUSTER {cluster_id} ({len(cluster_logs)} logs) ===")
        
        # Show first 2 examples
        for log in cluster_logs[:2]:
            print(f"  {log}")
    
    return stats

def build_distance_based_registry(raw_logs, preprocessed_logs, cluster_labels, embeddings):
    """Build stable cluster registry focused on distance-based (centroids + KNN) approach"""
    print(f"\n=== Building Distance-Based Stable Cluster Registry ===")
    
    # Create dataframe for signature generation (required for compatibility)
    results_df = pd.DataFrame({
        'cluster_id': cluster_labels,
        'raw_log': raw_logs,
        'preprocessed_log': preprocessed_logs
    })
    
    # Initialize stable registry
    stable_registry = StableClusterRegistry()
    
    # Create cluster signatures (minimal, just for registry structure)
    cluster_signatures = stable_registry.create_cluster_signatures(results_df)
    
    # Build stable registry with emphasis on distance-based data (centroids + samples)
    stable_registry.build_stable_registry(cluster_signatures, embeddings, cluster_labels)
    
    print(f"Registry Summary: {stable_registry.get_registry_summary()}")
    print(f"Distance-based matching enabled: {len(stable_registry.centroids)} centroids stored")
    print(f"KNN sample embeddings stored for: {len(stable_registry.sample_embeddings)} clusters")
    
    return stable_registry, cluster_signatures

def save_model_artifacts(clusterer, embeddings, stats, stable_registry, model_dir='model'):
    """Save all trained model artifacts with focus on distance-based components"""
    print(f"Saving model artifacts to {model_dir}/")
    
    os.makedirs(model_dir, exist_ok=True)
    
    # Save HDBSCAN clusterer
    clusterer_path = os.path.join(model_dir, 'hdbscan_clusterer.pkl')
    with open(clusterer_path, 'wb') as f:
        pickle.dump(clusterer, f)
    print(f"Saved clusterer: {clusterer_path}")
    
    # Save training embeddings
    embeddings_path = os.path.join(model_dir, 'training_embeddings.npy')
    np.save(embeddings_path, embeddings)
    print(f"Saved embeddings: {embeddings_path}")
    
    # Save cluster statistics
    stats_path = os.path.join(model_dir, 'cluster_stats.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"Saved statistics: {stats_path}")
    
    # Save stable cluster registry (includes centroids.pkl and sample_embeddings.pkl)
    stable_registry.save_registry(model_dir)
    
    # Save registry summary in stats
    registry_summary = stable_registry.get_registry_summary()
    stats['stable_registry_info'] = registry_summary

def main():
    """Main training pipeline with distance-based stable cluster ID system"""
    print("=== HDBSCAN Log Training with Distance-Based Stable IDs ===\n")
    
    # Configuration
    training_file = 'data/train/data.log'
    preprocessed_file = 'data/preprocessed/preprocessed_logs.log'
    model_dir = 'model'
    batch_size = 64
    aggression_level = 'moderate'  # Balanced approach for mixed data
    
    # Setup device
    device = setup_device()
    
    # Load training data
    raw_logs = load_training_data(training_file)
    
    # Preprocess logs and save
    preprocessed_logs = preprocess_and_save_logs(raw_logs, preprocessed_file)
    
    # Generate embeddings from preprocessed logs
    embeddings, _ = generate_embeddings(
        preprocessed_logs,  # Use preprocessed logs for embeddings
        device=device,
        batch_size=batch_size
    )
    
    # Train HDBSCAN clusterer
    clusterer, cluster_labels = train_hdbscan_clusterer(
        embeddings,
        logs_for_analysis=preprocessed_logs,
        aggression_level=aggression_level
    )
    
    # Analyze clusters
    stats = analyze_clusters(preprocessed_logs, cluster_labels, 'preprocessed', aggression_level)
    
    # Build distance-based stable cluster registry (centroids + KNN samples)
    stable_registry, cluster_signatures = build_distance_based_registry(
        raw_logs, preprocessed_logs, cluster_labels, embeddings
    )
    
    # Save all model artifacts
    save_model_artifacts(clusterer, embeddings, stats, stable_registry, model_dir)
    
    print(f"\n=== Training Complete ===")
    print(f"Model saved to: {model_dir}/")
    print(f"Distance-based registry created with {len(cluster_signatures)} base clusters")
    print(f"Centroids saved for inference clustering via cosine distance + KNN verification")
    print(f"Ready for inference on new logs using centroid/KNN matching")

if __name__ == "__main__":
    main()
