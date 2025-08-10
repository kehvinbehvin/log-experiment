# Log Analysis and Clustering System

A comprehensive log analysis system that uses machine learning to process, cluster, and extract insights from system logs. The system provides both training and inference capabilities with an intelligent MCP (Model Context Protocol) server for advanced log querying and field extraction.

## 🏗️ Architecture Overview

This project implements a complete log processing pipeline with three main components:

### 1. **Training Pipeline** (`train.py`)
- Preprocesses raw log data using semantic symbol extraction
- Trains HDBSCAN clustering models with distance-based stable cluster registry
- Generates centroids and KNN samples for consistent inference
- Produces stable cluster IDs that persist across training sessions

### 2. **Inference Pipeline** (`cluster.py`)
- Processes new logs using trained models and centroids
- Assigns stable cluster IDs without retraining
- Outputs structured CSV data for MCP consumption
- Maintains pattern consistency across datasets

### 3. **MCP Server** (`log_mcp.py`)
- Provides Model Context Protocol tools for log analysis
- Offers three specialized tools: `log_schema`, `log_query`, and `log_filter`
- Enables surgical data extraction with unified field indexing
- Supports shallow log reconstruction for privacy-preserving analysis

## 📁 Directory Structure

```
├── src/                          # Core implementation files
│   ├── train.py                  # HDBSCAN training pipeline
│   ├── cluster.py                # Inference clustering pipeline  
│   ├── log_mcp.py                # MCP server with log tools
│   ├── log_preprocessing.py      # Centralized preprocessing utilities
│   ├── log_filter_utils.py       # Field extraction utilities
│   ├── stable_cluster_registry.py # Stable cluster ID management
│   └── system_owner.py           # Placeholder for future agentic system
├── data/                         # Log data storage
│   ├── train/                    # Training log files
│   ├── inference/                # Raw logs for clustering
│   ├── preprocessed/             # Processed training logs
│   └── mcp/                      # CSV output for MCP tools
├── model/                        # Trained model artifacts
│   ├── hdbscan_clusterer.pkl     # Trained HDBSCAN model
│   ├── stable_cluster_registry.json # Cluster patterns and metadata
│   ├── centroids.pkl             # Distance-based cluster centroids
│   ├── sample_embeddings.pkl     # KNN verification samples
│   └── cluster_stats.json        # Training statistics
└── archive/                      # Reference implementations
    ├── masking.py                # Original preprocessing logic
    ├── filter.py                 # Field extraction reference
    └── extract_shallow.py        # Privacy-preserving extraction
```

## 🚀 Quick Start

### Prerequisites
```bash
# Install dependencies
pip install pandas numpy sentence-transformers hdbscan torch scikit-learn mcp[cli]>=0.4.0
```

### 1. Training a Model
```bash
python src/train.py
```
- Processes logs from `data/train/data.log`
- Generates preprocessed patterns using semantic symbol extraction
- Trains HDBSCAN clustering model with distance-based registry
- Saves model artifacts to `model/` directory

### 2. Clustering New Logs
```bash
python src/cluster.py
```
- Loads trained model and stable cluster registry
- Processes logs from `data/inference/*.log`
- Assigns stable cluster IDs using centroid + KNN approach
- Outputs structured CSV to `data/mcp/data.log`

### 3. Running MCP Server
```bash
uv run mcp dev src/log_mcp.py
```
- Starts MCP server with three specialized tools
- Provides schema information, log querying, and field extraction
- Supports integration with Claude Desktop and other MCP clients

## 🔧 MCP Tools

### `log_schema()`
Returns comprehensive schema information about all log clusters:
```json
{
  "total_clusters": 15,
  "total_training_logs": 50000,
  "clusters": {
    "3": {
      "patterns": [". - - [X] \"X\""],
      "training_size": 1200,
      "sample_preview": "192.168.1.1 - - [22/Jan/2019:14:18:06] \"GET /test\" 200"
    }
  }
}
```

### `log_query(cluster_id, pattern, limit, offset, shallow)`
Queries logs with filtering and optional shallow extraction:
```bash
# Get raw logs from cluster 3
log_query(cluster_id=3, pattern="GET", limit=10, shallow=false)

# Get privacy-preserving shallow logs
log_query(cluster_id=3, pattern="error", limit=5, shallow=true)
```

### `log_filter(cluster_id, pattern, field_indices, limit, offset)`
**Surgical field extraction** using unified indexing:
```bash
# Extract specific fields by index
log_filter(cluster_id=3, pattern=".", field_indices=[0, 4, 5], limit=10)
```

#### Unified Field Indexing Example
For pattern `". . . - [X] . <X>"` with log `.12.12.12 HI-DED [24/24] . <abc>`:
- **Index 0**: `12` (first whitespace field)
- **Index 1**: `12` (second whitespace field)  
- **Index 2**: `12 HI` (third whitespace field)
- **Index 3**: `DED` (fourth whitespace field)
- **Index 4**: `24/24` (X-masked field from `[X]`)
- **Index 5**: `abc` (X-masked field from `<X>`)

Returns tab-separated, newline-delimited text for easy processing.

## 🧠 Key Technical Features

### **Stable Cluster Registry**
- Maintains consistent cluster IDs across training sessions
- Uses distance-based matching (centroids + KNN verification)
- Preserves cluster assignments even when underlying model changes
- Supports pattern-based cluster identification

### **Semantic Preprocessing** 
- Extracts symbolic patterns from raw logs (`. - - [X] "X"`)
- Masks sensitive content while preserving structure
- Unified preprocessing logic shared across training and inference
- Handles complex nested symbols and IP addresses

### **Distance-Based Clustering**
- Primary: Cosine distance to cluster centroids
- Verification: KNN against sample embeddings  
- Fallback: Outlier classification for unmatched logs
- No retraining required for new log classification

### **Privacy-Preserving Analysis**
- Shallow extraction reconstructs logs with masked sensitive content
- Preserves structural information while hiding private data
- Configurable masking rules for different content types
- Maintains log utility for analysis while protecting privacy

## 📊 Example Workflow

1. **Training Phase**:
   ```bash
   # Place training logs in data/train/data.log
   python src/train.py
   # → Generates model artifacts in model/ directory
   ```

2. **Inference Phase**:
   ```bash
   # Place new logs in data/inference/*.log
   python src/cluster.py  
   # → Outputs clustered data to data/mcp/data.log
   ```

3. **Analysis Phase**:
   ```bash
   # Start MCP server
   uv run mcp dev src/log_mcp.py
   
   # Use tools for analysis
   log_schema()  # Get cluster overview
   log_filter(cluster_id=3, pattern="error", field_indices=[0, 2, 4], limit=20)
   ```

## 🔍 Pattern Examples

| Log Type | Pattern | Example Fields |
|----------|---------|----------------|
| Web Server | `. - - [X] "X"` | IP, timestamp, HTTP request, status |
| System Logs | `- : : . :` | Date, time, PID, component, message |
| Error Logs | `[X] : ` | Timestamp, level, error message |
| Database | `- . : ( ) =` | Date, query type, parameters, result |

## 🛠️ Development

### Core Modules
- **`log_preprocessing.py`**: Centralized preprocessing with consistent masking logic
- **`log_filter_utils.py`**: Advanced field extraction and unified indexing
- **`stable_cluster_registry.py`**: Manages persistent cluster identity mapping

### Testing
```bash
# Test preprocessing utilities
python src/log_preprocessing.py

# Test field extraction
python src/log_filter_utils.py

# Test MCP server
python src/log_mcp.py
```

## 🔮 Future Enhancements

- **Agentic System Owner**: LangGraph-based intelligent log monitoring
- **Real-time Processing**: Streaming log analysis capabilities  
- **Advanced Visualization**: Interactive cluster exploration tools
- **Custom Pattern Detection**: User-defined log pattern recognition

## 📄 License

This project provides a comprehensive foundation for log analysis and clustering workflows with advanced MCP integration for seamless data exploration and extraction.