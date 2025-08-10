## Directories
- data: Store any log files in any states
- model: Store trained models
- src: Store python workflow scripts

## Workflows
- Train (train.py)
- Cluster (cluster.py)
- System Owner (system_owner.py)

### Train
Involves using raw logs as training data to train a HDBSCAN Model to cluster logs.
Steps
- Preprocess training logs
- Feed logs to HDBSCAN for training
- Save model to models directory

### Inference
Involves using model to cluster raw logs using Centroids and KNN method.
- Preprocess inference logs
- Cluster logs using centroids and KNN
- Append output to data.log in data/mcp

## System Owner
An agentic system owner (via langgraph) who manages a knowledge base of the system by intelligently
querying system logs via an mcp tool and persisting its knowledge in a knowledge graph
via an mcp tool.