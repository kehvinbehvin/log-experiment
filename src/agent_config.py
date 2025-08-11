"""
Agent Configuration for Autonomous Log Analysis System

This module defines the configuration, prompts, and settings for the
autonomous log analysis agent that explores system logs and builds
a knowledge graph of discovered insights.
"""

import os
from typing import Dict, Any
from pathlib import Path

# Load .env file if it exists
def load_env_file():
    """Load environment variables from .env file if it exists."""
    env_file = Path(__file__).parent.parent / '.env'
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    # Only set if not already in environment
                    if key not in os.environ:
                        os.environ[key] = value.strip('"\'')

# Load .env on import
load_env_file()

# Get the absolute path to the project root
PROJECT_ROOT = Path(__file__).parent.parent.absolute()

# OpenAI Configuration
OPENAI_MODEL = "gpt-4o"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Agent behavior settings
MAX_ITERATIONS = 50  # Safety limit for autonomous operation
MIN_CLUSTERS_TO_ANALYZE = 5  # Minimum clusters to explore
MAX_SAMPLES_PER_CLUSTER = 10  # Max log samples per cluster
INSIGHT_SATURATION_THRESHOLD = 3  # Stop when no new insights for N iterations

# MCP Server configurations
MCP_SERVERS = {
    "log_analysis": {
        "command": str(PROJECT_ROOT / "venv" / "bin" / "python"),
        "args": [str(PROJECT_ROOT / "src" / "log_mcp.py")],
        "transport": "stdio",
        "env": {
            "PYTHONPATH": str(PROJECT_ROOT),
            "VIRTUAL_ENV": str(PROJECT_ROOT / "venv"),
            "PATH": f"{PROJECT_ROOT / 'venv' / 'bin'}:{os.environ.get('PATH', '')}"
        }
    },
    "neo4j_memory": {
        "url": "http://localhost:8000/api/mcp",
        "transport": "streamable_http"
    }
}

# System prompt for the autonomous agent  
SYSTEM_PROMPT = """You are an autonomous system administrator and log analyst responsible for understanding the systems under your management.

Your mission is to:
1. **Explore log data** systematically to understand system patterns, errors, and behaviors
2. **Extract actionable insights** about system health, performance, and potential issues  
3. **Build a knowledge graph** of your findings for future reference
4. **Work autonomously** until you have sufficient understanding of the systems

**Available Tools:**
You have access to both log analysis and knowledge graph tools:

**Log Analysis Tools:**
- log_schema(): Get overview of all available log clusters and patterns
- log_query(): Retrieve specific logs by cluster ID and pattern matching
- log_filter(): Extract specific fields from logs using surgical precision

**Knowledge Graph Tools:**
- read_graph(): Review the current knowledge graph
- create_entities(): Store discovered systems, components, and insights
- create_relations(): Connect related entities with meaningful relationships  
- add_observations(): Add new insights to existing entities
- search_memories(): Search for existing information
- find_memories_by_name(): Find specific entities

**Working Strategy:**
1. Start with log_schema() to understand what log clusters exist
2. Intelligently sample different clusters using log_query() and log_filter()
3. Look for patterns: errors, performance issues, security events, system changes
4. Extract insights about systems, services, and their relationships from logs
5. Store your findings in the knowledge graph as entities and relationships
6. Build connections between discovered systems based on log evidence
7. Terminate when you have adequate understanding of the systems

**Key Principles:**
- Be surgical and precise - use log_filter() to extract only needed fields
- Look for systems, services, error patterns, and relationships in logs
- Create entities for discovered systems, services, and components
- Build relationships based on log evidence (dependencies, interactions, etc.)
- Focus on actionable insights that help understand system health and behavior
- Work autonomously and decide when your analysis is sufficient

Remember: Your goal is understanding the systems through log analysis and building a persistent knowledge graph of your discoveries. Work smartly and efficiently."""

# Termination conditions
TERMINATION_CONDITIONS = {
    "max_iterations_reached": "Maximum iteration limit reached",
    "insufficient_data": "No log data available to analyze", 
    "mcp_connection_failed": "Unable to connect to required MCP servers",
    "analysis_complete": "Sufficient system understanding achieved",
    "insight_saturation": "No new insights discovered in recent iterations",
    "error_threshold_exceeded": "Too many consecutive errors encountered"
}

# Agent state tracking
AGENT_STATE_KEYS = [
    "iteration_count",
    "clusters_analyzed", 
    "insights_discovered",
    "entities_created",
    "relations_created",
    "consecutive_no_insights",
    "last_termination_check"
]

def get_openai_config() -> Dict[str, Any]:
    """Get OpenAI configuration for the agent."""
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY environment variable must be set")
    
    return {
        "model": OPENAI_MODEL,
        "api_key": OPENAI_API_KEY,
        "temperature": 0.1  # Low temperature for consistent, focused analysis
    }

def should_terminate(agent_state: Dict[str, Any]) -> tuple[bool, str]:
    """
    Determine if the agent should terminate based on current state.
    
    Returns:
        tuple: (should_terminate: bool, reason: str)
    """
    # Max iterations safety check
    if agent_state.get("iteration_count", 0) >= MAX_ITERATIONS:
        return True, TERMINATION_CONDITIONS["max_iterations_reached"]
    
    # Check if we've analyzed minimum clusters and have insights
    clusters_analyzed = agent_state.get("clusters_analyzed", 0)
    insights_discovered = agent_state.get("insights_discovered", 0)
    
    if (clusters_analyzed >= MIN_CLUSTERS_TO_ANALYZE and 
        insights_discovered > 0 and 
        agent_state.get("consecutive_no_insights", 0) >= INSIGHT_SATURATION_THRESHOLD):
        return True, TERMINATION_CONDITIONS["insight_saturation"]
    
    # Check for error conditions (to be implemented based on actual errors)
    if agent_state.get("consecutive_errors", 0) >= 5:
        return True, TERMINATION_CONDITIONS["error_threshold_exceeded"]
    
    return False, ""