# Demo Architecture for 1.77GB Dataset

## Relaxed Performance Requirements = Simpler Solution

### Target Performance:
- ✅ **15-20 second response time** (acceptable)
- ✅ **Demo usage** (1-2 concurrent users max)
- ✅ **Cost-effective** (under $100/month)
- ✅ **Simple deployment** (minimal infrastructure)

## Recommended: t3.large + File-Based Processing

### EC2 Configuration:
- **t3.large** (4 vCPU, 8GB RAM) - $60/month
- **50GB EBS storage** - $5/month
- **Total: ~$65/month**

### Memory Strategy (No Database Needed):
```python
# Chunked processing approach - never load full dataset
import pandas as pd
import polars as pl  # Faster than pandas for large files

class ChunkedKnowledgeGraph:
    def __init__(self):
        self.parquet_file = "data/knowledge_base.parquet"
        self.chunk_size = 50000  # Process 50K rows at a time
    
    def search_concepts(self, query, max_results=100):
        """Search across 34M rows in chunks - 15-20 seconds is fine"""
        results = []
        
        # Process parquet file in chunks
        for chunk in pl.scan_parquet(self.parquet_file).collect_streaming(chunk_size=self.chunk_size):
            # Filter chunk based on query
            matches = chunk.filter(
                pl.col("text").str.contains(query, case=False)
            ).head(max_results)
            
            if len(matches) > 0:
                results.append(matches)
            
            # Stop if we have enough results
            if sum(len(r) for r in results) >= max_results:
                break
        
        return pl.concat(results).head(max_results) if results else pl.DataFrame()
    
    def get_concept_relationships(self, concept_id):
        """Find related concepts - process in chunks"""
        # Similar chunked approach for relationships
        pass
```

### Polars vs Pandas for Large Data:
```python
# Polars is 5-10x faster than pandas for large datasets
import polars as pl

# Memory-efficient reading
df = pl.scan_parquet("large_file.parquet")  # Lazy - doesn't load into memory

# Query pushdown - filtering happens at read time
filtered = df.filter(pl.col("category") == "climate").collect()

# Much faster than pandas equivalent:
# df = pd.read_parquet("large_file.parquet")  # Loads everything into memory
# filtered = df[df["category"] == "climate"]  # Filters after loading
```

### Updated MCP Server:
```python
# mcp/large_knowledge_server.py
import polars as pl
from fastmcp import FastMCP
import time

mcp = FastMCP("large-knowledge-server")

@mcp.tool()
def SearchLargeKnowledgeBase(query: str, limit: int = 50) -> dict:
    """
    Search across 34M records using chunked processing.
    Expected response time: 10-20 seconds.
    """
    start_time = time.time()
    
    # Lazy scan - doesn't load into memory
    df = pl.scan_parquet("data/knowledge_base.parquet")
    
    # Query with limit for performance
    results = df.filter(
        pl.col("text").str.contains(query, case_False) |
        pl.col("title").str.contains(query, case_False)
    ).head(limit).collect()
    
    processing_time = time.time() - start_time
    
    return {
        "results": results.to_dicts(),
        "total_found": len(results),
        "processing_time_seconds": round(processing_time, 2),
        "query": query
    }

@mcp.tool() 
def GetConceptEmbeddings(concept: str) -> dict:
    """
    Get semantic embeddings for concept.
    Uses chunked processing if embeddings aren't pre-computed.
    """
    # Implementation with chunked embedding search
    pass
```

### Systemd Configuration for Large Data:
```ini
[Service]
# Increased memory limits
MemoryHigh=6G
MemoryMax=7G

# Longer timeouts for slow queries
TimeoutStartSec=300
TimeoutStopSec=120

# Environment for Polars optimization
Environment=POLARS_MAX_THREADS=4
Environment=POLARS_STREAMING_CHUNK_SIZE=50000

# Working directory with data access
WorkingDirectory=/opt/climate-api/app
ReadWritePaths=/opt/climate-api/app/data
```

### Progressive Loading Strategy:
```python
# Load data progressively as needed
class ProgressiveKG:
    def __init__(self):
        self.loaded_chunks = {}  # Cache recently used chunks
        self.max_cached_chunks = 5  # Keep 5 chunks in memory max
    
    def get_chunk(self, chunk_id):
        """Load chunk only when needed"""
        if chunk_id not in self.loaded_chunks:
            # Remove oldest chunk if at limit
            if len(self.loaded_chunks) >= self.max_cached_chunks:
                oldest = min(self.loaded_chunks.keys())
                del self.loaded_chunks[oldest]
            
            # Load new chunk
            start_row = chunk_id * 50000
            end_row = start_row + 50000
            
            chunk = pl.read_parquet(
                "data/knowledge_base.parquet",
                row_count_limit=50000,
                row_count_offset=start_row
            )
            
            self.loaded_chunks[chunk_id] = chunk
        
        return self.loaded_chunks[chunk_id]
```

## Data Processing Pipeline

### 1. Convert HuggingFace to Parquet:
```python
# convert_dataset.py
from datasets import load_dataset
import polars as pl

def convert_huggingface_to_parquet():
    # Load dataset (this might take a while)
    dataset = load_dataset("your-dataset-name")
    
    # Convert to polars DataFrame
    df = pl.DataFrame(dataset["train"])
    
    # Save as optimized parquet
    df.write_parquet(
        "data/knowledge_base.parquet",
        compression="snappy",  # Good compression + speed balance
        row_group_size=50000   # Optimize for chunked reading
    )
    
    print(f"Converted {len(df)} rows to parquet")
    print(f"File size: {os.path.getsize('data/knowledge_base.parquet') / 1024**3:.2f} GB")
```

### 2. Pre-process for Faster Queries:
```python
# preprocess_kg.py  
def create_search_indexes():
    """Pre-process text for faster searching"""
    df = pl.read_parquet("data/knowledge_base.parquet")
    
    # Add search-optimized columns
    df = df.with_columns([
        pl.col("text").str.to_lowercase().alias("text_lower"),
        pl.col("text").str.len().alias("text_length"),
        # Add other preprocessed columns for common searches
    ])
    
    # Save optimized version
    df.write_parquet("data/knowledge_base_optimized.parquet")
```

## Performance Expectations:

### Response Times (15-20 sec target):
- ✅ **Simple text search:** 5-10 seconds
- ✅ **Complex concept queries:** 10-15 seconds  
- ✅ **Relationship discovery:** 15-20 seconds
- ✅ **Multiple dataset queries:** 20+ seconds (acceptable for demo)

### Memory Usage:
- **Base system:** 2GB
- **Polars + chunk:** 1-2GB
- **MCP servers:** 1GB
- **Available:** 3-4GB headroom

## Deployment Steps:

1. **Setup t3.large** with 50GB storage
2. **Install Polars** (`pip install polars`)
3. **Convert HuggingFace dataset** to optimized parquet
4. **Deploy chunked MCP server**
5. **Test with actual queries**

## Cost Breakdown:
- **EC2 t3.large:** $60/month
- **50GB EBS:** $5/month
- **Data transfer:** $2/month
- **Total: ~$67/month**

**This approach gives you enterprise-scale data processing at demo-friendly costs and complexity!** 🎯

The key insight: With 15-20 second acceptable response times, we can use streaming/chunked processing instead of expensive in-memory solutions.