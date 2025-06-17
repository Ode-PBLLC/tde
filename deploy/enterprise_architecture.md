# Enterprise Architecture for 1.77GB Dataset

## Dataset Profile Analysis

### Massive Scale Challenge:
- **1.77GB Parquet files** (HuggingFace dataset)
- **34,185,184 rows** of data
- **Knowledge graph conversion** required
- **Embeddings generation** for semantic search
- **Real-time query performance** needed

## Architecture Options

### Option 1: All-in-Memory (t3.2xlarge) - $240/month
```
EC2 t3.2xlarge (8 vCPU, 32GB RAM)
+ 100GB EBS storage
+ Direct parquet loading with pandas/polars
```

**Pros:** Simple, fast queries
**Cons:** Expensive, single point of failure

### Option 2: Database-Centric (Recommended) - $150/month
```
EC2 t3.large (4 vCPU, 8GB RAM) - $60/month
+ RDS PostgreSQL db.t3.large (2 vCPU, 8GB) - $70/month  
+ Vector database (pgvector) for embeddings
+ Redis cache for hot queries - $20/month
```

**Pros:** Scalable, professional, cacheable
**Cons:** More complex setup

### Option 3: Hybrid Approach - $120/month
```
EC2 t3.xlarge (4 vCPU, 16GB RAM) - $120/month
+ Parquet files on EFS/S3
+ Lazy loading with chunking
+ SQLite for knowledge graph
```

**Pros:** Cost-effective, simpler than full DB
**Cons:** Performance limitations at scale

## Recommended: Database-Centric Architecture

### 1. Data Pipeline
```python
# Convert HuggingFace dataset to PostgreSQL
import pandas as pd
from datasets import load_dataset

# Load dataset in chunks
dataset = load_dataset("your-dataset-name", streaming=True)

# Process and insert in batches
def process_to_database():
    for batch in dataset.iter(batch_size=10000):
        df = pd.DataFrame(batch)
        
        # Insert to PostgreSQL
        df.to_sql('knowledge_base', 
                 engine, 
                 if_exists='append', 
                 index=False,
                 chunksize=1000)
```

### 2. Knowledge Graph in PostgreSQL
```sql
-- Create tables for knowledge graph
CREATE TABLE concepts (
    id SERIAL PRIMARY KEY,
    name TEXT UNIQUE,
    description TEXT,
    embedding VECTOR(1536)  -- Using pgvector
);

CREATE TABLE relationships (
    id SERIAL PRIMARY KEY,
    source_concept_id INT REFERENCES concepts(id),
    target_concept_id INT REFERENCES concepts(id),
    relationship_type TEXT,
    weight FLOAT
);

CREATE TABLE passages (
    id SERIAL PRIMARY KEY,
    content TEXT,
    source_document TEXT,
    embedding VECTOR(1536),
    concepts INT[] -- Array of concept IDs
);

-- Indexes for performance
CREATE INDEX ON concepts USING ivfflat (embedding vector_cosine_ops);
CREATE INDEX ON passages USING ivfflat (embedding vector_cosine_ops);
```

### 3. Updated MCP Server Architecture
```python
# mcp/knowledge_db_server.py - Database-backed KG server
import asyncpg
import numpy as np

class DatabaseKnowledgeGraph:
    def __init__(self):
        self.pool = None
    
    async def get_similar_concepts(self, query_embedding, limit=10):
        async with self.pool.acquire() as conn:
            # Vector similarity search
            return await conn.fetch("""
                SELECT name, description, 
                       1 - (embedding <=> $1) as similarity
                FROM concepts
                ORDER BY embedding <=> $1
                LIMIT $2
            """, query_embedding, limit)
    
    async def get_concept_relationships(self, concept_id):
        async with self.pool.acquire() as conn:
            return await conn.fetch("""
                SELECT c.name, r.relationship_type, r.weight
                FROM relationships r
                JOIN concepts c ON c.id = r.target_concept_id  
                WHERE r.source_concept_id = $1
                ORDER BY r.weight DESC
            """, concept_id)
```

### 4. Caching Strategy
```python
import redis
import json

class CachedKnowledgeGraph:
    def __init__(self):
        self.redis = redis.Redis(host='localhost', port=6379, db=0)
        self.db_kg = DatabaseKnowledgeGraph()
    
    async def get_concept_with_cache(self, concept_name):
        # Check cache first
        cached = self.redis.get(f"concept:{concept_name}")
        if cached:
            return json.loads(cached)
        
        # Query database
        result = await self.db_kg.get_concept(concept_name)
        
        # Cache for 1 hour
        self.redis.setex(f"concept:{concept_name}", 3600, json.dumps(result))
        return result
```

## Deployment Configuration

### PostgreSQL Setup
```bash
# Install PostgreSQL with extensions
sudo apt install postgresql postgresql-contrib

# Install pgvector for embeddings
sudo -u postgres psql -c "CREATE EXTENSION vector;"

# Configure for large datasets
echo "shared_buffers = 2GB" >> /etc/postgresql/14/main/postgresql.conf
echo "effective_cache_size = 6GB" >> /etc/postgresql/14/main/postgresql.conf
echo "random_page_cost = 1.1" >> /etc/postgresql/14/main/postgresql.conf
```

### Data Import Pipeline
```python
# import_dataset.py
import asyncio
import asyncpg
from datasets import load_dataset
import pandas as pd

async def import_huggingface_dataset():
    # Connect to database
    conn = await asyncpg.connect('postgresql://user:pass@localhost/climate_kg')
    
    # Load dataset in streaming mode
    dataset = load_dataset("your-dataset", streaming=True)
    
    batch_size = 5000
    batch = []
    
    for i, record in enumerate(dataset):
        batch.append(record)
        
        if len(batch) >= batch_size:
            # Process batch
            await insert_batch(conn, batch)
            batch = []
            
            if i % 50000 == 0:
                print(f"Processed {i} records...")
    
    # Process remaining records
    if batch:
        await insert_batch(conn, batch)

async def insert_batch(conn, batch):
    # Convert to DataFrame
    df = pd.DataFrame(batch)
    
    # Generate embeddings if needed
    if 'embedding' not in df.columns:
        df['embedding'] = await generate_embeddings(df['text'].tolist())
    
    # Bulk insert
    records = df.to_records(index=False)
    await conn.executemany("""
        INSERT INTO passages (content, source_document, embedding)
        VALUES ($1, $2, $3)
    """, records)
```

## Performance Optimization

### 1. Chunked Loading
```python
# Never load full 1.77GB into memory
def query_chunked(query, chunk_size=10000):
    for chunk in pd.read_parquet('data.parquet', chunksize=chunk_size):
        # Process chunk
        filtered = chunk[chunk['text'].str.contains(query)]
        if not filtered.empty:
            yield filtered
```

### 2. Async Processing
```python
# Background knowledge graph updates
from celery import Celery

@celery.task
def update_embeddings_background():
    # Generate embeddings for new data
    # Update relationship weights
    # Refresh caches
    pass
```

### 3. Connection Pooling
```python
# PostgreSQL connection pool
import asyncpg

class DatabasePool:
    def __init__(self):
        self.pool = None
    
    async def init_pool(self):
        self.pool = await asyncpg.create_pool(
            dsn='postgresql://user:pass@localhost/climate_kg',
            min_size=10,
            max_size=20
        )
```

## Cost Analysis

### Database-Centric Option ($150/month):
- **EC2 t3.large:** $60/month
- **RDS PostgreSQL db.t3.large:** $70/month  
- **ElastiCache Redis:** $20/month
- **Storage (100GB):** $10/month

### Benefits:
- ✅ Handles 34M+ rows efficiently
- ✅ Vector similarity search built-in
- ✅ Automatic backups and scaling
- ✅ Professional architecture
- ✅ Can handle concurrent users
- ✅ Cacheable queries

## Migration Strategy

1. **Week 1:** Setup PostgreSQL + pgvector
2. **Week 2:** Import HuggingFace dataset 
3. **Week 3:** Generate knowledge graph relationships
4. **Week 4:** Optimize queries and add caching
5. **Week 5:** Deploy and test performance

This is definitely enterprise-scale now! 🚀