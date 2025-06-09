# PyHologres

PyHologres is a Python client library for Hologres, providing both local database connections and cloud-based remote connections. It offers a LanceDB-compatible API for vector operations, full-text search, and traditional database operations.

## Features

- **Dual Connection Modes**: Support for both local PostgreSQL-compatible connections and Hologres Cloud API connections
- **Vector Operations**: Native support for vector storage, indexing, and similarity search
- **Full-Text Search**: Built-in full-text search capabilities with customizable tokenizers
- **LanceDB Compatibility**: Familiar API for users migrating from LanceDB
- **Async Support**: Full async/await support for non-blocking operations
- **Multiple Data Formats**: Support for PyArrow, Pandas, Polars, and native Python data structures
- **Embedding Functions**: Built-in support for popular embedding models (OpenAI, Sentence Transformers, etc.)

## Installation

```bash
pip install pyhologres
```

## Quick Start

### Local Connection

```python
import pyhologres

# Connect to local Hologres instance
db = pyhologres.connect("postgresql://user:password@localhost:5432/mydb")

# Create a table with vector data
import pyarrow as pa
schema = pa.schema([
    pa.field("id", pa.int64()),
    pa.field("text", pa.string()),
    pa.field("vector", pa.list_(pa.float32()))
])

table = db.create_table("documents", schema=schema)

# Add data
data = [
    {"id": 1, "text": "Hello world", "vector": [0.1, 0.2, 0.3]},
    {"id": 2, "text": "Goodbye world", "vector": [0.4, 0.5, 0.6]}
]
table.add(data)

# Vector search
results = table.search([0.1, 0.2, 0.3]).limit(10).to_pandas()
print(results)
```

### Cloud Connection

```python
import pyhologres

# Connect to Hologres Cloud
db = pyhologres.connect(
    "holo://your-endpoint/your-database",
    api_key="your-api-key",
    region="cn-hangzhou"
)

# Same API as local connection
table = db.open_table("documents")
results = table.search([0.1, 0.2, 0.3]).limit(10).to_pandas()
```

### Async Operations

```python
import asyncio
import pyhologres

async def main():
    # Async connection
    db = await pyhologres.connect_async("postgresql://user:password@localhost:5432/mydb")
    
    # Async operations
    table = await db.open_table("documents")
    results = await table.search([0.1, 0.2, 0.3]).limit(10).to_pandas_async()
    
    await db.close()

asyncio.run(main())
```

## Advanced Features

### Embedding Functions

```python
from pyhologres.embeddings import OpenAIEmbeddings

# Configure embedding function
embedding_fn = OpenAIEmbeddings(api_key="your-openai-key")

# Create table with embedding function
table = db.create_table(
    "documents",
    data=[
        {"id": 1, "text": "Hello world"},
        {"id": 2, "text": "Goodbye world"}
    ],
    embedding_function=embedding_fn,
    vector_column_name="text_vector"
)

# Search with text (automatically embedded)
results = table.search("Hello").limit(10).to_pandas()
```

### Full-Text Search

```python
# Create full-text search index
table.create_fts_index("text", language="English")

# Full-text search
from pyhologres.query import MatchQuery
results = table.search(MatchQuery("hello world")).limit(10).to_pandas()

# Hybrid search (vector + full-text)
results = table.search(
    vector=[0.1, 0.2, 0.3],
    query_type="hybrid",
    fts_query=MatchQuery("hello")
).limit(10).to_pandas()
```

### Vector Indexing

```python
# Create vector index for faster similarity search
table.create_index(
    "vector",
    index_type="IVFFLAT",
    metric="cosine",
    num_partitions=100
)
```

### Data Operations

```python
# Update data
table.update(
    where="id = 1",
    values={"text": "Updated text"}
)

# Delete data
table.delete("id = 1")

# Count rows
count = table.count_rows()
print(f"Total rows: {count}")

# Convert to different formats
pandas_df = table.to_pandas()
polars_df = table.to_polars()
arrow_table = table.to_arrow()
```

## Configuration

### Environment Variables

You can configure connections using environment variables:

```bash
# For local connections
export HOLOGRES_HOST=localhost
export HOLOGRES_PORT=5432
export HOLOGRES_USER=myuser
export HOLOGRES_PASSWORD=mypassword
export HOLOGRES_DATABASE=mydb

# For cloud connections
export HOLOGRES_API_KEY=your-api-key
export HOLOGRES_REGION=cn-hangzhou
export HOLOGRES_ENDPOINT=your-endpoint
```

### Connection Options

```python
# Local connection with options
db = pyhologres.connect(
    "postgresql://user:password@localhost:5432/mydb",
    pool_size=10,
    max_overflow=20,
    pool_timeout=30,
    pool_recycle=3600
)

# Cloud connection with options
db = pyhologres.connect(
    "holo://your-endpoint/your-database",
    api_key="your-api-key",
    region="cn-hangzhou",
    timeout=30,
    max_retries=3,
    retry_delay=1.0
)
```

## API Reference

### Database Connection

- `pyhologres.connect(uri, **kwargs)` - Create a database connection
- `pyhologres.connect_async(uri, **kwargs)` - Create an async database connection

### Database Operations

- `db.table_names()` - List all table names
- `db.open_table(name)` - Open an existing table
- `db.create_table(name, data=None, schema=None, **kwargs)` - Create a new table
- `db.drop_table(name)` - Drop a table
- `db.rename_table(old_name, new_name)` - Rename a table

### Table Operations

- `table.add(data, mode="append")` - Add data to table
- `table.search(query, **kwargs)` - Search the table
- `table.update(where, values)` - Update rows
- `table.delete(where)` - Delete rows
- `table.count_rows(filter=None)` - Count rows
- `table.to_pandas()` - Convert to pandas DataFrame
- `table.to_polars()` - Convert to polars DataFrame
- `table.to_arrow()` - Convert to PyArrow Table

### Query Operations

- `query.limit(n)` - Limit results
- `query.offset(n)` - Skip results
- `query.where(condition)` - Filter results
- `query.select(columns)` - Select specific columns
- `query.to_pandas()` - Execute and return pandas DataFrame
- `query.to_polars()` - Execute and return polars DataFrame
- `query.to_arrow()` - Execute and return PyArrow Table

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.