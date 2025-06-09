#!/usr/bin/env python3
"""
Basic usage example for PyHologres.

This example demonstrates:
- Connecting to Hologres
- Creating tables
- Adding data
- Vector search
- Full-text search
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

import pyhologres
import pyarrow as pa
import numpy as np


def main():
    # Connect to Hologres (using environment variables or direct connection)
    # Option 1: Local PostgreSQL-compatible connection
    db_uri = os.getenv('HOLOGRES_URI', 'postgresql://postgres:password@localhost:5432/test')
    
    # Option 2: Cloud connection
    # db_uri = "holo://your-endpoint/your-database"
    # api_key = os.getenv('HOLOGRES_API_KEY')
    # db = pyhologres.connect(db_uri, api_key=api_key, region="cn-hangzhou")
    
    print(f"Connecting to: {db_uri}")
    db = pyhologres.connect(db_uri)
    
    try:
        # Create a table schema
        schema = pa.schema([
            pa.field("id", pa.int64()),
            pa.field("title", pa.string()),
            pa.field("content", pa.string()),
            pa.field("embedding", pa.list_(pa.float32())),
            pa.field("category", pa.string()),
            pa.field("score", pa.float64())
        ])
        
        # Sample data
        documents = [
            {
                "id": 1,
                "title": "Introduction to Machine Learning",
                "content": "Machine learning is a subset of artificial intelligence that focuses on algorithms.",
                "embedding": np.random.rand(128).astype(np.float32).tolist(),
                "category": "AI",
                "score": 0.95
            },
            {
                "id": 2,
                "title": "Deep Learning Fundamentals",
                "content": "Deep learning uses neural networks with multiple layers to learn complex patterns.",
                "embedding": np.random.rand(128).astype(np.float32).tolist(),
                "category": "AI",
                "score": 0.92
            },
            {
                "id": 3,
                "title": "Database Systems Overview",
                "content": "Database systems are designed to store, retrieve, and manage large amounts of data.",
                "embedding": np.random.rand(128).astype(np.float32).tolist(),
                "category": "Database",
                "score": 0.88
            },
            {
                "id": 4,
                "title": "Vector Databases Explained",
                "content": "Vector databases are specialized for storing and querying high-dimensional vectors.",
                "embedding": np.random.rand(128).astype(np.float32).tolist(),
                "category": "Database",
                "score": 0.90
            },
            {
                "id": 5,
                "title": "Natural Language Processing",
                "content": "NLP combines computational linguistics with machine learning and deep learning.",
                "embedding": np.random.rand(128).astype(np.float32).tolist(),
                "category": "AI",
                "score": 0.93
            }
        ]
        
        # Create table
        table_name = "documents_example"
        print(f"Creating table: {table_name}")
        
        # Drop table if exists
        try:
            db.drop_table(table_name)
            print(f"Dropped existing table: {table_name}")
        except Exception:
            pass  # Table doesn't exist
        
        # Create new table
        table = db.create_table(table_name, data=documents, schema=schema)
        print(f"Created table with {table.count_rows()} rows")
        
        # List all tables
        print("\nAvailable tables:")
        for name in db.table_names():
            print(f"  - {name}")
        
        # Basic queries
        print("\n=== Basic Queries ===")
        
        # Convert to pandas for easy viewing
        df = table.to_pandas()
        print(f"\nTable shape: {df.shape}")
        print("\nFirst 3 rows:")
        print(df[['id', 'title', 'category', 'score']].head(3))
        
        # Vector search
        print("\n=== Vector Search ===")
        query_vector = np.random.rand(128).astype(np.float32).tolist()
        
        # Search for similar vectors
        results = table.search(query_vector, vector_column_name="embedding").limit(3).to_pandas()
        print("\nTop 3 similar documents:")
        print(results[['id', 'title', 'category', 'score']])
        
        # Create vector index for better performance
        print("\nCreating vector index...")
        try:
            table.create_index("embedding", index_type="IVFFLAT", metric="cosine")
            print("Vector index created successfully")
        except Exception as e:
            print(f"Vector index creation failed (may not be supported): {e}")
        
        # Full-text search
        print("\n=== Full-Text Search ===")
        
        # Create FTS index
        try:
            table.create_fts_index("content", language="English")
            print("Full-text search index created")
            
            # Search for documents containing specific terms
            from pyhologres.query import MatchQuery
            fts_results = table.search(MatchQuery("machine learning")).limit(3).to_pandas()
            print("\nDocuments matching 'machine learning':")
            print(fts_results[['id', 'title', 'category']])
            
        except Exception as e:
            print(f"Full-text search failed (may not be supported): {e}")
        
        # Filtering and aggregation
        print("\n=== Filtering and Aggregation ===")
        
        # Count rows by category
        ai_count = table.count_rows(filter="category = 'AI'")
        db_count = table.count_rows(filter="category = 'Database'")
        print(f"AI documents: {ai_count}")
        print(f"Database documents: {db_count}")
        
        # Update operations
        print("\n=== Update Operations ===")
        
        # Update a document
        table.update(
            where="id = 1",
            values={"score": 0.98, "category": "AI/ML"}
        )
        print("Updated document with id=1")
        
        # Verify update
        updated_doc = table.search().where("id = 1").to_pandas()
        print(f"Updated document: {updated_doc[['id', 'category', 'score']].iloc[0].to_dict()}")
        
        # Add more data
        print("\n=== Adding More Data ===")
        
        new_documents = [
            {
                "id": 6,
                "title": "Computer Vision Applications",
                "content": "Computer vision enables machines to interpret and understand visual information.",
                "embedding": np.random.rand(128).astype(np.float32).tolist(),
                "category": "AI",
                "score": 0.89
            }
        ]
        
        table.add(new_documents)
        print(f"Added {len(new_documents)} new documents")
        print(f"Total documents: {table.count_rows()}")
        
        # Complex query with multiple conditions
        print("\n=== Complex Queries ===")
        
        # Query with multiple filters
        complex_results = (
            table.search()
            .where("category = 'AI' AND score > 0.90")
            .select(["id", "title", "score"])
            .limit(5)
            .to_pandas()
        )
        print("\nHigh-scoring AI documents:")
        print(complex_results)
        
        # Cleanup (optional)
        print("\n=== Cleanup ===")
        
        # Delete a document
        table.delete("id = 6")
        print("Deleted document with id=6")
        print(f"Remaining documents: {table.count_rows()}")
        
        print("\n=== Example completed successfully! ===")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Close connection
        db.close()
        print("\nConnection closed.")


if __name__ == "__main__":
    main()