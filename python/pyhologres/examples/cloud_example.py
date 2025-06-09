#!/usr/bin/env python3
"""
Hologres Cloud connection example for PyHologres.

This example demonstrates:
- Connecting to Hologres Cloud
- Remote table operations
- Cloud-specific features
- API configuration
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

import pyhologres
from pyhologres.remote import ClientConfig
import pyarrow as pa
import numpy as np
from typing import Dict, Any


def get_cloud_config() -> Dict[str, Any]:
    """Get cloud configuration from environment variables."""
    config = {
        "api_key": os.getenv('HOLOGRES_API_KEY'),
        "region": os.getenv('HOLOGRES_REGION', 'cn-hangzhou'),
        "endpoint": os.getenv('HOLOGRES_ENDPOINT'),
        "database": os.getenv('HOLOGRES_DATABASE', 'default')
    }
    
    # Check required configuration
    if not config["api_key"]:
        raise ValueError("HOLOGRES_API_KEY environment variable is required")
    
    if not config["endpoint"]:
        raise ValueError("HOLOGRES_ENDPOINT environment variable is required")
    
    return config


def demo_cloud_connection():
    """Demonstrate basic cloud connection."""
    print("\n=== Cloud Connection Demo ===")
    
    try:
        config = get_cloud_config()
        
        # Build cloud URI
        cloud_uri = f"holo://{config['endpoint']}/{config['database']}"
        print(f"Connecting to: {cloud_uri}")
        
        # Create client configuration
        client_config = ClientConfig(
            timeout=30,
            max_retries=3,
            retry_delay=1.0,
            max_connections=10
        )
        
        # Connect to Hologres Cloud
        db = pyhologres.connect(
            cloud_uri,
            api_key=config["api_key"],
            region=config["region"],
            client_config=client_config
        )
        
        print("Successfully connected to Hologres Cloud")
        
        # List existing tables
        tables = db.table_names()
        print(f"Found {len(tables)} tables: {tables}")
        
        return db, config
        
    except Exception as e:
        print(f"Cloud connection failed: {e}")
        print("Make sure to set HOLOGRES_API_KEY and HOLOGRES_ENDPOINT environment variables")
        return None, None


def demo_cloud_table_operations(db):
    """Demonstrate cloud table operations."""
    print("\n=== Cloud Table Operations Demo ===")
    
    if not db:
        print("Skipping cloud table operations - no connection")
        return None
    
    try:
        # Create schema for cloud table
        schema = pa.schema([
            pa.field("id", pa.int64()),
            pa.field("title", pa.string()),
            pa.field("content", pa.string()),
            pa.field("embedding", pa.list_(pa.float32())),
            pa.field("category", pa.string()),
            pa.field("score", pa.float64()),
            pa.field("metadata", pa.string())
        ])
        
        # Sample data for cloud
        cloud_documents = [
            {
                "id": 1,
                "title": "Cloud Computing Basics",
                "content": "Introduction to cloud computing concepts and services.",
                "embedding": np.random.rand(128).astype(np.float32).tolist(),
                "category": "Cloud",
                "score": 0.92,
                "metadata": '{"source": "cloud_docs", "version": "1.0"}'
            },
            {
                "id": 2,
                "title": "Distributed Systems",
                "content": "Understanding distributed systems architecture and patterns.",
                "embedding": np.random.rand(128).astype(np.float32).tolist(),
                "category": "Architecture",
                "score": 0.89,
                "metadata": '{"source": "arch_docs", "version": "1.1"}'
            },
            {
                "id": 3,
                "title": "Microservices Design",
                "content": "Best practices for designing and implementing microservices.",
                "embedding": np.random.rand(128).astype(np.float32).tolist(),
                "category": "Architecture",
                "score": 0.94,
                "metadata": '{"source": "design_docs", "version": "2.0"}'
            },
            {
                "id": 4,
                "title": "API Gateway Patterns",
                "content": "Common patterns and practices for API gateway implementation.",
                "embedding": np.random.rand(128).astype(np.float32).tolist(),
                "category": "API",
                "score": 0.87,
                "metadata": '{"source": "api_docs", "version": "1.5"}'
            },
            {
                "id": 5,
                "title": "Container Orchestration",
                "content": "Managing containerized applications at scale using orchestration tools.",
                "embedding": np.random.rand(128).astype(np.float32).tolist(),
                "category": "DevOps",
                "score": 0.91,
                "metadata": '{"source": "devops_docs", "version": "1.3"}'
            }
        ]
        
        # Create cloud table
        table_name = "cloud_demo_table"
        
        # Drop table if exists
        try:
            db.drop_table(table_name)
            print(f"Dropped existing table: {table_name}")
        except Exception:
            pass
        
        # Create new table
        table = db.create_table(
            table_name,
            data=cloud_documents,
            schema=schema,
            mode="create"
        )
        
        print(f"Created cloud table with {table.count_rows()} rows")
        
        # Test cloud table operations
        print("\nTesting cloud table operations:")
        
        # Vector search
        query_vector = np.random.rand(128).astype(np.float32).tolist()
        search_results = (
            table.search(query_vector, vector_column_name="embedding")
            .limit(3)
            .to_pandas()
        )
        
        print(f"Vector search returned {len(search_results)} results")
        print(search_results[['id', 'title', 'category', 'score']])
        
        # Add more data to cloud table
        additional_docs = [
            {
                "id": 6,
                "title": "Serverless Computing",
                "content": "Understanding serverless architecture and Function-as-a-Service.",
                "embedding": np.random.rand(128).astype(np.float32).tolist(),
                "category": "Cloud",
                "score": 0.88,
                "metadata": '{"source": "serverless_docs", "version": "1.0"}'
            }
        ]
        
        table.add(additional_docs)
        print(f"\nAdded {len(additional_docs)} documents. Total: {table.count_rows()}")
        
        return table
        
    except Exception as e:
        print(f"Cloud table operations failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def demo_cloud_indexing(table):
    """Demonstrate cloud indexing features."""
    print("\n=== Cloud Indexing Demo ===")
    
    if not table:
        print("Skipping cloud indexing - no table")
        return
    
    try:
        # Create vector index
        print("Creating vector index...")
        table.create_index(
            "embedding",
            index_type="IVFFLAT",
            metric="cosine",
            num_partitions=10
        )
        print("Vector index created successfully")
        
        # Create full-text search index
        print("Creating full-text search index...")
        table.create_fts_index(
            "content",
            language="English",
            base_tokenizer="standard"
        )
        print("Full-text search index created successfully")
        
        # Create regular index
        print("Creating category index...")
        table.create_index(
            "category",
            index_type="BTREE",
            name="category_idx"
        )
        print("Category index created successfully")
        
        # List all indices
        indices = table.list_indices()
        print(f"\nTable indices: {len(indices)}")
        for idx in indices:
            print(f"  - {idx.get('name', 'unnamed')}: {idx.get('type', 'unknown')} on {idx.get('column', 'unknown')}")
        
    except Exception as e:
        print(f"Cloud indexing failed: {e}")
        # This is expected if the cloud service doesn't support certain index types


def demo_cloud_search_features(table):
    """Demonstrate cloud-specific search features."""
    print("\n=== Cloud Search Features Demo ===")
    
    if not table:
        print("Skipping cloud search - no table")
        return
    
    try:
        # Vector similarity search with different metrics
        print("Testing vector similarity search...")
        query_vector = np.random.rand(128).astype(np.float32).tolist()
        
        # Basic vector search
        results = (
            table.search(query_vector, vector_column_name="embedding")
            .limit(3)
            .to_pandas()
        )
        print(f"Basic vector search: {len(results)} results")
        
        # Search with filters
        filtered_results = (
            table.search(query_vector, vector_column_name="embedding")
            .where("category = 'Cloud'")
            .limit(2)
            .to_pandas()
        )
        print(f"Filtered vector search (Cloud category): {len(filtered_results)} results")
        
        # Full-text search (if supported)
        try:
            from pyhologres.query import MatchQuery
            fts_results = (
                table.search(MatchQuery("computing"))
                .limit(3)
                .to_pandas()
            )
            print(f"Full-text search: {len(fts_results)} results")
        except Exception as e:
            print(f"Full-text search not supported: {e}")
        
        # Hybrid search (if supported)
        try:
            hybrid_results = (
                table.search(
                    vector=query_vector,
                    query_type="hybrid",
                    fts_query=MatchQuery("architecture")
                )
                .limit(3)
                .to_pandas()
            )
            print(f"Hybrid search: {len(hybrid_results)} results")
        except Exception as e:
            print(f"Hybrid search not supported: {e}")
        
    except Exception as e:
        print(f"Cloud search features failed: {e}")


def demo_cloud_data_management(table):
    """Demonstrate cloud data management features."""
    print("\n=== Cloud Data Management Demo ===")
    
    if not table:
        print("Skipping cloud data management - no table")
        return
    
    try:
        # Update operations
        print("Testing update operations...")
        table.update(
            where="category = 'API'",
            values={"score": 0.95}
        )
        print("Updated API documents")
        
        # Verify update
        api_docs = (
            table.search()
            .where("category = 'API'")
            .select(["id", "title", "category", "score"])
            .to_pandas()
        )
        print(f"API documents after update: {len(api_docs)}")
        if len(api_docs) > 0:
            print(api_docs)
        
        # Batch operations
        print("\nTesting batch operations...")
        
        # Add multiple documents
        batch_docs = []
        for i in range(3):
            doc = {
                "id": 100 + i,
                "title": f"Batch Document {i + 1}",
                "content": f"This is batch document {i + 1} for testing cloud operations.",
                "embedding": np.random.rand(128).astype(np.float32).tolist(),
                "category": "Batch",
                "score": 0.80 + (i * 0.05),
                "metadata": f'{{"batch_id": {i + 1}, "test": true}}'
            }
            batch_docs.append(doc)
        
        table.add(batch_docs)
        print(f"Added {len(batch_docs)} batch documents")
        
        # Count by category
        total_count = table.count_rows()
        batch_count = table.count_rows(filter="category = 'Batch'")
        print(f"Total documents: {total_count}, Batch documents: {batch_count}")
        
        # Delete batch documents
        table.delete("category = 'Batch'")
        print("Deleted batch documents")
        
        final_count = table.count_rows()
        print(f"Final document count: {final_count}")
        
    except Exception as e:
        print(f"Cloud data management failed: {e}")


def demo_cloud_performance_monitoring():
    """Demonstrate cloud performance monitoring."""
    print("\n=== Cloud Performance Monitoring Demo ===")
    
    try:
        config = get_cloud_config()
        
        # Create client with monitoring configuration
        client_config = ClientConfig(
            timeout=30,
            max_retries=3,
            retry_delay=1.0,
            max_connections=5,
            enable_metrics=True,
            request_timeout=10
        )
        
        cloud_uri = f"holo://{config['endpoint']}/{config['database']}"
        
        with pyhologres.connect(
            cloud_uri,
            api_key=config["api_key"],
            region=config["region"],
            client_config=client_config
        ) as db:
            
            # Perform operations and monitor performance
            import time
            
            start_time = time.time()
            tables = db.table_names()
            list_time = time.time() - start_time
            
            print(f"Listed {len(tables)} tables in {list_time:.3f} seconds")
            
            if tables:
                table_name = tables[0]
                
                start_time = time.time()
                table = db.open_table(table_name)
                open_time = time.time() - start_time
                
                print(f"Opened table '{table_name}' in {open_time:.3f} seconds")
                
                start_time = time.time()
                count = table.count_rows()
                count_time = time.time() - start_time
                
                print(f"Counted {count} rows in {count_time:.3f} seconds")
        
        print("Performance monitoring completed")
        
    except Exception as e:
        print(f"Performance monitoring failed: {e}")


def main():
    """Main function."""
    print("PyHologres Cloud Connection Example")
    print("===================================")
    
    try:
        # Check if cloud configuration is available
        try:
            config = get_cloud_config()
            print(f"Cloud configuration found:")
            print(f"  Region: {config['region']}")
            print(f"  Endpoint: {config['endpoint']}")
            print(f"  Database: {config['database']}")
        except ValueError as e:
            print(f"Cloud configuration error: {e}")
            print("\nTo run this example, set the following environment variables:")
            print("  export HOLOGRES_API_KEY=your_api_key")
            print("  export HOLOGRES_ENDPOINT=your_endpoint")
            print("  export HOLOGRES_DATABASE=your_database  # optional, defaults to 'default'")
            print("  export HOLOGRES_REGION=cn-hangzhou     # optional, defaults to 'cn-hangzhou'")
            return
        
        # Demo cloud connection
        db, config = demo_cloud_connection()
        
        if db:
            try:
                # Demo table operations
                table = demo_cloud_table_operations(db)
                
                # Demo indexing
                demo_cloud_indexing(table)
                
                # Demo search features
                demo_cloud_search_features(table)
                
                # Demo data management
                demo_cloud_data_management(table)
                
                print("\n=== Cloud example completed successfully! ===")
                
            finally:
                # Cleanup
                try:
                    if table:
                        db.drop_table("cloud_demo_table")
                        print("Cleaned up demo table")
                except Exception:
                    pass
                
                db.close()
                print("Connection closed")
        
        # Demo performance monitoring
        demo_cloud_performance_monitoring()
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()