#!/usr/bin/env python3
"""
Async operations example for PyHologres.

This example demonstrates:
- Async database connections
- Async table operations
- Concurrent operations
- Async context managers
"""

import asyncio
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

import pyhologres
import pyarrow as pa
import numpy as np
import time
from typing import List, Dict, Any


async def create_sample_data(num_docs: int = 100) -> List[Dict[str, Any]]:
    """Create sample documents for testing."""
    print(f"Creating {num_docs} sample documents...")
    
    categories = ["AI", "Database", "Web", "Mobile", "Security"]
    
    documents = []
    for i in range(num_docs):
        doc = {
            "id": i + 1,
            "title": f"Document {i + 1}",
            "content": f"This is the content of document {i + 1}. It contains information about {categories[i % len(categories)]}.",
            "embedding": np.random.rand(128).astype(np.float32).tolist(),
            "category": categories[i % len(categories)],
            "score": np.random.uniform(0.5, 1.0),
            "created_at": f"2024-01-{(i % 30) + 1:02d}"
        }
        documents.append(doc)
    
    return documents


async def demo_basic_async_operations():
    """Demonstrate basic async operations."""
    print("\n=== Basic Async Operations Demo ===")
    
    # Async connection
    db_uri = os.getenv('HOLOGRES_URI', 'postgresql://postgres:password@localhost:5432/test')
    print(f"Connecting to: {db_uri}")
    
    async with pyhologres.connect_async(db_uri) as db:
        # Create schema
        schema = pa.schema([
            pa.field("id", pa.int64()),
            pa.field("title", pa.string()),
            pa.field("content", pa.string()),
            pa.field("embedding", pa.list_(pa.float32())),
            pa.field("category", pa.string()),
            pa.field("score", pa.float64()),
            pa.field("created_at", pa.string())
        ])
        
        # Create sample data
        documents = await create_sample_data(50)
        
        # Create table
        table_name = "async_demo_table"
        
        # Drop table if exists
        try:
            await db.drop_table_async(table_name)
            print(f"Dropped existing table: {table_name}")
        except Exception:
            pass
        
        # Create new table
        table = await db.create_table_async(table_name, data=documents, schema=schema)
        print(f"Created table with {await table.count_rows_async()} rows")
        
        # Async search
        query_vector = np.random.rand(128).astype(np.float32).tolist()
        results = await (
            table.search(query_vector, vector_column_name="embedding")
            .limit(5)
            .to_pandas_async()
        )
        
        print(f"\nAsync search results: {len(results)} rows")
        print(results[['id', 'title', 'category', 'score']].head())
        
        return table


async def demo_concurrent_operations(table):
    """Demonstrate concurrent operations."""
    print("\n=== Concurrent Operations Demo ===")
    
    # Define multiple search queries
    search_tasks = []
    
    for i in range(5):
        query_vector = np.random.rand(128).astype(np.float32).tolist()
        task = table.search(query_vector, vector_column_name="embedding").limit(3).to_pandas_async()
        search_tasks.append(task)
    
    # Execute searches concurrently
    start_time = time.time()
    results = await asyncio.gather(*search_tasks)
    end_time = time.time()
    
    print(f"Executed {len(search_tasks)} concurrent searches in {end_time - start_time:.2f} seconds")
    
    for i, result in enumerate(results):
        print(f"Search {i + 1}: {len(result)} results")
    
    return results


async def demo_async_data_operations(table):
    """Demonstrate async data operations."""
    print("\n=== Async Data Operations Demo ===")
    
    # Add more data asynchronously
    new_documents = await create_sample_data(20)
    
    # Modify IDs to avoid conflicts
    for i, doc in enumerate(new_documents):
        doc["id"] = 1000 + i
        doc["title"] = f"New Document {i + 1}"
    
    await table.add_async(new_documents)
    print(f"Added {len(new_documents)} new documents")
    
    # Count rows asynchronously
    total_rows = await table.count_rows_async()
    print(f"Total rows after addition: {total_rows}")
    
    # Async update
    await table.update_async(
        where="category = 'AI'",
        values={"score": 0.95}
    )
    print("Updated all AI documents with score 0.95")
    
    # Verify update
    ai_docs = await (
        table.search()
        .where("category = 'AI'")
        .select(["id", "title", "category", "score"])
        .limit(5)
        .to_pandas_async()
    )
    
    print("\nUpdated AI documents:")
    print(ai_docs)
    
    # Async delete
    await table.delete_async("id > 1010")
    print("Deleted documents with id > 1010")
    
    final_count = await table.count_rows_async()
    print(f"Final row count: {final_count}")


async def demo_async_batch_processing():
    """Demonstrate async batch processing."""
    print("\n=== Async Batch Processing Demo ===")
    
    db_uri = os.getenv('HOLOGRES_URI', 'postgresql://postgres:password@localhost:5432/test')
    
    async with pyhologres.connect_async(db_uri) as db:
        # Create multiple tables concurrently
        table_tasks = []
        
        for i in range(3):
            table_name = f"batch_table_{i + 1}"
            documents = await create_sample_data(30)
            
            # Modify IDs to avoid conflicts
            for doc in documents:
                doc["id"] = doc["id"] + (i * 1000)
            
            # Create table task
            task = db.create_table_async(
                table_name,
                data=documents,
                mode="overwrite"
            )
            table_tasks.append((table_name, task))
        
        # Execute table creation concurrently
        start_time = time.time()
        tables = []
        
        for table_name, task in table_tasks:
            try:
                table = await task
                tables.append((table_name, table))
                print(f"Created table: {table_name}")
            except Exception as e:
                print(f"Failed to create table {table_name}: {e}")
        
        end_time = time.time()
        print(f"Created {len(tables)} tables in {end_time - start_time:.2f} seconds")
        
        # Perform operations on all tables concurrently
        operation_tasks = []
        
        for table_name, table in tables:
            # Count rows
            count_task = table.count_rows_async()
            operation_tasks.append((f"{table_name}_count", count_task))
            
            # Search
            query_vector = np.random.rand(128).astype(np.float32).tolist()
            search_task = (
                table.search(query_vector, vector_column_name="embedding")
                .limit(3)
                .to_pandas_async()
            )
            operation_tasks.append((f"{table_name}_search", search_task))
        
        # Execute all operations concurrently
        start_time = time.time()
        
        for operation_name, task in operation_tasks:
            try:
                result = await task
                if "count" in operation_name:
                    print(f"{operation_name}: {result} rows")
                else:
                    print(f"{operation_name}: {len(result)} search results")
            except Exception as e:
                print(f"Failed operation {operation_name}: {e}")
        
        end_time = time.time()
        print(f"Completed {len(operation_tasks)} operations in {end_time - start_time:.2f} seconds")
        
        # Cleanup tables
        cleanup_tasks = []
        for table_name, _ in tables:
            cleanup_tasks.append(db.drop_table_async(table_name))
        
        await asyncio.gather(*cleanup_tasks, return_exceptions=True)
        print(f"Cleaned up {len(tables)} tables")


async def demo_async_streaming():
    """Demonstrate async streaming operations."""
    print("\n=== Async Streaming Demo ===")
    
    db_uri = os.getenv('HOLOGRES_URI', 'postgresql://postgres:password@localhost:5432/test')
    
    async with pyhologres.connect_async(db_uri) as db:
        table_name = "streaming_demo"
        
        # Create table
        schema = pa.schema([
            pa.field("id", pa.int64()),
            pa.field("data", pa.string()),
            pa.field("timestamp", pa.string())
        ])
        
        table = await db.create_table_async(table_name, schema=schema, mode="overwrite")
        
        # Simulate streaming data insertion
        async def stream_data_producer():
            """Produce streaming data."""
            for batch_id in range(5):
                batch_data = []
                for i in range(10):
                    doc = {
                        "id": batch_id * 10 + i,
                        "data": f"Streaming data batch {batch_id}, item {i}",
                        "timestamp": f"2024-01-01T{batch_id:02d}:{i:02d}:00"
                    }
                    batch_data.append(doc)
                
                yield batch_data
                await asyncio.sleep(0.1)  # Simulate delay
        
        # Process streaming data
        total_inserted = 0
        async for batch in stream_data_producer():
            await table.add_async(batch)
            total_inserted += len(batch)
            print(f"Inserted batch of {len(batch)} items (total: {total_inserted})")
        
        # Verify final count
        final_count = await table.count_rows_async()
        print(f"Final streaming table count: {final_count}")
        
        # Cleanup
        await db.drop_table_async(table_name)


async def demo_error_handling():
    """Demonstrate async error handling."""
    print("\n=== Async Error Handling Demo ===")
    
    db_uri = os.getenv('HOLOGRES_URI', 'postgresql://postgres:password@localhost:5432/test')
    
    try:
        async with pyhologres.connect_async(db_uri) as db:
            # Try to open non-existent table
            try:
                table = await db.open_table_async("non_existent_table")
            except Exception as e:
                print(f"Expected error opening non-existent table: {e}")
            
            # Try invalid operations
            try:
                # Create table with invalid schema
                invalid_data = [{"invalid_field": "value"}]
                table = await db.create_table_async(
                    "invalid_table",
                    data=invalid_data,
                    schema=pa.schema([pa.field("different_field", pa.string())])
                )
            except Exception as e:
                print(f"Expected error with schema mismatch: {e}")
            
            print("Error handling completed successfully")
    
    except Exception as e:
        print(f"Connection error: {e}")


async def main():
    """Main async function."""
    print("PyHologres Async Operations Example")
    print("===================================")
    
    try:
        # Basic async operations
        table = await demo_basic_async_operations()
        
        # Concurrent operations
        await demo_concurrent_operations(table)
        
        # Async data operations
        await demo_async_data_operations(table)
        
        # Batch processing
        await demo_async_batch_processing()
        
        # Streaming operations
        await demo_async_streaming()
        
        # Error handling
        await demo_error_handling()
        
        print("\n=== Async example completed successfully! ===")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())