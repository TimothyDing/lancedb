#!/usr/bin/env python3
"""
Embedding functions example for PyHologres.

This example demonstrates:
- Using different embedding functions
- Automatic text embedding
- Semantic search
- Hybrid search (vector + full-text)
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

import pyhologres
from pyhologres.embeddings import (
    OpenAIEmbeddings,
    SentenceTransformerEmbeddings,
    HuggingFaceEmbeddings
)
from pyhologres.query import MatchQuery, PhraseQuery
import pyarrow as pa


def demo_openai_embeddings():
    """Demonstrate OpenAI embeddings."""
    print("\n=== OpenAI Embeddings Demo ===")
    
    # Check if API key is available
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("Skipping OpenAI demo - OPENAI_API_KEY not set")
        return None
    
    # Create embedding function
    embedding_fn = OpenAIEmbeddings(
        api_key=api_key,
        model="text-embedding-ada-002"
    )
    
    print(f"Embedding dimensions: {embedding_fn.ndims()}")
    
    return embedding_fn


def demo_sentence_transformers():
    """Demonstrate Sentence Transformers embeddings."""
    print("\n=== Sentence Transformers Demo ===")
    
    try:
        # Create embedding function
        embedding_fn = SentenceTransformerEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )
        
        print(f"Embedding dimensions: {embedding_fn.ndims()}")
        
        # Test embedding
        test_text = "This is a test sentence."
        embedding = embedding_fn.compute_query_embeddings([test_text])[0]
        print(f"Sample embedding (first 5 dims): {embedding[:5]}")
        
        return embedding_fn
        
    except ImportError:
        print("Skipping Sentence Transformers demo - sentence-transformers not installed")
        return None


def demo_huggingface_embeddings():
    """Demonstrate HuggingFace embeddings."""
    print("\n=== HuggingFace Embeddings Demo ===")
    
    try:
        # Create embedding function
        embedding_fn = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        print(f"Embedding dimensions: {embedding_fn.ndims()}")
        
        return embedding_fn
        
    except ImportError:
        print("Skipping HuggingFace demo - transformers not installed")
        return None


def create_sample_documents():
    """Create sample documents for testing."""
    return [
        {
            "id": 1,
            "title": "Introduction to Machine Learning",
            "content": "Machine learning is a method of data analysis that automates analytical model building. It is a branch of artificial intelligence based on the idea that systems can learn from data, identify patterns and make decisions with minimal human intervention.",
            "category": "AI",
            "tags": ["machine learning", "AI", "data science"]
        },
        {
            "id": 2,
            "title": "Deep Learning Neural Networks",
            "content": "Deep learning is part of a broader family of machine learning methods based on artificial neural networks with representation learning. Learning can be supervised, semi-supervised or unsupervised.",
            "category": "AI",
            "tags": ["deep learning", "neural networks", "AI"]
        },
        {
            "id": 3,
            "title": "Natural Language Processing",
            "content": "Natural language processing is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language, in particular how to program computers to process and analyze large amounts of natural language data.",
            "category": "NLP",
            "tags": ["NLP", "linguistics", "text processing"]
        },
        {
            "id": 4,
            "title": "Computer Vision Applications",
            "content": "Computer vision is an interdisciplinary scientific field that deals with how computers can gain high-level understanding from digital images or videos. From the perspective of engineering, it seeks to understand and automate tasks that the human visual system can do.",
            "category": "CV",
            "tags": ["computer vision", "image processing", "visual recognition"]
        },
        {
            "id": 5,
            "title": "Reinforcement Learning Basics",
            "content": "Reinforcement learning is an area of machine learning concerned with how intelligent agents ought to take actions in an environment in order to maximize the notion of cumulative reward. Reinforcement learning is one of three basic machine learning paradigms.",
            "category": "AI",
            "tags": ["reinforcement learning", "agents", "rewards"]
        },
        {
            "id": 6,
            "title": "Data Science Methodology",
            "content": "Data science is an inter-disciplinary field that uses scientific methods, processes, algorithms and systems to extract knowledge and insights from many structural and unstructured data in various forms, similar to data mining.",
            "category": "Data Science",
            "tags": ["data science", "analytics", "insights"]
        },
        {
            "id": 7,
            "title": "Big Data Technologies",
            "content": "Big data refers to data sets that are too large or complex to be dealt with by traditional data-processing application software. Data with many cases offer greater statistical power, while data with higher complexity may lead to a higher false discovery rate.",
            "category": "Big Data",
            "tags": ["big data", "distributed systems", "scalability"]
        },
        {
            "id": 8,
            "title": "Cloud Computing Fundamentals",
            "content": "Cloud computing is the on-demand availability of computer system resources, especially data storage and computing power, without direct active management by the user. The term is generally used to describe data centers available to many users over the Internet.",
            "category": "Cloud",
            "tags": ["cloud computing", "distributed systems", "scalability"]
        }
    ]


def demo_semantic_search(db, embedding_fn):
    """Demonstrate semantic search with embeddings."""
    print("\n=== Semantic Search Demo ===")
    
    # Create table with embedding function
    documents = create_sample_documents()
    
    table_name = "semantic_search_demo"
    
    # Drop table if exists
    try:
        db.drop_table(table_name)
    except Exception:
        pass
    
    # Create table with automatic embedding
    table = db.create_table(
        table_name,
        data=documents,
        embedding_function=embedding_fn,
        vector_column_name="content_embedding"
    )
    
    print(f"Created table with {table.count_rows()} documents")
    
    # Semantic search queries
    search_queries = [
        "artificial intelligence algorithms",
        "neural network architectures",
        "text analysis and processing",
        "image recognition systems",
        "distributed computing platforms"
    ]
    
    for query in search_queries:
        print(f"\nSearching for: '{query}'")
        
        # Perform semantic search
        results = (
            table.search(query, vector_column_name="content_embedding")
            .limit(3)
            .to_pandas()
        )
        
        print("Top 3 results:")
        for _, row in results.iterrows():
            print(f"  - {row['title']} (Category: {row['category']})")
    
    return table


def demo_hybrid_search(table):
    """Demonstrate hybrid search (vector + full-text)."""
    print("\n=== Hybrid Search Demo ===")
    
    try:
        # Create full-text search index
        table.create_fts_index("content", language="English")
        print("Created full-text search index")
        
        # Hybrid search examples
        hybrid_queries = [
            {
                "vector_query": "machine learning algorithms",
                "text_query": MatchQuery("learning"),
                "description": "Learning-related content"
            },
            {
                "vector_query": "computer systems",
                "text_query": PhraseQuery("computer vision"),
                "description": "Computer vision systems"
            }
        ]
        
        for query_info in hybrid_queries:
            print(f"\nHybrid search: {query_info['description']}")
            print(f"Vector query: '{query_info['vector_query']}'")
            print(f"Text query: {query_info['text_query']}")
            
            # Perform hybrid search
            results = (
                table.search(
                    vector=query_info['vector_query'],
                    query_type="hybrid",
                    fts_query=query_info['text_query']
                )
                .limit(3)
                .to_pandas()
            )
            
            print("Results:")
            for _, row in results.iterrows():
                print(f"  - {row['title']} (Category: {row['category']})")
    
    except Exception as e:
        print(f"Hybrid search not supported: {e}")


def demo_embedding_comparison(db):
    """Compare different embedding functions."""
    print("\n=== Embedding Comparison Demo ===")
    
    # Get available embedding functions
    embedding_functions = []
    
    # Try different embedding functions
    openai_fn = demo_openai_embeddings()
    if openai_fn:
        embedding_functions.append(("OpenAI", openai_fn))
    
    st_fn = demo_sentence_transformers()
    if st_fn:
        embedding_functions.append(("SentenceTransformers", st_fn))
    
    hf_fn = demo_huggingface_embeddings()
    if hf_fn:
        embedding_functions.append(("HuggingFace", hf_fn))
    
    if not embedding_functions:
        print("No embedding functions available for comparison")
        return
    
    # Test query
    test_query = "machine learning neural networks"
    print(f"\nComparing embeddings for query: '{test_query}'")
    
    # Create tables with different embedding functions
    documents = create_sample_documents()[:3]  # Use fewer documents for comparison
    
    for name, embedding_fn in embedding_functions:
        table_name = f"comparison_{name.lower()}"
        
        try:
            # Drop table if exists
            try:
                db.drop_table(table_name)
            except Exception:
                pass
            
            # Create table
            table = db.create_table(
                table_name,
                data=documents,
                embedding_function=embedding_fn,
                vector_column_name="content_embedding"
            )
            
            # Search
            results = (
                table.search(test_query, vector_column_name="content_embedding")
                .limit(2)
                .to_pandas()
            )
            
            print(f"\n{name} results:")
            for _, row in results.iterrows():
                print(f"  - {row['title']}")
        
        except Exception as e:
            print(f"Error with {name}: {e}")


def main():
    """Main function."""
    print("PyHologres Embedding Functions Example")
    print("=====================================")
    
    # Connect to database
    db_uri = os.getenv('HOLOGRES_URI', 'postgresql://postgres:password@localhost:5432/test')
    print(f"Connecting to: {db_uri}")
    
    db = pyhologres.connect(db_uri)
    
    try:
        # Demo different embedding functions
        demo_openai_embeddings()
        st_fn = demo_sentence_transformers()
        demo_huggingface_embeddings()
        
        # Use Sentence Transformers for main demo (most likely to be available)
        if st_fn:
            table = demo_semantic_search(db, st_fn)
            demo_hybrid_search(table)
        else:
            print("\nNo embedding functions available for full demo")
        
        # Compare embedding functions
        demo_embedding_comparison(db)
        
        print("\n=== Embedding example completed successfully! ===")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        db.close()
        print("\nConnection closed.")


if __name__ == "__main__":
    main()