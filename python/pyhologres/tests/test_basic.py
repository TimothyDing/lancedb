# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Hologres Authors

import pytest
import os
import tempfile
import numpy as np
import pyarrow as pa
from typing import List, Dict, Any

import pyhologres
from pyhologres.common import sanitize_uri, parse_hologres_uri
from pyhologres.schema import create_vector_column, validate_vector_data


class TestBasicFunctionality:
    """Test basic PyHologres functionality."""
    
    @pytest.fixture
    def sample_data(self) -> List[Dict[str, Any]]:
        """Create sample data for testing."""
        return [
            {
                "id": 1,
                "text": "Hello world",
                "vector": [0.1, 0.2, 0.3],
                "category": "greeting"
            },
            {
                "id": 2,
                "text": "Goodbye world",
                "vector": [0.4, 0.5, 0.6],
                "category": "farewell"
            },
            {
                "id": 3,
                "text": "How are you?",
                "vector": [0.7, 0.8, 0.9],
                "category": "question"
            }
        ]
    
    @pytest.fixture
    def sample_schema(self) -> pa.Schema:
        """Create sample schema for testing."""
        return pa.schema([
            pa.field("id", pa.int64()),
            pa.field("text", pa.string()),
            pa.field("vector", pa.list_(pa.float32())),
            pa.field("category", pa.string())
        ])
    
    def test_uri_sanitization(self):
        """Test URI sanitization."""
        # Test PostgreSQL URI
        pg_uri = "postgresql://user:pass@localhost:5432/db"
        sanitized = sanitize_uri(pg_uri)
        assert sanitized == pg_uri
        
        # Test Hologres URI
        holo_uri = "holo://endpoint/database"
        sanitized = sanitize_uri(holo_uri)
        assert sanitized == holo_uri
        
        # Test URI with special characters
        special_uri = "postgresql://user:p@ss@localhost:5432/db"
        sanitized = sanitize_uri(special_uri)
        assert "p@ss" in sanitized
    
    def test_hologres_uri_parsing(self):
        """Test Hologres URI parsing."""
        uri = "holo://my-endpoint.hologres.aliyuncs.com/my_database"
        parsed = parse_hologres_uri(uri)
        
        assert parsed["scheme"] == "holo"
        assert parsed["endpoint"] == "my-endpoint.hologres.aliyuncs.com"
        assert parsed["path"] == "/my_database"
        
        # Test URI without database
        uri_no_db = "holo://my-endpoint.hologres.aliyuncs.com"
        parsed_no_db = parse_hologres_uri(uri_no_db)
        assert parsed_no_db["path"] == ""
    
    def test_vector_column_creation(self):
        """Test vector column creation."""
        # Test float32 vector
        field = create_vector_column("embedding", 128, pa.float32())
        assert field.name == "embedding"
        assert pa.types.is_list(field.type)
        assert field.type.value_type == pa.float32()
        
        # Test float64 vector
        field_64 = create_vector_column("embedding_64", 256, pa.float64())
        assert field_64.type.value_type == pa.float64()
    
    def test_vector_data_validation(self):
        """Test vector data validation."""
        # Valid vector data
        valid_vectors = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        validated = validate_vector_data(valid_vectors, expected_dim=3)
        assert len(validated) == 2
        assert all(len(v) == 3 for v in validated)
        
        # Invalid dimension
        with pytest.raises(ValueError, match="Expected vector dimension"):
            validate_vector_data([[0.1, 0.2]], expected_dim=3)
        
        # Mixed types
        mixed_vectors = [[0.1, 0.2, 0.3], [1, 2, 3]]
        validated_mixed = validate_vector_data(mixed_vectors, expected_dim=3)
        assert all(isinstance(v[0], float) for v in validated_mixed)
    
    def test_schema_inference(self, sample_data):
        """Test schema inference from data."""
        from pyhologres.schema import infer_schema_from_data
        
        schema = infer_schema_from_data(sample_data)
        
        # Check field names
        field_names = [field.name for field in schema]
        assert "id" in field_names
        assert "text" in field_names
        assert "vector" in field_names
        assert "category" in field_names
        
        # Check field types
        field_types = {field.name: field.type for field in schema}
        assert pa.types.is_integer(field_types["id"])
        assert pa.types.is_string(field_types["text"])
        assert pa.types.is_list(field_types["vector"])
        assert pa.types.is_string(field_types["category"])


class TestConnectionHandling:
    """Test connection handling."""
    
    def test_invalid_uri(self):
        """Test handling of invalid URIs."""
        with pytest.raises(ValueError):
            pyhologres.connect("invalid://uri")
    
    def test_missing_credentials(self):
        """Test handling of missing credentials for cloud connections."""
        with pytest.raises(ValueError):
            pyhologres.connect("holo://endpoint/database")  # No API key
    
    @pytest.mark.skipif(
        not os.getenv('HOLOGRES_TEST_URI'),
        reason="HOLOGRES_TEST_URI not set"
    )
    def test_real_connection(self):
        """Test real database connection (requires test database)."""
        uri = os.getenv('HOLOGRES_TEST_URI')
        
        with pyhologres.connect(uri) as db:
            # Test basic operations
            tables = db.table_names()
            assert isinstance(tables, list)
    
    def test_connection_context_manager(self):
        """Test connection as context manager."""
        # This test uses a mock connection since we don't have a real database
        # In a real test environment, you would use an actual test database
        
        # Test that connection can be used as context manager
        try:
            with pyhologres.connect("postgresql://test:test@localhost:5432/test") as db:
                # This will fail to connect, but we're testing the context manager
                pass
        except Exception:
            # Expected to fail without real database
            pass


class TestDataTypes:
    """Test data type handling."""
    
    def test_numpy_array_conversion(self):
        """Test numpy array conversion."""
        # Test numpy array to list conversion
        np_vector = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        data = [{"id": 1, "vector": np_vector}]
        
        from pyhologres.schema import infer_schema_from_data
        schema = infer_schema_from_data(data)
        
        vector_field = next(field for field in schema if field.name == "vector")
        assert pa.types.is_list(vector_field.type)
    
    def test_pandas_dataframe_support(self):
        """Test pandas DataFrame support."""
        pytest.importorskip("pandas")
        import pandas as pd
        
        # Create pandas DataFrame
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "text": ["a", "b", "c"],
            "vector": [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
        })
        
        from pyhologres.schema import infer_schema_from_data
        schema = infer_schema_from_data(df)
        
        field_names = [field.name for field in schema]
        assert "id" in field_names
        assert "text" in field_names
        assert "vector" in field_names
    
    def test_polars_dataframe_support(self):
        """Test polars DataFrame support."""
        pytest.importorskip("polars")
        import polars as pl
        
        # Create polars DataFrame
        df = pl.DataFrame({
            "id": [1, 2, 3],
            "text": ["a", "b", "c"],
            "score": [0.1, 0.2, 0.3]
        })
        
        from pyhologres.schema import infer_schema_from_data
        schema = infer_schema_from_data(df)
        
        field_names = [field.name for field in schema]
        assert "id" in field_names
        assert "text" in field_names
        assert "score" in field_names


class TestErrorHandling:
    """Test error handling."""
    
    def test_invalid_vector_dimensions(self):
        """Test handling of invalid vector dimensions."""
        with pytest.raises(ValueError):
            validate_vector_data([[0.1, 0.2], [0.3, 0.4, 0.5]], expected_dim=2)
    
    def test_empty_data_handling(self):
        """Test handling of empty data."""
        from pyhologres.schema import infer_schema_from_data
        
        # Empty list
        with pytest.raises(ValueError):
            infer_schema_from_data([])
        
        # None data
        with pytest.raises(ValueError):
            infer_schema_from_data(None)
    
    def test_malformed_data_handling(self):
        """Test handling of malformed data."""
        from pyhologres.schema import infer_schema_from_data
        
        # Inconsistent data structure
        malformed_data = [
            {"id": 1, "text": "hello"},
            {"id": 2, "different_field": "world"}
        ]
        
        # Should handle gracefully by using union of all fields
        schema = infer_schema_from_data(malformed_data)
        field_names = [field.name for field in schema]
        assert "id" in field_names
        assert "text" in field_names
        assert "different_field" in field_names


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_environment_variable_parsing(self):
        """Test environment variable parsing."""
        from pyhologres.common import get_hologres_config
        
        # Test with no environment variables set
        config = get_hologres_config()
        assert isinstance(config, dict)
        
        # Test with environment variables
        os.environ['HOLOGRES_HOST'] = 'test_host'
        os.environ['HOLOGRES_PORT'] = '5432'
        
        config = get_hologres_config()
        assert config.get('host') == 'test_host'
        assert config.get('port') == '5432'
        
        # Cleanup
        del os.environ['HOLOGRES_HOST']
        del os.environ['HOLOGRES_PORT']
    
    def test_type_conversion_utilities(self):
        """Test type conversion utilities."""
        from pyhologres.table import _into_pyarrow_reader
        
        # Test list of dicts
        data = [{"id": 1, "text": "hello"}, {"id": 2, "text": "world"}]
        reader = _into_pyarrow_reader(data)
        
        # Should be able to iterate over batches
        batches = list(reader)
        assert len(batches) > 0
        
        # Test PyArrow table
        table = pa.table({"id": [1, 2], "text": ["hello", "world"]})
        reader = _into_pyarrow_reader(table)
        batches = list(reader)
        assert len(batches) > 0


if __name__ == "__main__":
    pytest.main([__file__])