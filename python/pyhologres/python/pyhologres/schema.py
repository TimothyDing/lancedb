# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Hologres Authors

import pyarrow as pa
from typing import List, Union, Optional, Any
import numpy as np


def vector(dim: int, value_type: pa.DataType = pa.float32()) -> pa.FixedSizeListType:
    """Create a vector type for Hologres.
    
    Parameters
    ----------
    dim : int
        The dimension of the vector
    value_type : pa.DataType, default pa.float32()
        The data type of vector elements
        
    Returns
    -------
    pa.FixedSizeListType
        A PyArrow FixedSizeListType representing the vector
        
    Examples
    --------
    >>> import pyhologres
    >>> import pyarrow as pa
    >>> 
    >>> # Create a 128-dimensional float32 vector type
    >>> vec_type = pyhologres.vector(128)
    >>> 
    >>> # Create a 256-dimensional float64 vector type
    >>> vec_type = pyhologres.vector(256, pa.float64())
    """
    return pa.list_(value_type, dim)


def create_vector_column(name: str, dim: int, value_type: pa.DataType = pa.float32()) -> pa.Field:
    """Create a vector column field for Hologres schema.
    
    Parameters
    ----------
    name : str
        The name of the vector column
    dim : int
        The dimension of the vector
    value_type : pa.DataType, default pa.float32()
        The data type of vector elements
        
    Returns
    -------
    pa.Field
        A PyArrow Field representing the vector column
    """
    return pa.field(name, vector(dim, value_type))


def validate_vector_data(data: Any, expected_dim: Optional[int] = None) -> np.ndarray:
    """Validate and convert vector data to numpy array.
    
    Parameters
    ----------
    data : Any
        The vector data to validate
    expected_dim : int, optional
        The expected dimension of the vector
        
    Returns
    -------
    np.ndarray
        The validated vector data as numpy array
        
    Raises
    ------
    ValueError
        If the vector data is invalid
    """
    if isinstance(data, (list, tuple)):
        data = np.array(data, dtype=np.float32)
    elif isinstance(data, np.ndarray):
        if data.dtype not in [np.float32, np.float64]:
            data = data.astype(np.float32)
    else:
        raise ValueError(f"Invalid vector data type: {type(data)}")
    
    if data.ndim != 1:
        raise ValueError(f"Vector must be 1-dimensional, got {data.ndim}")
    
    if expected_dim is not None and len(data) != expected_dim:
        raise ValueError(f"Vector dimension mismatch: expected {expected_dim}, got {len(data)}")
    
    return data


def create_hologres_schema(
    columns: List[Union[pa.Field, tuple]], 
    vector_columns: Optional[List[tuple]] = None
) -> pa.Schema:
    """Create a PyArrow schema for Hologres table.
    
    Parameters
    ----------
    columns : List[Union[pa.Field, tuple]]
        List of column definitions. Each item can be:
        - pa.Field object
        - tuple of (name, type) or (name, type, nullable)
    vector_columns : List[tuple], optional
        List of vector column definitions. Each tuple should be (name, dimension, value_type)
        
    Returns
    -------
    pa.Schema
        The PyArrow schema for the table
        
    Examples
    --------
    >>> import pyhologres
    >>> import pyarrow as pa
    >>> 
    >>> # Create schema with regular and vector columns
    >>> schema = pyhologres.create_hologres_schema(
    ...     columns=[
    ...         ("id", pa.int64()),
    ...         ("text", pa.string()),
    ...         ("score", pa.float64())
    ...     ],
    ...     vector_columns=[
    ...         ("embedding", 128, pa.float32())
    ...     ]
    ... )
    """
    fields = []
    
    # Process regular columns
    for col in columns:
        if isinstance(col, pa.Field):
            fields.append(col)
        elif isinstance(col, tuple):
            if len(col) == 2:
                name, dtype = col
                fields.append(pa.field(name, dtype))
            elif len(col) == 3:
                name, dtype, nullable = col
                fields.append(pa.field(name, dtype, nullable=nullable))
            else:
                raise ValueError(f"Invalid column definition: {col}")
        else:
            raise ValueError(f"Invalid column type: {type(col)}")
    
    # Process vector columns
    if vector_columns:
        for vec_col in vector_columns:
            if len(vec_col) == 2:
                name, dim = vec_col
                value_type = pa.float32()
            elif len(vec_col) == 3:
                name, dim, value_type = vec_col
            else:
                raise ValueError(f"Invalid vector column definition: {vec_col}")
            
            fields.append(create_vector_column(name, dim, value_type))
    
    return pa.schema(fields)


def infer_schema_from_data(data: Union[pa.Table, pa.RecordBatch, dict, list]) -> pa.Schema:
    """Infer PyArrow schema from data.
    
    Parameters
    ----------
    data : Union[pa.Table, pa.RecordBatch, dict, list]
        The data to infer schema from
        
    Returns
    -------
    pa.Schema
        The inferred schema
    """
    if isinstance(data, (pa.Table, pa.RecordBatch)):
        return data.schema
    elif isinstance(data, dict):
        # Convert dict to PyArrow table to infer schema
        table = pa.table(data)
        return table.schema
    elif isinstance(data, list) and len(data) > 0:
        # Convert list of dicts to PyArrow table
        if isinstance(data[0], dict):
            table = pa.table(data)
            return table.schema
        else:
            raise ValueError("List data must contain dictionaries")
    else:
        raise ValueError(f"Cannot infer schema from data type: {type(data)}")