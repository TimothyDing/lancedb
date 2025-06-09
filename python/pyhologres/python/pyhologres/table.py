# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Hologres Authors

import asyncio
import json
import warnings
from abc import ABC, abstractmethod
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Union,
    TYPE_CHECKING,
)

import numpy as np
import pyarrow as pa
from sqlalchemy import create_engine, text, MetaData, Table as SQLTable, Column, Integer, String, Float, Boolean, DateTime
from sqlalchemy.dialects.postgresql import ARRAY, REAL
from sqlalchemy.orm import sessionmaker

from .query import Query, HologresQuery
from .schema import validate_vector_data

if TYPE_CHECKING:
    import pandas as pd
    import polars as pl


def _into_pyarrow_reader(data: Any) -> pa.RecordBatchReader:
    """Convert various data types to PyArrow RecordBatchReader."""
    if isinstance(data, pa.RecordBatchReader):
        return data
    elif isinstance(data, pa.Table):
        return data.to_reader()
    elif isinstance(data, pa.RecordBatch):
        return pa.Table.from_batches([data]).to_reader()
    elif isinstance(data, (list, tuple)) and len(data) > 0:
        if isinstance(data[0], pa.RecordBatch):
            return pa.Table.from_batches(data).to_reader()
        elif isinstance(data[0], dict):
            # List of dictionaries
            table = pa.table(data)
            return table.to_reader()
        else:
            raise ValueError(f"Unsupported list element type: {type(data[0])}")
    elif isinstance(data, dict):
        # Dictionary of lists
        table = pa.table(data)
        return table.to_reader()
    else:
        # Try to import pandas/polars and convert
        try:
            import pandas as pd
            if isinstance(data, pd.DataFrame):
                table = pa.Table.from_pandas(data)
                return table.to_reader()
        except ImportError:
            pass
        
        try:
            import polars as pl
            if isinstance(data, (pl.DataFrame, pl.LazyFrame)):
                if isinstance(data, pl.LazyFrame):
                    data = data.collect()
                table = data.to_arrow()
                return table.to_reader()
        except ImportError:
            pass
        
        raise ValueError(f"Cannot convert data of type {type(data)} to PyArrow RecordBatchReader")


class Table(ABC):
    """A table in a Hologres database."""

    @property
    @abstractmethod
    def name(self) -> str:
        """The name of the table."""
        pass

    @property
    @abstractmethod
    def schema(self) -> pa.Schema:
        """The schema of the table."""
        pass

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of rows in the table."""
        pass

    @abstractmethod
    def count_rows(self, filter: Optional[str] = None) -> int:
        """Count the number of rows in the table.
        
        Parameters
        ----------
        filter : str, optional
            SQL WHERE clause to filter rows
            
        Returns
        -------
        int
            Number of rows
        """
        pass

    @abstractmethod
    def to_pandas(self, **kwargs) -> "pd.DataFrame":
        """Convert the table to a Pandas DataFrame.
        
        Returns
        -------
        pd.DataFrame
            The table as a Pandas DataFrame
        """
        pass

    @abstractmethod
    def to_arrow(self) -> pa.Table:
        """Convert the table to a PyArrow Table.
        
        Returns
        -------
        pa.Table
            The table as a PyArrow Table
        """
        pass

    @abstractmethod
    def add(
        self,
        data: Union[
            pa.Table,
            pa.RecordBatch,
            Iterable[pa.RecordBatch],
            "pd.DataFrame",
            "pl.DataFrame",
            "pl.LazyFrame",
            List[dict],
            Dict[str, List],
        ],
        mode: str = "append",
        on_bad_vectors: str = "error",
        fill_value: float = 0.0,
    ) -> None:
        """Add data to the table.
        
        Parameters
        ----------
        data : various types
            The data to add to the table
        mode : str, default "append"
            The mode for adding data:
            - "append": Append to existing data
            - "overwrite": Overwrite existing data
        on_bad_vectors : str, default "error"
            How to handle bad vectors:
            - "error": Raise an error
            - "drop": Drop bad vectors
            - "fill": Fill with fill_value
        fill_value : float, default 0.0
            Value to use when filling bad vectors
        """
        pass

    @abstractmethod
    def search(
        self,
        query: Optional[Union[np.ndarray, List[float], str]] = None,
        vector_column_name: Optional[str] = None,
        query_type: str = "vector",
    ) -> Query:
        """Search the table.
        
        Parameters
        ----------
        query : np.ndarray, List[float], or str, optional
            The query vector or text
        vector_column_name : str, optional
            Name of the vector column to search
        query_type : str, default "vector"
            Type of query: "vector", "hybrid", or "fts"
            
        Returns
        -------
        Query
            A query object for further refinement
        """
        pass

    @abstractmethod
    def create_index(
        self,
        column: str,
        *,
        config: Optional[Dict[str, Any]] = None,
        replace: bool = False,
    ) -> None:
        """Create an index on a column.
        
        Parameters
        ----------
        column : str
            The column to create an index on
        config : dict, optional
            Index configuration
        replace : bool, default False
            Whether to replace existing index
        """
        pass

    @abstractmethod
    def create_fts_index(
        self,
        field_names: Union[str, List[str]],
        *,
        config: Optional[Dict[str, Any]] = None,
        replace: bool = False,
    ) -> None:
        """Create a full-text search index.
        
        Parameters
        ----------
        field_names : str or List[str]
            The field(s) to create FTS index on
        config : dict, optional
            FTS index configuration
        replace : bool, default False
            Whether to replace existing index
        """
        pass

    @abstractmethod
    def update(
        self,
        where: Optional[str] = None,
        values: Optional[Dict[str, Any]] = None,
        values_sql: Optional[Dict[str, str]] = None,
    ) -> None:
        """Update rows in the table.
        
        Parameters
        ----------
        where : str, optional
            SQL WHERE clause to filter rows to update
        values : dict, optional
            Dictionary of column names to new values
        values_sql : dict, optional
            Dictionary of column names to SQL expressions
        """
        pass

    @abstractmethod
    def delete(self, where: str) -> None:
        """Delete rows from the table.
        
        Parameters
        ----------
        where : str
            SQL WHERE clause to filter rows to delete
        """
        pass


class HologresTable(Table):
    """A table in Hologres database."""

    def __init__(
        self,
        engine,
        name: str,
        schema: pa.Schema,
        storage_options: Optional[Dict[str, str]] = None,
    ):
        """Initialize Hologres table.
        
        Parameters
        ----------
        engine : sqlalchemy.Engine
            SQLAlchemy engine for database connection
        name : str
            Table name
        schema : pa.Schema
            PyArrow schema of the table
        storage_options : dict, optional
            Additional storage options
        """
        self._engine = engine
        self._name = name
        self._schema = schema
        self._storage_options = storage_options or {}
        self._session_factory = sessionmaker(bind=engine)

    @property
    def name(self) -> str:
        """The name of the table."""
        return self._name

    @property
    def schema(self) -> pa.Schema:
        """The schema of the table."""
        return self._schema

    def __len__(self) -> int:
        """Return the number of rows in the table."""
        return self.count_rows()

    def count_rows(self, filter: Optional[str] = None) -> int:
        """Count the number of rows in the table."""
        with self._engine.connect() as conn:
            query = f"SELECT COUNT(*) FROM {self._name}"
            if filter:
                query += f" WHERE {filter}"
            result = conn.execute(text(query))
            return result.scalar()

    def to_pandas(self, **kwargs) -> "pd.DataFrame":
        """Convert the table to a Pandas DataFrame."""
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required for to_pandas()")
        
        with self._engine.connect() as conn:
            query = f"SELECT * FROM {self._name}"
            return pd.read_sql(query, conn, **kwargs)

    def to_arrow(self) -> pa.Table:
        """Convert the table to a PyArrow Table."""
        df = self.to_pandas()
        return pa.Table.from_pandas(df, schema=self._schema)

    def _create_table(self) -> None:
        """Create the table in the database."""
        metadata = MetaData()
        columns = []
        
        for field in self._schema:
            col_name = field.name
            col_type = field.type
            
            # Map PyArrow types to SQLAlchemy types
            if pa.types.is_string(col_type):
                sql_type = String
            elif pa.types.is_integer(col_type):
                sql_type = Integer
            elif pa.types.is_floating(col_type):
                sql_type = Float
            elif pa.types.is_boolean(col_type):
                sql_type = Boolean
            elif pa.types.is_timestamp(col_type):
                sql_type = DateTime
            elif pa.types.is_list(col_type):
                # Vector column - use PostgreSQL array type
                sql_type = ARRAY(REAL)
            else:
                # Default to string for unknown types
                sql_type = String
            
            columns.append(Column(col_name, sql_type))
        
        table = SQLTable(self._name, metadata, *columns)
        metadata.create_all(self._engine)

    def add(
        self,
        data: Union[
            pa.Table,
            pa.RecordBatch,
            Iterable[pa.RecordBatch],
            "pd.DataFrame",
            "pl.DataFrame",
            "pl.LazyFrame",
            List[dict],
            Dict[str, List],
        ],
        mode: str = "append",
        on_bad_vectors: str = "error",
        fill_value: float = 0.0,
    ) -> None:
        """Add data to the table."""
        if mode == "overwrite":
            # Clear existing data
            with self._engine.connect() as conn:
                conn.execute(text(f"DELETE FROM {self._name}"))
                conn.commit()
        
        # Convert data to PyArrow format
        reader = _into_pyarrow_reader(data)
        
        # Process data in batches
        for batch in reader:
            self._insert_batch(batch, on_bad_vectors, fill_value)

    def _insert_batch(
        self,
        batch: pa.RecordBatch,
        on_bad_vectors: str = "error",
        fill_value: float = 0.0,
    ) -> None:
        """Insert a batch of data into the table."""
        # Convert batch to pandas for easier manipulation
        df = batch.to_pandas()
        
        # Validate and process vector columns
        for field in self._schema:
            if pa.types.is_list(field.type) and field.name in df.columns:
                # This is a vector column
                vectors = []
                for i, vec in enumerate(df[field.name]):
                    try:
                        validated_vec = validate_vector_data(vec)
                        vectors.append(validated_vec.tolist())
                    except ValueError as e:
                        if on_bad_vectors == "error":
                            raise e
                        elif on_bad_vectors == "drop":
                            # Mark row for deletion
                            df = df.drop(index=i)
                            continue
                        elif on_bad_vectors == "fill":
                            # Fill with default value
                            dim = len(vec) if hasattr(vec, '__len__') else 128  # Default dimension
                            vectors.append([fill_value] * dim)
                        else:
                            raise ValueError(f"Invalid on_bad_vectors option: {on_bad_vectors}")
                
                df[field.name] = vectors
        
        # Insert data using pandas to_sql
        df.to_sql(
            self._name,
            self._engine,
            if_exists="append",
            index=False,
            method="multi",
        )

    def search(
        self,
        query: Optional[Union[np.ndarray, List[float], str]] = None,
        vector_column_name: Optional[str] = None,
        query_type: str = "vector",
    ) -> Query:
        """Search the table."""
        return HologresQuery(
            self._engine,
            self._name,
            self._schema,
            query=query,
            vector_column_name=vector_column_name,
            query_type=query_type,
        )

    def create_index(
        self,
        column: str,
        *,
        config: Optional[Dict[str, Any]] = None,
        replace: bool = False,
    ) -> None:
        """Create an index on a column."""
        index_name = f"idx_{self._name}_{column}"
        
        with self._engine.connect() as conn:
            if replace:
                conn.execute(text(f"DROP INDEX IF EXISTS {index_name}"))
            
            # Check if this is a vector column
            field = next((f for f in self._schema if f.name == column), None)
            if field and pa.types.is_list(field.type):
                # Create vector index (using GIN index for arrays in PostgreSQL)
                conn.execute(text(f"CREATE INDEX {index_name} ON {self._name} USING GIN ({column})"))
            else:
                # Create regular B-tree index
                conn.execute(text(f"CREATE INDEX {index_name} ON {self._name} ({column})"))
            
            conn.commit()

    def create_fts_index(
        self,
        field_names: Union[str, List[str]],
        *,
        config: Optional[Dict[str, Any]] = None,
        replace: bool = False,
    ) -> None:
        """Create a full-text search index."""
        if isinstance(field_names, str):
            field_names = [field_names]
        
        index_name = f"fts_idx_{self._name}_{'_'.join(field_names)}"
        
        with self._engine.connect() as conn:
            if replace:
                conn.execute(text(f"DROP INDEX IF EXISTS {index_name}"))
            
            # Create GIN index for full-text search
            columns_expr = " || ' ' || ".join(f"COALESCE({col}, '')" for col in field_names)
            conn.execute(text(
                f"CREATE INDEX {index_name} ON {self._name} "
                f"USING GIN (to_tsvector('english', {columns_expr}))"
            ))
            conn.commit()

    def update(
        self,
        where: Optional[str] = None,
        values: Optional[Dict[str, Any]] = None,
        values_sql: Optional[Dict[str, str]] = None,
    ) -> None:
        """Update rows in the table."""
        if not values and not values_sql:
            raise ValueError("Either values or values_sql must be provided")
        
        set_clauses = []
        params = {}
        
        if values:
            for col, val in values.items():
                set_clauses.append(f"{col} = :{col}")
                params[col] = val
        
        if values_sql:
            for col, expr in values_sql.items():
                set_clauses.append(f"{col} = {expr}")
        
        query = f"UPDATE {self._name} SET {', '.join(set_clauses)}"
        if where:
            query += f" WHERE {where}"
        
        with self._engine.connect() as conn:
            conn.execute(text(query), params)
            conn.commit()

    def delete(self, where: str) -> None:
        """Delete rows from the table."""
        query = f"DELETE FROM {self._name} WHERE {where}"
        
        with self._engine.connect() as conn:
            conn.execute(text(query))
            conn.commit()


class AsyncTable:
    """Async wrapper for Table operations."""

    def __init__(self, table: Table):
        self._table = table

    @property
    def name(self) -> str:
        return self._table.name

    @property
    def schema(self) -> pa.Schema:
        return self._table.schema

    async def count_rows(self, filter: Optional[str] = None) -> int:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._table.count_rows, filter)

    async def to_pandas(self, **kwargs) -> "pd.DataFrame":
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._table.to_pandas, **kwargs)

    async def to_arrow(self) -> pa.Table:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._table.to_arrow)

    async def add(self, data: Any, **kwargs) -> None:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._table.add, data, **kwargs)

    async def search(self, query: Any = None, **kwargs) -> "AsyncQuery":
        from .query import AsyncQuery
        
        loop = asyncio.get_event_loop()
        sync_query = await loop.run_in_executor(None, self._table.search, query, **kwargs)
        return AsyncQuery(sync_query)

    async def create_index(self, column: str, **kwargs) -> None:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._table.create_index, column, **kwargs)

    async def create_fts_index(self, field_names: Union[str, List[str]], **kwargs) -> None:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._table.create_fts_index, field_names, **kwargs)

    async def update(self, **kwargs) -> None:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._table.update, **kwargs)

    async def delete(self, where: str) -> None:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._table.delete, where)