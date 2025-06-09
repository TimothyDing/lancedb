# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Hologres Authors

import asyncio
import json
from abc import ABC, abstractmethod
from enum import Enum
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Union,
    TYPE_CHECKING,
)

import numpy as np
import pyarrow as pa
from sqlalchemy import text

from .schema import validate_vector_data

if TYPE_CHECKING:
    import pandas as pd


def ensure_vector_query(query: Union[np.ndarray, List[float], str]) -> np.ndarray:
    """Ensure query is a valid vector."""
    if isinstance(query, str):
        raise ValueError("String queries not supported for vector search")
    return validate_vector_data(query)


class FullTextOperator(str, Enum):
    """Full-text search operators."""
    AND = "&"
    OR = "|"
    NOT = "!"
    PHRASE = '"'


class Occur(str, Enum):
    """Occurrence types for full-text search."""
    MUST = "must"
    SHOULD = "should"
    MUST_NOT = "must_not"


class FullTextQuery(ABC):
    """Abstract base class for full-text queries."""

    def __and__(self, other: "FullTextQuery") -> "BooleanQuery":
        """Combine queries with AND operator."""
        return BooleanQuery([(self, Occur.MUST), (other, Occur.MUST)])

    def __or__(self, other: "FullTextQuery") -> "BooleanQuery":
        """Combine queries with OR operator."""
        return BooleanQuery([(self, Occur.SHOULD), (other, Occur.SHOULD)])

    @abstractmethod
    def to_sql(self) -> str:
        """Convert query to SQL expression."""
        pass


class MatchQuery(FullTextQuery):
    """Match query for full-text search."""

    def __init__(self, text: str, field: Optional[str] = None):
        """Initialize match query.
        
        Parameters
        ----------
        text : str
            Text to search for
        field : str, optional
            Field to search in
        """
        self.text = text
        self.field = field

    def to_sql(self) -> str:
        """Convert to SQL expression."""
        if self.field:
            return f"to_tsvector('english', {self.field}) @@ plainto_tsquery('english', '{self.text}')"
        else:
            return f"to_tsvector('english', *) @@ plainto_tsquery('english', '{self.text}')"


class PhraseQuery(FullTextQuery):
    """Phrase query for exact phrase matching."""

    def __init__(self, phrase: str, field: Optional[str] = None):
        """Initialize phrase query.
        
        Parameters
        ----------
        phrase : str
            Phrase to search for
        field : str, optional
            Field to search in
        """
        self.phrase = phrase
        self.field = field

    def to_sql(self) -> str:
        """Convert to SQL expression."""
        if self.field:
            return f"to_tsvector('english', {self.field}) @@ phraseto_tsquery('english', '{self.phrase}')"
        else:
            return f"to_tsvector('english', *) @@ phraseto_tsquery('english', '{self.phrase}')"


class BooleanQuery(FullTextQuery):
    """Boolean query combining multiple queries."""

    def __init__(self, clauses: List[tuple]):
        """Initialize boolean query.
        
        Parameters
        ----------
        clauses : List[tuple]
            List of (query, occur) tuples
        """
        self.clauses = clauses

    def to_sql(self) -> str:
        """Convert to SQL expression."""
        must_clauses = []
        should_clauses = []
        must_not_clauses = []
        
        for query, occur in self.clauses:
            sql_expr = query.to_sql()
            if occur == Occur.MUST:
                must_clauses.append(sql_expr)
            elif occur == Occur.SHOULD:
                should_clauses.append(sql_expr)
            elif occur == Occur.MUST_NOT:
                must_not_clauses.append(f"NOT ({sql_expr})")
        
        parts = []
        if must_clauses:
            parts.append(" AND ".join(must_clauses))
        if should_clauses:
            parts.append(f"({' OR '.join(should_clauses)})")
        if must_not_clauses:
            parts.extend(must_not_clauses)
        
        return " AND ".join(parts)


class Query(ABC):
    """Abstract base class for database queries."""

    @abstractmethod
    def limit(self, limit: int) -> "Query":
        """Limit the number of results.
        
        Parameters
        ----------
        limit : int
            Maximum number of results to return
            
        Returns
        -------
        Query
            Modified query object
        """
        pass

    @abstractmethod
    def offset(self, offset: int) -> "Query":
        """Skip a number of results.
        
        Parameters
        ----------
        offset : int
            Number of results to skip
            
        Returns
        -------
        Query
            Modified query object
        """
        pass

    @abstractmethod
    def where(self, condition: str, *args) -> "Query":
        """Add a WHERE condition.
        
        Parameters
        ----------
        condition : str
            SQL WHERE condition
        *args
            Parameters for the condition
            
        Returns
        -------
        Query
            Modified query object
        """
        pass

    @abstractmethod
    def select(self, columns: List[str]) -> "Query":
        """Select specific columns.
        
        Parameters
        ----------
        columns : List[str]
            List of column names to select
            
        Returns
        -------
        Query
            Modified query object
        """
        pass

    @abstractmethod
    def to_pandas(self) -> "pd.DataFrame":
        """Execute query and return results as Pandas DataFrame.
        
        Returns
        -------
        pd.DataFrame
            Query results
        """
        pass

    @abstractmethod
    def to_arrow(self) -> pa.Table:
        """Execute query and return results as PyArrow Table.
        
        Returns
        -------
        pa.Table
            Query results
        """
        pass

    @abstractmethod
    def to_list(self) -> List[Dict[str, Any]]:
        """Execute query and return results as list of dictionaries.
        
        Returns
        -------
        List[Dict[str, Any]]
            Query results
        """
        pass


class HologresQuery(Query):
    """Query implementation for Hologres."""

    def __init__(
        self,
        engine,
        table_name: str,
        schema: pa.Schema,
        query: Optional[Union[np.ndarray, List[float], str]] = None,
        vector_column_name: Optional[str] = None,
        query_type: str = "vector",
    ):
        """Initialize Hologres query.
        
        Parameters
        ----------
        engine : sqlalchemy.Engine
            Database engine
        table_name : str
            Name of the table to query
        schema : pa.Schema
            Table schema
        query : np.ndarray, List[float], or str, optional
            Query vector or text
        vector_column_name : str, optional
            Name of vector column for similarity search
        query_type : str, default "vector"
            Type of query: "vector", "hybrid", or "fts"
        """
        self._engine = engine
        self._table_name = table_name
        self._schema = schema
        self._query = query
        self._vector_column_name = vector_column_name
        self._query_type = query_type
        
        # Query building state
        self._limit_value = None
        self._offset_value = None
        self._where_conditions = []
        self._select_columns = None
        self._order_by = None

    def limit(self, limit: int) -> "HologresQuery":
        """Limit the number of results."""
        new_query = self._copy()
        new_query._limit_value = limit
        return new_query

    def offset(self, offset: int) -> "HologresQuery":
        """Skip a number of results."""
        new_query = self._copy()
        new_query._offset_value = offset
        return new_query

    def where(self, condition: str, *args) -> "HologresQuery":
        """Add a WHERE condition."""
        new_query = self._copy()
        new_query._where_conditions.append((condition, args))
        return new_query

    def select(self, columns: List[str]) -> "HologresQuery":
        """Select specific columns."""
        new_query = self._copy()
        new_query._select_columns = columns
        return new_query

    def metric(self, metric: str) -> "HologresQuery":
        """Set distance metric for vector search.
        
        Parameters
        ----------
        metric : str
            Distance metric: "cosine", "l2", "dot"
            
        Returns
        -------
        HologresQuery
            Modified query object
        """
        new_query = self._copy()
        new_query._metric = metric
        return new_query

    def nprobes(self, nprobes: int) -> "HologresQuery":
        """Set number of probes for vector search.
        
        Parameters
        ----------
        nprobes : int
            Number of probes
            
        Returns
        -------
        HologresQuery
            Modified query object
        """
        new_query = self._copy()
        new_query._nprobes = nprobes
        return new_query

    def refine_factor(self, refine_factor: int) -> "HologresQuery":
        """Set refine factor for vector search.
        
        Parameters
        ----------
        refine_factor : int
            Refine factor
            
        Returns
        -------
        HologresQuery
            Modified query object
        """
        new_query = self._copy()
        new_query._refine_factor = refine_factor
        return new_query

    def _copy(self) -> "HologresQuery":
        """Create a copy of the query."""
        new_query = HologresQuery(
            self._engine,
            self._table_name,
            self._schema,
            self._query,
            self._vector_column_name,
            self._query_type,
        )
        new_query._limit_value = self._limit_value
        new_query._offset_value = self._offset_value
        new_query._where_conditions = self._where_conditions.copy()
        new_query._select_columns = self._select_columns
        new_query._order_by = self._order_by
        return new_query

    def _build_sql(self) -> str:
        """Build SQL query string."""
        # Select clause
        if self._select_columns:
            select_clause = ", ".join(self._select_columns)
        else:
            select_clause = "*"
        
        # Add distance calculation for vector search
        if self._query_type == "vector" and self._query is not None and self._vector_column_name:
            query_vector = ensure_vector_query(self._query)
            vector_str = "[" + ",".join(map(str, query_vector)) + "]"
            
            # Use array distance functions (simplified for PostgreSQL)
            distance_expr = f"array_distance({self._vector_column_name}, '{vector_str}'::real[]) AS _distance"
            if self._select_columns:
                select_clause += f", {distance_expr}"
            else:
                select_clause = f"*, {distance_expr}"
            
            # Order by distance
            self._order_by = "_distance ASC"
        
        query = f"SELECT {select_clause} FROM {self._table_name}"
        
        # WHERE clause
        where_parts = []
        
        # Add vector search conditions
        if self._query_type == "vector" and self._query is not None and self._vector_column_name:
            # For now, we'll use a simple array comparison
            # In a real implementation, you'd use proper vector similarity functions
            pass  # Distance calculation is handled in SELECT clause
        
        # Add user-defined WHERE conditions
        for condition, args in self._where_conditions:
            where_parts.append(condition)
        
        if where_parts:
            query += " WHERE " + " AND ".join(where_parts)
        
        # ORDER BY clause
        if self._order_by:
            query += f" ORDER BY {self._order_by}"
        
        # LIMIT clause
        if self._limit_value:
            query += f" LIMIT {self._limit_value}"
        
        # OFFSET clause
        if self._offset_value:
            query += f" OFFSET {self._offset_value}"
        
        return query

    def to_pandas(self) -> "pd.DataFrame":
        """Execute query and return results as Pandas DataFrame."""
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required for to_pandas()")
        
        sql = self._build_sql()
        with self._engine.connect() as conn:
            return pd.read_sql(text(sql), conn)

    def to_arrow(self) -> pa.Table:
        """Execute query and return results as PyArrow Table."""
        df = self.to_pandas()
        return pa.Table.from_pandas(df)

    def to_list(self) -> List[Dict[str, Any]]:
        """Execute query and return results as list of dictionaries."""
        df = self.to_pandas()
        return df.to_dict("records")


class AsyncQuery:
    """Async wrapper for Query operations."""

    def __init__(self, query: Query):
        self._query = query

    async def limit(self, limit: int) -> "AsyncQuery":
        loop = asyncio.get_event_loop()
        new_query = await loop.run_in_executor(None, self._query.limit, limit)
        return AsyncQuery(new_query)

    async def offset(self, offset: int) -> "AsyncQuery":
        loop = asyncio.get_event_loop()
        new_query = await loop.run_in_executor(None, self._query.offset, offset)
        return AsyncQuery(new_query)

    async def where(self, condition: str, *args) -> "AsyncQuery":
        loop = asyncio.get_event_loop()
        new_query = await loop.run_in_executor(None, self._query.where, condition, *args)
        return AsyncQuery(new_query)

    async def select(self, columns: List[str]) -> "AsyncQuery":
        loop = asyncio.get_event_loop()
        new_query = await loop.run_in_executor(None, self._query.select, columns)
        return AsyncQuery(new_query)

    async def to_pandas(self) -> "pd.DataFrame":
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._query.to_pandas)

    async def to_arrow(self) -> pa.Table:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._query.to_arrow)

    async def to_list(self) -> List[Dict[str, Any]]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._query.to_list)