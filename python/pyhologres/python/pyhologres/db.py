# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Hologres Authors

import asyncio
import warnings
from abc import ABC, abstractmethod
from datetime import timedelta
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Union,
    TYPE_CHECKING,
)

import pyarrow as pa
import psycopg2
import sqlalchemy as sa
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from .common import URI, sanitize_uri, parse_hologres_uri
from .schema import infer_schema_from_data
from .table import Table, HologresTable

if TYPE_CHECKING:
    import pandas as pd
    import polars as pl


class DBConnection(ABC):
    """A connection to a Hologres database."""

    @abstractmethod
    def table_names(self, limit: Optional[int] = None) -> List[str]:
        """List all table names in the database.
        
        Parameters
        ----------
        limit : int, optional
            Maximum number of table names to return
            
        Returns
        -------
        List[str]
            List of table names
        """
        pass

    @abstractmethod
    def create_table(
        self,
        name: str,
        data: Optional[
            Union[
                pa.Table,
                pa.RecordBatch,
                Iterable[pa.RecordBatch],
                "pd.DataFrame",
                "pl.DataFrame",
                "pl.LazyFrame",
                List[dict],
                Dict[str, List],
            ]
        ] = None,
        *,
        schema: Optional[Union[pa.Schema, "pd.DataFrame"]] = None,
        mode: str = "create",
        exist_ok: bool = False,
        on_bad_vectors: str = "error",
        fill_value: float = 0.0,
        storage_options: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> Table:
        """Create a table in the database.
        
        Parameters
        ----------
        name : str
            The name of the table
        data : various types, optional
            The data to insert into the table. Can be:
            - PyArrow Table or RecordBatch
            - Pandas DataFrame
            - Polars DataFrame or LazyFrame
            - List of dictionaries
            - Dictionary of lists
        schema : pa.Schema or pd.DataFrame, optional
            The schema of the table
        mode : str, default "create"
            The mode for creating the table:
            - "create": Create a new table, error if exists
            - "overwrite": Overwrite existing table
            - "append": Append to existing table
        exist_ok : bool, default False
            If True, do not raise an error if the table already exists
        on_bad_vectors : str, default "error"
            How to handle bad vectors:
            - "error": Raise an error
            - "drop": Drop bad vectors
            - "fill": Fill with fill_value
        fill_value : float, default 0.0
            Value to use when filling bad vectors
        storage_options : dict, optional
            Additional storage options
            
        Returns
        -------
        Table
            The created table
        """
        pass

    def __getitem__(self, name: str) -> Table:
        """Get a table by name.
        
        Parameters
        ----------
        name : str
            The name of the table
            
        Returns
        -------
        Table
            The table
        """
        return self.open_table(name)

    @abstractmethod
    def open_table(self, name: str) -> Table:
        """Open an existing table.
        
        Parameters
        ----------
        name : str
            The name of the table
            
        Returns
        -------
        Table
            The opened table
        """
        pass

    @abstractmethod
    def drop_table(self, name: str) -> None:
        """Drop a table.
        
        Parameters
        ----------
        name : str
            The name of the table to drop
        """
        pass

    @abstractmethod
    def rename_table(self, old_name: str, new_name: str) -> None:
        """Rename a table.
        
        Parameters
        ----------
        old_name : str
            The current name of the table
        new_name : str
            The new name for the table
        """
        pass

    def drop_database(self) -> None:
        """Drop the database.
        
        This will delete all tables and data in the database.
        Use with caution!
        """
        warnings.warn(
            "drop_database is not supported for Hologres. "
            "Please use Hologres console to manage databases.",
            UserWarning,
        )


class HologresDBConnection(DBConnection):
    """A connection to a Hologres database using PostgreSQL protocol."""

    def __init__(
        self,
        uri: URI,
        *,
        username: Optional[str] = None,
        password: Optional[str] = None,
        database: Optional[str] = None,
        host: Optional[str] = None,
        port: int = 80,
        read_consistency_interval: Optional[timedelta] = None,
        storage_options: Optional[Dict[str, str]] = None,
    ):
        """Initialize Hologres database connection.
        
        Parameters
        ----------
        uri : str or Path
            The connection URI
        username : str, optional
            Username for authentication
        password : str, optional
            Password for authentication
        database : str, optional
            Database name
        host : str, optional
            Host address
        port : int, default 80
            Port number
        read_consistency_interval : timedelta, optional
            Read consistency interval
        storage_options : dict, optional
            Additional storage options
        """
        self._uri = sanitize_uri(uri)
        self._username = username
        self._password = password
        self._database = database
        self._host = host
        self._port = port
        self._read_consistency_interval = read_consistency_interval
        self._storage_options = storage_options or {}
        
        # Parse connection string
        if self._uri.startswith(("postgresql://", "postgres://")):
            parsed = parse_hologres_uri(self._uri)
            self._username = self._username or parsed.get("username")
            self._password = self._password or parsed.get("password")
            self._host = self._host or parsed.get("hostname")
            self._port = self._port or parsed.get("port", 80)
            self._database = self._database or parsed.get("database")
        
        # Create SQLAlchemy engine
        connection_string = self._build_connection_string()
        self._engine = create_engine(connection_string)
        self._session_factory = sessionmaker(bind=self._engine)

    def _build_connection_string(self) -> str:
        """Build PostgreSQL connection string for Hologres."""
        if self._uri.startswith(("postgresql://", "postgres://")):
            return self._uri
        
        # Build connection string from components
        if not all([self._username, self._password, self._host, self._database]):
            raise ValueError(
                "username, password, host, and database are required for Hologres connection"
            )
        
        return (
            f"postgresql://{self._username}:{self._password}@"
            f"{self._host}:{self._port}/{self._database}"
        )

    def table_names(self, limit: Optional[int] = None) -> List[str]:
        """List all table names in the database."""
        with self._engine.connect() as conn:
            query = text(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'public' AND table_type = 'BASE TABLE'"
            )
            if limit:
                query = text(str(query) + f" LIMIT {limit}")
            
            result = conn.execute(query)
            return [row[0] for row in result]

    def create_table(
        self,
        name: str,
        data: Optional[
            Union[
                pa.Table,
                pa.RecordBatch,
                Iterable[pa.RecordBatch],
                "pd.DataFrame",
                "pl.DataFrame",
                "pl.LazyFrame",
                List[dict],
                Dict[str, List],
            ]
        ] = None,
        *,
        schema: Optional[Union[pa.Schema, "pd.DataFrame"]] = None,
        mode: str = "create",
        exist_ok: bool = False,
        on_bad_vectors: str = "error",
        fill_value: float = 0.0,
        storage_options: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> Table:
        """Create a table in Hologres."""
        # Check if table exists
        table_exists = name in self.table_names()
        
        if table_exists:
            if mode == "create" and not exist_ok:
                raise ValueError(f"Table {name} already exists")
            elif mode == "overwrite":
                self.drop_table(name)
                table_exists = False
            elif mode == "append":
                # Return existing table for append
                table = self.open_table(name)
                if data is not None:
                    table.add(data)
                return table
        
        # Infer schema if not provided
        if schema is None and data is not None:
            schema = infer_schema_from_data(data)
        elif schema is None:
            raise ValueError("Either schema or data must be provided")
        
        # Create new table
        table = HologresTable(
            self._engine,
            name,
            schema,
            storage_options=storage_options or self._storage_options,
        )
        
        # Create table in database
        table._create_table()
        
        # Insert data if provided
        if data is not None:
            table.add(data, mode="append")
        
        return table

    def open_table(self, name: str) -> Table:
        """Open an existing table."""
        if name not in self.table_names():
            raise ValueError(f"Table {name} does not exist")
        
        # Get table schema from database
        with self._engine.connect() as conn:
            query = text(
                "SELECT column_name, data_type FROM information_schema.columns "
                "WHERE table_name = :table_name AND table_schema = 'public'"
            )
            result = conn.execute(query, {"table_name": name})
            columns = list(result)
        
        # Build PyArrow schema from database schema
        fields = []
        for col_name, col_type in columns:
            # Map PostgreSQL types to PyArrow types
            if col_type.startswith("character varying") or col_type == "text":
                pa_type = pa.string()
            elif col_type.startswith("integer") or col_type == "bigint":
                pa_type = pa.int64()
            elif col_type.startswith("double precision") or col_type == "real":
                pa_type = pa.float64()
            elif col_type == "boolean":
                pa_type = pa.bool_()
            elif col_type.startswith("timestamp"):
                pa_type = pa.timestamp("us")
            elif col_type.startswith("real[]"):  # Vector type
                # Extract dimension from array type
                pa_type = pa.list_(pa.float32())
            else:
                # Default to string for unknown types
                pa_type = pa.string()
            
            fields.append(pa.field(col_name, pa_type))
        
        schema = pa.schema(fields)
        
        return HologresTable(
            self._engine,
            name,
            schema,
            storage_options=self._storage_options,
        )

    def drop_table(self, name: str) -> None:
        """Drop a table."""
        with self._engine.connect() as conn:
            conn.execute(text(f"DROP TABLE IF EXISTS {name}"))
            conn.commit()

    def rename_table(self, old_name: str, new_name: str) -> None:
        """Rename a table."""
        with self._engine.connect() as conn:
            conn.execute(text(f"ALTER TABLE {old_name} RENAME TO {new_name}"))
            conn.commit()


class AsyncConnection:
    """Async connection to Hologres database."""

    def __init__(self, inner):
        self._inner = inner

    async def table_names(self, limit: Optional[int] = None) -> List[str]:
        """List all table names in the database asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._inner.table_names, limit)

    async def create_table(
        self,
        name: str,
        data: Optional[Any] = None,
        **kwargs: Any,
    ) -> "AsyncTable":
        """Create a table asynchronously."""
        from .table import AsyncTable
        
        loop = asyncio.get_event_loop()
        table = await loop.run_in_executor(
            None, self._inner.create_table, name, data, **kwargs
        )
        return AsyncTable(table)

    async def open_table(self, name: str) -> "AsyncTable":
        """Open a table asynchronously."""
        from .table import AsyncTable
        
        loop = asyncio.get_event_loop()
        table = await loop.run_in_executor(None, self._inner.open_table, name)
        return AsyncTable(table)

    async def drop_table(self, name: str) -> None:
        """Drop a table asynchronously."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._inner.drop_table, name)

    async def rename_table(self, old_name: str, new_name: str) -> None:
        """Rename a table asynchronously."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._inner.rename_table, old_name, new_name)