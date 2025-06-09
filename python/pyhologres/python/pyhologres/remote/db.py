# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Hologres Authors

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Iterable, List, Optional, Union, TYPE_CHECKING

import pyarrow as pa

from ..common import URI, parse_hologres_uri
from ..db import DBConnection
from ..schema import infer_schema_from_data
from ..table import Table
from .client import ClientConfig, HologresCloudClient
from .table import RemoteTable

if TYPE_CHECKING:
    import pandas as pd
    import polars as pl


class RemoteDBConnection(DBConnection):
    """Connection to a remote Hologres Cloud database."""

    def __init__(
        self,
        uri: URI,
        api_key: str,
        region: str = "cn-hangzhou",
        host_override: Optional[str] = None,
        *,
        username: Optional[str] = None,
        password: Optional[str] = None,
        database: Optional[str] = None,
        host: Optional[str] = None,
        port: int = 80,
        request_thread_pool: Optional[ThreadPoolExecutor] = None,
        client_config: Optional[Union[ClientConfig, Dict[str, Any]]] = None,
        storage_options: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ):
        """Initialize remote Hologres database connection.
        
        Parameters
        ----------
        uri : str or Path
            The Hologres Cloud URI
        api_key : str
            API key for authentication
        region : str, default "cn-hangzhou"
            Hologres Cloud region
        host_override : str, optional
            Override host URL
        username : str, optional
            Username for database connection
        password : str, optional
            Password for database connection
        database : str, optional
            Database name
        host : str, optional
            Database host
        port : int, default 80
            Database port
        request_thread_pool : ThreadPoolExecutor, optional
            Thread pool for async operations
        client_config : ClientConfig or dict, optional
            HTTP client configuration
        storage_options : dict, optional
            Additional storage options
        **kwargs
            Additional connection options
        """
        self._uri = str(uri)
        self._api_key = api_key
        self._region = region
        self._host_override = host_override
        self._username = username
        self._password = password
        self._database = database
        self._host = host
        self._port = port
        self._storage_options = storage_options or {}
        
        # Parse URI to extract database info
        if self._uri.startswith("holo://"):
            parsed = parse_hologres_uri(self._uri)
            self._endpoint = parsed.get("endpoint")
            # Extract database from path if not provided
            if not self._database and parsed.get("path"):
                self._database = parsed["path"].strip("/")
        else:
            raise ValueError(f"Invalid Hologres Cloud URI: {self._uri}")
        
        # Setup client configuration
        if isinstance(client_config, dict):
            client_config = ClientConfig(**client_config)
        elif client_config is None:
            client_config = ClientConfig()
        
        # Create HTTP client
        self._client = HologresCloudClient(
            api_key=self._api_key,
            region=self._region,
            host_override=self._host_override,
            config=client_config,
        )
        
        # Setup thread pool for async operations
        if request_thread_pool is None:
            self._thread_pool = ThreadPoolExecutor(max_workers=4)
            self._owns_thread_pool = True
        else:
            self._thread_pool = request_thread_pool
            self._owns_thread_pool = False

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
        if not self._database:
            raise ValueError("Database name is required")
        
        tables_info = self._client.list_tables(self._database)
        table_names = [table["name"] for table in tables_info]
        
        if limit:
            table_names = table_names[:limit]
        
        return table_names

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
        if not self._database:
            raise ValueError("Database name is required")
        
        # Get table info from remote
        table_info = self._client.get_table(self._database, name)
        
        # Convert schema info to PyArrow schema
        schema_info = table_info.get("schema", {})
        fields = []
        
        for field_info in schema_info.get("fields", []):
            field_name = field_info["name"]
            field_type_str = field_info["type"]
            
            # Map type strings to PyArrow types
            if field_type_str == "string":
                pa_type = pa.string()
            elif field_type_str == "int64":
                pa_type = pa.int64()
            elif field_type_str == "float64":
                pa_type = pa.float64()
            elif field_type_str == "bool":
                pa_type = pa.bool_()
            elif field_type_str.startswith("list<"):
                # Vector type
                inner_type_str = field_type_str[5:-1]  # Remove "list<" and ">"
                if inner_type_str == "float32":
                    inner_type = pa.float32()
                elif inner_type_str == "float64":
                    inner_type = pa.float64()
                else:
                    inner_type = pa.float32()  # Default
                pa_type = pa.list_(inner_type)
            else:
                pa_type = pa.string()  # Default
            
            fields.append(pa.field(field_name, pa_type))
        
        schema = pa.schema(fields)
        
        return RemoteTable(
            self._client,
            self._database,
            name,
            schema,
            storage_options=self._storage_options,
        )

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
        """Create a table in the remote database.
        
        Parameters
        ----------
        name : str
            The name of the table
        data : various types, optional
            The data to insert into the table
        schema : pa.Schema or pd.DataFrame, optional
            The schema of the table
        mode : str, default "create"
            The mode for creating the table
        exist_ok : bool, default False
            If True, do not raise an error if the table already exists
        on_bad_vectors : str, default "error"
            How to handle bad vectors
        fill_value : float, default 0.0
            Value to use when filling bad vectors
        storage_options : dict, optional
            Additional storage options
            
        Returns
        -------
        Table
            The created table
        """
        if not self._database:
            raise ValueError("Database name is required")
        
        # Check if table exists
        existing_tables = self.table_names()
        table_exists = name in existing_tables
        
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
        
        # Convert PyArrow schema to API format
        schema_dict = {
            "fields": [
                {
                    "name": field.name,
                    "type": self._pyarrow_type_to_string(field.type),
                    "nullable": field.nullable,
                }
                for field in schema
            ]
        }
        
        # Create table via API
        create_response = self._client.create_table(
            self._database,
            name,
            schema_dict,
            **kwargs,
        )
        
        # Create table object
        table = RemoteTable(
            self._client,
            self._database,
            name,
            schema,
            storage_options=storage_options or self._storage_options,
        )
        
        # Insert data if provided
        if data is not None:
            table.add(data, mode="append")
        
        return table

    def drop_table(self, name: str) -> None:
        """Drop a table.
        
        Parameters
        ----------
        name : str
            The name of the table to drop
        """
        if not self._database:
            raise ValueError("Database name is required")
        
        self._client.drop_table(self._database, name)

    def rename_table(self, old_name: str, new_name: str) -> None:
        """Rename a table.
        
        Parameters
        ----------
        old_name : str
            The current name of the table
        new_name : str
            The new name for the table
        """
        # For remote tables, we need to implement this via API
        # This is a placeholder - actual implementation would depend on API support
        raise NotImplementedError("Table renaming is not yet supported for remote tables")

    def _pyarrow_type_to_string(self, pa_type: pa.DataType) -> str:
        """Convert PyArrow type to string representation.
        
        Parameters
        ----------
        pa_type : pa.DataType
            PyArrow data type
            
        Returns
        -------
        str
            String representation of the type
        """
        if pa.types.is_string(pa_type):
            return "string"
        elif pa.types.is_integer(pa_type):
            return "int64"
        elif pa.types.is_floating(pa_type):
            if pa_type == pa.float32():
                return "float32"
            else:
                return "float64"
        elif pa.types.is_boolean(pa_type):
            return "bool"
        elif pa.types.is_timestamp(pa_type):
            return "timestamp"
        elif pa.types.is_list(pa_type):
            inner_type = self._pyarrow_type_to_string(pa_type.value_type)
            return f"list<{inner_type}>"
        else:
            return "string"  # Default fallback

    def close(self) -> None:
        """Close the connection and cleanup resources."""
        if self._owns_thread_pool:
            self._thread_pool.shutdown(wait=True)
        
        # Close HTTP client (sync version)
        if hasattr(self._client, '__exit__'):
            self._client.__exit__(None, None, None)

    async def close_async(self) -> None:
        """Close the connection asynchronously."""
        if self._owns_thread_pool:
            self._thread_pool.shutdown(wait=True)
        
        # Close HTTP client (async version)
        await self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close_async()