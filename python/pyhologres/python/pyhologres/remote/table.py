# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Hologres Authors

import asyncio
from typing import Any, Dict, Iterable, List, Optional, Union, TYPE_CHECKING

import pyarrow as pa
import pyarrow.compute as pc

from ..query import Query, HologresQuery
from ..table import Table, _into_pyarrow_reader
from .client import HologresCloudClient

if TYPE_CHECKING:
    import pandas as pd
    import polars as pl


class RemoteTable(Table):
    """A table stored in Hologres Cloud."""

    def __init__(
        self,
        client: HologresCloudClient,
        database: str,
        name: str,
        schema: pa.Schema,
        storage_options: Optional[Dict[str, str]] = None,
    ):
        """Initialize a remote table.
        
        Parameters
        ----------
        client : HologresCloudClient
            The HTTP client for API calls
        database : str
            Database name
        name : str
            Table name
        schema : pa.Schema
            Table schema
        storage_options : dict, optional
            Additional storage options
        """
        self._client = client
        self._database = database
        self._name = name
        self._schema = schema
        self._storage_options = storage_options or {}

    @property
    def name(self) -> str:
        """Get the table name."""
        return self._name

    @property
    def schema(self) -> pa.Schema:
        """Get the table schema."""
        return self._schema

    def count_rows(self, filter: Optional[str] = None) -> int:
        """Count the number of rows in the table.
        
        Parameters
        ----------
        filter : str, optional
            SQL WHERE clause filter
            
        Returns
        -------
        int
            Number of rows
        """
        # Build count query
        sql = f"SELECT COUNT(*) FROM {self._name}"
        if filter:
            sql += f" WHERE {filter}"
        
        # Execute query
        result = self._client.query_table(self._database, sql)
        
        # Extract count from result
        if result and len(result) > 0:
            return int(result[0][0])
        return 0

    def to_pandas(self, **kwargs) -> "pd.DataFrame":
        """Convert table to pandas DataFrame.
        
        Returns
        -------
        pd.DataFrame
            Table data as pandas DataFrame
        """
        return self.to_arrow().to_pandas(**kwargs)

    def to_polars(self, **kwargs) -> "pl.DataFrame":
        """Convert table to polars DataFrame.
        
        Returns
        -------
        pl.DataFrame
            Table data as polars DataFrame
        """
        try:
            import polars as pl
            return pl.from_arrow(self.to_arrow(), **kwargs)
        except ImportError:
            raise ImportError("polars is required for to_polars()")

    def to_arrow(self) -> pa.Table:
        """Convert table to PyArrow Table.
        
        Returns
        -------
        pa.Table
            Table data as PyArrow Table
        """
        # Query all data from table
        sql = f"SELECT * FROM {self._name}"
        result = self._client.query_table(self._database, sql)
        
        # Convert result to PyArrow Table
        if not result:
            # Return empty table with schema
            return pa.table([], schema=self._schema)
        
        # Convert rows to columns
        columns = {field.name: [] for field in self._schema}
        
        for row in result:
            for i, field in enumerate(self._schema):
                if i < len(row):
                    columns[field.name].append(row[i])
                else:
                    columns[field.name].append(None)
        
        # Create PyArrow arrays
        arrays = []
        for field in self._schema:
            column_data = columns[field.name]
            
            # Handle vector columns (list type)
            if pa.types.is_list(field.type):
                # Convert string representations back to lists
                processed_data = []
                for item in column_data:
                    if isinstance(item, str):
                        # Parse string representation of list
                        try:
                            import json
                            processed_data.append(json.loads(item))
                        except:
                            processed_data.append(None)
                    elif isinstance(item, list):
                        processed_data.append(item)
                    else:
                        processed_data.append(None)
                column_data = processed_data
            
            arrays.append(pa.array(column_data, type=field.type))
        
        return pa.table(arrays, schema=self._schema)

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
        *,
        mode: str = "append",
        on_bad_vectors: str = "error",
        fill_value: float = 0.0,
    ) -> None:
        """Add data to the table.
        
        Parameters
        ----------
        data : various types
            Data to add to the table
        mode : str, default "append"
            How to add the data ("append" or "overwrite")
        on_bad_vectors : str, default "error"
            How to handle bad vectors
        fill_value : float, default 0.0
            Value to use when filling bad vectors
        """
        # Convert data to PyArrow format
        reader = _into_pyarrow_reader(data)
        
        # Process batches and insert via API
        for batch in reader:
            # Convert batch to list of dictionaries
            batch_dict = batch.to_pydict()
            rows = []
            
            # Convert to row format
            num_rows = len(batch_dict[list(batch_dict.keys())[0]])
            for i in range(num_rows):
                row = {}
                for column_name, column_data in batch_dict.items():
                    value = column_data[i]
                    
                    # Handle vector columns
                    if isinstance(value, list):
                        # Convert to JSON string for API
                        import json
                        value = json.dumps(value)
                    
                    row[column_name] = value
                rows.append(row)
            
            # Insert batch via API
            self._client.insert_data(self._database, self._name, rows)

    def search(
        self,
        query: Optional[Union[str, List[float], "np.ndarray"]] = None,
        vector_column_name: Optional[str] = None,
        query_type: str = "vector",
    ) -> Query:
        """Search the table.
        
        Parameters
        ----------
        query : str, list, or np.ndarray, optional
            Query vector or text
        vector_column_name : str, optional
            Name of the vector column to search
        query_type : str, default "vector"
            Type of query ("vector", "fts", "hybrid")
            
        Returns
        -------
        Query
            Query object for further configuration
        """
        return HologresQuery(
            table=self,
            query=query,
            vector_column_name=vector_column_name,
            query_type=query_type,
        )

    def create_index(
        self,
        column: str,
        *,
        index_type: str = "BTREE",
        name: Optional[str] = None,
        replace: bool = True,
        **kwargs: Any,
    ) -> None:
        """Create an index on the table.
        
        Parameters
        ----------
        column : str
            Column name to index
        index_type : str, default "BTREE"
            Type of index to create
        name : str, optional
            Name of the index
        replace : bool, default True
            Whether to replace existing index
        **kwargs
            Additional index options
        """
        if name is None:
            name = f"{self._name}_{column}_idx"
        
        # Create index via API
        index_config = {
            "name": name,
            "column": column,
            "type": index_type,
            "replace": replace,
            **kwargs,
        }
        
        self._client.create_index(self._database, self._name, index_config)

    def create_fts_index(
        self,
        field: str,
        *,
        name: Optional[str] = None,
        replace: bool = True,
        base_tokenizer: str = "simple",
        language: str = "English",
        max_token_length: Optional[int] = None,
        lower_case: bool = True,
        stem: bool = False,
        remove_stop_words: bool = False,
        ascii_folding: bool = False,
        with_position: bool = True,
        **kwargs: Any,
    ) -> None:
        """Create a full-text search index.
        
        Parameters
        ----------
        field : str
            Field name to index
        name : str, optional
            Name of the index
        replace : bool, default True
            Whether to replace existing index
        base_tokenizer : str, default "simple"
            Base tokenizer to use
        language : str, default "English"
            Language for tokenization
        max_token_length : int, optional
            Maximum token length
        lower_case : bool, default True
            Whether to convert to lowercase
        stem : bool, default False
            Whether to apply stemming
        remove_stop_words : bool, default False
            Whether to remove stop words
        ascii_folding : bool, default False
            Whether to apply ASCII folding
        with_position : bool, default True
            Whether to store positions
        **kwargs
            Additional index options
        """
        if name is None:
            name = f"{self._name}_{field}_fts_idx"
        
        # Create FTS index via API
        index_config = {
            "name": name,
            "field": field,
            "type": "FTS",
            "replace": replace,
            "base_tokenizer": base_tokenizer,
            "language": language,
            "max_token_length": max_token_length,
            "lower_case": lower_case,
            "stem": stem,
            "remove_stop_words": remove_stop_words,
            "ascii_folding": ascii_folding,
            "with_position": with_position,
            **kwargs,
        }
        
        self._client.create_index(self._database, self._name, index_config)

    def update(
        self,
        where: Optional[str] = None,
        values: Optional[Dict[str, Any]] = None,
        *,
        values_sql: Optional[Dict[str, str]] = None,
    ) -> None:
        """Update rows in the table.
        
        Parameters
        ----------
        where : str, optional
            SQL WHERE clause
        values : dict, optional
            Column values to update
        values_sql : dict, optional
            Column values as SQL expressions
        """
        if not values and not values_sql:
            raise ValueError("Either values or values_sql must be provided")
        
        # Build UPDATE SQL
        set_clauses = []
        
        if values:
            for column, value in values.items():
                if isinstance(value, str):
                    set_clauses.append(f"{column} = '{value}'")
                elif isinstance(value, list):
                    # Handle vector columns
                    import json
                    value_str = json.dumps(value)
                    set_clauses.append(f"{column} = '{value_str}'")
                else:
                    set_clauses.append(f"{column} = {value}")
        
        if values_sql:
            for column, sql_expr in values_sql.items():
                set_clauses.append(f"{column} = {sql_expr}")
        
        sql = f"UPDATE {self._name} SET {', '.join(set_clauses)}"
        if where:
            sql += f" WHERE {where}"
        
        # Execute update via API
        self._client.query_table(self._database, sql)

    def delete(self, where: str) -> None:
        """Delete rows from the table.
        
        Parameters
        ----------
        where : str
            SQL WHERE clause for deletion
        """
        sql = f"DELETE FROM {self._name} WHERE {where}"
        
        # Execute delete via API
        self._client.query_table(self._database, sql)

    def cleanup_old_versions(
        self,
        older_than: Optional[Union[int, str]] = None,
        delete_unverified: bool = False,
    ) -> Dict[str, Any]:
        """Clean up old versions of the table.
        
        Parameters
        ----------
        older_than : int or str, optional
            Delete versions older than this
        delete_unverified : bool, default False
            Whether to delete unverified versions
            
        Returns
        -------
        dict
            Cleanup statistics
        """
        # For remote tables, this would be handled by the cloud service
        # Return empty stats for now
        return {"versions_deleted": 0, "bytes_freed": 0}

    def compact_files(
        self,
        *,
        target_rows_per_fragment: int = 1024 * 1024,
        max_rows_per_group: int = 1024,
        materialize_deletions: bool = True,
        materialize_deletions_threshold: float = 0.1,
        num_threads: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Compact table files.
        
        Parameters
        ----------
        target_rows_per_fragment : int, default 1048576
            Target rows per fragment
        max_rows_per_group : int, default 1024
            Maximum rows per group
        materialize_deletions : bool, default True
            Whether to materialize deletions
        materialize_deletions_threshold : float, default 0.1
            Threshold for materializing deletions
        num_threads : int, optional
            Number of threads to use
            
        Returns
        -------
        dict
            Compaction statistics
        """
        # For remote tables, compaction would be handled by the cloud service
        # Return empty stats for now
        return {"fragments_removed": 0, "fragments_added": 0, "files_removed": 0, "files_added": 0}

    def list_indices(self) -> List[Dict[str, Any]]:
        """List all indices on the table.
        
        Returns
        -------
        List[dict]
            List of index information
        """
        # Get indices via API
        try:
            indices = self._client.list_indices(self._database, self._name)
            return indices
        except Exception:
            # Return empty list if API doesn't support listing indices
            return []

    def __repr__(self) -> str:
        return f"RemoteTable(name='{self._name}', database='{self._database}', schema={self._schema})"

    def __str__(self) -> str:
        return self.__repr__()