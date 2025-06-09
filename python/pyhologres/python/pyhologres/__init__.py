# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Hologres Authors

import importlib.metadata
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from typing import Dict, Optional, Union, Any
import warnings

__version__ = "0.1.0"

from .common import URI, sanitize_uri
from .db import AsyncConnection, DBConnection, HologresDBConnection
from .remote import ClientConfig
from .remote.db import RemoteDBConnection
from .schema import vector
from .table import AsyncTable


def connect(
    uri: URI,
    *,
    username: Optional[str] = None,
    password: Optional[str] = None,
    database: Optional[str] = None,
    host: Optional[str] = None,
    port: int = 80,
    api_key: Optional[str] = None,
    region: str = "cn-hangzhou",
    host_override: Optional[str] = None,
    read_consistency_interval: Optional[timedelta] = None,
    request_thread_pool: Optional[Union[int, ThreadPoolExecutor]] = None,
    client_config: Union[ClientConfig, Dict[str, Any], None] = None,
    storage_options: Optional[Dict[str, str]] = None,
    **kwargs: Any,
) -> DBConnection:
    """Connect to a Hologres database.

    Parameters
    ----------
    uri: str or Path
        The uri of the database. Can be:
        - PostgreSQL connection string: "postgresql://user:pass@host:port/db"
        - Hologres cloud URI: "holo://endpoint"
        - Local file path for testing
    username: str, optional
        Username for Hologres connection
    password: str, optional
        Password for Hologres connection
    database: str, optional
        Database name
    host: str, optional
        Hologres endpoint host
    port: int, default 80
        Hologres endpoint port
    api_key: str, optional
        If presented, connect to Hologres cloud.
        Can be set via environment variable `HOLOGRES_API_KEY`.
    region: str, default "cn-hangzhou"
        The region to use for Hologres Cloud.
    host_override: str, optional
        The override url for Hologres Cloud.
    read_consistency_interval: timedelta, default None
        The interval at which to check for updates to the table from other
        processes. If None, then consistency is not checked.
    client_config: ClientConfig or dict, optional
        Configuration options for the Hologres Cloud HTTP client.
    storage_options: dict, optional
        Additional options for the storage backend.

    Examples
    --------

    For a PostgreSQL-style connection:

    >>> import pyhologres
    >>> db = pyhologres.connect("postgresql://user:pass@host:port/database")

    For Hologres cloud:

    >>> db = pyhologres.connect("holo://my_endpoint", api_key="holo_...",
    ...                         username="user", password="pass")

    Returns
    -------
    conn : DBConnection
        A connection to a Hologres database.
    """
    if isinstance(uri, str) and uri.startswith("holo://"):
        if api_key is None:
            api_key = os.environ.get("HOLOGRES_API_KEY")
        if api_key is None:
            raise ValueError(f"api_key is required to connect to Hologres cloud: {uri}")
        if isinstance(request_thread_pool, int):
            request_thread_pool = ThreadPoolExecutor(request_thread_pool)
        return RemoteDBConnection(
            uri,
            api_key,
            region,
            host_override,
            username=username,
            password=password,
            database=database,
            host=host,
            port=port,
            request_thread_pool=request_thread_pool,
            client_config=client_config,
            storage_options=storage_options,
            **kwargs,
        )

    if kwargs:
        raise ValueError(f"Unknown keyword arguments: {kwargs}")
    return HologresDBConnection(
        uri,
        username=username,
        password=password,
        database=database,
        host=host,
        port=port,
        read_consistency_interval=read_consistency_interval,
        storage_options=storage_options,
    )


async def connect_async(
    uri: URI,
    *,
    username: Optional[str] = None,
    password: Optional[str] = None,
    database: Optional[str] = None,
    host: Optional[str] = None,
    port: int = 80,
    api_key: Optional[str] = None,
    region: str = "cn-hangzhou",
    host_override: Optional[str] = None,
    read_consistency_interval: Optional[timedelta] = None,
    client_config: Optional[Union[ClientConfig, Dict[str, Any]]] = None,
    storage_options: Optional[Dict[str, str]] = None,
) -> AsyncConnection:
    """Connect to a Hologres database asynchronously.

    Parameters
    ----------
    uri: str or Path
        The uri of the database.
    username: str, optional
        Username for Hologres connection
    password: str, optional
        Password for Hologres connection
    database: str, optional
        Database name
    host: str, optional
        Hologres endpoint host
    port: int, default 80
        Hologres endpoint port
    api_key: str, optional
        If present, connect to Hologres cloud.
        Can be set via environment variable `HOLOGRES_API_KEY`.
    region: str, default "cn-hangzhou"
        The region to use for Hologres Cloud.
    host_override: str, optional
        The override url for Hologres Cloud.
    read_consistency_interval: timedelta, default None
        The interval at which to check for updates to the table from other
        processes.
    client_config: ClientConfig or dict, optional
        Configuration options for the Hologres Cloud HTTP client.
    storage_options: dict, optional
        Additional options for the storage backend.

    Examples
    --------

    >>> import pyhologres
    >>> async def doctest_example():
    ...     # For a PostgreSQL-style connection
    ...     db = await pyhologres.connect_async("postgresql://user:pass@host:port/db")
    ...     # For Hologres cloud
    ...     db = await pyhologres.connect_async("holo://my_endpoint", api_key="holo_...",
    ...                                         username="user", password="pass")

    Returns
    -------
    conn : AsyncConnection
        A connection to a Hologres database.
    """
    if read_consistency_interval is not None:
        read_consistency_interval_secs = read_consistency_interval.total_seconds()
    else:
        read_consistency_interval_secs = None

    if isinstance(client_config, dict):
        client_config = ClientConfig(**client_config)

    from .hologres_connection import hologres_connect_async
    
    return AsyncConnection(
        await hologres_connect_async(
            sanitize_uri(uri),
            username,
            password,
            database,
            host,
            port,
            api_key,
            region,
            host_override,
            read_consistency_interval_secs,
            client_config,
            storage_options,
        )
    )


__all__ = [
    "connect",
    "connect_async",
    "AsyncConnection",
    "AsyncTable",
    "URI",
    "sanitize_uri",
    "vector",
    "DBConnection",
    "HologresDBConnection",
    "RemoteDBConnection",
    "__version__",
]


def __warn_on_fork():
    warnings.warn(
        "pyhologres is not fork-safe. If you are using multiprocessing, use spawn instead.",
    )


if hasattr(os, "register_at_fork"):
    os.register_at_fork(before=__warn_on_fork)