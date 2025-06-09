# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Hologres Authors

import os
from pathlib import Path
from typing import Union
from urllib.parse import urlparse

# Type alias for URI
URI = Union[str, Path]


def sanitize_uri(uri: URI) -> str:
    """Sanitize a URI for use with Hologres.
    
    Parameters
    ----------
    uri : str or Path
        The URI to sanitize
        
    Returns
    -------
    str
        The sanitized URI
    """
    if isinstance(uri, Path):
        return str(uri.absolute())
    
    if isinstance(uri, str):
        # Handle special Hologres cloud URIs
        if uri.startswith("holo://"):
            return uri
        
        # Handle PostgreSQL connection strings
        if uri.startswith(("postgresql://", "postgres://")):
            return uri
            
        # Handle local paths
        if not uri.startswith(("http://", "https://", "s3://", "gs://", "azure://")):
            return str(Path(uri).absolute())
            
        return uri
    
    raise ValueError(f"Invalid URI type: {type(uri)}")


def parse_hologres_uri(uri: str) -> dict:
    """Parse a Hologres URI into components.
    
    Parameters
    ----------
    uri : str
        The Hologres URI to parse
        
    Returns
    -------
    dict
        Dictionary containing parsed URI components
    """
    if uri.startswith("holo://"):
        # Parse Hologres cloud URI
        parsed = urlparse(uri)
        return {
            "scheme": "holo",
            "endpoint": parsed.netloc,
            "path": parsed.path,
            "params": parsed.params,
            "query": parsed.query,
            "fragment": parsed.fragment,
        }
    elif uri.startswith(("postgresql://", "postgres://")):
        # Parse PostgreSQL connection string
        parsed = urlparse(uri)
        return {
            "scheme": parsed.scheme,
            "username": parsed.username,
            "password": parsed.password,
            "hostname": parsed.hostname,
            "port": parsed.port,
            "database": parsed.path.lstrip("/") if parsed.path else None,
            "params": parsed.params,
            "query": parsed.query,
            "fragment": parsed.fragment,
        }
    else:
        raise ValueError(f"Unsupported URI scheme: {uri}")


def get_hologres_config():
    """Get Hologres configuration from environment variables.
    
    Returns
    -------
    dict
        Dictionary containing configuration values
    """
    return {
        "api_key": os.environ.get("HOLOGRES_API_KEY"),
        "username": os.environ.get("HOLOGRES_USERNAME"),
        "password": os.environ.get("HOLOGRES_PASSWORD"),
        "database": os.environ.get("HOLOGRES_DATABASE"),
        "host": os.environ.get("HOLOGRES_HOST"),
        "port": int(os.environ.get("HOLOGRES_PORT", "80")),
        "region": os.environ.get("HOLOGRES_REGION", "cn-hangzhou"),
    }