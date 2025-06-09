# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Hologres Authors

import json
import time
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urljoin

import aiohttp
import requests
from pydantic import BaseModel, Field


class ClientConfig(BaseModel):
    """Configuration for Hologres Cloud HTTP client."""
    
    timeout: float = Field(default=30.0, description="Request timeout in seconds")
    max_retries: int = Field(default=3, description="Maximum number of retries")
    retry_delay: float = Field(default=1.0, description="Base delay between retries")
    max_connections: int = Field(default=100, description="Maximum number of connections")
    user_agent: str = Field(default="pyhologres/0.1.0", description="User agent string")
    verify_ssl: bool = Field(default=True, description="Whether to verify SSL certificates")
    
    class Config:
        """Pydantic configuration."""
        extra = "allow"


class HologresCloudClient:
    """HTTP client for Hologres Cloud API."""
    
    def __init__(
        self,
        api_key: str,
        region: str = "cn-hangzhou",
        host_override: Optional[str] = None,
        config: Optional[ClientConfig] = None,
    ):
        """Initialize Hologres Cloud client.
        
        Parameters
        ----------
        api_key : str
            API key for authentication
        region : str, default "cn-hangzhou"
            Region for Hologres Cloud
        host_override : str, optional
            Override host URL
        config : ClientConfig, optional
            Client configuration
        """
        self.api_key = api_key
        self.region = region
        self.config = config or ClientConfig()
        
        # Determine base URL
        if host_override:
            self.base_url = host_override
        else:
            self.base_url = f"https://hologres.{region}.aliyuncs.com"
        
        # Setup session
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "User-Agent": self.config.user_agent,
            "Content-Type": "application/json",
        })
        
        # Setup async session
        self._async_session = None
    
    def _get_async_session(self) -> aiohttp.ClientSession:
        """Get or create async session."""
        if self._async_session is None or self._async_session.closed:
            connector = aiohttp.TCPConnector(
                limit=self.config.max_connections,
                verify_ssl=self.config.verify_ssl,
            )
            timeout = aiohttp.ClientTimeout(total=self.config.timeout)
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "User-Agent": self.config.user_agent,
                "Content-Type": "application/json",
            }
            self._async_session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                headers=headers,
            )
        return self._async_session
    
    def _make_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Make HTTP request with retry logic.
        
        Parameters
        ----------
        method : str
            HTTP method
        endpoint : str
            API endpoint
        data : dict, optional
            Request body data
        params : dict, optional
            Query parameters
            
        Returns
        -------
        Dict[str, Any]
            Response data
            
        Raises
        ------
        requests.RequestException
            If request fails after all retries
        """
        url = urljoin(self.base_url, endpoint)
        
        for attempt in range(self.config.max_retries + 1):
            try:
                response = self.session.request(
                    method=method,
                    url=url,
                    json=data,
                    params=params,
                    timeout=self.config.timeout,
                    verify=self.config.verify_ssl,
                )
                response.raise_for_status()
                return response.json()
            
            except requests.RequestException as e:
                if attempt < self.config.max_retries:
                    delay = self.config.retry_delay * (2 ** attempt)
                    time.sleep(delay)
                    continue
                else:
                    raise e
    
    async def _make_async_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Make async HTTP request with retry logic.
        
        Parameters
        ----------
        method : str
            HTTP method
        endpoint : str
            API endpoint
        data : dict, optional
            Request body data
        params : dict, optional
            Query parameters
            
        Returns
        -------
        Dict[str, Any]
            Response data
            
        Raises
        ------
        aiohttp.ClientError
            If request fails after all retries
        """
        url = urljoin(self.base_url, endpoint)
        session = self._get_async_session()
        
        for attempt in range(self.config.max_retries + 1):
            try:
                async with session.request(
                    method=method,
                    url=url,
                    json=data,
                    params=params,
                ) as response:
                    response.raise_for_status()
                    return await response.json()
            
            except aiohttp.ClientError as e:
                if attempt < self.config.max_retries:
                    delay = self.config.retry_delay * (2 ** attempt)
                    await asyncio.sleep(delay)
                    continue
                else:
                    raise e
    
    def list_databases(self) -> List[Dict[str, Any]]:
        """List available databases.
        
        Returns
        -------
        List[Dict[str, Any]]
            List of database information
        """
        return self._make_request("GET", "/api/v1/databases")
    
    def list_tables(self, database: str) -> List[Dict[str, Any]]:
        """List tables in a database.
        
        Parameters
        ----------
        database : str
            Database name
            
        Returns
        -------
        List[Dict[str, Any]]
            List of table information
        """
        return self._make_request("GET", f"/api/v1/databases/{database}/tables")
    
    def create_table(
        self,
        database: str,
        table_name: str,
        schema: Dict[str, Any],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Create a table.
        
        Parameters
        ----------
        database : str
            Database name
        table_name : str
            Table name
        schema : Dict[str, Any]
            Table schema
        **kwargs
            Additional table creation options
            
        Returns
        -------
        Dict[str, Any]
            Table creation response
        """
        data = {
            "name": table_name,
            "schema": schema,
            **kwargs,
        }
        return self._make_request("POST", f"/api/v1/databases/{database}/tables", data=data)
    
    def get_table(
        self,
        database: str,
        table_name: str,
    ) -> Dict[str, Any]:
        """Get table information.
        
        Parameters
        ----------
        database : str
            Database name
        table_name : str
            Table name
            
        Returns
        -------
        Dict[str, Any]
            Table information
        """
        return self._make_request("GET", f"/api/v1/databases/{database}/tables/{table_name}")
    
    def drop_table(
        self,
        database: str,
        table_name: str,
    ) -> Dict[str, Any]:
        """Drop a table.
        
        Parameters
        ----------
        database : str
            Database name
        table_name : str
            Table name
            
        Returns
        -------
        Dict[str, Any]
            Drop response
        """
        return self._make_request("DELETE", f"/api/v1/databases/{database}/tables/{table_name}")
    
    def insert_data(
        self,
        database: str,
        table_name: str,
        data: List[Dict[str, Any]],
        mode: str = "append",
    ) -> Dict[str, Any]:
        """Insert data into a table.
        
        Parameters
        ----------
        database : str
            Database name
        table_name : str
            Table name
        data : List[Dict[str, Any]]
            Data to insert
        mode : str, default "append"
            Insert mode
            
        Returns
        -------
        Dict[str, Any]
            Insert response
        """
        payload = {
            "data": data,
            "mode": mode,
        }
        return self._make_request(
            "POST",
            f"/api/v1/databases/{database}/tables/{table_name}/data",
            data=payload,
        )
    
    def query_data(
        self,
        database: str,
        table_name: str,
        query: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Query data from a table.
        
        Parameters
        ----------
        database : str
            Database name
        table_name : str
            Table name
        query : Dict[str, Any]
            Query parameters
            
        Returns
        -------
        Dict[str, Any]
            Query results
        """
        return self._make_request(
            "POST",
            f"/api/v1/databases/{database}/tables/{table_name}/query",
            data=query,
        )
    
    def create_index(
        self,
        database: str,
        table_name: str,
        index_config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Create an index on a table.
        
        Parameters
        ----------
        database : str
            Database name
        table_name : str
            Table name
        index_config : Dict[str, Any]
            Index configuration
            
        Returns
        -------
        Dict[str, Any]
            Index creation response
        """
        return self._make_request(
            "POST",
            f"/api/v1/databases/{database}/tables/{table_name}/indexes",
            data=index_config,
        )
    
    async def close(self) -> None:
        """Close the client and cleanup resources."""
        if self._async_session and not self._async_session.closed:
            await self._async_session.close()
        
        if hasattr(self.session, 'close'):
            self.session.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # For sync context manager, we can't call async close
        # Just close the sync session
        if hasattr(self.session, 'close'):
            self.session.close()
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()