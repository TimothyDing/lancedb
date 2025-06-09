# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Hologres Authors

import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

import numpy as np
from pydantic import BaseModel


class EmbeddingFunction(BaseModel, ABC):
    """Abstract base class for embedding functions.
    
    This class defines the interface that all embedding functions must implement
    to be compatible with Hologres.
    """

    class Config:
        """Pydantic configuration."""
        arbitrary_types_allowed = True

    @abstractmethod
    def compute_query_embeddings(self, query: str) -> List[float]:
        """Compute embeddings for a query string.
        
        Parameters
        ----------
        query : str
            The query string to embed
            
        Returns
        -------
        List[float]
            The embedding vector
        """
        pass

    @abstractmethod
    def compute_source_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Compute embeddings for a list of source texts.
        
        Parameters
        ----------
        texts : List[str]
            List of texts to embed
            
        Returns
        -------
        List[List[float]]
            List of embedding vectors
        """
        pass

    def ndims(self) -> Optional[int]:
        """Return the number of dimensions of the embeddings.
        
        Returns
        -------
        Optional[int]
            Number of dimensions, or None if unknown
        """
        return None

    @classmethod
    def create(cls, **kwargs) -> "EmbeddingFunction":
        """Create an instance of the embedding function.
        
        Parameters
        ----------
        **kwargs
            Keyword arguments for initialization
            
        Returns
        -------
        EmbeddingFunction
            An instance of the embedding function
        """
        return cls(**kwargs)

    def __resolveVariables(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve variables in configuration data.
        
        Parameters
        ----------
        data : Dict[str, Any]
            Configuration data
            
        Returns
        -------
        Dict[str, Any]
            Resolved configuration data
        """
        # This is a placeholder for variable resolution logic
        # In a real implementation, this might resolve environment variables
        # or other dynamic configuration values
        return data

    def sensitive_keys(self) -> List[str]:
        """Return a list of sensitive configuration keys.
        
        Returns
        -------
        List[str]
            List of sensitive keys that should not be logged
        """
        return ["api_key", "secret_key", "password", "token"]

    def compute_query_embeddings_with_retry(
        self,
        query: str,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ) -> List[float]:
        """Compute query embeddings with retry logic.
        
        Parameters
        ----------
        query : str
            The query string to embed
        max_retries : int, default 3
            Maximum number of retry attempts
        retry_delay : float, default 1.0
            Delay between retries in seconds
            
        Returns
        -------
        List[float]
            The embedding vector
            
        Raises
        ------
        Exception
            If all retry attempts fail
        """
        last_exception = None
        
        for attempt in range(max_retries + 1):
            try:
                return self.compute_query_embeddings(query)
            except Exception as e:
                last_exception = e
                if attempt < max_retries:
                    time.sleep(retry_delay * (2 ** attempt))  # Exponential backoff
                    continue
                else:
                    break
        
        raise last_exception

    def compute_source_embeddings_with_retry(
        self,
        texts: List[str],
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ) -> List[List[float]]:
        """Compute source embeddings with retry logic.
        
        Parameters
        ----------
        texts : List[str]
            List of texts to embed
        max_retries : int, default 3
            Maximum number of retry attempts
        retry_delay : float, default 1.0
            Delay between retries in seconds
            
        Returns
        -------
        List[List[float]]
            List of embedding vectors
            
        Raises
        ------
        Exception
            If all retry attempts fail
        """
        last_exception = None
        
        for attempt in range(max_retries + 1):
            try:
                return self.compute_source_embeddings(texts)
            except Exception as e:
                last_exception = e
                if attempt < max_retries:
                    time.sleep(retry_delay * (2 ** attempt))  # Exponential backoff
                    continue
                else:
                    break
        
        raise last_exception

    def validate_embeddings(self, embeddings: List[List[float]]) -> bool:
        """Validate that embeddings are well-formed.
        
        Parameters
        ----------
        embeddings : List[List[float]]
            List of embedding vectors to validate
            
        Returns
        -------
        bool
            True if embeddings are valid, False otherwise
        """
        if not embeddings:
            return False
        
        # Check that all embeddings have the same dimension
        first_dim = len(embeddings[0]) if embeddings else 0
        for emb in embeddings:
            if len(emb) != first_dim:
                return False
            
            # Check for NaN or infinite values
            if any(not np.isfinite(x) for x in emb):
                return False
        
        return True

    def normalize_embeddings(self, embeddings: List[List[float]]) -> List[List[float]]:
        """Normalize embeddings to unit length.
        
        Parameters
        ----------
        embeddings : List[List[float]]
            List of embedding vectors to normalize
            
        Returns
        -------
        List[List[float]]
            List of normalized embedding vectors
        """
        normalized = []
        for emb in embeddings:
            emb_array = np.array(emb)
            norm = np.linalg.norm(emb_array)
            if norm > 0:
                normalized.append((emb_array / norm).tolist())
            else:
                normalized.append(emb)
        return normalized


class EmbeddingFunctionRegistry:
    """Registry for embedding functions.
    
    This class manages a registry of available embedding functions,
    allowing users to discover and instantiate them by name.
    """

    def __init__(self):
        self._functions: Dict[str, type] = {}

    def register(self, name: str, func_class: type) -> None:
        """Register an embedding function.
        
        Parameters
        ----------
        name : str
            Name of the embedding function
        func_class : type
            Class implementing the embedding function
        """
        self._functions[name] = func_class

    def get(self, name: str) -> Optional[type]:
        """Get an embedding function class by name.
        
        Parameters
        ----------
        name : str
            Name of the embedding function
            
        Returns
        -------
        Optional[type]
            The embedding function class, or None if not found
        """
        return self._functions.get(name)

    def list_functions(self) -> List[str]:
        """List all registered embedding functions.
        
        Returns
        -------
        List[str]
            List of registered function names
        """
        return list(self._functions.keys())

    def create(self, name: str, **kwargs) -> Optional[EmbeddingFunction]:
        """Create an instance of an embedding function.
        
        Parameters
        ----------
        name : str
            Name of the embedding function
        **kwargs
            Keyword arguments for initialization
            
        Returns
        -------
        Optional[EmbeddingFunction]
            An instance of the embedding function, or None if not found
        """
        func_class = self.get(name)
        if func_class:
            return func_class.create(**kwargs)
        return None