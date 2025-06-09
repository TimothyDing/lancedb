# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Hologres Authors

"""Embedding functions for Hologres.

This module provides various embedding functions that can be used with Hologres
for vector similarity search and retrieval.
"""

from .base import EmbeddingFunction, EmbeddingFunctionRegistry
from .openai_embeddings import OpenAIEmbeddings
from .sentence_transformers import SentenceTransformerEmbeddings
from .huggingface import HuggingFaceEmbeddings
from .cohere import CohereEmbeddingFunction
from .registry import get_registry

__all__ = [
    "EmbeddingFunction",
    "EmbeddingFunctionRegistry",
    "OpenAIEmbeddings",
    "SentenceTransformerEmbeddings",
    "HuggingFaceEmbeddings",
    "CohereEmbeddingFunction",
    "get_registry",
]