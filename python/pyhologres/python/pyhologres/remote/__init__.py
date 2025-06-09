# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Hologres Authors

"""Remote connection support for Hologres Cloud.

This module provides functionality for connecting to Hologres Cloud instances
through HTTP APIs.
"""

from .client import ClientConfig, HologresCloudClient
from .db import RemoteDBConnection
from .table import RemoteTable

__all__ = [
    "ClientConfig",
    "HologresCloudClient",
    "RemoteDBConnection",
    "RemoteTable",
]