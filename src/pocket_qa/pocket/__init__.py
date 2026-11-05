"""Clients for the public Pocket API.

The HTTP client speaks the documented public API and is unused unless an API
key is provided. The mock client is what the CLI and tests use. It is not
Pocket and it does not contact heypocketai.com.
"""

from pocket_qa.pocket.http import HttpPocketClient
from pocket_qa.pocket.mock import MockPocketClient

__all__ = ["HttpPocketClient", "MockPocketClient"]
