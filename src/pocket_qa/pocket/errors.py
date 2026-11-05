"""Errors raised by the Pocket API client."""


class PocketError(RuntimeError):
    """Base error for Pocket client calls."""


class PocketAuthError(PocketError):
    """The call needs a Pocket API key, or the key was rejected."""


class PocketNotFoundError(PocketError):
    """The recording id was not found."""


class PocketAPIError(PocketError):
    """The public API returned an error status."""

    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(f"Pocket API {status_code}: {message}")
        self.status_code = status_code
