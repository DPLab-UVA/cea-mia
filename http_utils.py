"""HTTP utilities with retry logic for resilient LLM API calls."""
from __future__ import annotations

import asyncio
import logging
from functools import wraps
from typing import Callable, TypeVar

import httpx

log = logging.getLogger(__name__)

T = TypeVar("T")

# Retryable HTTP status codes
RETRYABLE_STATUS_CODES = {
    408,  # Request Timeout
    429,  # Too Many Requests
    500,  # Internal Server Error
    502,  # Bad Gateway
    503,  # Service Unavailable
    504,  # Gateway Timeout
}

# Retryable exception types
RETRYABLE_EXCEPTIONS = (
    httpx.ConnectError,
    httpx.ConnectTimeout,
    httpx.ReadError,
    httpx.ReadTimeout,
    httpx.RemoteProtocolError,
    httpx.WriteError,
    httpx.WriteTimeout,
    httpx.PoolTimeout,
    ConnectionError,
    TimeoutError,
)


async def retry_async(
    func: Callable[..., T],
    *args,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    exponential_base: float = 2.0,
    **kwargs,
) -> T:
    """Execute an async function with exponential backoff retry.

    Args:
        func: Async function to execute
        *args: Positional arguments for func
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay in seconds
        max_delay: Maximum delay in seconds
        exponential_base: Base for exponential backoff
        **kwargs: Keyword arguments for func

    Returns:
        Result from the function

    Raises:
        The last exception if all retries fail
    """
    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            return await func(*args, **kwargs)
        except RETRYABLE_EXCEPTIONS as e:
            last_exception = e
            if attempt < max_retries:
                delay = min(base_delay * (exponential_base ** attempt), max_delay)
                log.warning(
                    "Request failed (attempt %d/%d, %s): %s. Retrying in %.1fs...",
                    attempt + 1,
                    max_retries + 1,
                    type(e).__name__,
                    str(e),
                    delay,
                )
                await asyncio.sleep(delay)
            else:
                log.error(
                    "Request failed after %d attempts (%s): %s",
                    max_retries + 1,
                    type(e).__name__,
                    str(e),
                )
        except httpx.HTTPStatusError as e:
            last_exception = e
            if e.response.status_code in RETRYABLE_STATUS_CODES and attempt < max_retries:
                delay = min(base_delay * (exponential_base ** attempt), max_delay)
                log.warning(
                    "Request failed with status %d (attempt %d/%d). Retrying in %.1fs...",
                    e.response.status_code,
                    attempt + 1,
                    max_retries + 1,
                    delay,
                )
                await asyncio.sleep(delay)
            else:
                raise

    raise last_exception


async def post_with_retry(
    client: httpx.AsyncClient,
    url: str,
    *,
    max_retries: int = 3,
    base_delay: float = 1.0,
    **kwargs,
) -> httpx.Response:
    """POST request with automatic retry on transient failures.

    Args:
        client: httpx AsyncClient instance
        url: URL to POST to
        max_retries: Maximum retry attempts
        base_delay: Initial delay between retries
        **kwargs: Additional arguments to client.post()

    Returns:
        httpx.Response object

    Raises:
        httpx.HTTPStatusError: If request fails with non-retryable status
        Other exceptions after all retries exhausted
    """
    async def _do_post():
        resp = await client.post(url, **kwargs)
        resp.raise_for_status()
        return resp

    return await retry_async(_do_post, max_retries=max_retries, base_delay=base_delay)
