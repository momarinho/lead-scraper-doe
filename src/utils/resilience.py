"""
Resilience patterns for self-annealing capabilities.

Implements Circuit Breaker pattern to prevent cascading failures
when external services become unavailable.
"""
import asyncio
import logging
import time
from typing import Callable, Any, Optional
from enum import Enum

logger = logging.getLogger(__name__)


class CircuitState(str, Enum):
    """Circuit breaker states"""
    CLOSED = "CLOSED"      # Normal operation
    OPEN = "OPEN"          # Failing, block requests
    HALF_OPEN = "HALF_OPEN"  # Testing recovery


class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open and blocking requests"""
    def __init__(self, service_name: str, retry_after: float):
        self.service_name = service_name
        self.retry_after = retry_after
        super().__init__(
            f"Circuit breaker OPEN for '{service_name}'. "
            f"Retry after {retry_after:.1f}s"
        )


class CircuitBreaker:
    """
    Circuit Breaker pattern for resilient service calls.

    Prevents cascading failures by:
    - Tracking consecutive failures
    - Opening circuit when threshold exceeded (fail fast)
    - Automatically testing recovery after timeout
    - Closing circuit when service recovers

    States:
        CLOSED: Normal operation, requests pass through
        OPEN: Service failing, requests blocked immediately
        HALF_OPEN: Testing if service recovered

    Usage:
        circuit = CircuitBreaker("google_maps", failure_threshold=5, timeout=60)

        try:
            result = await circuit.call(fetch_page, url)
        except CircuitBreakerOpenError as e:
            # Handle blocked request (service is down)
            logger.warning(f"Service unavailable: {e}")
    """

    def __init__(
        self,
        service_name: str,
        failure_threshold: int = 5,
        timeout: float = 60.0,
        success_threshold: int = 1
    ):
        """
        Initialize circuit breaker.

        Args:
            service_name: Name for logging/identification
            failure_threshold: Consecutive failures before opening
            timeout: Seconds before testing recovery (OPEN -> HALF_OPEN)
            success_threshold: Successes needed to close from HALF_OPEN
        """
        self.service_name = service_name
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.success_threshold = success_threshold

        self._state = CircuitState.CLOSED
        self._failures = 0
        self._successes = 0
        self._last_failure_time: Optional[float] = None
        self._lock = asyncio.Lock()

    @property
    def state(self) -> CircuitState:
        """Current circuit state"""
        return self._state

    @property
    def failures(self) -> int:
        """Current consecutive failure count"""
        return self._failures

    def _should_allow_request(self) -> bool:
        """Check if request should be allowed based on current state"""
        if self._state == CircuitState.CLOSED:
            return True

        if self._state == CircuitState.OPEN:
            # Check if timeout has passed
            if self._last_failure_time:
                elapsed = time.time() - self._last_failure_time
                if elapsed >= self.timeout:
                    # Transition to HALF_OPEN
                    self._state = CircuitState.HALF_OPEN
                    self._successes = 0
                    logger.info(
                        f"Circuit breaker '{self.service_name}' "
                        f"transitioning OPEN -> HALF_OPEN after {elapsed:.1f}s"
                    )
                    return True
            return False

        if self._state == CircuitState.HALF_OPEN:
            # Allow single test request
            return True

        return False

    def _on_success(self) -> None:
        """Handle successful call"""
        if self._state == CircuitState.HALF_OPEN:
            self._successes += 1
            if self._successes >= self.success_threshold:
                self._state = CircuitState.CLOSED
                self._failures = 0
                self._successes = 0
                logger.info(
                    f"Circuit breaker '{self.service_name}' "
                    f"CLOSED after successful recovery"
                )
        elif self._state == CircuitState.CLOSED:
            # Reset failure count on success
            self._failures = 0

    def _on_failure(self, error: Exception) -> None:
        """Handle failed call"""
        self._failures += 1
        self._last_failure_time = time.time()

        if self._state == CircuitState.HALF_OPEN:
            # Single failure returns to OPEN
            self._state = CircuitState.OPEN
            logger.warning(
                f"Circuit breaker '{self.service_name}' "
                f"HALF_OPEN -> OPEN after test failure: {error}"
            )
        elif self._state == CircuitState.CLOSED:
            if self._failures >= self.failure_threshold:
                self._state = CircuitState.OPEN
                logger.error(
                    f"Circuit breaker '{self.service_name}' OPEN "
                    f"after {self._failures} consecutive failures"
                )

    async def call(
        self,
        func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """
        Execute function through circuit breaker.

        Args:
            func: Async function to call
            *args: Positional arguments for func
            **kwargs: Keyword arguments for func

        Returns:
            Result from func

        Raises:
            CircuitBreakerOpenError: If circuit is open
            Exception: Any exception from func (also tracked as failure)
        """
        async with self._lock:
            if not self._should_allow_request():
                retry_after = self.timeout
                if self._last_failure_time:
                    elapsed = time.time() - self._last_failure_time
                    retry_after = max(0, self.timeout - elapsed)
                raise CircuitBreakerOpenError(self.service_name, retry_after)

        try:
            # Execute the function
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)

            async with self._lock:
                self._on_success()

            return result

        except CircuitBreakerOpenError:
            raise  # Re-raise circuit breaker errors
        except Exception as e:
            async with self._lock:
                self._on_failure(e)
            raise

    def reset(self) -> None:
        """Manually reset circuit breaker to CLOSED state"""
        self._state = CircuitState.CLOSED
        self._failures = 0
        self._successes = 0
        self._last_failure_time = None
        logger.info(f"Circuit breaker '{self.service_name}' manually reset to CLOSED")

    def get_status(self) -> dict:
        """Get current circuit breaker status for monitoring"""
        return {
            "service_name": self.service_name,
            "state": self._state.value,
            "failures": self._failures,
            "failure_threshold": self.failure_threshold,
            "timeout_seconds": self.timeout,
            "last_failure_time": self._last_failure_time,
            "time_until_retry": (
                max(0, self.timeout - (time.time() - self._last_failure_time))
                if self._last_failure_time and self._state == CircuitState.OPEN
                else None
            )
        }
