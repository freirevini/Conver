"""
Async Utilities for FastAPI Application.

Provides:
- Async file operations
- Concurrent task execution
- Async context managers
- Background task helpers
"""
import asyncio
import aiofiles
import logging
from pathlib import Path
from typing import List, TypeVar, Callable, Awaitable, Any, Optional
from concurrent.futures import ThreadPoolExecutor
from functools import partial

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Shared thread pool for CPU-bound operations
_executor: Optional[ThreadPoolExecutor] = None


def get_executor(max_workers: int = 4) -> ThreadPoolExecutor:
    """Get shared thread pool executor."""
    global _executor
    if _executor is None:
        _executor = ThreadPoolExecutor(max_workers=max_workers)
    return _executor


async def run_in_thread(func: Callable[..., T], *args, **kwargs) -> T:
    """
    Run a sync function in a thread pool.
    
    Useful for CPU-bound operations that would block the event loop.
    
    Args:
        func: Synchronous function to run
        *args: Positional arguments
        **kwargs: Keyword arguments
        
    Returns:
        Result of the function
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        get_executor(),
        partial(func, *args, **kwargs)
    )


async def gather_with_concurrency(
    limit: int,
    *tasks: Awaitable[T]
) -> List[T]:
    """
    Run async tasks with concurrency limit.
    
    Args:
        limit: Maximum concurrent tasks
        *tasks: Async tasks to run
        
    Returns:
        List of results in order
    """
    semaphore = asyncio.Semaphore(limit)
    
    async def limited_task(task: Awaitable[T]) -> T:
        async with semaphore:
            return await task
    
    return await asyncio.gather(
        *[limited_task(t) for t in tasks],
        return_exceptions=True
    )


async def read_file_async(path: str | Path, encoding: str = "utf-8") -> str:
    """
    Read file contents asynchronously.
    
    Args:
        path: Path to file
        encoding: File encoding
        
    Returns:
        File contents as string
    """
    async with aiofiles.open(path, mode="r", encoding=encoding) as f:
        return await f.read()


async def write_file_async(
    path: str | Path, 
    content: str, 
    encoding: str = "utf-8"
) -> None:
    """
    Write content to file asynchronously.
    
    Args:
        path: Path to file
        content: Content to write
        encoding: File encoding
    """
    async with aiofiles.open(path, mode="w", encoding=encoding) as f:
        await f.write(content)


async def read_binary_async(path: str | Path) -> bytes:
    """Read binary file asynchronously."""
    async with aiofiles.open(path, mode="rb") as f:
        return await f.read()


async def write_binary_async(path: str | Path, content: bytes) -> None:
    """Write binary content asynchronously."""
    async with aiofiles.open(path, mode="wb") as f:
        await f.write(content)


class AsyncTaskQueue:
    """
    Simple async task queue for background processing.
    
    Usage:
        queue = AsyncTaskQueue(max_concurrent=5)
        await queue.start()
        
        await queue.enqueue(my_async_task())
        
        await queue.stop()
    """
    
    def __init__(self, max_concurrent: int = 5):
        self.max_concurrent = max_concurrent
        self._queue: asyncio.Queue = asyncio.Queue()
        self._workers: List[asyncio.Task] = []
        self._running = False
    
    async def start(self):
        """Start worker tasks."""
        self._running = True
        for i in range(self.max_concurrent):
            worker = asyncio.create_task(self._worker(f"worker-{i}"))
            self._workers.append(worker)
    
    async def stop(self, timeout: float = 30.0):
        """Stop all workers gracefully."""
        self._running = False
        
        # Wait for queue to empty
        try:
            await asyncio.wait_for(self._queue.join(), timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning("Task queue stop timeout, cancelling workers")
        
        # Cancel workers
        for worker in self._workers:
            worker.cancel()
        
        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()
    
    async def enqueue(self, coro: Awaitable[Any]) -> None:
        """Add a coroutine to the queue."""
        await self._queue.put(coro)
    
    async def _worker(self, name: str):
        """Worker that processes tasks from queue."""
        while self._running:
            try:
                coro = await asyncio.wait_for(
                    self._queue.get(), 
                    timeout=1.0
                )
                try:
                    await coro
                except Exception as e:
                    logger.error(f"[{name}] Task error: {e}")
                finally:
                    self._queue.task_done()
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break


async def timeout_wrapper(
    coro: Awaitable[T],
    timeout_seconds: float,
    default: Optional[T] = None
) -> Optional[T]:
    """
    Wrap a coroutine with a timeout.
    
    Args:
        coro: Coroutine to run
        timeout_seconds: Maximum execution time
        default: Value to return on timeout
        
    Returns:
        Result or default value
    """
    try:
        return await asyncio.wait_for(coro, timeout=timeout_seconds)
    except asyncio.TimeoutError:
        logger.warning(f"Operation timed out after {timeout_seconds}s")
        return default


def create_background_task(coro: Awaitable[Any], name: str = "bg-task") -> asyncio.Task:
    """
    Create a background task with error logging.
    
    Args:
        coro: Coroutine to run
        name: Task name for logging
        
    Returns:
        Created task
    """
    async def wrapper():
        try:
            return await coro
        except Exception as e:
            logger.error(f"Background task '{name}' failed: {e}")
            raise
    
    return asyncio.create_task(wrapper(), name=name)
