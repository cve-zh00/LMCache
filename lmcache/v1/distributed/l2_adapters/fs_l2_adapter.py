# SPDX-License-Identifier: Apache-2.0
"""
File-system based L2 adapter using aiofiles or optional asyncfiles for async I/O.

Stores KV cache objects as raw tensor bytes on disk (no metadata
header).  Each ObjectKey maps to a separate ``.data`` file whose
name encodes all key fields so it can be reversed on startup.
"""

# Future
from __future__ import annotations

# Standard
from collections.abc import Coroutine
from concurrent.futures import Future
from contextlib import ExitStack
from pathlib import Path
from typing import TYPE_CHECKING, Optional, TypeVar, Union
import asyncio
import ctypes
import errno
import importlib
import os
import threading

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.distributed.internal_api import (
        L1MemoryDesc,
    )
    from lmcache.v1.memory_management import MemoryObj

# Third Party
import aiofiles
import aiofiles.os

# First Party
from lmcache.lmcache_native import Bitmap
from lmcache.logging import init_logger
from lmcache.v1.distributed.api import (
    MemoryLayoutDesc,
    MemoryResidency,
    MemorySegment,
    MemoryTags,
    ObjectKey,
)
from lmcache.v1.distributed.internal_api import L2StoreResult
from lmcache.v1.distributed.key_codec import decode_fs_filename, encode_fs_filename
from lmcache.v1.distributed.l2_adapters.base import (
    L2AdapterInterface,
    L2TaskId,
)
from lmcache.v1.distributed.l2_adapters.config import (
    L2AdapterConfigBase,
    register_l2_adapter_type,
)
from lmcache.v1.distributed.l2_adapters.factory import (
    register_l2_adapter_factory,
)
from lmcache.v1.platform import create_event_notifier

logger = init_logger(__name__)

# errnos with which a filesystem or device refuses O_DIRECT, at open() or
# at the transfer itself. Any of these means "use buffered I/O instead",
# not "this key is lost". (ENOTSUP equals EOPNOTSUPP on Linux.)
_DIRECT_IO_REFUSED_ERRNOS = frozenset({errno.EINVAL, errno.EOPNOTSUPP})
_T = TypeVar("_T")


class _FSIOWorker:
    """Own an I/O thread; optional-provider adapters share one reference-counted worker.

    The released asyncfiles libuv backend uses a process-global native loop.
    All its LMCache callbacks must therefore run on one asyncio thread.
    """

    _condition = threading.Condition()
    _asyncfiles_worker: _FSIOWorker | None = None

    def __init__(self) -> None:
        self.loop = asyncio.new_event_loop()
        self._users = 1  # Guarded by _condition, including during final shutdown.
        try:
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()
        except BaseException:
            self.loop.close()
            raise

    @classmethod
    def acquire_asyncfiles(cls) -> _FSIOWorker:
        """Acquire the shared worker, waiting for any retiring worker to exit."""
        with cls._condition:
            while cls._asyncfiles_worker is not None:
                if cls._asyncfiles_worker._users:
                    cls._asyncfiles_worker._users += 1
                    return cls._asyncfiles_worker
                cls._condition.wait()
            cls._asyncfiles_worker = cls()
            return cls._asyncfiles_worker

    def release(self) -> bool:
        """Drop one owner; stop only the final owner's worker, without self-joining."""
        with self._condition:
            self._users -= 1
            if self._users:
                return False
            self.loop.call_soon_threadsafe(self.loop.stop)
        if threading.current_thread() is not self.thread:
            self.thread.join()
        return True

    def _run(self) -> None:
        asyncio.set_event_loop(self.loop)
        try:
            self.loop.run_forever()
        finally:
            try:
                self.loop.close()
            finally:
                with self._condition:
                    if type(self)._asyncfiles_worker is self:
                        type(self)._asyncfiles_worker = None
                        self._condition.notify_all()


def _device_logical_block_size(path: Path) -> int:
    """Return the logical block size of the block device holding *path*.

    O_DIRECT alignment is set by the device's logical block size, not by
    ``statvfs().f_bsize``, which is the filesystem's preferred I/O size and
    is often larger (4096 on a disk whose logical block size is 512).

    Returns 0 when it cannot be determined, for example on a filesystem
    without a backing block device or on a platform without sysfs.
    """
    try:
        st = os.stat(path)
    except OSError:
        return 0
    dev = Path(f"/sys/dev/block/{os.major(st.st_dev)}:{os.minor(st.st_dev)}")
    # A partition has no queue/ of its own; its parent disk does.
    for queue in (dev / "queue", dev / ".." / "queue"):
        try:
            return int((queue / "logical_block_size").read_text())
        except (OSError, ValueError):
            continue
    return 0


def _buffer_is_aligned(
    buf: Union[bytearray, memoryview, bytes],
    align: int,
) -> bool:
    """Whether *buf*'s start address is a multiple of *align*.

    O_DIRECT needs the buffer address aligned as well as the offset and
    length. A read-only or empty buffer exposes no address to check, so it
    counts as unaligned and takes the buffered path.
    """
    if align <= 0:
        return False
    try:
        ptr = ctypes.addressof((ctypes.c_byte * 1).from_buffer(buf))
    except (TypeError, ValueError, BufferError):
        return False
    return ptr % align == 0


def _write_full(fd: int, buf: Union[bytearray, memoryview, bytes]) -> None:
    """Loop ``os.write`` until all of *buf* is written.

    A single ``os.write`` may write fewer bytes than asked for. Ignoring
    that would publish a truncated file under the key's final name.
    """
    mv = memoryview(buf).cast("B")
    total = 0
    while total < len(mv):
        n = os.write(fd, mv[total:])
        if n <= 0:
            raise OSError(errno.EIO, "write returned no progress")
        total += n


def _readinto_full(
    f,  # typing: IO[bytes]
    buf: Union[bytearray, memoryview, bytes],
) -> int:
    """Loop readinto() until *buf* is full or EOF.

    A single ``readinto()`` may return fewer bytes than
    *len(buf)* even when more data is available.  This
    helper keeps reading until the buffer is completely
    filled or the file reaches EOF.

    Returns:
        Total number of bytes read.
    """
    mv = memoryview(buf) if not isinstance(buf, memoryview) else buf
    total = 0
    while total < len(mv):
        n = f.readinto(mv[total:])
        if n is None or n == 0:
            break
        total += n
    return total


async def _async_readinto_full(
    f,  # selected provider's async file handle
    buf: Union[bytearray, memoryview, bytes],
) -> int:
    """Async version of :func:`_readinto_full`."""
    mv = memoryview(buf) if not isinstance(buf, memoryview) else buf
    total = 0
    while total < len(mv):
        n = await f.readinto(mv[total:])
        if n is None or n == 0:
            break
        total += n
    return total


def _object_key_to_filename(key: ObjectKey) -> str:
    """Return the filename shared by Python and native FS."""
    return encode_fs_filename(key)


def _filename_to_object_key(
    filename: str,
) -> Optional[ObjectKey]:
    """Recover an ObjectKey, or return None for a foreign/invalid filename."""
    try:
        return decode_fs_filename(filename)
    except ValueError:
        return None


class FSL2AdapterConfig(L2AdapterConfigBase):
    """
    Config for the filesystem-backed L2 adapter.

    Fields:
    - base_path: directory for storing KV cache files.
    - relative_tmp_dir: optional relative sub-dir for
      temp files (same as fs_connector_relative_tmp_dir).
    - io_backend: aiofiles (default) or optional asyncfiles.
    """

    def __init__(
        self,
        base_path: str,
        relative_tmp_dir: Optional[str] = None,
        read_ahead_size: Optional[int] = None,
        use_odirect: bool = False,
        io_backend: str = "aiofiles",
    ) -> None:
        """Initialize FSL2AdapterConfig.

        Args:
            base_path: Directory for storing KV cache files.
            relative_tmp_dir: Relative sub-dir under
                base_path for temp files during writes.
            read_ahead_size: If set, trigger filesystem
                readahead by issuing a small initial read
                of this many bytes before reading the rest.
            use_odirect: If True, bypass the OS page cache
                using O_DIRECT for both reads and writes.
                Requires buffer sizes aligned to the
                filesystem block size.
            io_backend: Async I/O provider, "aiofiles" (default) or
                "asyncfiles" (requires pip install py-asyncfiles).

        Raises:
            ValueError: If io_backend is not a supported provider string.
        """
        if not isinstance(io_backend, str) or io_backend not in (
            "aiofiles",
            "asyncfiles",
        ):
            raise ValueError("io_backend must be 'aiofiles' or 'asyncfiles'")
        self.base_path = base_path
        self.relative_tmp_dir = relative_tmp_dir
        self.read_ahead_size = read_ahead_size
        self.use_odirect = use_odirect
        self.io_backend = io_backend

    @classmethod
    def from_dict(cls, d: dict) -> "FSL2AdapterConfig":
        base_path = d.get("base_path")
        if not isinstance(base_path, str) or not base_path:
            raise ValueError("base_path must be a non-empty string")
        relative_tmp_dir = d.get("relative_tmp_dir", None)
        if relative_tmp_dir is not None:
            if not isinstance(relative_tmp_dir, str):
                raise ValueError("relative_tmp_dir must be a string")
        read_ahead_size = d.get("read_ahead_size", None)
        if read_ahead_size is not None:
            if not isinstance(read_ahead_size, int) or read_ahead_size <= 0:
                raise ValueError("read_ahead_size must be a positive integer")
        use_odirect = d.get("use_odirect", False)
        if not isinstance(use_odirect, bool):
            raise ValueError("use_odirect must be a boolean")
        return cls(
            base_path=base_path,
            relative_tmp_dir=relative_tmp_dir,
            read_ahead_size=read_ahead_size,
            use_odirect=use_odirect,
            io_backend=d.get("io_backend", "aiofiles"),
        )

    @classmethod
    def help(cls) -> str:
        return (
            "FS L2 adapter config fields:\n"
            "- base_path (str): directory for KV cache "
            "files (required)\n"
            "- relative_tmp_dir (str): relative sub-dir "
            "for temp files (optional, same as "
            "fs_connector_relative_tmp_dir)\n"
            "- read_ahead_size (int): trigger fs "
            "readahead by reading this many bytes first "
            "(optional)\n"
            "- use_odirect (bool): bypass page cache "
            "via O_DIRECT (optional, default false)\n"
            "- io_backend (str): aiofiles or asyncfiles "
            "(optional, default aiofiles; asyncfiles requires "
            "pip install py-asyncfiles)"
        )


class FSL2Adapter(L2AdapterInterface):
    """
    File-system backed L2 adapter with per-instance async I/O providers.

    Uses aiofiles by default; asyncfiles is imported only when selected.

    Each file stores **only** the raw tensor bytes (no metadata
    header), which gives maximum I/O throughput.  The file name
    itself encodes the full ``ObjectKey`` so it is reversible.

    Thread safety is ensured via a lock for shared bookkeeping. aiofiles
    instances own separate workers; asyncfiles instances share one worker
    to preserve native callback thread affinity.
    """

    _DELETE_CONCURRENCY = 64

    def __init__(self, config: FSL2AdapterConfig) -> None:
        """Select the I/O provider before allocating background resources.

        Args:
            config: Filesystem paths, tuning options, and I/O provider.

        Raises:
            ImportError: If the selected asyncfiles provider cannot be imported.
        """
        if config.io_backend == "asyncfiles":
            try:
                self._io_lib = importlib.import_module("asyncfiles")
            except ModuleNotFoundError as exc:
                if exc.name != "asyncfiles":
                    raise
                raise ImportError(
                    "io_backend='asyncfiles' requires py-asyncfiles; "
                    "pip install py-asyncfiles"
                ) from exc
            self._io_os = importlib.import_module("asyncfiles.os")
        else:
            self._io_lib = aiofiles
            self._io_os = aiofiles.os

        super().__init__()
        self._config = config
        base = config.base_path
        self._base_path = Path(base)
        self._base_path.mkdir(parents=True, exist_ok=True)

        # Temp-file strategy aligned with FSConnector:
        # if relative_tmp_dir is set, write to a sub-dir;
        # otherwise fall back to a .tmp suffix.
        self._relative_tmp_dir: Optional[Path] = None
        if config.relative_tmp_dir is not None:
            self._relative_tmp_dir = Path(config.relative_tmp_dir)
            if (
                self._relative_tmp_dir.is_absolute()
                or ".." in self._relative_tmp_dir.parts
            ):
                raise ValueError("Invalid relative_tmp_dir: " + config.relative_tmp_dir)
            (self._base_path / self._relative_tmp_dir).mkdir(
                parents=False, exist_ok=True
            )

        # I/O tuning options aligned with FSConnector
        self._read_ahead_size = config.read_ahead_size
        self._use_odirect = config.use_odirect
        self._os_disk_bs = 0
        # O_DIRECT fallbacks repeat for every object of a misaligned or
        # refusing deployment, so each kind is logged once per adapter.
        self._warned_unaligned = False
        self._warned_refused = False
        if self._use_odirect:
            self._os_disk_bs = (
                _device_logical_block_size(self._base_path)
                or os.statvfs(self._base_path).f_bsize
            )

        # Task bookkeeping
        self._next_task_id: L2TaskId = 0
        self._completed_store_tasks: dict[L2TaskId, L2StoreResult] = {}
        self._completed_lookup_tasks: dict[L2TaskId, Bitmap] = {}
        self._completed_load_tasks: dict[L2TaskId, Bitmap] = {}
        self._lock = threading.Lock()
        # Only the owning worker accesses the task set.
        self._pending_tasks: set[asyncio.Task[object]] = set()
        self._close_future: Future[bool] | None = None

        with ExitStack() as cleanup:
            self._store_efd = create_event_notifier()
            cleanup.callback(self._store_efd.close)
            self._lookup_efd = create_event_notifier()
            cleanup.callback(self._lookup_efd.close)
            self._load_efd = create_event_notifier()
            cleanup.callback(self._load_efd.close)
            self._worker = (
                _FSIOWorker.acquire_asyncfiles()
                if config.io_backend == "asyncfiles"
                else _FSIOWorker()
            )
            cleanup.callback(self._worker.release)
            self._loop = self._worker.loop
            self._loop_thread = self._worker.thread

            logger.info(
                "Initialized FSL2Adapter with base_path=%s, "
                "relative_tmp_dir=%s, "
                "read_ahead_size=%s, use_odirect=%s",
                self._base_path,
                self._relative_tmp_dir,
                self._read_ahead_size,
                self._use_odirect,
            )
            cleanup.pop_all()

    # ------------------------------------------------------------------
    # Event Fd Interface
    # ------------------------------------------------------------------

    def get_store_event_fd(self) -> int:
        return self._store_efd.fileno()

    def get_lookup_and_lock_event_fd(self) -> int:
        return self._lookup_efd.fileno()

    def get_load_event_fd(self) -> int:
        return self._load_efd.fileno()

    # ------------------------------------------------------------------
    # Store Interface
    # ------------------------------------------------------------------

    def submit_store_task(
        self,
        keys: list[ObjectKey],
        objects: list[MemoryObj],
    ) -> L2TaskId:
        with self._lock:
            task_id = self._get_next_task_id()

        self._submit(self._execute_store(keys, objects, task_id))
        return task_id

    def pop_completed_store_tasks(
        self,
    ) -> dict[L2TaskId, L2StoreResult]:
        """Pop all completed store tasks.

        Returns:
            dict[L2TaskId, L2StoreResult]: a dictionary mapping the task
            id to an ``L2StoreResult`` that encodes both the success flag
            and the bytes actually transferred.
        """
        with self._lock:
            completed = self._completed_store_tasks
            self._completed_store_tasks = {}
        return completed

    # ------------------------------------------------------------------
    # Lookup and Lock Interface
    # ------------------------------------------------------------------

    def submit_lookup_and_lock_task(
        self, keys: list[ObjectKey], group_layout_descs: dict[int, MemoryLayoutDesc]
    ) -> L2TaskId:
        with self._lock:
            task_id = self._get_next_task_id()

        self._submit(self._execute_lookup(keys, task_id))
        return task_id

    def query_lookup_and_lock_result(self, task_id: L2TaskId) -> Bitmap | None:
        with self._lock:
            return self._completed_lookup_tasks.pop(task_id, None)

    def submit_unlock(self, keys: list[ObjectKey]) -> None:
        # No-op: the FS adapter tracks no per-key locks — a delete
        # racing the lookup→load window degrades the load to a miss.
        pass

    # ------------------------------------------------------------------
    # Load Interface
    # ------------------------------------------------------------------

    def submit_load_task(
        self,
        keys: list[ObjectKey],
        objects: list[MemoryObj],
    ) -> L2TaskId:
        with self._lock:
            task_id = self._get_next_task_id()

        self._submit(self._execute_load(keys, objects, task_id))
        return task_id

    def query_load_result(self, task_id: L2TaskId) -> Bitmap | None:
        with self._lock:
            return self._completed_load_tasks.pop(task_id, None)

    # ------------------------------------------------------------------
    # Status Interface
    # ------------------------------------------------------------------

    def report_status(self) -> dict:
        """Return a status dict for the FS L2 adapter."""
        with self._lock:
            healthy = self._close_future is None and self._loop_thread.is_alive()
        return {
            "is_healthy": healthy,
            "type": "FSL2Adapter",
            "base_path": str(self._base_path),
            "use_odirect": self._use_odirect,
            "event_loop_alive": self._loop_thread.is_alive(),
        }

    # ------------------------------------------------------------------
    # Eviction Interface
    # ------------------------------------------------------------------

    def delete(self, keys: list[ObjectKey]) -> None:
        """Delete each key's backing file and notify listeners.

        Args:
            keys: The object keys to delete.

        Note:
            No per-key locks: a delete racing a load turns that load
            into a miss; racing a store of the same key may leave the
            key re-stored.
        """
        if not keys:
            return
        try:
            fut = self._submit(self._execute_delete(keys))
            deleted_keys, deleted_sizes = fut.result(timeout=30.0)
        except Exception as e:
            logger.warning("FSL2Adapter delete failed: %s", e)
            return
        if deleted_keys:
            self._notify_keys_deleted(deleted_keys, deleted_sizes)

    # ``get_usage()`` is inherited from ``L2AdapterInterface``. The base
    # class maintains byte totals via ``_notify_keys_stored`` /
    # ``_notify_keys_deleted``, but the FS adapter declares no max capacity
    # (default 0) so ``supports_global_eviction`` returns ``False`` and
    # ``usage_fraction == -1.0`` — the eviction controller treats this as
    # "no eviction signal" and skips the adapter entirely.

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def get_memory_tags(self) -> MemoryTags:
        """Return local storage tags; remote when the adapter is shared."""
        residency = (
            MemoryResidency.REMOTE if self._config.shared else MemoryResidency.LOCAL
        )
        return MemoryTags(MemorySegment.STORAGE, residency)

    def close(self) -> None:
        """Cancel this instance's tasks and release its worker reference.

        Calls from the I/O thread schedule cleanup without blocking that thread.
        Other callers wait for cleanup and, for the last owner, thread shutdown.

        Raises:
            TimeoutError: If cancellation does not finish within five seconds.
        """
        with self._lock:
            if self._close_future is None:
                self._close_future = asyncio.run_coroutine_threadsafe(
                    self._shutdown(), self._loop
                )
            future = self._close_future
        if threading.current_thread() is self._loop_thread:
            return
        if future.result(timeout=5):
            self._loop_thread.join()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _submit(self, operation: Coroutine[object, object, _T]) -> Future[_T]:
        with self._lock:
            if self._close_future is not None:
                operation.close()
                raise RuntimeError("FSL2Adapter is closed")
            return asyncio.run_coroutine_threadsafe(
                self._run_task(operation), self._loop
            )

    async def _run_task(self, operation: Coroutine[object, object, _T]) -> _T:
        task = asyncio.current_task()
        assert task is not None
        self._pending_tasks.add(task)
        try:
            return await operation
        finally:
            self._pending_tasks.remove(task)

    async def _shutdown(self) -> bool:
        tasks = list(self._pending_tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._store_efd.close()
        self._lookup_efd.close()
        self._load_efd.close()
        logger.info("FSL2Adapter closed")
        return self._worker.release()

    def _get_next_task_id(self) -> L2TaskId:
        tid = self._next_task_id
        self._next_task_id += 1
        return tid

    def _key_to_path(self, key: ObjectKey) -> Path:
        return self._base_path / _object_key_to_filename(key)

    async def _key_exists_on_disk(
        self,
        key: ObjectKey,
    ) -> bool:
        """Check whether the file for *key* exists on disk.

        Uses the selected provider's async ``os.path.exists`` so the check is
        non-blocking and always reflects the real FS state,
        which is critical for multi-node shared-FS setups.
        """
        path = self._key_to_path(key)
        return await self._io_os.path.exists(path)

    def _key_to_file_and_tmp_path(self, key: ObjectKey) -> tuple[Path, Path]:
        """Return ``(final_path, tmp_path)``.

        When ``relative_tmp_dir`` is configured, the temp file
        is placed under that sub-directory (same behaviour as
        ``FSConnector._get_file_and_tmp_path``).  Otherwise a
        ``.tmp`` suffix is used.
        """
        fname = _object_key_to_filename(key)
        final = self._base_path / fname
        if self._relative_tmp_dir is not None:
            tmp = self._base_path / self._relative_tmp_dir / fname
        else:
            tmp = final.with_suffix(".tmp")
        return final, tmp

    # ---- O_DIRECT helpers -----------------------------------------------

    def _odirect_usable(
        self, file_path: Path, buf: Union[bytearray, memoryview, bytes]
    ) -> bool:
        """Whether *buf* meets O_DIRECT's length and address alignment."""
        bs = self._os_disk_bs
        if bs > 0 and len(buf) % bs == 0 and _buffer_is_aligned(buf, bs):
            return True
        if not self._warned_unaligned:
            self._warned_unaligned = True
            logger.warning(
                "Cannot use O_DIRECT for %s: buffer of %d bytes is not aligned "
                "to block size %d, using buffered I/O. Further unaligned "
                "buffers will not be logged.",
                file_path.name,
                len(buf),
                bs,
            )
        return False

    def _warn_refused(self, op: str, file_path: Path, e: OSError) -> None:
        if self._warned_refused:
            return
        self._warned_refused = True
        logger.warning(
            "O_DIRECT %s of %s refused (%s), using buffered I/O. Further "
            "refusals will not be logged.",
            op,
            file_path.name,
            errno.errorcode.get(e.errno or 0, e.errno),
        )

    def _read_with_odirect(
        self,
        file_path: Path,
        dst_buf: Union[bytearray, memoryview, bytes],
    ) -> int:
        """Synchronous O_DIRECT read into *dst_buf*.

        Falls back to a buffered read when *dst_buf* is not aligned, or when
        the filesystem refuses O_DIRECT. Any other error, including a missing
        file, propagates to the caller, so a failed read is not mistaken for
        a short one.

        Returns the number of bytes actually read.
        Runs in an executor (not on the event loop).
        """
        if self._odirect_usable(file_path, dst_buf):
            try:
                fd = os.open(
                    str(file_path),
                    os.O_RDONLY | getattr(os, "O_DIRECT", 0),
                )
                try:
                    fdo = os.fdopen(fd, "rb", buffering=0)
                except BaseException:
                    os.close(fd)
                    raise
                with fdo:
                    return _readinto_full(fdo, dst_buf)
            except OSError as e:
                if e.errno not in _DIRECT_IO_REFUSED_ERRNOS:
                    raise
                self._warn_refused("read", file_path, e)
        with open(file_path, "rb") as f:
            return _readinto_full(f, dst_buf)

    def _write_with_odirect(
        self, file_path: Path, buf: Union[bytearray, memoryview, bytes]
    ) -> None:
        """Synchronous O_DIRECT write of *buf*.

        Falls back to a buffered write when *buf* is not aligned, or when
        the filesystem refuses O_DIRECT. The buffered write starts over, so a
        partial O_DIRECT write never survives. Any other error propagates.

        Runs in an executor (not on the event loop).
        """
        if self._odirect_usable(file_path, buf):
            try:
                fd = os.open(
                    str(file_path),
                    os.O_CREAT | os.O_WRONLY | os.O_TRUNC | getattr(os, "O_DIRECT", 0),
                    0o644,
                )
                try:
                    _write_full(fd, buf)
                finally:
                    os.close(fd)
                return
            except OSError as e:
                if e.errno not in _DIRECT_IO_REFUSED_ERRNOS:
                    raise
                self._warn_refused("write", file_path, e)
        with open(file_path, "wb") as f:
            f.write(buf)

    # ---- store ----------------------------------------------------------

    async def _execute_store(
        self,
        keys: list[ObjectKey],
        objects: list[MemoryObj],
        task_id: L2TaskId,
    ) -> None:
        success = True
        bytes_written = 0
        stored_keys: list[ObjectKey] = []
        stored_sizes: list[int] = []
        try:
            for key, obj in zip(keys, objects, strict=True):
                file_path, tmp_path = self._key_to_file_and_tmp_path(key)

                # Skip if already stored on disk
                if await self._io_os.path.exists(file_path):
                    continue
                buf = obj.byte_array
                size = len(buf)

                try:
                    if self._use_odirect:
                        await self._loop.run_in_executor(
                            None,
                            self._write_with_odirect,
                            tmp_path,
                            buf,
                        )
                    else:
                        async with self._io_lib.open(tmp_path, "wb") as f:
                            written = await f.write(buf)
                            if written != size:
                                raise OSError(
                                    errno.EIO,
                                    f"Incomplete write: expected {size}, got {written}",
                                )

                    await self._io_os.replace(tmp_path, file_path)
                    bytes_written += size
                    stored_keys.append(key)
                    stored_sizes.append(size)
                    logger.debug(
                        "FSL2Adapter stored key %s (%d bytes)",
                        file_path.name,
                        size,
                    )
                except Exception:
                    logger.exception(
                        "FSL2Adapter failed to store %s",
                        file_path,
                    )
                    if await self._io_os.path.exists(tmp_path):
                        await self._io_os.unlink(tmp_path)
                    success = False
        except Exception:
            logger.exception(
                "FSL2Adapter store task %s failed",
                task_id,
            )
            success = False

        if stored_keys:
            self._notify_keys_stored(stored_keys, stored_sizes)

        with self._lock:
            self._completed_store_tasks[task_id] = L2StoreResult(success, bytes_written)
        self._store_efd.notify()

    # ---- lookup ---------------------------------------------------------

    async def _execute_lookup(
        self,
        keys: list[ObjectKey],
        task_id: L2TaskId,
    ) -> None:
        bitmap = Bitmap(len(keys))
        for i, key in enumerate(keys):
            if not await self._key_exists_on_disk(key):
                continue
            bitmap.set(i)

        with self._lock:
            self._completed_lookup_tasks[task_id] = bitmap
        self._lookup_efd.notify()

    # ---- load -----------------------------------------------------------

    async def _execute_load(
        self,
        keys: list[ObjectKey],
        objects: list[MemoryObj],
        task_id: L2TaskId,
    ) -> None:
        bitmap = Bitmap(len(keys))
        for i, key in enumerate(keys):
            file_path = self._key_to_path(key)
            try:
                dst_buf: Union[bytearray, memoryview, bytes] = objects[i].byte_array
                expected = len(dst_buf)
                num_read: Optional[int] = None

                # O_DIRECT path (sync, via executor)
                if self._use_odirect:
                    num_read = await self._loop.run_in_executor(
                        None,
                        self._read_with_odirect,
                        file_path,
                        dst_buf,
                    )
                    if num_read != expected:
                        logger.warning(
                            "Incomplete O_DIRECT read for %s: expected %d, got %d",
                            file_path.name,
                            expected,
                            num_read or 0,
                        )
                    else:
                        bitmap.set(i)
                        logger.debug(
                            "FSL2Adapter loaded key %s (%d bytes, O_DIRECT)",
                            file_path.name,
                            num_read,
                        )
                    continue

                # Standard async path with optional
                # read-ahead
                expected = len(dst_buf)
                async with self._io_lib.open(file_path, "rb") as f:
                    if self._read_ahead_size is None:
                        num_read = await _async_readinto_full(f, dst_buf)
                    else:
                        if not isinstance(dst_buf, memoryview):
                            dst_buf = memoryview(dst_buf)
                        # Trigger readahead with a
                        # small initial read
                        ra = self._read_ahead_size
                        n_head = await _async_readinto_full(f, dst_buf[:ra])
                        if n_head == ra:
                            n_tail = await _async_readinto_full(f, dst_buf[ra:])
                            num_read = n_head + n_tail
                        else:
                            num_read = n_head

                    if num_read != expected:
                        logger.warning(
                            "Incomplete read for %s: expected %d, got %d",
                            file_path.name,
                            expected,
                            num_read,
                        )
                        continue

                    bitmap.set(i)
                    logger.debug(
                        "FSL2Adapter loaded key %s (%d bytes)",
                        file_path.name,
                        num_read,
                    )
            except FileNotFoundError:
                continue
            except Exception:
                logger.exception(
                    "FSL2Adapter failed to load %s",
                    file_path,
                )
                continue

        loaded_keys = [keys[i] for i in bitmap.get_indices_list()]
        if loaded_keys:
            self._notify_keys_accessed(loaded_keys)

        with self._lock:
            self._completed_load_tasks[task_id] = bitmap
        self._load_efd.notify()

    # ---- delete ---------------------------------------------------------

    async def _execute_delete(
        self, keys: list[ObjectKey]
    ) -> tuple[list[ObjectKey], list[int]]:
        """Unlink each key's file concurrently; return removed keys + sizes."""
        sem = asyncio.Semaphore(self._DELETE_CONCURRENCY)

        async def _delete_one(key: ObjectKey) -> tuple[ObjectKey, int] | None:
            file_path = self._key_to_path(key)
            async with sem:
                try:
                    size = (await self._io_os.stat(file_path)).st_size
                    await self._io_os.unlink(file_path)
                except FileNotFoundError:
                    return None
                except Exception:
                    logger.exception(
                        "FSL2Adapter failed to delete %s",
                        file_path,
                    )
                    return None
            return key, size

        results = await asyncio.gather(*(_delete_one(k) for k in keys))

        deleted_keys: list[ObjectKey] = []
        deleted_sizes: list[int] = []
        for res in results:
            if res is not None:
                key, size = res
                deleted_keys.append(key)
                deleted_sizes.append(size)
        return deleted_keys, deleted_sizes


# Self-register config type and adapter factory
register_l2_adapter_type("fs", FSL2AdapterConfig)


def _create_fs_adapter(
    config: L2AdapterConfigBase,
    l1_memory_desc: "Optional[L1MemoryDesc]" = None,
) -> L2AdapterInterface:
    """Create an FSL2Adapter from config."""
    return FSL2Adapter(config)  # type: ignore[arg-type]


register_l2_adapter_factory("fs", _create_fs_adapter)
