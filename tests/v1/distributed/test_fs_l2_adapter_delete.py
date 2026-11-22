# SPDX-License-Identifier: Apache-2.0
"""Tests for FSL2Adapter delete and listener notifications.

``delete`` unlinks each key's backing file and fires
``on_l2_keys_deleted``; store/load fire ``on_l2_keys_stored`` /
``on_l2_keys_accessed``. Together these drive the base class's byte
accounting and feed the coordinator's cache-event stream.
"""

# Standard
from collections.abc import AsyncIterator, Iterator
from contextlib import ExitStack, asynccontextmanager, closing
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, patch
import asyncio
import importlib
import importlib.util
import os
import subprocess
import sys
import threading
import time

# Third Party
import aiofiles
import pytest

# First Party
from lmcache.lmcache_native import Bitmap
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.internal_api import L2AdapterListener
from lmcache.v1.distributed.l2_adapters import fs_l2_adapter
from lmcache.v1.distributed.l2_adapters.fs_l2_adapter import (
    FSL2Adapter,
    FSL2AdapterConfig,
)
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.platform.event_notifier import EventNotifier


class _RecordingListener(L2AdapterListener):
    """Capture callbacks through the L2 adapter listener interface."""

    def __init__(self) -> None:
        self.stored: list[tuple[ObjectKey, int]] = []
        self.accessed: list[ObjectKey] = []
        self.deleted: list[ObjectKey] = []

    def on_l2_keys_stored(self, keys: list[ObjectKey], sizes: list[int]) -> None:
        self.stored.extend(zip(keys, sizes, strict=True))

    def on_l2_keys_accessed(self, keys: list[ObjectKey]) -> None:
        self.accessed.extend(keys)

    def on_l2_keys_deleted(self, keys: list[ObjectKey]) -> None:
        self.deleted.extend(keys)


class _Buf:
    """Minimal MemoryObj stand-in: just the ``byte_array`` the FS
    adapter's store/load paths read and write."""

    def __init__(self, data: bytes) -> None:
        self._data = bytearray(data)

    @property
    def byte_array(self) -> memoryview:
        return memoryview(self._data)


# (adapter, listener) pair produced by the ``adapter`` fixture.
AdapterFixture = tuple[FSL2Adapter, _RecordingListener]


def _key(h: bytes = b"\xde\xad\xbe\xef", salt: str = "alice") -> ObjectKey:
    return ObjectKey(
        chunk_hash=h,
        model_name="llama",
        kv_rank=42,
        cache_salt=salt,
    )


@pytest.fixture(params=["aiofiles", "asyncfiles"])
def io_backend(request: pytest.FixtureRequest) -> str:
    # Only an absent optional package is skippable, not a broken native install.
    if request.param == "asyncfiles" and importlib.util.find_spec("asyncfiles") is None:
        pytest.importorskip("asyncfiles")
    return cast(str, request.param)


@pytest.fixture
def adapter(
    tmp_path: Path,
    io_backend: str,
) -> Iterator[AdapterFixture]:
    adp = FSL2Adapter(FSL2AdapterConfig(base_path=str(tmp_path), io_backend=io_backend))
    listener = _RecordingListener()
    adp.register_listener(listener)
    try:
        yield adp, listener
    finally:
        adp.close()


def _bufs(payloads: list[bytes]) -> list[MemoryObj]:
    """Wrap raw payloads as MemoryObj-shaped buffers (see ``_Buf``)."""
    return cast("list[MemoryObj]", [_Buf(p) for p in payloads])


def _lookup_and_wait(adp: FSL2Adapter, keys: list[ObjectKey]) -> list[bool]:
    """Submit a lookup and poll until its hit bitmap is available."""
    task_id = adp.submit_lookup_and_lock_task(
        keys, {0: MemoryLayoutDesc(shapes=[], dtypes=[])}
    )
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        bitmap = adp.query_lookup_and_lock_result(task_id)
        if bitmap is not None:
            return [bitmap.test(i) for i in range(len(keys))]
        time.sleep(0.01)
    raise AssertionError("lookup task did not complete within 5s")


def _store_and_wait(
    adp: FSL2Adapter, keys: list[ObjectKey], payloads: list[bytes]
) -> None:
    """Submit a store and poll until its result is available."""
    task_id = adp.submit_store_task(keys, _bufs(payloads))
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        completed = adp.pop_completed_store_tasks()
        if task_id in completed:
            # L2StoreResult is an int: >= 0 success, -1 failure.
            assert int(completed[task_id]) >= 0
            return
        time.sleep(0.01)
    pytest.fail("store task did not complete within 5s")


class TestIOBackendConfig:
    def test_default_preserves_positional_arguments(self) -> None:
        config = FSL2AdapterConfig("/cache", "tmp", 4096, True)
        assert config.io_backend == "aiofiles"
        assert config.relative_tmp_dir == "tmp"
        assert config.read_ahead_size == 4096
        assert config.use_odirect is True
        assert FSL2AdapterConfig.from_dict({"base_path": "/cache"}).io_backend == (
            "aiofiles"
        )

    @pytest.mark.parametrize("backend", ["aiofiles", "asyncfiles"])
    def test_from_dict_and_help(self, backend: str) -> None:
        config = FSL2AdapterConfig.from_dict(
            {
                "base_path": "/cache",
                "relative_tmp_dir": "tmp",
                "read_ahead_size": 4096,
                "use_odirect": True,
                "io_backend": backend,
            }
        )
        assert config.io_backend == backend
        assert config.base_path == "/cache"
        assert config.relative_tmp_dir == "tmp"
        assert config.read_ahead_size == 4096
        assert config.use_odirect is True
        assert backend in FSL2AdapterConfig.help()
        assert "default aiofiles" in FSL2AdapterConfig.help()

    @pytest.mark.parametrize(
        "backend", ["", "AIOFILES", "os", None, 0, False, [], {}, b"aiofiles"]
    )
    def test_invalid_backend(self, backend: object) -> None:
        with pytest.raises(ValueError, match="io_backend"):
            FSL2AdapterConfig("/cache", io_backend=cast(str, backend))
        with pytest.raises(ValueError, match="io_backend"):
            FSL2AdapterConfig.from_dict({"base_path": "/cache", "io_backend": backend})


class TestIOBackendImports:
    def test_missing_optional_dependency(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        real_import = importlib.import_module
        optional_imports: list[str] = []

        def without_asyncfiles(name: str, package: str | None = None) -> ModuleType:
            if name == "asyncfiles" or name.startswith("asyncfiles."):
                optional_imports.append(name)
                raise ModuleNotFoundError(
                    "No module named 'asyncfiles'", name="asyncfiles"
                )
            return real_import(name, package)

        monkeypatch.setattr(importlib, "import_module", without_asyncfiles)
        with closing(FSL2Adapter(FSL2AdapterConfig(str(tmp_path)))) as default:
            key = _key()
            _store_and_wait(default, [key], [b"default remains usable"])
            assert _lookup_and_wait(default, [key]) == [True]
            default.delete([key])
            assert _lookup_and_wait(default, [key]) == [False]
        assert optional_imports == []

        with (
            patch.object(fs_l2_adapter, "create_event_notifier") as notifier,
            patch.object(fs_l2_adapter.asyncio, "new_event_loop") as new_loop,
            patch.object(fs_l2_adapter.threading, "Thread") as thread,
        ):
            with pytest.raises(ImportError, match="pip install py-asyncfiles"):
                FSL2Adapter(FSL2AdapterConfig(str(tmp_path), io_backend="asyncfiles"))
            notifier.assert_not_called()
            new_loop.assert_not_called()
            thread.assert_not_called()
        assert optional_imports == ["asyncfiles"]

    @pytest.mark.parametrize(
        ("failing_module", "error"),
        [
            ("asyncfiles", ModuleNotFoundError(name="asyncfiles._native")),
            ("asyncfiles", ModuleNotFoundError(name="transitive_dependency")),
            ("asyncfiles", ImportError("native library could not be loaded")),
            ("asyncfiles.os", ModuleNotFoundError(name="asyncfiles.os")),
        ],
    )
    def test_broken_install_is_not_hidden(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        failing_module: str,
        error: ImportError,
    ) -> None:
        real_import = importlib.import_module

        def broken_import(name: str, package: str | None = None) -> ModuleType:
            if name == failing_module:
                raise error
            if name == "asyncfiles":
                return ModuleType("asyncfiles")
            return real_import(name, package)

        monkeypatch.setattr(importlib, "import_module", broken_import)
        with pytest.raises(ImportError) as exc_info:
            FSL2Adapter(FSL2AdapterConfig(str(tmp_path), io_backend="asyncfiles"))
        assert exc_info.value is error

    def test_instances_keep_their_own_provider(self, tmp_path: Path) -> None:
        if importlib.util.find_spec("asyncfiles") is None:
            pytest.importorskip("asyncfiles")
        optional_provider = importlib.import_module("asyncfiles")
        with (
            closing(FSL2Adapter(FSL2AdapterConfig(str(tmp_path)))) as default,
            closing(
                FSL2Adapter(FSL2AdapterConfig(str(tmp_path), io_backend="asyncfiles"))
            ) as optional,
            patch.object(aiofiles, "open", wraps=aiofiles.open) as default_open,
            patch.object(
                optional_provider, "open", wraps=optional_provider.open
            ) as optional_open,
        ):
            k1, k2 = _key(b"\x01" * 4), _key(b"\x02" * 4)
            _store_and_wait(default, [k1], [b"default"])
            default_open.assert_called_once()
            optional_open.assert_not_called()
            _store_and_wait(optional, [k2], [b"optional"])
            default_open.assert_called_once()
            optional_open.assert_called_once()
            assert _lookup_and_wait(default, [k2]) == [True]
            assert _lookup_and_wait(optional, [k1]) == [True]


class _ClosingListener(_RecordingListener):
    """Exercise close() from a callback running on the adapter's I/O thread."""

    def __init__(self, adapter: FSL2Adapter) -> None:
        super().__init__()
        self.adapter = adapter

    def on_l2_keys_stored(self, keys: list[ObjectKey], sizes: list[int]) -> None:
        super().on_l2_keys_stored(keys, sizes)
        self.adapter.close()


def _exercise_shared_native_worker() -> None:
    """Run real native concurrency and shutdown checks in a bounded subprocess."""
    errors: list[str] = []
    sys.unraisablehook = lambda event: errors.append(str(event.exc_value))
    with TemporaryDirectory() as directory:
        base = Path(directory)
        before = set(threading.enumerate())
        first, second = [
            FSL2Adapter(FSL2AdapterConfig(str(base / name), io_backend="asyncfiles"))
            for name in ("first", "second")
        ]
        release = threading.Event()
        try:
            keys = [_key(i.to_bytes(4, "big")) for i in range(32)]
            adapters = [first, second]
            tasks = [adp.submit_lookup_and_lock_task(keys, {}) for adp in adapters]
            results: list[Bitmap | None] = [None, None]
            deadline = time.monotonic() + 3
            while time.monotonic() < deadline and any(r is None for r in results):
                for i, adp in enumerate(adapters):
                    if results[i] is None:
                        results[i] = adp.query_lookup_and_lock_result(tasks[i])
                time.sleep(0.01)
            assert all(r is not None and r.popcount() == 0 for r in results), errors
            assert not errors, errors
            workers = set(threading.enumerate()) - before
            assert len(workers) == 1
            worker = workers.pop()

            with patch.object(
                fs_l2_adapter.logger,
                "info",
                side_effect=RuntimeError("constructor failed"),
            ):
                with pytest.raises(RuntimeError, match="constructor failed"):
                    FSL2Adapter(
                        FSL2AdapterConfig(str(base / "failed"), io_backend="asyncfiles")
                    )

            io_os = importlib.import_module("asyncfiles.os")
            real_exists = io_os.path.exists
            entered = threading.Event()

            async def gated_exists(path: Path) -> bool:
                if path.parent == base / "second":
                    entered.set()
                    while not release.is_set():
                        await asyncio.sleep(0.01)
                return await real_exists(path)

            with patch.object(io_os.path, "exists", gated_exists):
                pending = second.submit_lookup_and_lock_task(keys, {})
                assert entered.wait(3)
                first.register_listener(_ClosingListener(first))
                _store_and_wait(first, [_key()], [b"close first on its worker"])
                first.close()
                assert worker.is_alive()
                assert second.report_status()["is_healthy"]
                release.set()
                deadline = time.monotonic() + 3
                result = None
                while time.monotonic() < deadline and result is None:
                    result = second.query_lookup_and_lock_result(pending)
                    time.sleep(0.01)
                assert result is not None and result.popcount() == 0

            second.register_listener(_ClosingListener(second))
            _store_and_wait(second, [_key()], [b"close last on its worker"])
            second.close()
            assert not worker.is_alive()
            assert not errors, errors
        finally:
            release.set()
            first.close()
            second.close()

        # A later adapter must not reuse a retired native-completion loop.
        with closing(
            FSL2Adapter(FSL2AdapterConfig(str(base / "third"), io_backend="asyncfiles"))
        ) as third:
            assert _lookup_and_wait(third, keys) == [False] * len(keys)
        assert not errors, errors


class TestSharedNativeWorker:
    def test_concurrent_instances_and_lifetime(self) -> None:
        if importlib.util.find_spec("asyncfiles") is None:
            pytest.importorskip("asyncfiles")
        result = subprocess.run(
            [
                sys.executable,
                "-B",
                "-c",
                "from tests.v1.distributed.test_fs_l2_adapter_delete "
                "import _exercise_shared_native_worker; "
                "_exercise_shared_native_worker()",
            ],
            env={
                **os.environ,
                "PYTHONASYNCIODEBUG": "1",
                "PYTHONDONTWRITEBYTECODE": "1",
            },
            capture_output=True,
            text=True,
            timeout=25,
            check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr

    @pytest.mark.parametrize("failure", ["start", "after-acquire"])
    def test_failed_constructor_cleans_up(
        self, tmp_path: Path, io_backend: str, failure: str
    ) -> None:
        before = set(threading.enumerate())
        notifiers: list[EventNotifier] = []
        real_notifier = fs_l2_adapter.create_event_notifier

        def allocate_notifier() -> EventNotifier:
            notifier = real_notifier()
            notifiers.append(notifier)
            return notifier

        with ExitStack() as patches:
            patches.enter_context(
                patch.object(fs_l2_adapter, "create_event_notifier", allocate_notifier)
            )
            target = threading.Thread if failure == "start" else fs_l2_adapter.logger
            attribute = "start" if failure == "start" else "info"
            patches.enter_context(
                patch.object(target, attribute, side_effect=RuntimeError("ctor failed"))
            )
            with pytest.raises(RuntimeError, match="ctor failed"):
                FSL2Adapter(FSL2AdapterConfig(str(tmp_path), io_backend=io_backend))
        assert len(notifiers) == 3
        assert all(notifier.fileno() == -1 for notifier in notifiers)
        assert set(threading.enumerate()) == before


class TestStoreIntegrity:
    @pytest.mark.parametrize("written", [0, 5])
    def test_short_write_is_not_published(
        self,
        adapter: AdapterFixture,
        io_backend: str,
        tmp_path: Path,
        written: int,
    ) -> None:
        adp, listener = adapter
        provider = importlib.import_module(io_backend)
        real_open = provider.open
        payload = b"not a complete write"
        write = AsyncMock(return_value=written)

        @asynccontextmanager
        async def incomplete_open(
            path: Path, mode: str
        ) -> AsyncIterator[SimpleNamespace]:
            async with real_open(path, mode) as handle:
                if written:
                    await handle.write(payload[:written])
                yield SimpleNamespace(write=write)

        with (
            patch.object(provider, "open", incomplete_open),
            patch.object(
                provider.os, "replace", new=AsyncMock(wraps=provider.os.replace)
            ) as replace,
        ):
            task = adp.submit_store_task([_key()], _bufs([payload]))
            deadline = time.monotonic() + 5
            result = None
            while time.monotonic() < deadline and result is None:
                result = adp.pop_completed_store_tasks().get(task)
                time.sleep(0.01)
            assert result is not None, "store task did not complete"
            assert not result.is_successful()
            assert result.bytes_transferred() == 0
            write.assert_awaited_once()
            replace.assert_not_awaited()
        assert listener.stored == []
        assert adp.get_usage().total_bytes_used == 0
        assert list(tmp_path.iterdir()) == []


class TestDelete:
    def test_delete_removes_keys_and_notifies(self, adapter: AdapterFixture) -> None:
        adp, listener = adapter
        k1, k2 = _key(b"\x01" * 4), _key(b"\x02" * 4)
        _store_and_wait(adp, [k1, k2], [b"x" * 100, b"y" * 40])
        assert dict(listener.stored) == {k1: 100, k2: 40}
        assert adp.get_usage().total_bytes_used == 140

        adp.delete([k1])

        assert listener.deleted == [k1]
        # Byte accounting shrank by exactly k1's stat'd size.
        assert adp.get_usage().total_bytes_used == 40
        # k1 is gone, k2 survives.
        assert _lookup_and_wait(adp, [k1, k2]) == [False, True]

    def test_delete_missing_key_is_noop(self, adapter: AdapterFixture) -> None:
        adp, listener = adapter
        adp.delete([_key(b"\xff" * 4)])
        assert listener.deleted == []
        assert adp.get_usage().total_bytes_used == 0

    def test_double_delete_is_idempotent(self, adapter: AdapterFixture) -> None:
        adp, listener = adapter
        k = _key()
        _store_and_wait(adp, [k], [b"z" * 10])
        adp.delete([k])
        adp.delete([k])
        assert listener.deleted == [k]
        assert adp.get_usage().total_bytes_used == 0

    def test_large_batch_deletes_all_and_notifies(
        self, adapter: AdapterFixture
    ) -> None:
        """A batch far wider than _DELETE_CONCURRENCY is fully deleted
        with per-key sizes intact — exercises the bounded concurrent
        fan-out (order-preserving gather + semaphore)."""
        adp, listener = adapter
        n = 300
        keys = [_key(i.to_bytes(4, "big")) for i in range(n)]
        payloads = [b"x" * (10 + i % 7) for i in range(n)]
        _store_and_wait(adp, keys, payloads)

        adp.delete(keys)

        assert sorted(listener.deleted, key=str) == sorted(keys, key=str)
        assert adp.get_usage().total_bytes_used == 0
        assert _lookup_and_wait(adp, keys[:8]) == [False] * 8

    def test_delete_after_close_does_not_raise(self, adapter: AdapterFixture) -> None:
        """delete() on a closed adapter is a logged no-op, never a
        crash: run_coroutine_threadsafe raises RuntimeError once the
        background loop is stopped, and delete is best-effort."""
        adp, listener = adapter
        adp.close()
        adp.delete([_key()])
        assert listener.deleted == []

    def test_empty_key_list_is_noop(self, adapter: AdapterFixture) -> None:
        adp, listener = adapter
        adp.delete([])
        assert listener.deleted == []

    def test_load_after_delete_is_miss(self, adapter: AdapterFixture) -> None:
        """The documented race outcome: a load for a deleted key
        degrades to a miss, never an error."""
        adp, listener = adapter
        k = _key()
        _store_and_wait(adp, [k], [b"w" * 16])
        adp.delete([k])

        task_id = adp.submit_load_task([k], _bufs([b"\x00" * 16]))
        deadline = time.monotonic() + 5.0
        bitmap = None
        while time.monotonic() < deadline:
            bitmap = adp.query_load_result(task_id)
            if bitmap is not None:
                break
            time.sleep(0.01)
        assert bitmap is not None, "load task did not complete within 5s"
        assert bitmap.popcount() == 0
        assert listener.accessed == []


class TestStoreLoadNotifications:
    def test_restore_of_existing_key_does_not_renotify(
        self, adapter: AdapterFixture
    ) -> None:
        """A store that skips (file already on disk) fires no stored
        event, so byte accounting is never double-counted."""
        adp, listener = adapter
        k = _key()
        _store_and_wait(adp, [k], [b"a" * 8])
        _store_and_wait(adp, [k], [b"a" * 8])
        assert listener.stored == [(k, 8)]
        assert adp.get_usage().total_bytes_used == 8

    def test_load_notifies_accessed(self, adapter: AdapterFixture) -> None:
        adp, listener = adapter
        k = _key()
        _store_and_wait(adp, [k], [b"b" * 12])

        dst = _Buf(b"\x00" * 12)
        task_id = adp.submit_load_task([k], cast("list[MemoryObj]", [dst]))
        deadline = time.monotonic() + 5.0
        bitmap = None
        while time.monotonic() < deadline:
            bitmap = adp.query_load_result(task_id)
            if bitmap is not None:
                break
            time.sleep(0.01)
        assert bitmap is not None and bitmap.popcount() == 1
        assert bytes(dst.byte_array) == b"b" * 12
        assert listener.accessed == [k]

    def test_store_and_load_with_read_ahead_and_relative_tmp_dir(
        self, tmp_path: Path, io_backend: str
    ) -> None:
        adp = FSL2Adapter(
            FSL2AdapterConfig(
                base_path=str(tmp_path),
                relative_tmp_dir="tmp",
                read_ahead_size=4096,
                io_backend=io_backend,
            )
        )
        try:
            k = _key()
            payload = bytes(range(256)) * 64  # 16 KiB
            _store_and_wait(adp, [k], [payload])

            dst = _Buf(b"\x00" * len(payload))
            task_id = adp.submit_load_task([k], cast("list[MemoryObj]", [dst]))
            deadline = time.monotonic() + 5.0
            bitmap = None
            while time.monotonic() < deadline:
                bitmap = adp.query_load_result(task_id)
                if bitmap is not None:
                    break
                time.sleep(0.01)
            assert bitmap is not None and bitmap.popcount() == 1
            assert bytes(dst.byte_array) == payload
            assert list((tmp_path / "tmp").iterdir()) == []
            assert len(list(tmp_path.glob("*.data"))) == 1
        finally:
            adp.close()
