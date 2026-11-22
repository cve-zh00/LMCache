FileSystem
==========

A pure file-system L2 adapter using async I/O (``aiofiles`` by default).
Each KV cache object is stored as a raw ``.data`` file whose name encodes the full
``ObjectKey``.  Does **not** require NIXL -- works on any POSIX file system.

**Required fields:**

- ``base_path``: Directory for storing KV cache files.

**Optional fields:**

- ``relative_tmp_dir``: Relative sub-directory for temporary files during
  writes (atomic rename on completion).
- ``read_ahead_size``: Trigger file-system read-ahead by reading this many
  bytes first (positive integer, optional).
- ``use_odirect``: ``true`` or ``false`` (default ``false``) -- bypass the
  page cache via ``O_DIRECT``.
- ``io_backend``: ``"aiofiles"`` (default) or ``"asyncfiles"``.  Selected per
  adapter instance; ``asyncfiles`` requires the optional ``py-asyncfiles`` package.
  Other values, including non-strings, raise ``ValueError``.

To opt into ``asyncfiles``, install it separately:

.. code-block:: bash

    pip install py-asyncfiles

Only explicit selection imports ``asyncfiles``.  If the package is missing,
adapter initialization raises ``ImportError`` with installation instructions
before creating background resources; broken installations propagate their
original import errors.  There is no silent fallback to ``aiofiles``.

All ``asyncfiles`` adapters share one reference-counted I/O worker because the
released native backend requires completion callbacks on the same thread.
Closing one adapter cancels only its own requests; the last owner stops the worker.
``aiofiles`` adapters retain independent workers.  This does not make unrelated
host-application use of ``asyncfiles`` on other event loops safe.

Incomplete buffered writes fail the store and remove the temporary file without
publishing or accounting the object as stored.

``use_odirect`` retains the same synchronous OS transfers in an executor with
either provider; selecting ``asyncfiles`` does not enable a native direct-I/O
path.  Direct I/O needs aligned buffers and filesystem support; unaligned buffers
or filesystems refusing ``O_DIRECT`` fall back to buffered I/O.

**Configuration examples:**

.. code-block:: bash

    # Basic FS adapter
    --l2-adapter '{"type": "fs", "base_path": "/data/lmcache/l2"}'

    # With temp directory
    --l2-adapter '{"type": "fs", "base_path": "/data/lmcache/l2", "relative_tmp_dir": ".tmp"}'

    # With O_DIRECT for bypassing page cache
    --l2-adapter '{"type": "fs", "base_path": "/data/lmcache/l2", "use_odirect": true}'

    # Optional asyncfiles provider (after installing py-asyncfiles)
    --l2-adapter '{"type": "fs", "base_path": "/data/lmcache/l2", "io_backend": "asyncfiles"}'
