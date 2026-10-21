"""Redirect all runtime output to a temporary directory before any project import."""

import atexit
import os
import shutil
import tempfile

if not os.environ.get("MEDALLION_RUNTIME_DIR"):
    _runtime_dir = tempfile.mkdtemp(prefix="medallion-tests-")
    os.environ["MEDALLION_RUNTIME_DIR"] = _runtime_dir
    atexit.register(shutil.rmtree, _runtime_dir, ignore_errors=True)

# Plotly's JS is ~4.7 MB per report when inlined; use the CDN tag in tests (one test covers inline mode).
os.environ.setdefault("REPORT_PLOTLY_JS", "cdn")
