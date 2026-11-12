"""Skip the tests that need a dataset on disk when it is absent.

Data roots come from the environment (see README.md): SM_ASDPOSE_CACHE or SM_SMM_ROOT (ASDPose
segment cache), SM_MMASD_ROOT, SM_UTD_ROOT, SM_EXTERNAL_CACHE. Every other test runs on
synthetic data on the CPU.
"""
import os

import pytest

from safe_motion.data import asdpose as D

_ASDPOSE = os.path.exists(f"{D.CACHE_DIR}/index.csv")
_MMASD = os.path.isdir(os.environ.get("SM_MMASD_ROOT", "data/mmasd"))
_UTD = os.path.isdir(f"{os.environ.get('SM_UTD_ROOT', 'data/utd-mhad')}/Skeleton")
_EXTERNAL = os.path.exists(f"{os.environ.get('SM_EXTERNAL_CACHE', f'{D.SMM_ROOT}/cache/external')}/index.csv")

NEEDS = {
    "test_core.py::test_data_and_model": (_ASDPOSE, "ASDPose cache (SM_ASDPOSE_CACHE)"),
    "test_repair_v14.py::test_cleaned_validation_excludes_alias": (_ASDPOSE, "ASDPose cache (SM_ASDPOSE_CACHE)"),
    "test_new_datasets.py::test_utd_mhad_files": (_UTD, "UTD-MHAD skeletons (SM_UTD_ROOT)"),
    "test_new_datasets.py::test_mmasd_json_clip": (_MMASD, "MMASD OpenPose files (SM_MMASD_ROOT)"),
    "test_new_datasets.py::test_mmasd_ados_table": (_MMASD, "MMASD ADOS table (SM_MMASD_ROOT)"),
    "test_v9.py::test_tiny_training_each_host": (_MMASD and _UTD, "MMASD and UTD-MHAD caches"),
    "test_v11.py::test_external_cache_loads": (_EXTERNAL, "external cache (SM_EXTERNAL_CACHE)"),
}


def pytest_collection_modifyitems(config, items):
    for item in items:
        key = item.nodeid.split("tests/")[-1]
        for name, (ok, what) in NEEDS.items():
            if key.startswith(name) and not ok:
                item.add_marker(pytest.mark.skip(reason=f"needs {what}"))
