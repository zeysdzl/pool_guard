import os
import pytest

if os.environ.get("CI") == "true":
    pytest.skip("Skipping pipeline test in CI (requires OpenCV)", allow_module_level=True)
