# conftest.py – konfigurasi pytest untuk StackMatch backend tests
import pytest


# Atur mode asyncio pytest-asyncio secara global
# Dengan ini tidak perlu dekorator @pytest.mark.asyncio per test
def pytest_configure(config):
    config.addinivalue_line(
        "markers", "asyncio: mark test as async"
    )
