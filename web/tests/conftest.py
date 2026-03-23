"""Web API 测试公共 fixtures"""

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from web.app.main import create_app


@pytest.fixture
def app():
    return create_app()


@pytest_asyncio.fixture
async def client(app):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c
