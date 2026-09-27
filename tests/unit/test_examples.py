"""
Unit test suite validating that all top-level examples in `examples/` execute successfully.
"""

from __future__ import annotations

import pytest

from examples.async_agent_demo import run_async_agent_demo
from examples.framework_integrations_demo import run_integrations_demo
from examples.multi_tenant_compliance_demo import run_compliance_demo
from examples.quickstart_demo import run_quickstart_demo


def test_example_quickstart() -> None:
    """Ensure examples/quickstart_demo.py runs to completion without exception."""
    run_quickstart_demo()


def test_example_compliance() -> None:
    """Ensure examples/multi_tenant_compliance_demo.py runs to completion without exception."""
    run_compliance_demo()


@pytest.mark.asyncio
async def test_example_async_agent() -> None:
    """Ensure examples/async_agent_demo.py runs to completion asynchronously without exception."""
    await run_async_agent_demo()


def test_example_integrations() -> None:
    """Ensure examples/framework_integrations_demo.py runs to completion without exception."""
    run_integrations_demo()
