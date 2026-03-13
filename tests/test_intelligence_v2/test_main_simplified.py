"""Verify main.py no longer has legacy agent dispatch block."""
import ast
import inspect


def test_no_legacy_agent_dispatch_in_ask_handler():
    """The ask handler should not contain legacy domain agent dispatch code.
    All domain routing now goes through CoreAgent's DomainDispatcher."""
    # Read main.py source and find the handle_ask / ask function
    from pathlib import Path
    source = Path("src/main.py").read_text()

    # The old block had these distinctive patterns
    assert "get_domain_agent" not in source or "from src.agentic.domain_agents import get_domain_agent" not in source, \
        "Legacy get_domain_agent import still in main.py ask handler"

    # Check the specific legacy dispatch pattern is gone
    assert "_agent_name = getattr(request, \"agent_name\", None)" not in source, \
        "Legacy agent_name dispatch variable still in main.py"

    assert "_retrieve_rag_context" not in source, \
        "Legacy _retrieve_rag_context call still in main.py"


def test_execute_request_still_called():
    """execute_request should still be called for all requests."""
    from pathlib import Path
    source = Path("src/main.py").read_text()
    assert "execute_request" in source, "execute_request call missing from main.py"
