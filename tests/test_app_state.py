from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest


def test_search_question_survives_source_view_and_return():
    project_root = Path(__file__).parents[1]
    if not any((project_root / "indexes" / strategy / "manifest.json").is_file()
               for strategy in ("fixed_size", "recursive")):
        pytest.skip("Project indexes are unavailable")

    app = AppTest.from_file(
        str(project_root / "src" / "app.py"),
        default_timeout=45,
    )
    saved_query = "Đợt học 26"
    app.session_state["last_search_result"] = {
        "query": saved_query,
        "citation_token": "saved-answer",
        "response": {
            "answer": "Tra cứu đã lưu.",
            "sources": [],
            "generated": False,
            "invalid_citations": [],
        },
    }
    app.session_state["active_citation"] = {
        "token": "missing-answer",
        "citation_id": "E1",
        "navigation_id": "source-view",
    }

    app.run()
    assert not app.exception

    app.session_state["active_citation"] = None
    app.run()

    assert not app.exception
    assert app.text_input(key="search_question_widget").value == saved_query
    assert app.session_state["last_search_result"]["response"]["answer"] == "Tra cứu đã lưu."
