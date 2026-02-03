from src.rag.table_renderer import render_markdown_table


def test_table_renderer_sanitizes_cells():
    columns = ["Name", "Value"]
    records = [{"Name": "Alice", "Value": "Line1\nLine2"}]
    table = render_markdown_table(columns, records)
    lines = table.splitlines()
    assert lines[0] == "| Name | Value |"
    assert lines[1] == "| --- | --- |"
    assert "; " in lines[2]
