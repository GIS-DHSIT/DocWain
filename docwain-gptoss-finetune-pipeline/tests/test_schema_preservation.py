import re

from tests.conftest import load_eval_outputs


JSON_KEYS_PATTERN = re.compile(r"keys: (.+?)\.")
TABLE_PATTERN = re.compile(r"columns: (.+?)\.")
LINE_PATTERN = re.compile(r"single line: (.+?)\.")


def test_schema_preservation():
    rows = load_eval_outputs()
    for row in rows:
        rules = row.get("output_rules", "")
        answer = row["answer"]

        json_match = JSON_KEYS_PATTERN.search(rules)
        if json_match:
            keys = [k.strip() for k in json_match.group(1).split(",")]
            for key in keys:
                assert f'"{key}"' in answer

        table_match = TABLE_PATTERN.search(rules)
        if table_match:
            headers = [h.strip() for h in table_match.group(1).split("|")]
            for header in headers:
                assert header in answer

        line_match = LINE_PATTERN.search(rules)
        if line_match:
            required_prefix = line_match.group(1).split(":")[0]
            assert answer.strip().startswith(required_prefix)
