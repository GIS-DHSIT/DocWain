import re

TASK_1 = "TASK_1"
TASK_2 = "TASK_2"
TASK_3 = "TASK_3"
TASK_4 = "TASK_4"
TASK_5 = "TASK_5"
TASK_6 = "TASK_6"
GENERIC_EXTRACT = "GENERIC_EXTRACT"


def route_intent(prompt: str) -> str:
    text = (prompt or "").lower()

    if re.search(r"compare\s+resume\s+vs\s+linkedin|linkedin\s+vs\s+resume|resume\s+and\s+linkedin\s+compare", text):
        return TASK_5

    if re.search(r"top\s*5.*5\s*-\s*10\s*years.*certif|5\s*to\s*10\s*years.*certif", text):
        return TASK_4

    if re.search(r"top\s*5.*scm|supply\s+chain", text) and "top" in text:
        return TASK_3

    if re.search(r"top\s*5.*5\s*-\s*10\s*years|top\s*5.*5\s*to\s*10\s*years", text):
        return TASK_2

    if re.search(r"list\s*1:|list\s*2:|list\s*3:|top\s*5\s+candidates\s+with\s+5\s*-\s*10\s*years\s+experience\s+\+\s+certifications", text):
        return TASK_6

    if re.search(r"extract\s+fields|resume\s+summary|candidate\s+summary|profile\s+summary", text):
        return TASK_1

    return TASK_1 if text else GENERIC_EXTRACT


__all__ = ["TASK_1", "TASK_2", "TASK_3", "TASK_4", "TASK_5", "TASK_6", "GENERIC_EXTRACT", "route_intent"]
