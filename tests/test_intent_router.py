from src.nlp.dialogue_intel import detect_intent


def test_detect_intent_greeting():
    result = detect_intent("Hi there")
    assert result.intent == "GREETING"
    assert result.confidence >= 0.8


def test_detect_intent_thanks():
    result = detect_intent("Thank you, great job")
    assert result.intent == "THANKS_OR_PRAISE"


def test_detect_intent_negative_feedback():
    result = detect_intent("This is bad and doesn't work")
    assert result.intent == "NEGATIVE_FEEDBACK"


def test_detect_intent_document_task_default():
    result = detect_intent("Summarize the key terms in the document")
    assert result.intent == "DOCUMENT_TASK"


def test_detect_intent_clarification():
    result = detect_intent("Can you repeat that?")
    assert result.intent == "CLARIFICATION"
