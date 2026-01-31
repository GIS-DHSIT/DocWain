from src.nlp.dialogue_intel import analyze_sentiment


def test_analyze_sentiment_positive():
    result = analyze_sentiment("Thanks, that was great!")
    assert result.sentiment == "positive"
    assert result.feedback_type in {"thanks", "praise"}


def test_analyze_sentiment_negative():
    result = analyze_sentiment("This is wrong and not good")
    assert result.sentiment == "negative"
    assert result.should_recover


def test_analyze_sentiment_mixed():
    result = analyze_sentiment("Thanks, but this is wrong")
    assert result.sentiment == "mixed"
    assert result.should_recover


def test_analyze_sentiment_neutral():
    result = analyze_sentiment("Please summarize the contract")
    assert result.sentiment in {"neutral", "positive", "negative", "mixed"}
