"""Comprehensive tests for all domain agents.

Tests agent registration, capability listing, task detection (NLU-first with fallback),
agent execution, and the agentic API router.
"""
import pytest
from unittest.mock import patch, MagicMock


# ---------------------------------------------------------------------------
# Agent registry tests
# ---------------------------------------------------------------------------

class TestAgentRegistry:
    def test_list_available_agents_returns_all_14(self):
        from src.agentic.domain_agents import list_available_agents
        agents = list_available_agents()
        assert len(agents) == 14

    def test_all_expected_domains_present(self):
        from src.agentic.domain_agents import list_available_agents
        agents = list_available_agents()
        expected = {"hr", "medical", "legal", "invoice", "content", "translation",
                    "education", "image", "web", "analytics", "screening", "cloud",
                    "customer_service", "analytics_viz"}
        assert set(agents.keys()) == expected

    def test_total_capabilities_count(self):
        from src.agentic.domain_agents import list_available_agents
        agents = list_available_agents()
        total = sum(len(caps) for caps in agents.values())
        assert total >= 61  # 61 capabilities across 14 agents

    def test_get_domain_agent_valid(self):
        from src.agentic.domain_agents import get_domain_agent
        for domain in ["hr", "medical", "legal", "invoice", "content",
                       "translation", "education", "image", "web",
                       "analytics", "screening", "cloud", "cloud_platform"]:
            agent = get_domain_agent(domain)
            assert agent is not None, f"Agent for domain '{domain}' should exist"

    def test_get_domain_agent_aliases(self):
        from src.agentic.domain_agents import get_domain_agent
        aliases = {
            "resume": "hr", "policy": "legal", "financial": "invoice",
            "email": "content", "translator": "translation",
            "tutor": "education", "insights": "analytics",
            "web_search": "web", "screen_pii": "screening",
            "jira": "cloud_platform", "confluence": "cloud_platform",
            "sharepoint": "cloud_platform", "jira_confluence": "cloud_platform",
        }
        for alias, expected_domain in aliases.items():
            agent = get_domain_agent(alias)
            assert agent is not None, f"Alias '{alias}' should resolve to an agent"
            assert agent.domain == expected_domain

    def test_get_domain_agent_invalid(self):
        from src.agentic.domain_agents import get_domain_agent
        assert get_domain_agent("nonexistent") is None

    def test_each_agent_has_capabilities(self):
        from src.agentic.domain_agents import get_domain_agent
        for domain in ["hr", "medical", "legal", "invoice", "content",
                       "translation", "education", "image", "web",
                       "analytics", "screening"]:
            agent = get_domain_agent(domain)
            caps = agent.get_capabilities()
            assert len(caps) >= 4, f"{domain} agent should have >= 4 capabilities"

    def test_can_handle_valid_task(self):
        from src.agentic.domain_agents import get_domain_agent
        agent = get_domain_agent("hr")
        assert agent.can_handle("generate_interview_questions")
        assert agent.can_handle("skill_gap_analysis")
        assert not agent.can_handle("nonexistent_task")


# ---------------------------------------------------------------------------
# Agent task detection tests (ML-first with keyword fallback)
# ---------------------------------------------------------------------------

class TestDetectAgentTask:
    def test_hr_interview_questions(self):
        from src.agentic.domain_agents import detect_agent_task
        r = detect_agent_task("generate interview questions for this resume")
        assert r is not None
        assert r["domain"] == "hr"
        assert r["task_type"] == "generate_interview_questions"

    def test_hr_skill_gap(self):
        from src.agentic.domain_agents import detect_agent_task
        r = detect_agent_task("what is the skill gap for this candidate")
        assert r is not None
        assert r["domain"] == "hr"
        assert r["task_type"] == "skill_gap_analysis"

    def test_hr_role_fit(self):
        from src.agentic.domain_agents import detect_agent_task
        r = detect_agent_task("is this candidate a good fit for the role")
        assert r is not None
        assert r["domain"] == "hr"
        assert r["task_type"] == "role_fit_assessment"

    def test_medical_drug_interaction(self):
        from src.agentic.domain_agents import detect_agent_task
        r = detect_agent_task("check for drug interactions in this prescription")
        assert r is not None
        assert r["domain"] == "medical"
        assert r["task_type"] == "drug_interaction_check"

    def test_medical_treatment_plan(self):
        from src.agentic.domain_agents import detect_agent_task
        r = detect_agent_task("review this treatment plan")
        assert r is not None
        assert r["domain"] == "medical"
        assert r["task_type"] == "treatment_plan_review"

    def test_medical_lab_result(self):
        from src.agentic.domain_agents import detect_agent_task
        r = detect_agent_task("interpret these lab results")
        assert r is not None
        assert r["domain"] == "medical"
        assert r["task_type"] == "lab_result_interpretation"

    def test_legal_clause_risk(self):
        from src.agentic.domain_agents import detect_agent_task
        r = detect_agent_task("find risky clauses in this contract")
        assert r is not None
        assert r["domain"] == "legal"
        assert r["task_type"] == "clause_risk_assessment"

    def test_legal_compliance(self):
        from src.agentic.domain_agents import detect_agent_task
        r = detect_agent_task("check compliance with regulations")
        assert r is not None
        assert r["domain"] == "legal"
        assert r["task_type"] == "compliance_check"

    def test_invoice_anomaly(self):
        from src.agentic.domain_agents import detect_agent_task
        r = detect_agent_task("detect payment anomalies in these invoices")
        assert r is not None
        assert r["domain"] == "invoice"
        assert r["task_type"] == "payment_anomaly_detection"

    def test_invoice_expense_categorization(self):
        from src.agentic.domain_agents import detect_agent_task
        r = detect_agent_task("categorize these expenses from invoices")
        assert r is not None
        assert r["domain"] == "invoice"
        assert r["task_type"] == "expense_categorization"

    def test_content_draft_email(self):
        from src.agentic.domain_agents import detect_agent_task
        r = detect_agent_task("draft an email about the meeting")
        assert r is not None
        assert r["domain"] == "content"
        assert r["task_type"] == "draft_email"

    def test_content_documentation(self):
        from src.agentic.domain_agents import detect_agent_task
        r = detect_agent_task("create documentation for this API")
        assert r is not None
        assert r["domain"] == "content"
        assert r["task_type"] == "generate_documentation"

    def test_content_rewrite(self):
        from src.agentic.domain_agents import detect_agent_task
        r = detect_agent_task("rewrite this paragraph for clarity")
        assert r is not None
        assert r["domain"] == "content"
        assert r["task_type"] == "rewrite_text"

    def test_translation_translate(self):
        from src.agentic.domain_agents import detect_agent_task
        r = detect_agent_task("translate this document to French")
        assert r is not None
        assert r["domain"] == "translation"
        assert r["task_type"] == "translate_text"

    def test_translation_detect_language(self):
        from src.agentic.domain_agents import detect_agent_task
        r = detect_agent_task("what language is this text written in")
        assert r is not None
        assert r["domain"] == "translation"
        assert r["task_type"] == "detect_language"

    def test_education_quiz(self):
        from src.agentic.domain_agents import detect_agent_task
        r = detect_agent_task("create a quiz from this material")
        assert r is not None
        assert r["domain"] == "education"
        assert r["task_type"] == "generate_quiz"

    def test_education_explain(self):
        from src.agentic.domain_agents import detect_agent_task
        r = detect_agent_task("explain this concept to me")
        assert r is not None
        assert r["domain"] == "education"
        assert r["task_type"] == "explain_concept"

    def test_education_study_guide(self):
        from src.agentic.domain_agents import detect_agent_task
        r = detect_agent_task("create a study guide for this topic")
        assert r is not None
        assert r["domain"] == "education"
        assert r["task_type"] == "study_guide"

    def test_web_search(self):
        from src.agentic.domain_agents import detect_agent_task
        r = detect_agent_task("search the web for latest news")
        assert r is not None
        assert r["domain"] == "web"
        assert r["task_type"] == "search_web"

    def test_web_fact_check(self):
        from src.agentic.domain_agents import detect_agent_task
        r = detect_agent_task("fact check this claim")
        assert r is not None
        assert r["domain"] == "web"
        assert r["task_type"] == "fact_check"

    def test_analytics_anomalies(self):
        from src.agentic.domain_agents import detect_agent_task
        r = detect_agent_task("find anomalies in the data")
        assert r is not None
        assert r["domain"] == "analytics"
        assert r["task_type"] == "detect_anomalies"

    def test_analytics_action_items(self):
        from src.agentic.domain_agents import detect_agent_task
        r = detect_agent_task("extract action items from the meeting notes")
        assert r is not None
        assert r["domain"] == "analytics"
        assert r["task_type"] == "extract_action_items"

    def test_analytics_risk_assessment(self):
        from src.agentic.domain_agents import detect_agent_task
        r = detect_agent_task("perform a risk assessment on the project data")
        assert r is not None
        assert r["domain"] == "analytics"
        assert r["task_type"] == "risk_assessment"

    def test_screening_pii(self):
        from src.agentic.domain_agents import detect_agent_task
        r = detect_agent_task("check for PII in this document")
        assert r is not None
        assert r["domain"] == "screening"
        assert r["task_type"] == "screen_pii"

    def test_screening_ai_content(self):
        from src.agentic.domain_agents import detect_agent_task
        r = detect_agent_task("detect if this was AI generated")
        assert r is not None
        assert r["domain"] == "screening"
        assert r["task_type"] == "detect_ai_content"

    def test_screening_readability(self):
        from src.agentic.domain_agents import detect_agent_task
        r = detect_agent_task("assess the readability of this text")
        assert r is not None
        assert r["domain"] == "screening"
        assert r["task_type"] == "assess_readability"

    def test_explicit_domain_overrides(self):
        from src.agentic.domain_agents import detect_agent_task
        # When domain is explicitly set, should use it even without strong keywords
        r = detect_agent_task("tell me about this", domain="hr")
        # No task detected — query has no task-specific keywords
        assert r is None

    def test_no_detection_for_simple_query(self):
        from src.agentic.domain_agents import detect_agent_task
        r = detect_agent_task("what is the weather today")
        assert r is None

    def test_explicit_domain_with_task_keywords(self):
        from src.agentic.domain_agents import detect_agent_task
        r = detect_agent_task("do a risk assessment", domain="analytics")
        assert r is not None
        assert r["domain"] == "analytics"
        assert r["task_type"] == "risk_assessment"


# ---------------------------------------------------------------------------
# Agent execution tests (with mocked LLM)
# ---------------------------------------------------------------------------

class TestAgentExecution:
    def _mock_agent(self, domain):
        from src.agentic.domain_agents import get_domain_agent
        agent = get_domain_agent(domain)
        agent._llm = MagicMock()
        agent._llm.generate_with_metadata = MagicMock(return_value=("Mock LLM output for testing.", {}))
        return agent

    def test_resume_interview_questions(self):
        agent = self._mock_agent("hr")
        result = agent.execute("generate_interview_questions", {
            "text": "John Doe, 5 years Python experience, AWS certified, worked at Google.",
            "job_role": "Senior Python Developer",
        })
        assert result.success
        assert result.output
        assert result.task_type == "generate_interview_questions"

    def test_resume_skill_gap(self):
        agent = self._mock_agent("hr")
        result = agent.execute("skill_gap_analysis", {
            "text": "Java developer with 3 years experience",
            "job_description": "Senior Python developer, 5+ years, AWS, Docker",
        })
        assert result.success

    def test_resume_candidate_summary(self):
        agent = self._mock_agent("hr")
        result = agent.execute("candidate_summary", {
            "text": "Experienced software engineer with ML expertise",
        })
        assert result.success

    def test_medical_drug_interaction(self):
        agent = self._mock_agent("medical")
        result = agent.execute("drug_interaction_check", {
            "text": "Patient prescribed Warfarin 5mg and Aspirin 81mg daily",
        })
        assert result.success

    def test_medical_clinical_summary(self):
        agent = self._mock_agent("medical")
        result = agent.execute("clinical_summary", {
            "text": "Patient presents with chest pain, ECG shows ST elevation",
        })
        assert result.success

    def test_legal_clause_risk(self):
        agent = self._mock_agent("legal")
        result = agent.execute("clause_risk_assessment", {
            "text": "The party shall indemnify without limitation all damages",
        })
        assert result.success

    def test_legal_compliance(self):
        agent = self._mock_agent("legal")
        result = agent.execute("compliance_check", {
            "text": "Data processing agreement under GDPR",
            "framework": "GDPR",
        })
        assert result.success

    def test_invoice_anomaly(self):
        agent = self._mock_agent("invoice")
        result = agent.execute("payment_anomaly_detection", {
            "text": "Invoice #123: $50,000 from Vendor A. Invoice #124: $500,000 from Vendor A.",
        })
        assert result.success

    def test_content_draft_email(self):
        agent = self._mock_agent("content")
        result = agent.execute("draft_email", {
            "text": "Meeting notes from Q4 review",
            "query": "draft email summarizing key decisions",
        })
        assert result.success

    def test_content_generate(self):
        agent = self._mock_agent("content")
        result = agent.execute("generate_content", {
            "text": "Product launch details for Q1 2026",
            "content_type": "summary",
        })
        assert result.success

    def test_content_rewrite(self):
        agent = self._mock_agent("content")
        result = agent.execute("rewrite_text", {
            "text": "This is a bad sentence that needs to be made more better.",
            "tone": "professional",
        })
        assert result.success

    def test_translator_translate(self):
        agent = self._mock_agent("translation")
        result = agent.execute("translate_text", {
            "text": "Hello, how are you?",
            "target_language": "French",
        })
        assert result.success

    def test_translator_detect_language(self):
        agent = self._mock_agent("translation")
        result = agent.execute("detect_language", {
            "text": "Bonjour le monde",
        })
        assert result.success

    def test_tutor_create_lesson(self):
        agent = self._mock_agent("education")
        result = agent.execute("create_lesson", {
            "text": "Quantum computing uses qubits that can be in superposition",
            "learning_level": "beginner",
        })
        assert result.success

    def test_tutor_generate_quiz(self):
        agent = self._mock_agent("education")
        result = agent.execute("generate_quiz", {
            "text": "The mitochondria is the powerhouse of the cell",
            "num_questions": 5,
        })
        assert result.success

    def test_tutor_study_guide(self):
        agent = self._mock_agent("education")
        result = agent.execute("study_guide", {
            "text": "Chapter on neural networks: layers, activation functions, backpropagation",
        })
        assert result.success

    def test_image_analyze(self):
        agent = self._mock_agent("image")
        result = agent.execute("analyze_image", {
            "text": "[Image content: A receipt showing Total: $45.99, Tax: $3.50]",
        })
        assert result.success

    def test_web_search(self):
        agent = self._mock_agent("web")
        result = agent.execute("search_web", {
            "query": "latest Python 3.13 features",
        })
        assert result.success

    def test_web_fact_check(self):
        agent = self._mock_agent("web")
        result = agent.execute("fact_check", {
            "text": "The Earth is flat",
            "claim": "The Earth is flat",
        })
        assert result.success

    def test_insights_detect_anomalies(self):
        agent = self._mock_agent("analytics")
        result = agent.execute("detect_anomalies", {
            "text": "Invoice amounts: $100, $150, $120, $95, $50000, $110",
        })
        assert result.success

    def test_insights_action_items(self):
        agent = self._mock_agent("analytics")
        result = agent.execute("extract_action_items", {
            "text": "Action: John must complete the report by March 15. Sarah needs to review by end of week.",
        })
        assert result.success

    def test_insights_risk_assessment(self):
        agent = self._mock_agent("analytics")
        result = agent.execute("risk_assessment", {
            "text": "Contract with unlimited liability clause and no termination provision",
        })
        assert result.success

    def test_screening_pii(self):
        agent = self._mock_agent("screening")
        result = agent.execute("screen_pii", {
            "text": "John Smith, SSN: 123-45-6789, email: john@example.com",
        })
        assert result.success

    def test_screening_ai_content(self):
        agent = self._mock_agent("screening")
        result = agent.execute("detect_ai_content", {
            "text": "As an AI language model, I can help you understand...",
        })
        assert result.success

    def test_screening_readability(self):
        agent = self._mock_agent("screening")
        result = agent.execute("assess_readability", {
            "text": "The quick brown fox jumps over the lazy dog. Simple clear sentence.",
        })
        assert result.success

    def test_invalid_task_type(self):
        agent = self._mock_agent("hr")
        result = agent.execute("nonexistent_task", {"text": "test"})
        assert not result.success
        assert result.error

    def test_agent_result_to_dict(self):
        from src.agentic.domain_agents import AgentTaskResult
        result = AgentTaskResult(
            task_type="test_task",
            success=True,
            output="Test output",
            structured_data={"key": "value"},
        )
        d = result.to_dict()
        assert d["task_type"] == "test_task"
        assert d["success"] is True
        assert d["output"] == "Test output"
        assert d["structured_data"] == {"key": "value"}


# ---------------------------------------------------------------------------
# Agentic API router tests
# ---------------------------------------------------------------------------

class TestAgenticRouter:
    def test_router_has_4_routes(self):
        from src.agentic.api_router import agents_router
        assert len(agents_router.routes) == 4

    def test_list_agents_endpoint(self):
        from src.agentic.api_router import agents_router
        # Find the list agents route
        for route in agents_router.routes:
            if hasattr(route, "path") and route.path == "" and "GET" in (route.methods or set()):
                # Call the endpoint function directly
                response = route.endpoint()
                assert "agents" in response
                assert "total" in response
                assert response["total"] == 11
                break

    def test_get_agent_info_valid(self):
        from src.agentic.api_router import agents_router
        for route in agents_router.routes:
            if hasattr(route, "path") and route.path == "/{agent_name}" and "GET" in (route.methods or set()):
                response = route.endpoint("hr")
                assert response["name"] == "hr"
                assert response["domain"] == "hr"
                assert "generate_interview_questions" in response["capabilities"]
                break

    def test_tool_to_agent_mapping(self):
        from src.agentic.api_router import _TOOL_TO_AGENT_MAP
        # Every old tool should map to an agent
        expected_tools = {
            "resumes", "medical", "lawhere", "insights", "action_items",
            "creator", "email_drafting", "code_docs", "translator", "tutor",
            "image_analysis", "web_search", "web_extract", "screen_pii",
            "screen_ai_authorship", "screen_resume", "screen_readability",
            "content_generate",
        }
        assert expected_tools.issubset(set(_TOOL_TO_AGENT_MAP.keys()))


# ---------------------------------------------------------------------------
# NLU-based detection tests
# ---------------------------------------------------------------------------

class TestNLUDetection:
    def test_nlu_classification_with_domain_hint(self):
        """NLU classification resolves translation tasks when domain is hinted."""
        from src.agentic.domain_agents import detect_agent_task
        # With a domain hint, domain-filtered NLU scoring finds the best
        # match even for queries that might be ambiguous without a hint.
        r = detect_agent_task("translate this document to french", domain="translation")
        assert r is not None
        assert r["domain"] == "translation"
        assert r["task_type"] == "translate_text"

    def test_explicit_domain_takes_priority(self):
        """When domain is explicitly provided, domain-filtered classification
        resolves ambiguous queries correctly."""
        from src.agentic.domain_agents import detect_agent_task
        r = detect_agent_task("find anomalies", domain="analytics")
        assert r is not None
        assert r["domain"] == "analytics"
        assert r["task_type"] == "detect_anomalies"

    def test_classify_domain_task_direct(self):
        """classify_domain_task resolves domain:task_type via NLU scoring."""
        from src.nlp.nlu_engine import classify_domain_task
        # With domain hint, domain-filtered scoring finds the best match
        r = classify_domain_task("translate this to french", domain="translation")
        assert r is not None
        assert r["task_type"] == "translate_text"

        r = classify_domain_task("create a quiz", domain="education")
        assert r is not None
        assert r["task_type"] == "generate_quiz"

        r = classify_domain_task("find anomalies", domain="analytics")
        assert r is not None
        assert r["task_type"] == "detect_anomalies"

        # No task detected for a generic, non-task query even with domain
        r = classify_domain_task("what is the weather today", domain="hr")
        assert r is None
