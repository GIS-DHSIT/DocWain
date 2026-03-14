"""Domain-specialized agents that wrap existing tools with reasoning capabilities.

Each agent performs domain-specific tasks beyond simple extraction:
- ResumeAgent: interview question generation, skill gap analysis, candidate comparison
- MedicalAgent: drug interaction analysis, treatment plan review, clinical summaries
- LegalAgent: clause risk assessment, compliance checking, contract comparison
- InvoiceAgent: payment anomaly detection, vendor analysis, financial summarization
- ContentAgent: content generation, email drafting, documentation, rewriting
- TranslatorAgent: translation, language detection, multilingual summaries
- TutorAgent: lesson creation, quiz generation, concept explanation
- ImageAgent: image analysis, OCR, image description, data extraction
- WebAgent: web search, URL fetching, research, fact checking
- InsightsAgent: anomaly detection, pattern finding, action items, risk assessment
- ScreeningAgent: PII detection, AI content detection, resume screening, readability
- CustomerServiceAgent: issue resolution, troubleshooting, escalation, customer responses
- AnalyticsVisualizationAgent: chart generation, distributions, comparisons, dashboards

Agents use the existing tool handlers for data extraction, then add an LLM reasoning
layer for higher-order analysis tasks.
"""
from __future__ import annotations

from src.utils.logging_utils import get_logger
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Agent result
# ---------------------------------------------------------------------------

@dataclass
class AgentTaskResult:
    """Result from a domain agent task execution."""
    task_type: str
    success: bool
    output: str = ""
    structured_data: Dict[str, Any] = field(default_factory=dict)
    sources: List[Dict[str, Any]] = field(default_factory=list)
    reasoning: str = ""
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_type": self.task_type,
            "success": self.success,
            "output": self.output,
            "structured_data": self.structured_data,
            "sources": self.sources,
            "reasoning": self.reasoning,
            "error": self.error,
        }

# ---------------------------------------------------------------------------
# Base domain agent
# ---------------------------------------------------------------------------

class DomainAgent(ABC):
    """Base class for domain-specialized agents.

    MoE routing:
      - Agents with ``use_thinking_model = True`` (reasoning-heavy) use lfm2.5-thinking
        when a thinking_client is provided, falling back to DocWain-Agent.
      - Agents with ``use_thinking_model = False`` (generation-heavy) always use DocWain-Agent.
    """

    domain: str = "generic"
    capabilities: List[str] = []
    use_thinking_model: bool = True  # Override to False for generation-heavy agents

    def __init__(self, llm_client: Any = None, thinking_client: Any = None):
        self._llm = llm_client
        self._thinking = thinking_client  # lfm2.5-thinking for reasoning steps

    def _get_llm(self) -> Any:
        """Return the appropriate LLM client based on MoE routing.

        Reasoning agents → lfm2.5-thinking (fast, parallel-safe)
        Generation agents → DocWain-Agent (full power, long text)
        """
        # For reasoning tasks, prefer thinking model when available
        if self.use_thinking_model and self._thinking is not None:
            return self._thinking
        # Fall back to base model
        if self._llm is None:
            try:
                from src.llm.clients import OllamaClient
                self._llm = OllamaClient()
            except Exception as exc:
                logger.warning("Failed to initialize LLM client: %s", exc)
        return self._llm

    def _get_base_llm(self) -> Any:
        """Always return the base (DocWain-Agent) LLM, bypassing thinking model."""
        if self._llm is None:
            try:
                from src.llm.clients import OllamaClient
                self._llm = OllamaClient()
            except Exception as exc:
                logger.warning("Failed to initialize base LLM client: %s", exc)
        return self._llm

    def _generate(self, prompt: str, temperature: float = 0.3, max_tokens: int = 2048) -> str:
        """Generate text using the LLM client (MoE-routed)."""
        llm = self._get_llm()
        if llm is None:
            return ""
        try:
            if hasattr(llm, "generate_with_metadata"):
                text, _ = llm.generate_with_metadata(
                    prompt, options={"temperature": temperature, "num_predict": max_tokens, "num_ctx": 8192}
                )
                return text
            return llm.generate(prompt)
        except Exception as exc:
            # Fallback to base model if thinking model failed
            if llm is not self._llm and self._llm is not None:
                logger.debug("%s agent thinking model failed, falling back to base: %s", self.domain, exc)
                try:
                    base = self._get_base_llm()
                    if base and hasattr(base, "generate_with_metadata"):
                        text, _ = base.generate_with_metadata(
                            prompt, options={"temperature": temperature, "num_predict": max_tokens, "num_ctx": 8192}
                        )
                        return text
                except Exception as exc2:
                    exc = exc2
            logger.warning("%s agent LLM generation failed: %s", self.domain, exc)
            return ""

    @abstractmethod
    def get_capabilities(self) -> List[str]:
        """Return list of task types this agent can handle."""
        ...

    @abstractmethod
    def execute(self, task_type: str, context: Dict[str, Any]) -> AgentTaskResult:
        """Execute a specific task with the given context."""
        ...

    def can_handle(self, task_type: str) -> bool:
        """Check if this agent can handle the given task type."""
        return task_type in self.get_capabilities()

# ---------------------------------------------------------------------------
# Resume Agent
# ---------------------------------------------------------------------------

class ResumeAgent(DomainAgent):
    """Specialized agent for HR/resume analysis tasks."""

    domain = "hr"
    enable_internet = True

    def get_capabilities(self) -> List[str]:
        return [
            "generate_interview_questions",
            "skill_gap_analysis",
            "candidate_summary",
            "experience_timeline",
            "role_fit_assessment",
            "linkedin_verification",
            "certification_lookup",
        ]

    def execute(self, task_type: str, context: Dict[str, Any]) -> AgentTaskResult:
        handlers = {
            "generate_interview_questions": self._generate_interview_questions,
            "skill_gap_analysis": self._skill_gap_analysis,
            "candidate_summary": self._candidate_summary,
            "experience_timeline": self._experience_timeline,
            "role_fit_assessment": self._role_fit_assessment,
            "linkedin_verification": self._linkedin_verification,
            "certification_lookup": self._certification_lookup,
        }
        handler = handlers.get(task_type)
        if not handler:
            return AgentTaskResult(task_type=task_type, success=False, error=f"Unknown task: {task_type}")
        try:
            return handler(context)
        except Exception as exc:
            logger.warning("ResumeAgent.%s failed: %s", task_type, exc)
            return AgentTaskResult(task_type=task_type, success=False, error=str(exc))

    def _generate_interview_questions(self, context: Dict[str, Any]) -> AgentTaskResult:
        """Generate relevant interview questions based on resume content."""
        resume_text = context.get("text", "") or context.get("resume", "")
        job_role = context.get("job_role", "")
        num_questions = min(context.get("num_questions", 10), 20)

        role_context = f" for the role of **{job_role}**" if job_role else ""
        prompt = (
            f"You are an expert technical interviewer. Based on the following resume, "
            f"generate {num_questions} targeted interview questions{role_context}.\n\n"
            f"Include a mix of:\n"
            f"- Technical questions that probe depth of their claimed skills\n"
            f"- Behavioral questions (STAR format) about their specific experience\n"
            f"- Situational questions relevant to their career level\n"
            f"- Questions that verify claims made in the resume\n\n"
            f"OUTPUT FORMAT — use this exact structure for each question:\n"
            f"1. [TECHNICAL] Question text here\n"
            f"   *Rationale: Why this question matters for this candidate*\n\n"
            f"EXAMPLE:\n"
            f"1. [TECHNICAL] You mention leading a migration to Kubernetes. "
            f"Walk me through how you handled rollback strategy for stateful services.\n"
            f"   *Rationale: Tests depth of K8s knowledge beyond basic deployment*\n\n"
            f"Resume:\n{resume_text[:4000]}\n\n"
            f"Generate exactly {num_questions} questions. Reference specific details from the resume."
        )
        output = self._generate(prompt, temperature=0.4, max_tokens=2048)
        return AgentTaskResult(
            task_type="generate_interview_questions",
            success=bool(output),
            output=output,
            structured_data={"num_questions": num_questions, "job_role": job_role},
        )

    def _skill_gap_analysis(self, context: Dict[str, Any]) -> AgentTaskResult:
        """Analyze skill gaps against a job description."""
        resume_text = context.get("text", "") or context.get("resume", "")
        job_description = context.get("job_description", "")
        if not job_description:
            return AgentTaskResult(
                task_type="skill_gap_analysis", success=False,
                error="job_description is required for skill gap analysis"
            )
        prompt = (
            f"Analyze the following resume against the job description. Identify:\n"
            f"1. MATCHING SKILLS: Skills the candidate has that match the requirements\n"
            f"2. SKILL GAPS: Required skills the candidate appears to lack\n"
            f"3. BONUS SKILLS: Additional skills the candidate has beyond requirements\n"
            f"4. OVERALL FIT SCORE: Rate 1-10 with justification\n\n"
            f"Resume:\n{resume_text[:3000]}\n\n"
            f"Job Description:\n{job_description[:2000]}"
        )
        output = self._generate(prompt, temperature=0.2)
        return AgentTaskResult(task_type="skill_gap_analysis", success=bool(output), output=output)

    def _candidate_summary(self, context: Dict[str, Any]) -> AgentTaskResult:
        """Generate an executive summary of a candidate."""
        resume_text = context.get("text", "") or context.get("resume", "")
        prompt = (
            f"Write a concise executive summary (3-5 sentences) of this candidate, "
            f"highlighting their strongest qualifications, career trajectory, and unique value.\n\n"
            f"Resume:\n{resume_text[:4000]}"
        )
        output = self._generate(prompt, temperature=0.3, max_tokens=512)
        return AgentTaskResult(task_type="candidate_summary", success=bool(output), output=output)

    def _experience_timeline(self, context: Dict[str, Any]) -> AgentTaskResult:
        """Extract and organize career timeline."""
        resume_text = context.get("text", "") or context.get("resume", "")
        prompt = (
            f"Extract a chronological career timeline from this resume. "
            f"For each position, list: dates, company, role, and key accomplishment.\n"
            f"Order from most recent to oldest.\n\n"
            f"Resume:\n{resume_text[:4000]}"
        )
        output = self._generate(prompt, temperature=0.1)
        return AgentTaskResult(task_type="experience_timeline", success=bool(output), output=output)

    def _role_fit_assessment(self, context: Dict[str, Any]) -> AgentTaskResult:
        """Assess candidate fit for a specific role."""
        resume_text = context.get("text", "") or context.get("resume", "")
        role = context.get("role", context.get("job_role", ""))
        if not role:
            return AgentTaskResult(
                task_type="role_fit_assessment", success=False, error="role is required"
            )
        prompt = (
            f"Assess this candidate's fit for the role of '{role}'.\n"
            f"Consider: relevant experience, skills match, career progression, and potential risks.\n"
            f"Provide a structured assessment with strengths, concerns, and recommendation.\n\n"
            f"Resume:\n{resume_text[:4000]}"
        )
        output = self._generate(prompt, temperature=0.3)
        return AgentTaskResult(task_type="role_fit_assessment", success=bool(output), output=output)

    def _linkedin_verification(self, context: Dict[str, Any]) -> AgentTaskResult:
        """Verify candidate details against LinkedIn profile."""
        name = context.get("name", "") or context.get("text", "")[:100]
        prompt = (
            f"Based on the following candidate information, formulate a LinkedIn "
            f"verification strategy. What should we look for to verify:\n"
            f"1. Employment history accuracy\n"
            f"2. Skills endorsements\n"
            f"3. Education verification\n"
            f"4. Professional network quality\n\n"
            f"Candidate: {name}\nContext: {context.get('text', '')[:2000]}"
        )
        output = self._generate(prompt, temperature=0.3)
        return AgentTaskResult(task_type="linkedin_verification", success=bool(output), output=output)

    def _certification_lookup(self, context: Dict[str, Any]) -> AgentTaskResult:
        """Verify certifications mentioned in resume."""
        text = context.get("text", "")
        prompt = (
            f"Identify all certifications mentioned in this resume and for each:\n"
            f"1. Certification name and issuing body\n"
            f"2. Whether it appears to be current or expired\n"
            f"3. Industry relevance and recognition level\n"
            f"4. Verification method (how to confirm validity)\n\n"
            f"Resume:\n{text[:4000]}"
        )
        output = self._generate(prompt, temperature=0.2)
        return AgentTaskResult(task_type="certification_lookup", success=bool(output), output=output)

# ---------------------------------------------------------------------------
# Medical Agent
# ---------------------------------------------------------------------------

class MedicalAgent(DomainAgent):
    """Specialized agent for medical document analysis tasks."""

    domain = "medical"
    use_thinking_model = False  # Generation-heavy (summaries, interpretations) — not reasoning

    def get_capabilities(self) -> List[str]:
        return [
            "drug_interaction_check",
            "treatment_plan_review",
            "clinical_summary",
            "lab_result_interpretation",
            "patient_history_timeline",
            "nice_guidance_lookup",
            "evidence_based_review",
        ]

    def execute(self, task_type: str, context: Dict[str, Any]) -> AgentTaskResult:
        handlers = {
            "drug_interaction_check": self._drug_interaction_check,
            "treatment_plan_review": self._treatment_plan_review,
            "clinical_summary": self._clinical_summary,
            "lab_result_interpretation": self._lab_result_interpretation,
            "patient_history_timeline": self._patient_history_timeline,
            "nice_guidance_lookup": self._nice_guidance_lookup,
            "evidence_based_review": self._evidence_based_review,
        }
        handler = handlers.get(task_type)
        if not handler:
            return AgentTaskResult(task_type=task_type, success=False, error=f"Unknown task: {task_type}")
        try:
            return handler(context)
        except Exception as exc:
            logger.warning("MedicalAgent.%s failed: %s", task_type, exc)
            return AgentTaskResult(task_type=task_type, success=False, error=str(exc))

    def _drug_interaction_check(self, context: Dict[str, Any]) -> AgentTaskResult:
        """Analyze potential drug interactions from medical records."""
        text = context.get("text", "")
        prompt = (
            f"Analyze the following medical document for potential drug interactions.\n\n"
            f"For each medication found:\n"
            f"1. **Name** and dosage as stated in the document\n"
            f"2. Drug class (e.g., SSRI, beta-blocker, anticoagulant)\n\n"
            f"Then for each potential interaction pair:\n"
            f"- **Drugs involved**: Drug A + Drug B\n"
            f"- **Interaction type**: pharmacokinetic/pharmacodynamic\n"
            f"- **Severity**: HIGH / MEDIUM / LOW\n"
            f"- **Clinical effect**: What could happen\n"
            f"- **Management**: How to mitigate\n\n"
            f"Present interactions in a markdown table.\n"
            f"NOTE: For informational purposes only. Always consult a healthcare provider.\n\n"
            f"Document:\n{text[:4000]}"
        )
        output = self._generate(prompt, temperature=0.1)
        return AgentTaskResult(
            task_type="drug_interaction_check", success=bool(output), output=output,
            reasoning="Analyzed medications and cross-referenced for known interactions"
        )

    def _treatment_plan_review(self, context: Dict[str, Any]) -> AgentTaskResult:
        """Review and summarize a treatment plan."""
        text = context.get("text", "")
        prompt = (
            f"Review the following treatment plan and provide:\n"
            f"1. Treatment objectives\n"
            f"2. Prescribed interventions (medications, procedures, therapies)\n"
            f"3. Timeline and milestones\n"
            f"4. Follow-up schedule\n"
            f"5. Potential concerns or gaps in the plan\n\n"
            f"Document:\n{text[:4000]}"
        )
        output = self._generate(prompt, temperature=0.2)
        return AgentTaskResult(task_type="treatment_plan_review", success=bool(output), output=output)

    def _clinical_summary(self, context: Dict[str, Any]) -> AgentTaskResult:
        """Generate a clinical summary from medical records."""
        text = context.get("text", "")
        prompt = (
            f"Generate a concise clinical summary from the following medical document.\n"
            f"Include: presenting complaints, diagnosis, key findings, treatment plan, and prognosis.\n\n"
            f"Document:\n{text[:4000]}"
        )
        output = self._generate(prompt, temperature=0.2, max_tokens=1024)
        return AgentTaskResult(task_type="clinical_summary", success=bool(output), output=output)

    def _lab_result_interpretation(self, context: Dict[str, Any]) -> AgentTaskResult:
        """Interpret lab results with clinical context."""
        text = context.get("text", "")
        prompt = (
            f"Interpret the following lab results using this format:\n\n"
            f"| Test | Value | Reference Range | Status | Clinical Significance |\n"
            f"|------|-------|-----------------|--------|----------------------|\n"
            f"| HbA1c | 7.2% | <7% | **HIGH** | Sub-optimal glycemic control |\n\n"
            f"For each abnormal result:\n"
            f"- **Status**: HIGH / LOW / CRITICAL\n"
            f"- **Clinical significance**: What this means for the patient\n"
            f"- **Possible conditions**: Only list conditions supported by the lab pattern\n\n"
            f"End with an overall assessment paragraph.\n"
            f"NOTE: For informational purposes only.\n\n"
            f"Lab Results:\n{text[:4000]}"
        )
        output = self._generate(prompt, temperature=0.1)
        return AgentTaskResult(task_type="lab_result_interpretation", success=bool(output), output=output)

    def _patient_history_timeline(self, context: Dict[str, Any]) -> AgentTaskResult:
        """Build a chronological patient history timeline."""
        text = context.get("text", "")
        prompt = (
            f"Extract a chronological medical history timeline from this document.\n"
            f"For each event: date, type (diagnosis/procedure/medication/visit), description.\n"
            f"Order chronologically.\n\n"
            f"Document:\n{text[:4000]}"
        )
        output = self._generate(prompt, temperature=0.1)
        return AgentTaskResult(task_type="patient_history_timeline", success=bool(output), output=output)

    def _nice_guidance_lookup(self, context: Dict[str, Any]) -> AgentTaskResult:
        """Look up NICE guidance for a clinical condition or medication."""
        condition = context.get("condition", "") or context.get("query", "")
        text = context.get("text", "")
        prompt = (
            f"For the condition/medication '{condition}', provide:\n"
            f"1. Expected NICE guidance recommendations\n"
            f"2. Key clinical pathways\n"
            f"3. Recommended monitoring and follow-up\n"
            f"4. Red flags requiring urgent referral\n"
            f"NOTE: For informational purposes only.\n\n"
            f"Clinical context:\n{text[:3000]}"
        )
        output = self._generate(prompt, temperature=0.1)
        return AgentTaskResult(
            task_type="nice_guidance_lookup", success=bool(output), output=output,
            reasoning="Referenced NICE clinical guidelines framework"
        )

    def _evidence_based_review(self, context: Dict[str, Any]) -> AgentTaskResult:
        """Review clinical decisions against evidence-based guidelines."""
        text = context.get("text", "")
        prompt = (
            f"Review the following clinical document against evidence-based guidelines.\n"
            f"Evaluate:\n"
            f"1. Alignment with current best practice guidelines (NICE, WHO, etc.)\n"
            f"2. Evidence level for each treatment decision\n"
            f"3. Any deviations from standard care pathways\n"
            f"4. Recommendations for improvement\n"
            f"NOTE: For informational purposes only.\n\n"
            f"Document:\n{text[:4000]}"
        )
        output = self._generate(prompt, temperature=0.1)
        return AgentTaskResult(task_type="evidence_based_review", success=bool(output), output=output)

# ---------------------------------------------------------------------------
# Legal Agent
# ---------------------------------------------------------------------------

class LegalAgent(DomainAgent):
    """Specialized agent for legal document analysis tasks."""

    domain = "legal"

    def get_capabilities(self) -> List[str]:
        return [
            "clause_risk_assessment",
            "compliance_check",
            "contract_comparison",
            "key_terms_extraction",
            "obligation_tracker",
            "jurisdiction_analysis",
            "country_specific_compliance",
        ]

    def execute(self, task_type: str, context: Dict[str, Any]) -> AgentTaskResult:
        handlers = {
            "clause_risk_assessment": self._clause_risk_assessment,
            "compliance_check": self._compliance_check,
            "contract_comparison": self._contract_comparison,
            "key_terms_extraction": self._key_terms_extraction,
            "obligation_tracker": self._obligation_tracker,
            "jurisdiction_analysis": self._jurisdiction_analysis,
            "country_specific_compliance": self._country_specific_compliance,
        }
        handler = handlers.get(task_type)
        if not handler:
            return AgentTaskResult(task_type=task_type, success=False, error=f"Unknown task: {task_type}")
        try:
            return handler(context)
        except Exception as exc:
            logger.warning("LegalAgent.%s failed: %s", task_type, exc)
            return AgentTaskResult(task_type=task_type, success=False, error=str(exc))

    def _clause_risk_assessment(self, context: Dict[str, Any]) -> AgentTaskResult:
        text = context.get("text", "")
        prompt = (
            f"Analyze the following legal document for risky clauses.\n\n"
            f"RISK CLASSIFICATION:\n"
            f"- **HIGH**: Creates unilateral liability, unreasonable indemnification, "
            f"unlimited penalties, or is potentially unenforceable\n"
            f"- **MEDIUM**: Ambiguous language, missing definitions, one-sided but common\n"
            f"- **LOW**: Standard boilerplate, well-balanced provisions\n\n"
            f"For each risky clause found:\n\n"
            f"| # | Clause (quoted) | Risk | Impact | Suggested Alternative |\n"
            f"|---|-----------------|------|--------|----------------------|\n\n"
            f"EXAMPLE:\n"
            f"| 1 | \"Party A shall bear unlimited liability...\" | **HIGH** | "
            f"Unrestricted financial exposure | Add liability cap: \"not to exceed 2x contract value\" |\n\n"
            f"End with a **Risk Summary**: total HIGH/MEDIUM/LOW counts and overall assessment.\n\n"
            f"Document:\n{text[:4000]}"
        )
        output = self._generate(prompt, temperature=0.1)
        return AgentTaskResult(task_type="clause_risk_assessment", success=bool(output), output=output)

    def _compliance_check(self, context: Dict[str, Any]) -> AgentTaskResult:
        text = context.get("text", "")
        framework = context.get("framework", "general")
        prompt = (
            f"Check the following document for compliance with {framework} standards.\n"
            f"Identify: compliant areas, non-compliant areas, and recommendations.\n\n"
            f"Document:\n{text[:4000]}"
        )
        output = self._generate(prompt, temperature=0.1)
        return AgentTaskResult(task_type="compliance_check", success=bool(output), output=output)

    def _contract_comparison(self, context: Dict[str, Any]) -> AgentTaskResult:
        text1 = context.get("text", "") or context.get("contract1", "")
        text2 = context.get("text2", "") or context.get("contract2", "")
        if not text2:
            return AgentTaskResult(
                task_type="contract_comparison", success=False,
                error="Two contracts (text and text2) are required for comparison"
            )
        prompt = (
            f"Compare the following two contracts and identify:\n"
            f"1. Key differences in terms\n"
            f"2. Clauses present in one but not the other\n"
            f"3. Conflicting provisions\n"
            f"4. Which contract is more favorable and why\n\n"
            f"Contract 1:\n{text1[:2500]}\n\n"
            f"Contract 2:\n{text2[:2500]}"
        )
        output = self._generate(prompt, temperature=0.2, max_tokens=2048)
        return AgentTaskResult(task_type="contract_comparison", success=bool(output), output=output)

    def _key_terms_extraction(self, context: Dict[str, Any]) -> AgentTaskResult:
        text = context.get("text", "")
        prompt = (
            f"Extract all key legal terms and definitions from this document.\n"
            f"For each: term, definition, section where it appears, and significance.\n\n"
            f"Document:\n{text[:4000]}"
        )
        output = self._generate(prompt, temperature=0.1)
        return AgentTaskResult(task_type="key_terms_extraction", success=bool(output), output=output)

    def _obligation_tracker(self, context: Dict[str, Any]) -> AgentTaskResult:
        text = context.get("text", "")
        prompt = (
            f"Extract all obligations and deadlines from this legal document.\n"
            f"For each: who is obligated, what they must do, deadline, and consequences of non-compliance.\n\n"
            f"Document:\n{text[:4000]}"
        )
        output = self._generate(prompt, temperature=0.1)
        return AgentTaskResult(task_type="obligation_tracker", success=bool(output), output=output)

    def _jurisdiction_analysis(self, context: Dict[str, Any]) -> AgentTaskResult:
        """Analyze the jurisdictional context of a legal document."""
        text = context.get("text", "")
        country = context.get("country", "")
        country_ctx = f" under {country} law" if country else ""
        prompt = (
            f"Analyze the jurisdictional aspects of this legal document{country_ctx}.\n"
            f"Identify:\n"
            f"1. Governing law and jurisdiction clauses\n"
            f"2. Applicable legal system (common law, civil law, hybrid)\n"
            f"3. Court hierarchy and dispute resolution mechanism\n"
            f"4. Cross-border implications if any\n"
            f"5. Enforceability considerations\n\n"
            f"Document:\n{text[:4000]}"
        )
        output = self._generate(prompt, temperature=0.1)
        return AgentTaskResult(task_type="jurisdiction_analysis", success=bool(output), output=output)

    def _country_specific_compliance(self, context: Dict[str, Any]) -> AgentTaskResult:
        """Check compliance against country-specific regulations."""
        text = context.get("text", "")
        country = context.get("country", "")
        if not country:
            return AgentTaskResult(
                task_type="country_specific_compliance", success=False,
                error="country parameter is required"
            )
        prompt = (
            f"Check this document for compliance with {country} legal requirements.\n"
            f"Consider:\n"
            f"1. Country-specific regulatory requirements\n"
            f"2. Data protection laws applicable in {country}\n"
            f"3. Consumer protection requirements\n"
            f"4. Industry-specific regulations\n"
            f"5. Employment law requirements (if applicable)\n\n"
            f"Document:\n{text[:4000]}"
        )
        output = self._generate(prompt, temperature=0.1)
        return AgentTaskResult(
            task_type="country_specific_compliance", success=bool(output), output=output,
            structured_data={"country": country},
        )

# ---------------------------------------------------------------------------
# Invoice/Financial Agent
# ---------------------------------------------------------------------------

class InvoiceAgent(DomainAgent):
    """Specialized agent for invoice and financial document analysis."""

    domain = "invoice"

    def get_capabilities(self) -> List[str]:
        return [
            "payment_anomaly_detection",
            "vendor_analysis",
            "financial_summary",
            "expense_categorization",
            "duplicate_detection",
        ]

    def execute(self, task_type: str, context: Dict[str, Any]) -> AgentTaskResult:
        handlers = {
            "payment_anomaly_detection": self._payment_anomaly_detection,
            "vendor_analysis": self._vendor_analysis,
            "financial_summary": self._financial_summary,
            "expense_categorization": self._expense_categorization,
            "duplicate_detection": self._duplicate_detection,
        }
        handler = handlers.get(task_type)
        if not handler:
            return AgentTaskResult(task_type=task_type, success=False, error=f"Unknown task: {task_type}")
        try:
            return handler(context)
        except Exception as exc:
            logger.warning("InvoiceAgent.%s failed: %s", task_type, exc)
            return AgentTaskResult(task_type=task_type, success=False, error=str(exc))

    def _payment_anomaly_detection(self, context: Dict[str, Any]) -> AgentTaskResult:
        text = context.get("text", "")
        prompt = (
            f"Analyze the following invoice/financial data for anomalies:\n"
            f"1. Unusual amounts (significantly higher/lower than typical)\n"
            f"2. Duplicate charges\n"
            f"3. Missing information (dates, reference numbers)\n"
            f"4. Inconsistencies in calculations\n\n"
            f"Document:\n{text[:4000]}"
        )
        output = self._generate(prompt, temperature=0.1)
        return AgentTaskResult(task_type="payment_anomaly_detection", success=bool(output), output=output)

    def _vendor_analysis(self, context: Dict[str, Any]) -> AgentTaskResult:
        text = context.get("text", "")
        prompt = (
            f"Analyze vendor information from the following documents:\n"
            f"1. Vendor identification and contact details\n"
            f"2. Payment terms and conditions\n"
            f"3. Pricing patterns\n"
            f"4. Historical transaction summary\n\n"
            f"Document:\n{text[:4000]}"
        )
        output = self._generate(prompt, temperature=0.2)
        return AgentTaskResult(task_type="vendor_analysis", success=bool(output), output=output)

    def _financial_summary(self, context: Dict[str, Any]) -> AgentTaskResult:
        text = context.get("text", "")
        prompt = (
            f"Generate a financial summary from the following document:\n"
            f"Include: total amounts, line item breakdown, tax details, payment terms, and key dates.\n\n"
            f"Document:\n{text[:4000]}"
        )
        output = self._generate(prompt, temperature=0.1)
        return AgentTaskResult(task_type="financial_summary", success=bool(output), output=output)

    def _expense_categorization(self, context: Dict[str, Any]) -> AgentTaskResult:
        text = context.get("text", "")
        prompt = (
            f"Categorize expenses from the following financial document:\n"
            f"Group items into categories (e.g., supplies, services, travel, utilities).\n"
            f"Provide totals per category and percentage breakdown.\n\n"
            f"Document:\n{text[:4000]}"
        )
        output = self._generate(prompt, temperature=0.1)
        return AgentTaskResult(task_type="expense_categorization", success=bool(output), output=output)

    def _duplicate_detection(self, context: Dict[str, Any]) -> AgentTaskResult:
        text = context.get("text", "")
        prompt = (
            f"Check the following financial records for potential duplicates:\n"
            f"Look for: same amounts on similar dates, identical descriptions, matching reference numbers.\n\n"
            f"Document:\n{text[:4000]}"
        )
        output = self._generate(prompt, temperature=0.1)
        return AgentTaskResult(task_type="duplicate_detection", success=bool(output), output=output)

# ---------------------------------------------------------------------------
# Content Agent
# ---------------------------------------------------------------------------

class ContentAgent(DomainAgent):
    """Specialized agent for content generation tasks.

    Wraps: creator, email_drafting, code_docs tools.
    """

    domain = "content"
    use_thinking_model = False  # Generation-heavy — always uses DocWain-Agent

    def get_capabilities(self) -> List[str]:
        return [
            "generate_content",
            "draft_email",
            "generate_documentation",
            "rewrite_text",
            "create_presentation",
        ]

    def execute(self, task_type: str, context: Dict[str, Any]) -> AgentTaskResult:
        handlers = {
            "generate_content": self._generate_content,
            "draft_email": self._draft_email,
            "generate_documentation": self._generate_documentation,
            "rewrite_text": self._rewrite_text,
            "create_presentation": self._create_presentation,
        }
        handler = handlers.get(task_type)
        if not handler:
            return AgentTaskResult(task_type=task_type, success=False, error=f"Unknown task: {task_type}")
        try:
            return handler(context)
        except Exception as exc:
            logger.warning("ContentAgent.%s failed: %s", task_type, exc)
            return AgentTaskResult(task_type=task_type, success=False, error=str(exc))

    def _generate_content(self, context: Dict[str, Any]) -> AgentTaskResult:
        """Create summaries, blogs, SOPs, FAQs, or slide outlines from source text."""
        text = context.get("text", "")
        content_type = context.get("content_type", "summary")
        audience = context.get("audience", "general")
        prompt = (
            f"You are a professional content creator. Based on the following source material, "
            f"generate a {content_type} targeted at a {audience} audience.\n\n"
            f"Requirements:\n"
            f"- Maintain factual accuracy from the source\n"
            f"- Use clear, engaging language appropriate for the audience\n"
            f"- Structure the output with headings and bullet points where appropriate\n"
            f"- If creating an SOP, use numbered steps with clear actions\n"
            f"- If creating a FAQ, use Q&A format\n\n"
            f"Source Material:\n{text[:4000]}"
        )
        output = self._generate(prompt, temperature=0.4, max_tokens=2048)
        return AgentTaskResult(
            task_type="generate_content", success=bool(output), output=output,
            structured_data={"content_type": content_type, "audience": audience},
        )

    def _draft_email(self, context: Dict[str, Any]) -> AgentTaskResult:
        """Compose professional emails based on context and instructions."""
        text = context.get("text", "")
        tone = context.get("tone", "professional")
        recipient = context.get("recipient", "")
        subject = context.get("subject", "")
        recipient_ctx = f" to {recipient}" if recipient else ""
        subject_ctx = f" regarding '{subject}'" if subject else ""
        prompt = (
            f"Draft a {tone} email{recipient_ctx}{subject_ctx}.\n\n"
            f"Use the following content as the basis for the email:\n{text[:4000]}\n\n"
            f"Requirements:\n"
            f"- Include a clear subject line\n"
            f"- Use appropriate greeting and sign-off\n"
            f"- Be concise but comprehensive\n"
            f"- Maintain a {tone} tone throughout\n"
            f"- Include a clear call-to-action if applicable"
        )
        output = self._generate(prompt, temperature=0.4, max_tokens=1024)
        return AgentTaskResult(
            task_type="draft_email", success=bool(output), output=output,
            structured_data={"tone": tone, "recipient": recipient, "subject": subject},
        )

    def _generate_documentation(self, context: Dict[str, Any]) -> AgentTaskResult:
        """Create technical documentation, API docs, or user guides."""
        text = context.get("text", "")
        doc_type = context.get("doc_type", "technical documentation")
        prompt = (
            f"Generate {doc_type} from the following source material.\n\n"
            f"Requirements:\n"
            f"- Use clear technical language\n"
            f"- Include sections: Overview, Details, Usage/Examples, Notes\n"
            f"- Add code examples or usage patterns if applicable\n"
            f"- Define any technical terms on first use\n"
            f"- Include parameter descriptions and return values for API docs\n\n"
            f"Source Material:\n{text[:4000]}"
        )
        output = self._generate(prompt, temperature=0.3, max_tokens=2048)
        return AgentTaskResult(
            task_type="generate_documentation", success=bool(output), output=output,
            structured_data={"doc_type": doc_type},
        )

    def _rewrite_text(self, context: Dict[str, Any]) -> AgentTaskResult:
        """Improve or rewrite text for clarity, tone, or target audience."""
        text = context.get("text", "")
        target_tone = context.get("tone", "professional")
        target_audience = context.get("audience", "general")
        instructions = context.get("instructions", "")
        extra = f"\nAdditional instructions: {instructions}" if instructions else ""
        prompt = (
            f"Rewrite the following text to improve clarity and readability.\n"
            f"Target tone: {target_tone}\n"
            f"Target audience: {target_audience}\n{extra}\n\n"
            f"Requirements:\n"
            f"- Preserve all factual content and key information\n"
            f"- Improve sentence structure and flow\n"
            f"- Remove redundancy and jargon where possible\n"
            f"- Ensure consistency in style and voice\n\n"
            f"Original Text:\n{text[:4000]}"
        )
        output = self._generate(prompt, temperature=0.4, max_tokens=2048)
        return AgentTaskResult(
            task_type="rewrite_text", success=bool(output), output=output,
            structured_data={"tone": target_tone, "audience": target_audience},
        )

    def _create_presentation(self, context: Dict[str, Any]) -> AgentTaskResult:
        """Create slide deck content from document material."""
        text = context.get("text", "")
        num_slides = min(context.get("num_slides", 10), 30)
        topic = context.get("topic", "")
        topic_ctx = f" on '{topic}'" if topic else ""
        prompt = (
            f"Create a {num_slides}-slide presentation outline{topic_ctx} "
            f"from the following source material.\n\n"
            f"For each slide provide:\n"
            f"- Slide title\n"
            f"- 3-5 bullet points (concise, impactful)\n"
            f"- Speaker notes (1-2 sentences)\n\n"
            f"Include an opening slide, agenda, main content slides, and a closing slide.\n\n"
            f"Source Material:\n{text[:4000]}"
        )
        output = self._generate(prompt, temperature=0.4, max_tokens=2048)
        return AgentTaskResult(
            task_type="create_presentation", success=bool(output), output=output,
            structured_data={"num_slides": num_slides, "topic": topic},
        )

# ---------------------------------------------------------------------------
# Translator Agent
# ---------------------------------------------------------------------------

class TranslatorAgent(DomainAgent):
    """Specialized agent for translation and multilingual tasks.

    Wraps: translator tool.
    """

    domain = "translation"
    use_thinking_model = False  # Generation-heavy — always uses DocWain-Agent

    def get_capabilities(self) -> List[str]:
        return [
            "translate_text",
            "detect_language",
            "multilingual_summary",
            "localize_content",
        ]

    def execute(self, task_type: str, context: Dict[str, Any]) -> AgentTaskResult:
        handlers = {
            "translate_text": self._translate_text,
            "detect_language": self._detect_language,
            "multilingual_summary": self._multilingual_summary,
            "localize_content": self._localize_content,
        }
        handler = handlers.get(task_type)
        if not handler:
            return AgentTaskResult(task_type=task_type, success=False, error=f"Unknown task: {task_type}")
        try:
            return handler(context)
        except Exception as exc:
            logger.warning("TranslatorAgent.%s failed: %s", task_type, exc)
            return AgentTaskResult(task_type=task_type, success=False, error=str(exc))

    def _translate_text(self, context: Dict[str, Any]) -> AgentTaskResult:
        """Translate text to a target language."""
        text = context.get("text", "")
        target_language = context.get("target_language", "English")
        source_language = context.get("source_language", "")
        source_ctx = f" from {source_language}" if source_language else ""
        prompt = (
            f"Translate the following text{source_ctx} to {target_language}.\n\n"
            f"Requirements:\n"
            f"- Preserve the original meaning and nuance\n"
            f"- Maintain formatting and structure\n"
            f"- Use natural, idiomatic expressions in {target_language}\n"
            f"- Preserve technical terms where appropriate\n\n"
            f"Text to translate:\n{text[:4000]}"
        )
        output = self._generate(prompt, temperature=0.2, max_tokens=2048)
        return AgentTaskResult(
            task_type="translate_text", success=bool(output), output=output,
            structured_data={"target_language": target_language, "source_language": source_language},
        )

    def _detect_language(self, context: Dict[str, Any]) -> AgentTaskResult:
        """Identify the language of input text."""
        text = context.get("text", "")
        prompt = (
            f"Identify the language of the following text.\n"
            f"Provide:\n"
            f"1. Primary language name (e.g., English, Spanish, Mandarin Chinese)\n"
            f"2. ISO 639-1 language code (e.g., en, es, zh)\n"
            f"3. Confidence level (high/medium/low)\n"
            f"4. Any secondary languages detected (if mixed-language text)\n\n"
            f"Text:\n{text[:4000]}"
        )
        output = self._generate(prompt, temperature=0.1, max_tokens=512)
        return AgentTaskResult(task_type="detect_language", success=bool(output), output=output)

    def _multilingual_summary(self, context: Dict[str, Any]) -> AgentTaskResult:
        """Summarize a document and provide key points in a target language."""
        text = context.get("text", "")
        target_language = context.get("target_language", "English")
        prompt = (
            f"Summarize the following document and provide the summary in {target_language}.\n\n"
            f"Requirements:\n"
            f"- First identify the source language\n"
            f"- Create a concise summary (5-7 sentences) in {target_language}\n"
            f"- List 5-8 key points in {target_language}\n"
            f"- Note any culturally significant terms with their original form in parentheses\n\n"
            f"Document:\n{text[:4000]}"
        )
        output = self._generate(prompt, temperature=0.3, max_tokens=1024)
        return AgentTaskResult(
            task_type="multilingual_summary", success=bool(output), output=output,
            structured_data={"target_language": target_language},
        )

    def _localize_content(self, context: Dict[str, Any]) -> AgentTaskResult:
        """Adapt content for a specific regional or cultural context."""
        text = context.get("text", "")
        target_region = context.get("target_region", "")
        target_language = context.get("target_language", "")
        if not target_region and not target_language:
            return AgentTaskResult(
                task_type="localize_content", success=False,
                error="target_region or target_language is required for localization"
            )
        region_ctx = target_region or target_language
        prompt = (
            f"Localize the following content for {region_ctx}.\n\n"
            f"Requirements:\n"
            f"- Adapt cultural references to the target region\n"
            f"- Convert units, currencies, and date formats as appropriate\n"
            f"- Adjust tone and formality to match regional conventions\n"
            f"- Preserve the core message and intent\n"
            f"- Note any content that cannot be directly localized\n\n"
            f"Content:\n{text[:4000]}"
        )
        output = self._generate(prompt, temperature=0.3, max_tokens=2048)
        return AgentTaskResult(
            task_type="localize_content", success=bool(output), output=output,
            structured_data={"target_region": target_region, "target_language": target_language},
        )

# ---------------------------------------------------------------------------
# Tutor Agent
# ---------------------------------------------------------------------------

class TutorAgent(DomainAgent):
    """Specialized agent for educational content and tutoring tasks.

    Wraps: tutor tool.
    """

    domain = "education"
    use_thinking_model = False  # Generation-heavy — always uses DocWain-Agent

    def get_capabilities(self) -> List[str]:
        return [
            "create_lesson",
            "generate_quiz",
            "explain_concept",
            "study_guide",
        ]

    def execute(self, task_type: str, context: Dict[str, Any]) -> AgentTaskResult:
        handlers = {
            "create_lesson": self._create_lesson,
            "generate_quiz": self._generate_quiz,
            "explain_concept": self._explain_concept,
            "study_guide": self._study_guide,
        }
        handler = handlers.get(task_type)
        if not handler:
            return AgentTaskResult(task_type=task_type, success=False, error=f"Unknown task: {task_type}")
        try:
            return handler(context)
        except Exception as exc:
            logger.warning("TutorAgent.%s failed: %s", task_type, exc)
            return AgentTaskResult(task_type=task_type, success=False, error=str(exc))

    def _create_lesson(self, context: Dict[str, Any]) -> AgentTaskResult:
        """Build a structured lesson from document content."""
        text = context.get("text", "")
        level = context.get("level", "intermediate")
        topic = context.get("topic", "")
        topic_ctx = f" on '{topic}'" if topic else ""
        prompt = (
            f"Create a structured lesson{topic_ctx} at the {level} level "
            f"based on the following material.\n\n"
            f"Structure the lesson with:\n"
            f"1. Learning Objectives (3-5 clear outcomes)\n"
            f"2. Prerequisites (what the learner should already know)\n"
            f"3. Introduction (hook and context)\n"
            f"4. Main Content (broken into 3-4 sections with explanations and examples)\n"
            f"5. Practice Exercises (2-3 hands-on activities)\n"
            f"6. Summary and Key Takeaways\n\n"
            f"Source Material:\n{text[:4000]}"
        )
        output = self._generate(prompt, temperature=0.4, max_tokens=2048)
        return AgentTaskResult(
            task_type="create_lesson", success=bool(output), output=output,
            structured_data={"level": level, "topic": topic},
        )

    def _generate_quiz(self, context: Dict[str, Any]) -> AgentTaskResult:
        """Create quiz questions from document material."""
        text = context.get("text", "")
        num_questions = min(context.get("num_questions", 10), 25)
        difficulty = context.get("difficulty", "mixed")
        prompt = (
            f"Generate {num_questions} quiz questions (difficulty: {difficulty}) "
            f"from the following material.\n\n"
            f"Include a mix of question types:\n"
            f"- Multiple choice (4 options, mark correct answer)\n"
            f"- True/False with explanation\n"
            f"- Short answer\n"
            f"- Application/scenario-based questions\n\n"
            f"For each question, provide:\n"
            f"- The question\n"
            f"- Answer options (if applicable)\n"
            f"- Correct answer\n"
            f"- Brief explanation of why the answer is correct\n\n"
            f"Source Material:\n{text[:4000]}"
        )
        output = self._generate(prompt, temperature=0.4, max_tokens=2048)
        return AgentTaskResult(
            task_type="generate_quiz", success=bool(output), output=output,
            structured_data={"num_questions": num_questions, "difficulty": difficulty},
        )

    def _explain_concept(self, context: Dict[str, Any]) -> AgentTaskResult:
        """Break down complex concepts at a specified learning level."""
        text = context.get("text", "")
        concept = context.get("concept", "")
        level = context.get("level", "beginner")
        concept_ctx = f"the concept of '{concept}'" if concept else "the key concepts"
        prompt = (
            f"Explain {concept_ctx} at a {level} level "
            f"based on the following material.\n\n"
            f"Requirements:\n"
            f"- Start with a simple, relatable analogy\n"
            f"- Break down into digestible parts\n"
            f"- Use concrete examples from the source material\n"
            f"- Build from simple to complex\n"
            f"- Define technical terms in plain language\n"
            f"- Include a 'Check Your Understanding' question at the end\n\n"
            f"Source Material:\n{text[:4000]}"
        )
        output = self._generate(prompt, temperature=0.4, max_tokens=1024)
        return AgentTaskResult(
            task_type="explain_concept", success=bool(output), output=output,
            structured_data={"concept": concept, "level": level},
        )

    def _study_guide(self, context: Dict[str, Any]) -> AgentTaskResult:
        """Create a study guide with key points, flashcards, and review questions."""
        text = context.get("text", "")
        topic = context.get("topic", "")
        topic_ctx = f" for '{topic}'" if topic else ""
        prompt = (
            f"Create a comprehensive study guide{topic_ctx} "
            f"from the following material.\n\n"
            f"Include:\n"
            f"1. Key Concepts Summary (bullet points)\n"
            f"2. Important Definitions and Terms\n"
            f"3. Flashcards (10-15 cards in Term | Definition format)\n"
            f"4. Review Questions (5-8 questions covering main topics)\n"
            f"5. Common Misconceptions to Avoid\n"
            f"6. Suggested Study Strategy\n\n"
            f"Source Material:\n{text[:4000]}"
        )
        output = self._generate(prompt, temperature=0.3, max_tokens=2048)
        return AgentTaskResult(
            task_type="study_guide", success=bool(output), output=output,
            structured_data={"topic": topic},
        )

# ---------------------------------------------------------------------------
# Image Agent
# ---------------------------------------------------------------------------

class ImageAgent(DomainAgent):
    """Specialized agent for image analysis and processing tasks.

    MoE: Uses glm-ocr for multimodal vision analysis when image_bytes are
    provided in context, falls back to text-based analysis via LLM.
    """

    domain = "image"

    def get_capabilities(self) -> List[str]:
        return [
            "analyze_image",
            "extract_text_from_image",
            "describe_image",
            "extract_data_from_image",
        ]

    def _get_vision_client(self):
        """Get VisionOCRClient for multimodal image analysis."""
        try:
            from src.llm.vision_ocr import get_vision_ocr_client
            client = get_vision_ocr_client()
            if client and client.is_available():
                return client
        except Exception:
            pass
        return None

    def execute(self, task_type: str, context: Dict[str, Any]) -> AgentTaskResult:
        handlers = {
            "analyze_image": self._analyze_image,
            "extract_text_from_image": self._extract_text_from_image,
            "describe_image": self._describe_image,
            "extract_data_from_image": self._extract_data_from_image,
        }
        handler = handlers.get(task_type)
        if not handler:
            return AgentTaskResult(task_type=task_type, success=False, error=f"Unknown task: {task_type}")
        try:
            return handler(context)
        except Exception as exc:
            logger.warning("ImageAgent.%s failed: %s", task_type, exc)
            return AgentTaskResult(task_type=task_type, success=False, error=str(exc))

    def _analyze_image(self, context: Dict[str, Any]) -> AgentTaskResult:
        """Perform image analysis — vision-first (glm-ocr), text fallback."""
        image_bytes = context.get("image_bytes") or context.get("image")
        text = context.get("text", "")

        # Try multimodal vision analysis first
        if image_bytes:
            vision = self._get_vision_client()
            if vision:
                output, confidence = vision.analyze_image(image_bytes, analysis_type="general")
                if output:
                    return AgentTaskResult(
                        task_type="analyze_image", success=True, output=output,
                        structured_data={"method": "vision_multimodal", "confidence": confidence},
                    )

        # Fallback: text-based analysis
        prompt = (
            f"Analyze the following image content (OCR/extracted text) and provide:\n"
            f"1. Document type identification (form, letter, receipt, diagram, etc.)\n"
            f"2. Layout structure (headers, sections, tables, figures)\n"
            f"3. Key information extracted with locations\n"
            f"4. Quality assessment (readability, completeness)\n"
            f"5. Any notable visual elements or formatting\n\n"
            f"Extracted Content:\n{text[:4000]}"
        )
        output = self._generate(prompt, temperature=0.2)
        return AgentTaskResult(
            task_type="analyze_image", success=bool(output), output=output,
            structured_data={"method": "text_based"},
        )

    def _extract_text_from_image(self, context: Dict[str, Any]) -> AgentTaskResult:
        """Extract text — vision OCR first, then text cleanup fallback."""
        image_bytes = context.get("image_bytes") or context.get("image")
        text = context.get("text", "")

        # Try vision OCR first
        if image_bytes:
            vision = self._get_vision_client()
            if vision:
                output, confidence = vision.ocr_image(image_bytes)
                if output:
                    return AgentTaskResult(
                        task_type="extract_text_from_image", success=True, output=output,
                        structured_data={"method": "vision_ocr", "confidence": confidence},
                    )

        # Fallback: clean up existing OCR text
        prompt = (
            f"The following text was extracted from an image. Clean it up and organize it:\n\n"
            f"Requirements:\n"
            f"- Fix OCR errors (common character substitutions: 0/O, 1/l/I, etc.)\n"
            f"- Restore paragraph and line structure\n"
            f"- Preserve tables in tabular format\n"
            f"- Mark any uncertain or illegible sections with [unclear]\n"
            f"- Maintain original language and formatting intent\n\n"
            f"Raw OCR Text:\n{text[:4000]}"
        )
        output = self._generate(prompt, temperature=0.1, max_tokens=2048)
        return AgentTaskResult(
            task_type="extract_text_from_image", success=bool(output), output=output,
            structured_data={"method": "text_cleanup"},
        )

    def _describe_image(self, context: Dict[str, Any]) -> AgentTaskResult:
        """Describe image — vision-first, text fallback."""
        image_bytes = context.get("image_bytes") or context.get("image")
        text = context.get("text", "")

        if image_bytes:
            vision = self._get_vision_client()
            if vision:
                output, confidence = vision.analyze_image(image_bytes, analysis_type="photo")
                if output:
                    return AgentTaskResult(
                        task_type="describe_image", success=True, output=output,
                        structured_data={"method": "vision_description", "confidence": confidence},
                    )

        prompt = (
            f"Based on the following extracted content from an image, "
            f"generate a natural language description of what the image contains.\n\n"
            f"Include:\n"
            f"- Overall description (what type of document/image is this)\n"
            f"- Main content and purpose\n"
            f"- Key data points or information visible\n"
            f"- Layout and visual organization\n"
            f"- Any logos, stamps, signatures, or visual markers mentioned\n\n"
            f"Extracted Content:\n{text[:4000]}"
        )
        output = self._generate(prompt, temperature=0.3, max_tokens=1024)
        return AgentTaskResult(
            task_type="describe_image", success=bool(output), output=output,
            structured_data={"method": "text_based"},
        )

    def _extract_data_from_image(self, context: Dict[str, Any]) -> AgentTaskResult:
        """Extract structured data — vision-first for tables/charts, text fallback."""
        image_bytes = context.get("image_bytes") or context.get("image")
        text = context.get("text", "")

        if image_bytes:
            vision = self._get_vision_client()
            if vision:
                # Try table analysis first (most structured)
                output, confidence = vision.analyze_image(image_bytes, analysis_type="table")
                if output:
                    return AgentTaskResult(
                        task_type="extract_data_from_image", success=True, output=output,
                        structured_data={"method": "vision_table", "confidence": confidence},
                    )

        prompt = (
            f"Extract structured data from the following image content.\n\n"
            f"Look for and extract:\n"
            f"1. Tables: Present in markdown table format\n"
            f"2. Forms: Extract as field_name: value pairs\n"
            f"3. Key-Value Pairs: Any labeled data points\n"
            f"4. Lists: Numbered or bulleted items\n"
            f"5. Dates, amounts, and identifiers\n\n"
            f"Present all extracted data in a clean, structured format.\n\n"
            f"Image Content:\n{text[:4000]}"
        )
        output = self._generate(prompt, temperature=0.1, max_tokens=2048)
        return AgentTaskResult(
            task_type="extract_data_from_image", success=bool(output), output=output,
            structured_data={"method": "text_based"},
        )

# ---------------------------------------------------------------------------
# Web Agent
# ---------------------------------------------------------------------------

class WebAgent(DomainAgent):
    """Specialized agent for web search and research tasks.

    Wraps: web_search, web_extract tools.
    """

    domain = "web"

    def get_capabilities(self) -> List[str]:
        return [
            "search_web",
            "fetch_url",
            "research_topic",
            "fact_check",
        ]

    def execute(self, task_type: str, context: Dict[str, Any]) -> AgentTaskResult:
        handlers = {
            "search_web": self._search_web,
            "fetch_url": self._fetch_url,
            "research_topic": self._research_topic,
            "fact_check": self._fact_check,
        }
        handler = handlers.get(task_type)
        if not handler:
            return AgentTaskResult(task_type=task_type, success=False, error=f"Unknown task: {task_type}")
        try:
            return handler(context)
        except Exception as exc:
            logger.warning("WebAgent.%s failed: %s", task_type, exc)
            return AgentTaskResult(task_type=task_type, success=False, error=str(exc))

    def _search_web(self, context: Dict[str, Any]) -> AgentTaskResult:
        """Search the internet for information on a topic."""
        text = context.get("text", "")
        query = context.get("query", text)
        prompt = (
            f"You are a research assistant. Based on the following query, "
            f"formulate an effective search strategy and summarize what you would "
            f"look for to answer this question comprehensively.\n\n"
            f"Query: {query[:4000]}\n\n"
            f"Provide:\n"
            f"1. Reformulated search queries (2-3 variations for best results)\n"
            f"2. Key terms and concepts to search for\n"
            f"3. Recommended source types (academic, news, official docs, etc.)\n"
            f"4. What to look for in the results to answer the query"
        )
        output = self._generate(prompt, temperature=0.3, max_tokens=1024)
        return AgentTaskResult(
            task_type="search_web", success=bool(output), output=output,
            structured_data={"query": query[:200]},
        )

    def _fetch_url(self, context: Dict[str, Any]) -> AgentTaskResult:
        """Fetch and extract content from a URL, then analyze it."""
        text = context.get("text", "")
        url = context.get("url", "")
        prompt = (
            f"Analyze the following content fetched from a web page.\n"
            f"{'URL: ' + url + chr(10) if url else ''}\n"
            f"Provide:\n"
            f"1. Main topic and purpose of the page\n"
            f"2. Key information and data points\n"
            f"3. Summary (3-5 sentences)\n"
            f"4. Credibility assessment (source type, recency, authority)\n\n"
            f"Page Content:\n{text[:4000]}"
        )
        output = self._generate(prompt, temperature=0.2, max_tokens=1024)
        return AgentTaskResult(
            task_type="fetch_url", success=bool(output), output=output,
            structured_data={"url": url},
        )

    def _research_topic(self, context: Dict[str, Any]) -> AgentTaskResult:
        """Conduct multi-query research combining search and summarization."""
        text = context.get("text", "")
        topic = context.get("topic", text)
        depth = context.get("depth", "standard")
        prompt = (
            f"Conduct a comprehensive research analysis on the following topic.\n"
            f"Research depth: {depth}\n\n"
            f"Topic: {topic[:4000]}\n\n"
            f"Provide a research report with:\n"
            f"1. Executive Summary\n"
            f"2. Background and Context\n"
            f"3. Key Findings (organized by theme)\n"
            f"4. Different Perspectives or Viewpoints\n"
            f"5. Current State and Recent Developments\n"
            f"6. Open Questions and Areas for Further Research\n"
            f"7. Conclusions and Recommendations"
        )
        output = self._generate(prompt, temperature=0.3, max_tokens=2048)
        return AgentTaskResult(
            task_type="research_topic", success=bool(output), output=output,
            structured_data={"topic": topic[:200], "depth": depth},
        )

    def _fact_check(self, context: Dict[str, Any]) -> AgentTaskResult:
        """Verify claims by analyzing for corroborating or contradicting evidence."""
        text = context.get("text", "")
        claim = context.get("claim", text)
        prompt = (
            f"Fact-check the following claim or statement.\n\n"
            f"Claim: {claim[:4000]}\n\n"
            f"Provide:\n"
            f"1. Claim Identification: Break down the specific claims being made\n"
            f"2. Verdict: For each claim (Supported/Unsupported/Partially Supported/Unverifiable)\n"
            f"3. Supporting Evidence: What supports the claim\n"
            f"4. Contradicting Evidence: What contradicts the claim\n"
            f"5. Context: Important context that affects interpretation\n"
            f"6. Confidence Level: How confident in the assessment (high/medium/low)\n"
            f"7. Caveats: Any limitations of this analysis"
        )
        output = self._generate(prompt, temperature=0.1, max_tokens=1024)
        return AgentTaskResult(
            task_type="fact_check", success=bool(output), output=output,
            structured_data={"claim": claim[:200]},
        )

# ---------------------------------------------------------------------------
# Insights Agent
# ---------------------------------------------------------------------------

class InsightsAgent(DomainAgent):
    """Specialized agent for analytics, pattern detection, and reporting tasks.

    Wraps: insights, action_items tools.
    """

    domain = "analytics"

    def get_capabilities(self) -> List[str]:
        return [
            "detect_anomalies",
            "find_patterns",
            "extract_action_items",
            "risk_assessment",
            "generate_report",
        ]

    def execute(self, task_type: str, context: Dict[str, Any]) -> AgentTaskResult:
        handlers = {
            "detect_anomalies": self._detect_anomalies,
            "find_patterns": self._find_patterns,
            "extract_action_items": self._extract_action_items,
            "risk_assessment": self._risk_assessment,
            "generate_report": self._generate_report,
        }
        handler = handlers.get(task_type)
        if not handler:
            return AgentTaskResult(task_type=task_type, success=False, error=f"Unknown task: {task_type}")
        try:
            return handler(context)
        except Exception as exc:
            logger.warning("InsightsAgent.%s failed: %s", task_type, exc)
            return AgentTaskResult(task_type=task_type, success=False, error=str(exc))

    def _detect_anomalies(self, context: Dict[str, Any]) -> AgentTaskResult:
        """Find anomalies and outliers in document data."""
        text = context.get("text", "")
        prompt = (
            f"Analyze the following document data for anomalies and outliers.\n\n"
            f"Look for:\n"
            f"1. Statistical outliers (values significantly different from the norm)\n"
            f"2. Inconsistencies (contradictory information within the document)\n"
            f"3. Missing data patterns (gaps that suggest omission)\n"
            f"4. Unexpected patterns (sequences or values that deviate from expected norms)\n"
            f"5. Temporal anomalies (dates, timelines that seem incorrect)\n\n"
            f"For each anomaly, provide: description, severity (high/medium/low), "
            f"and recommended action.\n\n"
            f"Document:\n{text[:4000]}"
        )
        output = self._generate(prompt, temperature=0.1, max_tokens=1024)
        return AgentTaskResult(task_type="detect_anomalies", success=bool(output), output=output)

    def _find_patterns(self, context: Dict[str, Any]) -> AgentTaskResult:
        """Identify recurring patterns across document data."""
        text = context.get("text", "")
        prompt = (
            f"Analyze the following document for recurring patterns and trends.\n\n"
            f"Identify:\n"
            f"1. Recurring themes or topics\n"
            f"2. Numerical trends (increasing/decreasing patterns)\n"
            f"3. Structural patterns (repeated formats, sections, or layouts)\n"
            f"4. Keyword frequency patterns\n"
            f"5. Relationships between entities or data points\n"
            f"6. Seasonal or temporal patterns if applicable\n\n"
            f"For each pattern: describe it, note frequency, and assess significance.\n\n"
            f"Document:\n{text[:4000]}"
        )
        output = self._generate(prompt, temperature=0.2, max_tokens=1024)
        return AgentTaskResult(task_type="find_patterns", success=bool(output), output=output)

    def _extract_action_items(self, context: Dict[str, Any]) -> AgentTaskResult:
        """Extract tasks, deadlines, and obligations from documents."""
        text = context.get("text", "")
        prompt = (
            f"Extract all action items, tasks, and obligations from the following document.\n\n"
            f"For each action item, provide:\n"
            f"1. Description: What needs to be done\n"
            f"2. Assignee: Who is responsible (if mentioned)\n"
            f"3. Deadline: When it must be completed (if mentioned)\n"
            f"4. Priority: HIGH/MEDIUM/LOW (based on language urgency)\n"
            f"5. Category: Type of action (follow-up, deliverable, decision, review)\n"
            f"6. Status: If any status is indicated (pending, in-progress, completed)\n\n"
            f"Look for modal verbs (must, shall, will, should, needs to), "
            f"deadline phrases, and assignment language.\n\n"
            f"Document:\n{text[:4000]}"
        )
        output = self._generate(prompt, temperature=0.1, max_tokens=1024)
        return AgentTaskResult(task_type="extract_action_items", success=bool(output), output=output)

    def _risk_assessment(self, context: Dict[str, Any]) -> AgentTaskResult:
        """Identify potential risks and red flags in document content."""
        text = context.get("text", "")
        prompt = (
            f"Perform a risk assessment on the following document.\n\n"
            f"Identify:\n"
            f"1. Potential Risks: What could go wrong based on the content\n"
            f"2. Red Flags: Warning signs or concerning elements\n"
            f"3. Compliance Risks: Potential regulatory or legal issues\n"
            f"4. Financial Risks: Monetary exposure or liability\n"
            f"5. Operational Risks: Process or execution concerns\n\n"
            f"For each risk:\n"
            f"- Severity: CRITICAL/HIGH/MEDIUM/LOW\n"
            f"- Likelihood: LIKELY/POSSIBLE/UNLIKELY\n"
            f"- Impact description\n"
            f"- Recommended mitigation\n\n"
            f"Document:\n{text[:4000]}"
        )
        output = self._generate(prompt, temperature=0.1, max_tokens=1024)
        return AgentTaskResult(task_type="risk_assessment", success=bool(output), output=output)

    def _generate_report(self, context: Dict[str, Any]) -> AgentTaskResult:
        """Create an analytical report summarizing document findings."""
        text = context.get("text", "")
        report_type = context.get("report_type", "analytical summary")
        prompt = (
            f"Generate a {report_type} from the following document data.\n\n"
            f"Structure the report with:\n"
            f"1. Executive Summary (2-3 sentences)\n"
            f"2. Key Metrics and Data Points\n"
            f"3. Analysis and Findings\n"
            f"4. Trends and Patterns Observed\n"
            f"5. Risk Areas and Concerns\n"
            f"6. Recommendations\n"
            f"7. Conclusion\n\n"
            f"Use data from the source to support all findings.\n\n"
            f"Document:\n{text[:4000]}"
        )
        output = self._generate(prompt, temperature=0.3, max_tokens=2048)
        return AgentTaskResult(
            task_type="generate_report", success=bool(output), output=output,
            structured_data={"report_type": report_type},
        )

# ---------------------------------------------------------------------------
# Screening Agent
# ---------------------------------------------------------------------------

class ScreeningAgent(DomainAgent):
    """Specialized agent for document screening and compliance tasks.

    Wraps: screen_pii, screen_ai_authorship, screen_resume, screen_readability tools.
    """

    domain = "screening"

    def get_capabilities(self) -> List[str]:
        return [
            "screen_pii",
            "detect_ai_content",
            "screen_resume",
            "assess_readability",
            "compliance_scan",
        ]

    def execute(self, task_type: str, context: Dict[str, Any]) -> AgentTaskResult:
        handlers = {
            "screen_pii": self._screen_pii,
            "detect_ai_content": self._detect_ai_content,
            "screen_resume": self._screen_resume,
            "assess_readability": self._assess_readability,
            "compliance_scan": self._compliance_scan,
        }
        handler = handlers.get(task_type)
        if not handler:
            return AgentTaskResult(task_type=task_type, success=False, error=f"Unknown task: {task_type}")
        try:
            return handler(context)
        except Exception as exc:
            logger.warning("ScreeningAgent.%s failed: %s", task_type, exc)
            return AgentTaskResult(task_type=task_type, success=False, error=str(exc))

    def _screen_pii(self, context: Dict[str, Any]) -> AgentTaskResult:
        """Detect personally identifiable information in text."""
        text = context.get("text", "")
        prompt = (
            f"Scan the following text for Personally Identifiable Information (PII).\n\n"
            f"Check for:\n"
            f"1. Names (full names, partial names)\n"
            f"2. Contact Information (email, phone, address)\n"
            f"3. Government IDs (SSN, passport, driver's license, national ID)\n"
            f"4. Financial Information (credit card, bank account, tax ID)\n"
            f"5. Health Information (medical record numbers, conditions)\n"
            f"6. Biometric Data (fingerprints, facial recognition references)\n"
            f"7. Location Data (precise addresses, GPS coordinates)\n"
            f"8. Online Identifiers (IP addresses, usernames, login credentials)\n\n"
            f"For each PII item found:\n"
            f"- Type of PII\n"
            f"- Location in text (quote the surrounding context)\n"
            f"- Sensitivity level (HIGH/MEDIUM/LOW)\n"
            f"- Recommended redaction action\n\n"
            f"Text:\n{text[:4000]}"
        )
        output = self._generate(prompt, temperature=0.1, max_tokens=1024)
        return AgentTaskResult(task_type="screen_pii", success=bool(output), output=output)

    def _detect_ai_content(self, context: Dict[str, Any]) -> AgentTaskResult:
        """Check if content appears to be AI-generated."""
        text = context.get("text", "")
        prompt = (
            f"Analyze the following text for indicators of AI-generated content.\n\n"
            f"Evaluate:\n"
            f"1. Writing Style: Uniformity, lack of personal voice, generic phrasing\n"
            f"2. Structure: Overly formulaic organization, predictable patterns\n"
            f"3. Content Signals: Hedging language, overly balanced viewpoints\n"
            f"4. Vocabulary: Unusual word choices, repetitive phrase patterns\n"
            f"5. Coherence: Superficial coherence without deep reasoning\n"
            f"6. Factual Patterns: Generic facts, lack of specific citations\n\n"
            f"Provide:\n"
            f"- Overall Assessment: LIKELY AI / POSSIBLY AI / LIKELY HUMAN / INCONCLUSIVE\n"
            f"- Confidence: HIGH/MEDIUM/LOW\n"
            f"- Key indicators found\n"
            f"- Sections of most concern\n\n"
            f"Text:\n{text[:4000]}"
        )
        output = self._generate(prompt, temperature=0.1, max_tokens=1024)
        return AgentTaskResult(task_type="detect_ai_content", success=bool(output), output=output)

    def _screen_resume(self, context: Dict[str, Any]) -> AgentTaskResult:
        """Perform comprehensive resume screening analysis."""
        text = context.get("text", "")
        job_role = context.get("job_role", "")
        criteria = context.get("criteria", "")
        role_ctx = f" for the role of {job_role}" if job_role else ""
        criteria_ctx = f"\nScreening Criteria: {criteria}" if criteria else ""
        prompt = (
            f"Perform a comprehensive screening of this resume{role_ctx}.{criteria_ctx}\n\n"
            f"Evaluate:\n"
            f"1. Overall Quality: Format, completeness, professionalism\n"
            f"2. Experience: Relevance, progression, achievements\n"
            f"3. Skills: Technical and soft skills assessment\n"
            f"4. Education: Relevance and credentials\n"
            f"5. Red Flags: Gaps, inconsistencies, exaggerations\n"
            f"6. Strengths: Top 3 candidate strengths\n"
            f"7. Concerns: Top 3 areas of concern\n"
            f"8. Recommendation: ADVANCE / HOLD / REJECT with justification\n\n"
            f"Resume:\n{text[:4000]}"
        )
        output = self._generate(prompt, temperature=0.2, max_tokens=1024)
        return AgentTaskResult(
            task_type="screen_resume", success=bool(output), output=output,
            structured_data={"job_role": job_role},
        )

    def _assess_readability(self, context: Dict[str, Any]) -> AgentTaskResult:
        """Evaluate text readability and style quality."""
        text = context.get("text", "")
        prompt = (
            f"Assess the readability and writing quality of the following text.\n\n"
            f"Evaluate:\n"
            f"1. Readability Level: Approximate grade level (Flesch-Kincaid equivalent)\n"
            f"2. Sentence Complexity: Average sentence length, variance\n"
            f"3. Vocabulary Level: Simple/Intermediate/Advanced/Technical\n"
            f"4. Clarity: How easy is it to understand the main points\n"
            f"5. Structure: Organization, use of headings, paragraphing\n"
            f"6. Tone: Formal/Informal/Technical/Conversational\n"
            f"7. Jargon Usage: Amount of domain-specific terminology\n"
            f"8. Suggestions: 3-5 specific improvements for readability\n\n"
            f"Provide an overall readability score: A (excellent) through F (poor).\n\n"
            f"Text:\n{text[:4000]}"
        )
        output = self._generate(prompt, temperature=0.2, max_tokens=1024)
        return AgentTaskResult(task_type="assess_readability", success=bool(output), output=output)

    def _compliance_scan(self, context: Dict[str, Any]) -> AgentTaskResult:
        """Scan document for compliance issues combining PII and readability checks."""
        text = context.get("text", "")
        standards = context.get("standards", "general data protection and accessibility")
        prompt = (
            f"Perform a compliance scan of the following document against {standards} standards.\n\n"
            f"Check for:\n"
            f"1. PII Exposure: Any unprotected personal information\n"
            f"2. Data Protection: Compliance with data handling requirements\n"
            f"3. Accessibility: Readability and clarity for target audience\n"
            f"4. Required Disclosures: Any missing mandatory disclosures\n"
            f"5. Language Requirements: Appropriate and non-discriminatory language\n"
            f"6. Documentation Standards: Completeness and formatting requirements\n\n"
            f"For each issue:\n"
            f"- Severity: CRITICAL/HIGH/MEDIUM/LOW\n"
            f"- Description of the issue\n"
            f"- Specific location in the document\n"
            f"- Recommended remediation\n\n"
            f"Provide an overall compliance score: COMPLIANT / PARTIALLY COMPLIANT / NON-COMPLIANT.\n\n"
            f"Document:\n{text[:4000]}"
        )
        output = self._generate(prompt, temperature=0.1, max_tokens=1024)
        return AgentTaskResult(
            task_type="compliance_scan", success=bool(output), output=output,
            structured_data={"standards": standards},
        )

# ---------------------------------------------------------------------------
# Cloud Platform Agent
# ---------------------------------------------------------------------------

class CloudPlatformAgent(DomainAgent):
    """Specialized agent for cloud platform integration (Jira, Confluence, SharePoint)."""

    domain = "cloud_platform"

    def get_capabilities(self) -> List[str]:
        return [
            "jira_analysis",
            "confluence_analysis",
            "sharepoint_analysis",
            "cross_platform_summary",
        ]

    def execute(self, task_type: str, context: Dict[str, Any]) -> AgentTaskResult:
        handlers = {
            "jira_analysis": self._jira_analysis,
            "confluence_analysis": self._confluence_analysis,
            "sharepoint_analysis": self._sharepoint_analysis,
            "cross_platform_summary": self._cross_platform_summary,
        }
        handler = handlers.get(task_type)
        if not handler:
            return AgentTaskResult(task_type=task_type, success=False, error=f"Unknown task: {task_type}")
        try:
            return handler(context)
        except Exception as exc:
            logger.warning("CloudPlatformAgent.%s failed: %s", task_type, exc)
            return AgentTaskResult(task_type=task_type, success=False, error=str(exc))

    def _jira_analysis(self, context: Dict[str, Any]) -> AgentTaskResult:
        """Analyze Jira issues for trends, blockers, and sprint health."""
        text = context.get("text", "")
        prompt = (
            f"Analyze the following Jira data and provide:\n"
            f"1. Issue status summary (open/in-progress/done breakdown)\n"
            f"2. Blockers and dependencies\n"
            f"3. Sprint health assessment\n"
            f"4. Resource allocation insights\n"
            f"5. Risk areas and recommendations\n\n"
            f"Data:\n{text[:4000]}"
        )
        output = self._generate(prompt, temperature=0.2)
        return AgentTaskResult(task_type="jira_analysis", success=bool(output), output=output)

    def _confluence_analysis(self, context: Dict[str, Any]) -> AgentTaskResult:
        """Analyze Confluence documentation for completeness and quality."""
        text = context.get("text", "")
        prompt = (
            f"Analyze the following Confluence documentation:\n"
            f"1. Content completeness assessment\n"
            f"2. Documentation quality (clarity, structure, currency)\n"
            f"3. Knowledge gaps identified\n"
            f"4. Cross-references and link integrity\n"
            f"5. Recommendations for improvement\n\n"
            f"Content:\n{text[:4000]}"
        )
        output = self._generate(prompt, temperature=0.2)
        return AgentTaskResult(task_type="confluence_analysis", success=bool(output), output=output)

    def _sharepoint_analysis(self, context: Dict[str, Any]) -> AgentTaskResult:
        """Analyze SharePoint documents and site structure."""
        text = context.get("text", "")
        prompt = (
            f"Analyze the following SharePoint content:\n"
            f"1. Document organization and structure\n"
            f"2. Content relevance and currency\n"
            f"3. Access patterns and usage insights\n"
            f"4. Metadata completeness\n"
            f"5. Recommendations for document management\n\n"
            f"Content:\n{text[:4000]}"
        )
        output = self._generate(prompt, temperature=0.2)
        return AgentTaskResult(task_type="sharepoint_analysis", success=bool(output), output=output)

    def _cross_platform_summary(self, context: Dict[str, Any]) -> AgentTaskResult:
        """Summarize data across Jira, Confluence, and SharePoint."""
        text = context.get("text", "")
        prompt = (
            f"Create a cross-platform summary from the following data spanning "
            f"Jira, Confluence, and/or SharePoint.\n\n"
            f"Provide:\n"
            f"1. Unified project overview\n"
            f"2. Documentation coverage vs issue tracking alignment\n"
            f"3. Knowledge base gaps relative to active work items\n"
            f"4. Cross-platform recommendations\n\n"
            f"Data:\n{text[:4000]}"
        )
        output = self._generate(prompt, temperature=0.3, max_tokens=2048)
        return AgentTaskResult(task_type="cross_platform_summary", success=bool(output), output=output)

# ---------------------------------------------------------------------------
# Customer Service Agent
# ---------------------------------------------------------------------------

class CustomerServiceAgent(DomainAgent):
    """Specialized agent for customer support, issue resolution, and FAQ tasks.

    Provides document-grounded support: resolves user issues, troubleshoots
    problems, assesses escalation needs, drafts customer responses, and
    searches knowledge base / FAQ content.
    """

    domain = "customer_service"
    use_thinking_model = True  # reasoning needed for issue resolution

    def get_capabilities(self) -> List[str]:
        return [
            "resolve_issue",
            "troubleshoot",
            "escalation_assessment",
            "generate_response",
            "faq_search",
        ]

    def execute(self, task_type: str, context: Dict[str, Any]) -> AgentTaskResult:
        handlers = {
            "resolve_issue": self._resolve_issue,
            "troubleshoot": self._troubleshoot,
            "escalation_assessment": self._escalation_assessment,
            "generate_response": self._generate_response,
            "faq_search": self._faq_search,
        }
        handler = handlers.get(task_type)
        if not handler:
            return AgentTaskResult(task_type=task_type, success=False, error=f"Unknown task: {task_type}")
        try:
            return handler(context)
        except Exception as exc:
            logger.warning("CustomerServiceAgent.%s failed: %s", task_type, exc)
            return AgentTaskResult(task_type=task_type, success=False, error=str(exc))

    def _resolve_issue(self, context: Dict[str, Any]) -> AgentTaskResult:
        """Analyze query against document knowledge to resolve customer questions."""
        text = context.get("text", "")
        query = context.get("query", "")
        prompt = (
            f"You are an expert customer service agent. Resolve the customer's "
            f"issue using ONLY the provided document evidence.\n\n"
            f"Customer query: {query}\n\n"
            f"Guidelines:\n"
            f"1. Address the customer's concern directly and specifically\n"
            f"2. Reference relevant policy sections, terms, or procedures from the documents\n"
            f"3. Provide clear next steps the customer can take\n"
            f"4. If the documents don't fully answer the question, state what IS covered\n"
            f"5. Use a professional, empathetic tone\n\n"
            f"Document evidence:\n{text[:4000]}"
        )
        output = self._generate(prompt, temperature=0.2, max_tokens=1024)
        return AgentTaskResult(task_type="resolve_issue", success=bool(output), output=output)

    def _troubleshoot(self, context: Dict[str, Any]) -> AgentTaskResult:
        """Step-by-step troubleshooting from manuals/FAQs/guides."""
        text = context.get("text", "")
        query = context.get("query", "")
        prompt = (
            f"Provide clear step-by-step troubleshooting for the customer's problem. "
            f"Number each step.\n\n"
            f"Customer problem: {query}\n\n"
            f"Guidelines:\n"
            f"1. Start with the simplest, most common fix\n"
            f"2. Progress to more complex solutions\n"
            f"3. Include any prerequisites or warnings for each step\n"
            f"4. Reference specific sections from the documentation\n"
            f"5. End with escalation guidance if the steps don't resolve the issue\n\n"
            f"Reference documentation:\n{text[:4000]}"
        )
        output = self._generate(prompt, temperature=0.2, max_tokens=1024)
        return AgentTaskResult(task_type="troubleshoot", success=bool(output), output=output)

    def _escalation_assessment(self, context: Dict[str, Any]) -> AgentTaskResult:
        """Determine if issue needs escalation based on complexity."""
        text = context.get("text", "")
        query = context.get("query", "")
        prompt = (
            f"Assess whether this customer issue needs escalation.\n\n"
            f"Customer issue: {query}\n\n"
            f"Provide:\n"
            f"1. **Severity**: Low / Medium / High / Critical\n"
            f"2. **Category**: Technical / Billing / Policy / Compliance / Safety\n"
            f"3. **Escalation Recommended**: Yes / No\n"
            f"4. **Escalation Path**: Which team or level should handle this\n"
            f"5. **Justification**: Why this level of escalation is appropriate\n"
            f"6. **Immediate Actions**: What can be done before escalation\n"
            f"7. **SLA Impact**: Any time-sensitive considerations\n\n"
            f"Context from documents:\n{text[:4000]}"
        )
        output = self._generate(prompt, temperature=0.1, max_tokens=1024)
        return AgentTaskResult(
            task_type="escalation_assessment", success=bool(output), output=output,
            structured_data={"assessment_type": "escalation"},
        )

    def _generate_response(self, context: Dict[str, Any]) -> AgentTaskResult:
        """Draft customer-facing response (empathetic, formal tone)."""
        text = context.get("text", "")
        query = context.get("query", "")
        prompt = (
            f"Draft a professional, empathetic customer response. "
            f"Address the issue directly.\n\n"
            f"Customer query: {query}\n\n"
            f"Guidelines:\n"
            f"1. Open with acknowledgement of the customer's concern\n"
            f"2. Provide a clear, actionable resolution\n"
            f"3. Reference relevant policy or documentation where helpful\n"
            f"4. Close with next steps and an offer for further assistance\n"
            f"5. Keep tone warm but professional\n"
            f"6. Avoid jargon — use plain language\n\n"
            f"Supporting documentation:\n{text[:4000]}"
        )
        output = self._generate(prompt, temperature=0.3, max_tokens=1024)
        return AgentTaskResult(task_type="generate_response", success=bool(output), output=output)

    def _faq_search(self, context: Dict[str, Any]) -> AgentTaskResult:
        """Find most relevant FAQ/KB entries for the query."""
        text = context.get("text", "")
        query = context.get("query", "")
        prompt = (
            f"Find the most relevant knowledge base entries for this customer query. "
            f"Format as Q&A pairs.\n\n"
            f"Customer query: {query}\n\n"
            f"Instructions:\n"
            f"1. Extract the most relevant Q&A pairs from the documents\n"
            f"2. Format each as:\n"
            f"   **Q:** [Question]\n"
            f"   **A:** [Answer]\n"
            f"3. Rank by relevance to the customer's query\n"
            f"4. Include at most 5 Q&A pairs\n"
            f"5. If no direct FAQ match, synthesize answers from the document content\n\n"
            f"Knowledge base content:\n{text[:4000]}"
        )
        output = self._generate(prompt, temperature=0.2, max_tokens=1024)
        return AgentTaskResult(task_type="faq_search", success=bool(output), output=output)

# ---------------------------------------------------------------------------
# Analytics Visualization Agent
# ---------------------------------------------------------------------------

class AnalyticsVisualizationAgent(DomainAgent):
    """Specialized agent for generating charts, graphs, and visualizations
    from document data using matplotlib.

    Generates base64-encoded PNG images that can be embedded in responses.
    """

    domain = "analytics_viz"
    use_thinking_model = False  # generation-heavy, not reasoning

    def get_capabilities(self) -> List[str]:
        return [
            "generate_chart",
            "generate_distribution",
            "generate_comparison_chart",
            "generate_timeline_chart",
            "generate_summary_dashboard",
            "compute_statistics",
        ]

    def execute(self, task_type: str, context: Dict[str, Any]) -> AgentTaskResult:
        handlers = {
            "generate_chart": self._generate_chart,
            "generate_distribution": self._generate_distribution,
            "generate_comparison_chart": self._generate_comparison_chart,
            "generate_timeline_chart": self._generate_timeline_chart,
            "generate_summary_dashboard": self._generate_summary_dashboard,
            "compute_statistics": self._compute_statistics,
        }
        handler = handlers.get(task_type)
        if not handler:
            return AgentTaskResult(task_type=task_type, success=False, error=f"Unknown task: {task_type}")
        try:
            return handler(context)
        except Exception as exc:
            logger.warning("AnalyticsVisualizationAgent.%s failed: %s", task_type, exc)
            return AgentTaskResult(task_type=task_type, success=False, error=str(exc))

    def _extract_data(self, text: str, query: str) -> List[Dict[str, Any]]:
        """Use LLM to extract structured chart data from chunk text."""
        try:
            from src.tools.analytics_visualization import extract_chart_data
            llm = self._get_base_llm()
            return extract_chart_data(text, query, llm)
        except Exception as exc:
            logger.warning("Chart data extraction failed: %s", exc)
            return []

    def _generate_chart(self, context: Dict[str, Any]) -> AgentTaskResult:
        """Auto-select and generate the best chart type from document data."""
        text = context.get("text", "")
        query = context.get("query", "")
        data = self._extract_data(text, query)
        if not data:
            # Fall back to LLM text summary
            prompt = (
                f"The user asked for a chart/visualization but the data could not be "
                f"extracted into a structured format. Provide a text-based data summary "
                f"instead, formatted as a table.\n\n"
                f"Query: {query}\n\nData:\n{text[:4000]}"
            )
            output = self._generate(prompt, temperature=0.2, max_tokens=1024)
            return AgentTaskResult(task_type="generate_chart", success=bool(output), output=output)

        try:
            from src.tools.analytics_visualization import (
                select_chart_type, generate_bar_chart, generate_pie_chart,
                generate_line_chart,
            )
            chart_type = select_chart_type(query, data)
            labels = [d.get("label", "") for d in data]
            values = [d.get("value", 0) for d in data]
            title = query[:80]

            if chart_type == "pie":
                img_b64 = generate_pie_chart(labels, values, title)
            elif chart_type == "line":
                img_b64 = generate_line_chart(labels, values, title)
            else:
                img_b64 = generate_bar_chart(labels, values, title)

            summary = self._generate(
                f"Briefly describe this chart data in 2-3 sentences:\n"
                f"Title: {title}\nData: {data[:10]}",
                temperature=0.2, max_tokens=256,
            )
            output = f"{summary}\n\n![chart](data:image/png;base64,{img_b64})"
            return AgentTaskResult(
                task_type="generate_chart", success=True, output=output,
                structured_data={
                    "charts": [{"title": title, "image_base64": img_b64, "type": chart_type}],
                    "media": [{"type": "image/png", "title": title, "data": img_b64}],
                },
            )
        except Exception as exc:
            logger.debug("Chart generation failed, falling back to text: %s", exc)
            prompt = f"Summarize this data as a text table:\nQuery: {query}\nData: {data[:10]}"
            output = self._generate(prompt, temperature=0.2, max_tokens=1024)
            return AgentTaskResult(task_type="generate_chart", success=bool(output), output=output)

    def _generate_distribution(self, context: Dict[str, Any]) -> AgentTaskResult:
        """Generate distribution/histogram visualization."""
        text = context.get("text", "")
        query = context.get("query", "")
        data = self._extract_data(text, query)
        if not data:
            prompt = f"Provide a text-based frequency/distribution analysis.\n\nQuery: {query}\n\nData:\n{text[:4000]}"
            output = self._generate(prompt, temperature=0.2, max_tokens=1024)
            return AgentTaskResult(task_type="generate_distribution", success=bool(output), output=output)

        try:
            from src.tools.analytics_visualization import generate_histogram
            values = [d.get("value", 0) for d in data]
            title = query[:80]
            img_b64 = generate_histogram(values, title)
            summary = self._generate(
                f"Describe this distribution in 2-3 sentences:\nTitle: {title}\nValues: {values[:20]}",
                temperature=0.2, max_tokens=256,
            )
            output = f"{summary}\n\n![histogram](data:image/png;base64,{img_b64})"
            return AgentTaskResult(
                task_type="generate_distribution", success=True, output=output,
                structured_data={
                    "charts": [{"title": title, "image_base64": img_b64, "type": "histogram"}],
                    "media": [{"type": "image/png", "title": title, "data": img_b64}],
                },
            )
        except Exception as exc:
            logger.warning("Histogram generation failed: %s", exc)
            prompt = f"Provide a text distribution analysis.\nQuery: {query}\nData: {data[:10]}"
            output = self._generate(prompt, temperature=0.2, max_tokens=1024)
            return AgentTaskResult(task_type="generate_distribution", success=bool(output), output=output)

    def _generate_comparison_chart(self, context: Dict[str, Any]) -> AgentTaskResult:
        """Side-by-side comparison chart."""
        text = context.get("text", "")
        query = context.get("query", "")
        data = self._extract_data(text, query)
        if not data:
            prompt = f"Provide a text-based comparison table.\n\nQuery: {query}\n\nData:\n{text[:4000]}"
            output = self._generate(prompt, temperature=0.2, max_tokens=1024)
            return AgentTaskResult(task_type="generate_comparison_chart", success=bool(output), output=output)

        try:
            from src.tools.analytics_visualization import generate_grouped_bar_chart
            title = query[:80]
            img_b64 = generate_grouped_bar_chart(data, title)
            summary = self._generate(
                f"Describe this comparison in 2-3 sentences:\nTitle: {title}\nData: {data[:10]}",
                temperature=0.2, max_tokens=256,
            )
            output = f"{summary}\n\n![comparison](data:image/png;base64,{img_b64})"
            return AgentTaskResult(
                task_type="generate_comparison_chart", success=True, output=output,
                structured_data={
                    "charts": [{"title": title, "image_base64": img_b64, "type": "grouped_bar"}],
                    "media": [{"type": "image/png", "title": title, "data": img_b64}],
                },
            )
        except Exception as exc:
            logger.warning("Comparison chart failed: %s", exc)
            prompt = f"Provide a text comparison.\nQuery: {query}\nData: {data[:10]}"
            output = self._generate(prompt, temperature=0.2, max_tokens=1024)
            return AgentTaskResult(task_type="generate_comparison_chart", success=bool(output), output=output)

    def _generate_timeline_chart(self, context: Dict[str, Any]) -> AgentTaskResult:
        """Temporal progression chart."""
        text = context.get("text", "")
        query = context.get("query", "")
        data = self._extract_data(text, query)
        if not data:
            prompt = f"Provide a text-based timeline analysis.\n\nQuery: {query}\n\nData:\n{text[:4000]}"
            output = self._generate(prompt, temperature=0.2, max_tokens=1024)
            return AgentTaskResult(task_type="generate_timeline_chart", success=bool(output), output=output)

        try:
            from src.tools.analytics_visualization import generate_line_chart
            labels = [d.get("label", "") for d in data]
            values = [d.get("value", 0) for d in data]
            title = query[:80]
            img_b64 = generate_line_chart(labels, values, title, xlabel="Time")
            summary = self._generate(
                f"Describe this timeline trend in 2-3 sentences:\nTitle: {title}\nData: {data[:10]}",
                temperature=0.2, max_tokens=256,
            )
            output = f"{summary}\n\n![timeline](data:image/png;base64,{img_b64})"
            return AgentTaskResult(
                task_type="generate_timeline_chart", success=True, output=output,
                structured_data={
                    "charts": [{"title": title, "image_base64": img_b64, "type": "line"}],
                    "media": [{"type": "image/png", "title": title, "data": img_b64}],
                },
            )
        except Exception as exc:
            logger.warning("Timeline chart failed: %s", exc)
            prompt = f"Provide a text timeline.\nQuery: {query}\nData: {data[:10]}"
            output = self._generate(prompt, temperature=0.2, max_tokens=1024)
            return AgentTaskResult(task_type="generate_timeline_chart", success=bool(output), output=output)

    def _generate_summary_dashboard(self, context: Dict[str, Any]) -> AgentTaskResult:
        """Multi-chart summary dashboard."""
        text = context.get("text", "")
        query = context.get("query", "")
        data = self._extract_data(text, query)
        if not data:
            prompt = (
                f"Create a comprehensive text-based dashboard summary.\n\n"
                f"Query: {query}\n\nData:\n{text[:4000]}"
            )
            output = self._generate(prompt, temperature=0.2, max_tokens=2048)
            return AgentTaskResult(task_type="generate_summary_dashboard", success=bool(output), output=output)

        try:
            from src.tools.analytics_visualization import generate_bar_chart, generate_pie_chart
            labels = [d.get("label", "") for d in data]
            values = [d.get("value", 0) for d in data]
            title = query[:80]
            bar_b64 = generate_bar_chart(labels, values, f"{title} — Bar Chart")
            pie_b64 = generate_pie_chart(labels, values, f"{title} — Pie Chart")
            summary = self._generate(
                f"Provide an executive summary of this data in 3-4 sentences:\n"
                f"Title: {title}\nData: {data[:10]}",
                temperature=0.2, max_tokens=512,
            )
            output = (
                f"## Dashboard: {title}\n\n"
                f"{summary}\n\n"
                f"### Distribution\n"
                f"![bar](data:image/png;base64,{bar_b64})\n\n"
                f"### Proportions\n"
                f"![pie](data:image/png;base64,{pie_b64})"
            )
            return AgentTaskResult(
                task_type="generate_summary_dashboard", success=True, output=output,
                structured_data={
                    "charts": [
                        {"title": f"{title} — Bar", "image_base64": bar_b64, "type": "bar"},
                        {"title": f"{title} — Pie", "image_base64": pie_b64, "type": "pie"},
                    ],
                    "media": [
                        {"type": "image/png", "title": f"{title} — Bar", "data": bar_b64},
                        {"type": "image/png", "title": f"{title} — Pie", "data": pie_b64},
                    ],
                },
            )
        except Exception as exc:
            logger.warning("Dashboard generation failed: %s", exc)
            prompt = f"Create a text dashboard summary.\nQuery: {query}\nData: {data[:10]}"
            output = self._generate(prompt, temperature=0.2, max_tokens=2048)
            return AgentTaskResult(task_type="generate_summary_dashboard", success=bool(output), output=output)

    def _compute_statistics(self, context: Dict[str, Any]) -> AgentTaskResult:
        """Statistical analysis with visual output."""
        text = context.get("text", "")
        query = context.get("query", "")
        data = self._extract_data(text, query)

        # Always generate LLM statistical analysis
        prompt = (
            f"Perform a statistical analysis on the following document data.\n\n"
            f"Query: {query}\n\n"
            f"Provide:\n"
            f"1. Key metrics (count, sum, average, min, max, median)\n"
            f"2. Distribution characteristics\n"
            f"3. Notable patterns or outliers\n"
            f"4. Summary statistics table\n\n"
            f"Data:\n{text[:4000]}"
        )
        analysis = self._generate(prompt, temperature=0.1, max_tokens=1024)

        if data:
            try:
                from src.tools.analytics_visualization import generate_bar_chart
                labels = [d.get("label", "") for d in data]
                values = [d.get("value", 0) for d in data]
                title = query[:80]
                img_b64 = generate_bar_chart(labels, values, title)
                output = f"{analysis}\n\n![statistics](data:image/png;base64,{img_b64})"
                return AgentTaskResult(
                    task_type="compute_statistics", success=True, output=output,
                    structured_data={
                        "charts": [{"title": title, "image_base64": img_b64, "type": "bar"}],
                        "media": [{"type": "image/png", "title": title, "data": img_b64}],
                    },
                )
            except Exception:
                pass

        return AgentTaskResult(task_type="compute_statistics", success=bool(analysis), output=analysis)

# ---------------------------------------------------------------------------
# Agent Registry
# ---------------------------------------------------------------------------

_DOMAIN_AGENTS: Dict[str, type] = {
    "hr": ResumeAgent,
    "resume": ResumeAgent,
    "medical": MedicalAgent,
    "legal": LegalAgent,
    "policy": LegalAgent,
    "invoice": InvoiceAgent,
    "financial": InvoiceAgent,
    "content": ContentAgent,
    "content_generate": ContentAgent,
    "email": ContentAgent,
    "translation": TranslatorAgent,
    "translator": TranslatorAgent,
    "education": TutorAgent,
    "tutor": TutorAgent,
    "image": ImageAgent,
    "image_analysis": ImageAgent,
    "web": WebAgent,
    "web_search": WebAgent,
    "analytics": InsightsAgent,
    "insights": InsightsAgent,
    "action_items": InsightsAgent,
    "screening": ScreeningAgent,
    "screen_pii": ScreeningAgent,
    "screen_resume": ScreeningAgent,
    "cloud": CloudPlatformAgent,
    "cloud_platform": CloudPlatformAgent,
    "jira_confluence": CloudPlatformAgent,
    "sharepoint": CloudPlatformAgent,
    "jira": CloudPlatformAgent,
    "confluence": CloudPlatformAgent,
    "customer_service": CustomerServiceAgent,
    "support": CustomerServiceAgent,
    "helpdesk": CustomerServiceAgent,
    "analytics_viz": AnalyticsVisualizationAgent,
    "chart": AnalyticsVisualizationAgent,
    "visualization": AnalyticsVisualizationAgent,
}

def get_domain_agent(domain: str, llm_client: Any = None, thinking_client: Any = None) -> Optional[DomainAgent]:
    """Get a domain-specialized agent by domain name.

    MoE routing: ``thinking_client`` (lfm2.5-thinking) is passed to reasoning-heavy
    agents; generation-heavy agents ignore it and always use DocWain-Agent.
    """
    agent_class = _DOMAIN_AGENTS.get(domain.lower())
    if agent_class is None:
        return None
    return agent_class(llm_client=llm_client, thinking_client=thinking_client)

def list_available_agents() -> Dict[str, List[str]]:
    """List all available agents and their capabilities."""
    result = {}
    seen = set()
    for domain, agent_class in _DOMAIN_AGENTS.items():
        if agent_class in seen:
            continue
        seen.add(agent_class)
        agent = agent_class()
        result[domain] = agent.get_capabilities()
    return result

def _ml_detect_domain(query: str) -> Optional[str]:
    """Use ML intent classifier to detect domain. Returns domain string or None."""
    try:
        from src.intent.intent_classifier import get_intent_classifier
        classifier = get_intent_classifier()
        if classifier is None or not getattr(classifier, "_trained", False):
            return None
        # Get embedder for encoding
        from src.api import rag_state
        state = rag_state.get_app_state()
        if not state or not state.embedding_model:
            return None
        import numpy as np
        emb = state.embedding_model.encode([query])
        if isinstance(emb, list):
            emb = np.array(emb)
        _, _, domain_probs = classifier._forward(emb)
        domain_idx = int(np.argmax(domain_probs[0]))
        confidence = float(domain_probs[0][domain_idx])
        if confidence < 0.5:
            return None
        domain_name = classifier.domain_names[domain_idx]
        # Map ML domain names to agent domain names
        _ML_DOMAIN_TO_AGENT = {
            "resume": "hr", "invoice": "invoice", "legal": "legal",
            "policy": "legal", "medical": "medical",
            "report": "analytics", "generic": None,
        }
        return _ML_DOMAIN_TO_AGENT.get(domain_name)
    except Exception:
        return None

# Task detection now uses NLU engine — see src/nlp/nlu_engine.py domain_task registry.
# No hardcoded keyword patterns needed.

def _customer_service_fast_path(query: str) -> Optional[Dict[str, str]]:
    """Keyword fast-path for customer service tasks.

    Fires only on high-confidence signals that embedding similarity may miss.
    """
    import re
    ql = query.lower().strip()

    # resolve_issue: "resolve" + customer/support context
    if re.search(r"\bresolv\w*\b", ql) and re.search(r"\b(customer|support|service|issue|complaint|request)\b", ql):
        return {"domain": "customer_service", "task_type": "resolve_issue"}

    # troubleshoot: explicit troubleshooting
    if re.search(r"\btroubleshoot\b", ql):
        return {"domain": "customer_service", "task_type": "troubleshoot"}
    if re.search(r"\bdiagnos\w*\b.*\b(issue|problem|error)\b", ql):
        return {"domain": "customer_service", "task_type": "troubleshoot"}

    # escalation_assessment: escalation keywords
    if re.search(r"\bescalat\w*\b", ql):
        return {"domain": "customer_service", "task_type": "escalation_assessment"}

    # generate_response: draft/generate + customer/support reply context
    if re.search(r"\b(generate|draft|compose|write)\b", ql) and re.search(r"\b(customer|support|service)\b", ql) and re.search(r"\b(reply|response|message|answer)\b", ql):
        return {"domain": "customer_service", "task_type": "generate_response"}

    # faq_search: FAQ or knowledge base search
    if re.search(r"\b(faq|frequently\s+asked)\b", ql):
        return {"domain": "customer_service", "task_type": "faq_search"}
    if re.search(r"\b(search|find|look\s+up)\b.*\b(help\s+article|knowledge\s+base|support\s+doc)\b", ql):
        return {"domain": "customer_service", "task_type": "faq_search"}

    return None

def _keyword_task_fast_path(query: str) -> Optional[Dict[str, str]]:
    """Keyword fast-path for domain tasks where embedding similarity
    between closely related tasks (e.g., anomaly vs duplicate detection)
    produces ambiguous scores.
    """
    import re
    ql = query.lower().strip()

    # Invoice: payment anomaly detection
    if re.search(r"\banomal\w*\b", ql) and re.search(r"\b(payment|invoice|billing|charge)\b", ql):
        return {"domain": "invoice", "task_type": "payment_anomaly_detection"}

    # Legal: compliance check
    if re.search(r"\bcompl\w*\b", ql) and re.search(r"\b(regulat|gdpr|standard|law|rule|requirement)\b", ql):
        return {"domain": "legal", "task_type": "compliance_check"}

    # Translation: detect language
    if re.search(r"\b(what\s+)?language\b", ql) and re.search(r"\b(written|detect|identify|what)\b", ql):
        return {"domain": "translation", "task_type": "detect_language"}

    return None

def detect_agent_task(query: str, domain: str = "") -> Optional[Dict[str, str]]:
    """Detect if a query requires a specialized agent task.

    Uses the centralized NLU engine with embedding similarity + structural NLP
    to match queries to domain-specific task types. No hardcoded keywords.

    Returns {"domain": ..., "task_type": ...} or None.
    """
    # Strategy 0a: keyword fast-path for customer service tasks
    cs_match = _customer_service_fast_path(query)
    if cs_match:
        return cs_match

    # Strategy 0b: keyword fast-path for ambiguous embedding pairs
    kw_match = _keyword_task_fast_path(query)
    if kw_match:
        return kw_match

    # Strategy 1: NLU-based task classification
    try:
        from src.nlp.nlu_engine import classify_domain_task
        result = classify_domain_task(query, domain=domain)
        if result:
            return result
    except Exception as exc:
        logger.debug("NLU domain task classification failed: %s", exc)

    # Strategy 2: ML-based domain detection (fallback)
    ml_domain = _ml_detect_domain(query)
    if ml_domain:
        try:
            from src.nlp.nlu_engine import classify_domain_task
            result = classify_domain_task(query, domain=ml_domain)
            if result:
                return result
        except Exception:
            pass

    return None

__all__ = [
    "DomainAgent",
    "AgentTaskResult",
    "ResumeAgent",
    "MedicalAgent",
    "LegalAgent",
    "InvoiceAgent",
    "ContentAgent",
    "TranslatorAgent",
    "TutorAgent",
    "ImageAgent",
    "WebAgent",
    "InsightsAgent",
    "ScreeningAgent",
    "CustomerServiceAgent",
    "AnalyticsVisualizationAgent",
    "get_domain_agent",
    "list_available_agents",
    "detect_agent_task",
]
