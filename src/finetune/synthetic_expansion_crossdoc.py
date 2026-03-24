"""Cross-document reasoning and content generation expansion.

Generates ~100 training examples across two capability areas:
- Cross-document reasoning (~50): pattern detection, inconsistency
  detection, multi-doc comparison, aggregation, timeline/sequence.
- Content generation (~50): executive summaries, interview questions,
  compliance reports, candidate summaries, meeting prep.

Each example uses short evidence blocks (2-3 lines per document)
and focused responses (200-400 chars).
"""
from __future__ import annotations
from typing import Any, Dict, List


SYSTEM_PROMPT = (
    "You are DocWain \u2014 Document Wise AI Node \u2014 an intelligent document intelligence "
    "platform engineered by DHS IT Solutions Ltd. You process enterprise documents and "
    "provide grounded, evidence-backed answers. You never hallucinate. Every claim "
    "traces to uploaded documents. You work within profile-isolated scopes and never "
    "leak data across users. You analyze documents through metadata reasoning, "
    "structural understanding, entity recognition, and cross-document intelligence."
)


def _c(user: str, assistant: str) -> Dict[str, Any]:
    """Build a chat training example."""
    return {"messages": [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]}


# ---------------------------------------------------------------------------
# Cross-document reasoning examples (~50)
# ---------------------------------------------------------------------------

def generate_crossdoc_expanded() -> List[Dict]:
    """Return ~50 cross-document reasoning training examples."""
    examples: List[Dict] = []

    # === PATTERN DETECTION (10) ===

    examples.append(_c(
        "[DOC-1: Q1 Sales Report]\nRevenue: $2.1M. Top region: Northeast (+18%).\n"
        "[DOC-2: Q2 Sales Report]\nRevenue: $2.4M. Top region: Northeast (+22%).\n"
        "[DOC-3: Q3 Sales Report]\nRevenue: $2.9M. Top region: Northeast (+25%).\n\n"
        "What pattern do you see across these quarterly reports?",
        "## Cross-Document Pattern\n\n"
        "A consistent **upward revenue trend** is evident: $2.1M \u2192 $2.4M \u2192 $2.9M "
        "(+38% cumulative). The Northeast region drives growth each quarter with "
        "accelerating gains (+18% \u2192 +22% \u2192 +25%). This suggests compounding "
        "market penetration in the Northeast."
    ))

    examples.append(_c(
        "[DOC-1: Employee Survey Jan]\nEngagement score: 72%. Top concern: workload.\n"
        "[DOC-2: Employee Survey Apr]\nEngagement score: 68%. Top concern: workload.\n"
        "[DOC-3: Employee Survey Jul]\nEngagement score: 63%. Top concern: burnout.\n\n"
        "Identify the recurring pattern across these surveys.",
        "## Pattern: Declining Engagement\n\n"
        "Engagement scores are steadily declining: 72% \u2192 68% \u2192 63%. The persistent "
        "concern shifted from **workload** to **burnout**, indicating the original "
        "workload issue was never addressed and escalated. Intervention is needed "
        "before scores drop below 60%."
    ))

    examples.append(_c(
        "[DOC-1: Incident Log Jan-Mar]\n14 incidents. Root cause: config drift (9/14).\n"
        "[DOC-2: Incident Log Apr-Jun]\n11 incidents. Root cause: config drift (7/11).\n"
        "[DOC-3: Incident Log Jul-Sep]\n16 incidents. Root cause: config drift (12/16).\n\n"
        "What recurring pattern exists across these incident logs?",
        "## Pattern: Configuration Drift Dominance\n\n"
        "**Config drift** is the root cause in 64\u201375% of incidents every quarter "
        "(9/14, 7/11, 12/16). Despite a brief dip in Q2, incidents rebounded in Q3. "
        "The pattern indicates a systemic lack of configuration management controls "
        "rather than isolated events."
    ))

    examples.append(_c(
        "[DOC-1: Vendor A Invoice History]\nAvg delivery: 3.2 days. Late rate: 5%.\n"
        "[DOC-2: Vendor B Invoice History]\nAvg delivery: 4.8 days. Late rate: 18%.\n"
        "[DOC-3: Vendor C Invoice History]\nAvg delivery: 3.5 days. Late rate: 7%.\n\n"
        "What delivery pattern emerges across these vendor records?",
        "## Vendor Delivery Pattern\n\n"
        "- **Vendor A** and **Vendor C** deliver reliably (3.2\u20133.5 days, <7% late).\n"
        "- **Vendor B** is an outlier: 4.8 days average and **18% late rate**.\n\n"
        "Vendor B's performance is roughly 3x worse on late deliveries, suggesting "
        "capacity or logistics issues that warrant review."
    ))

    examples.append(_c(
        "[DOC-1: Support Tickets Jan]\n342 tickets. Category: login issues (38%).\n"
        "[DOC-2: Support Tickets Feb]\n389 tickets. Category: login issues (41%).\n"
        "[DOC-3: Support Tickets Mar]\n415 tickets. Category: login issues (44%).\n\n"
        "What pattern do these support ticket reports reveal?",
        "## Pattern: Escalating Login Issues\n\n"
        "Login-related tickets are growing both in volume and proportion: 130 \u2192 160 "
        "\u2192 183 tickets (38% \u2192 41% \u2192 44%). This accelerating trend points to a "
        "systemic authentication problem that is worsening, not a transient spike."
    ))

    examples.append(_c(
        "[DOC-1: Project Alpha Status]\nBudget used: 45%. Timeline: on track.\n"
        "[DOC-2: Project Beta Status]\nBudget used: 72%. Timeline: 3 weeks behind.\n"
        "[DOC-3: Project Gamma Status]\nBudget used: 40%. Timeline: on track.\n\n"
        "What pattern distinguishes the projects?",
        "## Pattern Analysis\n\n"
        "Projects **Alpha** and **Gamma** share a healthy pattern: moderate budget "
        "consumption (\u226445%) and on-track timelines. **Project Beta** breaks the "
        "pattern with 72% budget burn and a 3-week delay, indicating scope creep or "
        "resource misallocation unique to that project."
    ))

    examples.append(_c(
        "[DOC-1: Hiring Report Q1]\n12 hires. Avg time-to-fill: 34 days. Offer acceptance: 82%.\n"
        "[DOC-2: Hiring Report Q2]\n9 hires. Avg time-to-fill: 41 days. Offer acceptance: 75%.\n"
        "[DOC-3: Hiring Report Q3]\n7 hires. Avg time-to-fill: 48 days. Offer acceptance: 68%.\n\n"
        "Identify the hiring trend across quarters.",
        "## Hiring Trend: Slowing Pipeline\n\n"
        "All three metrics are deteriorating each quarter:\n"
        "- **Hires:** 12 \u2192 9 \u2192 7\n- **Time-to-fill:** 34 \u2192 41 \u2192 48 days\n"
        "- **Acceptance rate:** 82% \u2192 75% \u2192 68%\n\n"
        "The pattern suggests declining employer competitiveness, possibly due to "
        "compensation or brand perception issues."
    ))

    examples.append(_c(
        "[DOC-1: Marketing Campaign A]\nSpend: $50K. Leads: 1,200. CPL: $41.67.\n"
        "[DOC-2: Marketing Campaign B]\nSpend: $30K. Leads: 900. CPL: $33.33.\n"
        "[DOC-3: Marketing Campaign C]\nSpend: $80K. Leads: 1,800. CPL: $44.44.\n\n"
        "What cost pattern do you detect across campaigns?",
        "## Cost-Per-Lead Pattern\n\n"
        "**Campaign B** is the most cost-efficient at $33.33/lead despite the "
        "smallest budget. Campaigns A and C show diminishing returns as spend "
        "increases ($41.67 and $44.44 respectively). Higher budgets do not "
        "proportionally increase lead volume; smaller targeted campaigns outperform."
    ))

    examples.append(_c(
        "[DOC-1: Server Metrics Jan]\nCPU avg: 45%. Memory avg: 62%. Alerts: 3.\n"
        "[DOC-2: Server Metrics Feb]\nCPU avg: 52%. Memory avg: 68%. Alerts: 7.\n"
        "[DOC-3: Server Metrics Mar]\nCPU avg: 61%. Memory avg: 74%. Alerts: 14.\n\n"
        "What infrastructure pattern is emerging?",
        "## Pattern: Resource Exhaustion Trajectory\n\n"
        "Both CPU and memory utilization are climbing steadily (CPU: 45% \u2192 61%, "
        "Memory: 62% \u2192 74%) and alerts are doubling monthly (3 \u2192 7 \u2192 14). "
        "At this rate, resources will hit critical thresholds within 2\u20133 months. "
        "Capacity planning or scaling is urgently needed."
    ))

    examples.append(_c(
        "[DOC-1: Client NPS Q1]\nScore: 42. Detractors cite: slow response times.\n"
        "[DOC-2: Client NPS Q2]\nScore: 38. Detractors cite: slow response times.\n"
        "[DOC-3: Client NPS Q3]\nScore: 35. Detractors cite: poor follow-up.\n\n"
        "What client satisfaction pattern exists?",
        "## Pattern: Eroding Client Satisfaction\n\n"
        "NPS dropped from 42 \u2192 38 \u2192 35 across three quarters. The detractor theme "
        "shifted from **slow response times** to **poor follow-up**, suggesting "
        "that while initial responsiveness may have improved, the service "
        "delivery chain is failing downstream."
    ))

    # === INCONSISTENCY DETECTION (10) ===

    examples.append(_c(
        "[DOC-1: Annual Report 2025]\nTotal employees: 1,247. Revenue: $45.2M.\n"
        "[DOC-2: HR Headcount Report Dec 2025]\nTotal active employees: 1,312.\n\n"
        "Are there inconsistencies between these documents?",
        "## Inconsistency Detected\n\n"
        "**Employee count mismatch:** The Annual Report states **1,247** employees "
        "while the HR Headcount Report lists **1,312** \u2014 a discrepancy of 65 people "
        "(5.2%). This may indicate the annual report excluded contractors or used "
        "a different reporting date. Reconciliation is needed."
    ))

    examples.append(_c(
        "[DOC-1: Contract with Vendor X]\nPayment terms: Net 45. Penalty: 2% per month.\n"
        "[DOC-2: Vendor X Invoice #4421]\nPayment terms: Net 30.\n\n"
        "Check these documents for inconsistencies.",
        "## Inconsistency: Payment Terms\n\n"
        "The contract specifies **Net 45** but the invoice states **Net 30**. "
        "This 15-day discrepancy could result in early-payment pressure or "
        "unwarranted late-payment penalties. The invoice should be corrected "
        "to match the contractual Net 45 terms."
    ))

    examples.append(_c(
        "[DOC-1: Board Minutes Mar 2025]\nApproved budget: $3.2M for Project Phoenix.\n"
        "[DOC-2: Project Phoenix Charter]\nAllocated budget: $2.8M.\n\n"
        "Do these documents align?",
        "## Budget Inconsistency\n\n"
        "The board approved **$3.2M** but the project charter only allocates "
        "**$2.8M** \u2014 a $400K gap. Either the charter was drafted before final "
        "approval, or $400K was redirected elsewhere. The charter should be "
        "updated to reflect the board-approved figure."
    ))

    examples.append(_c(
        "[DOC-1: Product Spec v2.1]\nMax concurrent users: 10,000. SLA: 99.9%.\n"
        "[DOC-2: Sales Proposal to Client Y]\nMax concurrent users: 50,000. SLA: 99.99%.\n\n"
        "Are there inconsistencies between these documents?",
        "## Critical Inconsistency\n\n"
        "The sales proposal **overstates capabilities** compared to the product spec:\n"
        "- **Users:** 50,000 claimed vs. 10,000 supported (5x overstatement)\n"
        "- **SLA:** 99.99% promised vs. 99.9% actual\n\n"
        "This creates contractual risk. The proposal must be corrected before signing."
    ))

    examples.append(_c(
        "[DOC-1: Insurance Policy #PP-8821]\nCoverage limit: $500,000. Deductible: $5,000.\n"
        "[DOC-2: Certificate of Insurance]\nCoverage limit: $500,000. Deductible: $10,000.\n\n"
        "Compare these for inconsistencies.",
        "## Deductible Inconsistency\n\n"
        "Coverage limits match at $500,000, but the **deductible** differs: "
        "policy states **$5,000** while the certificate shows **$10,000**. "
        "The certificate may reflect an outdated endorsement. The insurer "
        "should reissue the certificate to match the policy."
    ))

    examples.append(_c(
        "[DOC-1: Job Description \u2014 Senior Developer]\nRequired: 5+ years Java, AWS.\n"
        "[DOC-2: Offer Letter \u2014 Candidate J. Smith]\nTitle: Junior Developer. Salary: $65K.\n\n"
        "Identify inconsistencies between the job description and offer letter.",
        "## Title Mismatch\n\n"
        "The job description is for **Senior Developer** (5+ years required) "
        "but the offer letter titles the role **Junior Developer** at $65K. "
        "Either the wrong template was used or the candidate is being "
        "under-leveled relative to the position requirements."
    ))

    examples.append(_c(
        "[DOC-1: Audit Report 2025]\nInventory value: $8.4M as of Dec 31.\n"
        "[DOC-2: Balance Sheet Dec 2025]\nInventory: $7.1M.\n\n"
        "Check for discrepancies between these financial documents.",
        "## Inventory Valuation Gap\n\n"
        "The audit report values inventory at **$8.4M** while the balance sheet "
        "shows **$7.1M** \u2014 a **$1.3M discrepancy** (15.5%). This could indicate "
        "write-downs not reflected in the audit, timing differences, or an "
        "accounting error requiring immediate investigation."
    ))

    examples.append(_c(
        "[DOC-1: Training Certificate]\nEmployee: R. Patel. Course: HIPAA Compliance. Date: 2024-06-15.\n"
        "[DOC-2: HR Records]\nR. Patel: HIPAA training \u2014 Not Completed.\n\n"
        "Are these records consistent?",
        "## Record Inconsistency\n\n"
        "R. Patel holds a **training certificate** dated 2024-06-15 for HIPAA "
        "Compliance, but HR records show the training as **Not Completed**. "
        "The HR system was likely not updated after certification. "
        "The HR record should be corrected to reflect completion."
    ))

    examples.append(_c(
        "[DOC-1: Lease Agreement]\nMonthly rent: $4,200. Start: 2025-01-01. Term: 24 months.\n"
        "[DOC-2: Accounts Payable Ledger]\nRent payments to landlord: $3,800/month since Jan 2025.\n\n"
        "Do these documents match?",
        "## Rent Payment Discrepancy\n\n"
        "The lease specifies **$4,200/month** but AP records show payments of "
        "**$3,800/month** \u2014 a $400/month shortfall. This could indicate a "
        "negotiated reduction not documented in the lease, or an underpayment "
        "that may trigger breach-of-lease provisions."
    ))

    examples.append(_c(
        "[DOC-1: Safety Inspection Report]\nFire extinguishers: 12 inspected, all passed.\n"
        "[DOC-2: Facility Inventory]\nFire extinguishers on premises: 15.\n\n"
        "Identify any inconsistencies.",
        "## Inspection Coverage Gap\n\n"
        "The facility has **15 fire extinguishers** but only **12 were inspected**. "
        "Three units (20%) were missed during the safety inspection. All 15 must "
        "be inspected to maintain compliance. The missing 3 should be located "
        "and inspected immediately."
    ))

    # === MULTI-DOC COMPARISON (10) ===

    examples.append(_c(
        "[DOC-1: Vendor A Proposal]\nPrice: $120K. Timeline: 8 weeks. Support: 12 months.\n"
        "[DOC-2: Vendor B Proposal]\nPrice: $95K. Timeline: 12 weeks. Support: 6 months.\n"
        "[DOC-3: Vendor C Proposal]\nPrice: $140K. Timeline: 6 weeks. Support: 24 months.\n\n"
        "Compare these vendor proposals.",
        "## Vendor Comparison\n\n"
        "| Criteria | Vendor A | Vendor B | Vendor C |\n"
        "|----------|----------|----------|----------|\n"
        "| Price | $120K | $95K | $140K |\n"
        "| Timeline | 8 weeks | 12 weeks | 6 weeks |\n"
        "| Support | 12 months | 6 months | 24 months |\n\n"
        "**Vendor B** is cheapest but slowest with minimal support. "
        "**Vendor C** is fastest with best support but most expensive. "
        "**Vendor A** offers a balanced middle ground."
    ))

    examples.append(_c(
        "[DOC-1: Cloud Provider X]\nUptime SLA: 99.95%. Storage: $0.023/GB. Regions: 25.\n"
        "[DOC-2: Cloud Provider Y]\nUptime SLA: 99.99%. Storage: $0.021/GB. Regions: 34.\n\n"
        "Compare these cloud providers.",
        "## Cloud Provider Comparison\n\n"
        "| Metric | Provider X | Provider Y |\n"
        "|--------|-----------|------------|\n"
        "| Uptime SLA | 99.95% | 99.99% |\n"
        "| Storage Cost | $0.023/GB | $0.021/GB |\n"
        "| Regions | 25 | 34 |\n\n"
        "**Provider Y** is superior across all three metrics: higher SLA, "
        "lower cost, and broader regional coverage."
    ))

    examples.append(_c(
        "[DOC-1: Policy A \u2014 Remote Work]\nEligibility: all employees. Days: 3/week max.\n"
        "[DOC-2: Policy B \u2014 Remote Work (Draft)]\nEligibility: tenure >1 year. Days: 5/week.\n\n"
        "Compare the current and draft remote work policies.",
        "## Policy Comparison\n\n"
        "| Aspect | Current (Policy A) | Draft (Policy B) |\n"
        "|--------|--------------------|-------------------|\n"
        "| Eligibility | All employees | Tenure >1 year |\n"
        "| Remote days | 3/week max | 5/week (full remote) |\n\n"
        "The draft **restricts eligibility** to tenured employees but "
        "**expands flexibility** to full-time remote. New hires lose remote "
        "access for their first year under the proposed policy."
    ))

    examples.append(_c(
        "[DOC-1: Resume \u2014 A. Chen]\nExperience: 8 years Python/ML. Education: MS CS, Stanford.\n"
        "[DOC-2: Resume \u2014 B. Kumar]\nExperience: 5 years Python/ML. Education: MS CS, MIT.\n"
        "[DOC-3: Resume \u2014 C. Lee]\nExperience: 10 years Python/ML. Education: PhD CS, CMU.\n\n"
        "Compare these candidates for a senior ML engineer role.",
        "## Candidate Comparison\n\n"
        "| Attribute | A. Chen | B. Kumar | C. Lee |\n"
        "|-----------|---------|----------|--------|\n"
        "| Experience | 8 years | 5 years | 10 years |\n"
        "| Education | MS, Stanford | MS, MIT | PhD, CMU |\n\n"
        "**C. Lee** is strongest with 10 years and a PhD. **A. Chen** offers "
        "solid experience. **B. Kumar** has the least experience but strong "
        "academic credentials. All are viable for the senior role."
    ))

    examples.append(_c(
        "[DOC-1: Health Plan A]\nPremium: $450/mo. Deductible: $2,000. Copay: $30.\n"
        "[DOC-2: Health Plan B]\nPremium: $320/mo. Deductible: $5,000. Copay: $50.\n\n"
        "Compare these health plan options.",
        "## Health Plan Comparison\n\n"
        "| Feature | Plan A | Plan B |\n"
        "|---------|--------|--------|\n"
        "| Monthly Premium | $450 | $320 |\n"
        "| Deductible | $2,000 | $5,000 |\n"
        "| Copay | $30 | $50 |\n\n"
        "**Plan A** costs $130/mo more but saves $3,000 on the deductible. "
        "For employees expecting frequent medical visits, Plan A is more "
        "cost-effective. Plan B suits those who rarely need care."
    ))

    examples.append(_c(
        "[DOC-1: Software License \u2014 Tool X]\nSeats: 100. Cost: $12K/year. Term: annual.\n"
        "[DOC-2: Software License \u2014 Tool Y]\nSeats: unlimited. Cost: $18K/year. Term: 3-year.\n\n"
        "Compare these software licenses.",
        "## License Comparison\n\n"
        "| Aspect | Tool X | Tool Y |\n"
        "|--------|--------|--------|\n"
        "| Seats | 100 | Unlimited |\n"
        "| Annual Cost | $12K | $18K |\n"
        "| Commitment | Annual | 3-year |\n\n"
        "Tool X is cheaper at $120/seat/year but caps at 100 users. Tool Y "
        "costs 50% more but removes seat limits. For teams likely to grow "
        "past 100, Tool Y is more scalable despite the longer commitment."
    ))

    examples.append(_c(
        "[DOC-1: Office Lease \u2014 Location A]\nRent: $28/sqft. Size: 5,000 sqft. Parking: 20 spots.\n"
        "[DOC-2: Office Lease \u2014 Location B]\nRent: $22/sqft. Size: 6,500 sqft. Parking: 10 spots.\n\n"
        "Compare these office lease options.",
        "## Office Lease Comparison\n\n"
        "| Factor | Location A | Location B |\n"
        "|--------|-----------|------------|\n"
        "| Rent | $28/sqft ($140K/yr) | $22/sqft ($143K/yr) |\n"
        "| Size | 5,000 sqft | 6,500 sqft |\n"
        "| Parking | 20 spots | 10 spots |\n\n"
        "Total annual costs are nearly identical. Location B offers 30% more "
        "space but half the parking. If parking is critical, choose A; if "
        "space is the priority, choose B."
    ))

    examples.append(_c(
        "[DOC-1: Penetration Test \u2014 App A]\nCritical: 2. High: 5. Medium: 12.\n"
        "[DOC-2: Penetration Test \u2014 App B]\nCritical: 0. High: 3. Medium: 8.\n\n"
        "Compare the security posture of these applications.",
        "## Security Posture Comparison\n\n"
        "| Severity | App A | App B |\n"
        "|----------|-------|-------|\n"
        "| Critical | 2 | 0 |\n"
        "| High | 5 | 3 |\n"
        "| Medium | 12 | 8 |\n\n"
        "**App B** has a significantly stronger security posture with zero "
        "critical findings. **App A** requires immediate remediation of its "
        "2 critical vulnerabilities before production deployment."
    ))

    examples.append(_c(
        "[DOC-1: Training Program A]\nDuration: 2 weeks. Cost: $1,500. Format: in-person.\n"
        "[DOC-2: Training Program B]\nDuration: 4 weeks. Cost: $800. Format: online.\n"
        "[DOC-3: Training Program C]\nDuration: 1 week. Cost: $2,200. Format: bootcamp.\n\n"
        "Compare these training programs.",
        "## Training Program Comparison\n\n"
        "| Aspect | Program A | Program B | Program C |\n"
        "|--------|-----------|-----------|----------|\n"
        "| Duration | 2 weeks | 4 weeks | 1 week |\n"
        "| Cost | $1,500 | $800 | $2,200 |\n"
        "| Format | In-person | Online | Bootcamp |\n\n"
        "**Program B** is most affordable but takes longest. **Program C** "
        "is the fastest but most expensive. **Program A** balances time and "
        "cost with hands-on learning."
    ))

    examples.append(_c(
        "[DOC-1: CRM System Eval]\nEase of use: 4.2/5. Integrations: 150+. Price: $50/user/mo.\n"
        "[DOC-2: CRM System Eval]\nEase of use: 3.8/5. Integrations: 300+. Price: $35/user/mo.\n\n"
        "Compare these CRM systems.",
        "## CRM System Comparison\n\n"
        "| Metric | CRM 1 | CRM 2 |\n"
        "|--------|-------|-------|\n"
        "| Ease of Use | 4.2/5 | 3.8/5 |\n"
        "| Integrations | 150+ | 300+ |\n"
        "| Price | $50/user/mo | $35/user/mo |\n\n"
        "**CRM 2** is cheaper with double the integrations, while **CRM 1** "
        "scores higher on usability. For integration-heavy workflows, CRM 2 "
        "is the better value."
    ))

    # === AGGREGATION (10) ===

    examples.append(_c(
        "[DOC-1: Dept A Budget]\nHeadcount: 45. Annual spend: $3.2M.\n"
        "[DOC-2: Dept B Budget]\nHeadcount: 32. Annual spend: $2.1M.\n"
        "[DOC-3: Dept C Budget]\nHeadcount: 28. Annual spend: $1.8M.\n\n"
        "Aggregate the total headcount and spend across all departments.",
        "## Aggregated Totals\n\n"
        "| Department | Headcount | Spend |\n"
        "|-----------|-----------|-------|\n"
        "| Dept A | 45 | $3.2M |\n"
        "| Dept B | 32 | $2.1M |\n"
        "| Dept C | 28 | $1.8M |\n"
        "| **Total** | **105** | **$7.1M** |\n\n"
        "Average spend per employee: **$67,619**. Dept A has the highest "
        "per-capita spend at $71,111."
    ))

    examples.append(_c(
        "[DOC-1: Branch North]\nNew accounts: 142. Deposits: $4.8M.\n"
        "[DOC-2: Branch South]\nNew accounts: 98. Deposits: $3.1M.\n"
        "[DOC-3: Branch Central]\nNew accounts: 167. Deposits: $5.5M.\n\n"
        "Provide aggregated figures across all branches.",
        "## Branch Aggregation\n\n"
        "- **Total new accounts:** 407 (Central leads with 167)\n"
        "- **Total deposits:** $13.4M (Central leads with $5.5M)\n"
        "- **Average deposits per account:** ~$32,923\n\n"
        "Central branch outperforms on both metrics, contributing 41% of "
        "all new accounts and 41% of total deposits."
    ))

    examples.append(_c(
        "[DOC-1: Product Line A]\nUnits sold: 12,400. Returns: 310.\n"
        "[DOC-2: Product Line B]\nUnits sold: 8,700. Returns: 435.\n"
        "[DOC-3: Product Line C]\nUnits sold: 15,200. Returns: 228.\n\n"
        "Aggregate sales and calculate return rates.",
        "## Sales Aggregation\n\n"
        "| Product Line | Units Sold | Returns | Return Rate |\n"
        "|-------------|-----------|---------|-------------|\n"
        "| A | 12,400 | 310 | 2.5% |\n"
        "| B | 8,700 | 435 | 5.0% |\n"
        "| C | 15,200 | 228 | 1.5% |\n"
        "| **Total** | **36,300** | **973** | **2.7%** |\n\n"
        "Product Line B has a **5.0% return rate** \u2014 double the average \u2014 "
        "warranting a quality investigation."
    ))

    examples.append(_c(
        "[DOC-1: Project X Timesheet]\nHours logged: 340. Team size: 4.\n"
        "[DOC-2: Project Y Timesheet]\nHours logged: 520. Team size: 6.\n"
        "[DOC-3: Project Z Timesheet]\nHours logged: 280. Team size: 3.\n\n"
        "Aggregate hours and compute per-person averages.",
        "## Timesheet Aggregation\n\n"
        "- **Total hours:** 1,140 across 13 team members\n"
        "- **Average hours per person:** 87.7\n"
        "- Project Y consumed the most hours (520) with the largest team (6)\n"
        "- Per-person hours: X=85, Y=86.7, Z=93.3\n\n"
        "Project Z has the highest per-person load despite the smallest team."
    ))

    examples.append(_c(
        "[DOC-1: Region EMEA]\nRevenue: $12.4M. Customers: 342.\n"
        "[DOC-2: Region APAC]\nRevenue: $8.7M. Customers: 218.\n"
        "[DOC-3: Region Americas]\nRevenue: $18.2M. Customers: 567.\n\n"
        "Aggregate global revenue and customer metrics.",
        "## Global Aggregation\n\n"
        "- **Total revenue:** $39.3M\n- **Total customers:** 1,127\n"
        "- **Revenue per customer:** ~$34,870\n\n"
        "Americas dominates (46% of revenue, 50% of customers). "
        "EMEA has the highest per-customer revenue at $36,257, "
        "suggesting higher-value deals in that region."
    ))

    examples.append(_c(
        "[DOC-1: Warehouse A Inventory]\nSKUs: 1,240. Value: $2.3M.\n"
        "[DOC-2: Warehouse B Inventory]\nSKUs: 890. Value: $1.7M.\n\n"
        "Aggregate inventory across warehouses.",
        "## Inventory Aggregation\n\n"
        "- **Total SKUs:** 2,130\n- **Total value:** $4.0M\n"
        "- **Avg value per SKU:** ~$1,878\n\n"
        "Warehouse A holds 58% of SKUs and 57.5% of inventory value, "
        "making it the primary storage facility."
    ))

    examples.append(_c(
        "[DOC-1: Jan Expenses]\nTravel: $14K. Software: $8K. Office: $3K.\n"
        "[DOC-2: Feb Expenses]\nTravel: $11K. Software: $8K. Office: $3.5K.\n"
        "[DOC-3: Mar Expenses]\nTravel: $16K. Software: $8K. Office: $2.8K.\n\n"
        "Aggregate Q1 expenses by category.",
        "## Q1 Expense Aggregation\n\n"
        "| Category | Jan | Feb | Mar | Q1 Total |\n"
        "|----------|-----|-----|-----|----------|\n"
        "| Travel | $14K | $11K | $16K | **$41K** |\n"
        "| Software | $8K | $8K | $8K | **$24K** |\n"
        "| Office | $3K | $3.5K | $2.8K | **$9.3K** |\n"
        "| **Total** | $25K | $22.5K | $26.8K | **$74.3K** |\n\n"
        "Travel is the largest category at 55% of total Q1 spend."
    ))

    examples.append(_c(
        "[DOC-1: Team Alpha Sprint 12]\nStories completed: 18. Bugs fixed: 7.\n"
        "[DOC-2: Team Beta Sprint 12]\nStories completed: 14. Bugs fixed: 12.\n"
        "[DOC-3: Team Gamma Sprint 12]\nStories completed: 22. Bugs fixed: 4.\n\n"
        "Aggregate sprint delivery metrics.",
        "## Sprint 12 Aggregation\n\n"
        "- **Total stories completed:** 54\n- **Total bugs fixed:** 23\n"
        "- **Total delivery items:** 77\n\n"
        "Team Gamma leads on feature delivery (22 stories) while Team Beta "
        "focused heavily on bug fixing (12 bugs, 46% of their output). "
        "Overall bug-to-story ratio is 0.43."
    ))

    examples.append(_c(
        "[DOC-1: Cert Tracker \u2014 Engineering]\nActive certs: 34. Expiring <90 days: 8.\n"
        "[DOC-2: Cert Tracker \u2014 Operations]\nActive certs: 21. Expiring <90 days: 5.\n"
        "[DOC-3: Cert Tracker \u2014 Security]\nActive certs: 18. Expiring <90 days: 3.\n\n"
        "Aggregate certification status across departments.",
        "## Certification Aggregation\n\n"
        "- **Total active certifications:** 73\n"
        "- **Expiring within 90 days:** 16 (21.9%)\n\n"
        "Engineering has the most at-risk certs (8 of 16). Renewal "
        "workflows should prioritize those 16 expirations to avoid "
        "compliance gaps."
    ))

    examples.append(_c(
        "[DOC-1: Call Center Jan]\nCalls handled: 4,200. Avg wait: 2.1 min.\n"
        "[DOC-2: Call Center Feb]\nCalls handled: 4,800. Avg wait: 3.4 min.\n"
        "[DOC-3: Call Center Mar]\nCalls handled: 5,100. Avg wait: 4.2 min.\n\n"
        "Aggregate Q1 call center performance.",
        "## Q1 Call Center Aggregation\n\n"
        "- **Total calls handled:** 14,100\n"
        "- **Avg wait time (weighted):** 3.28 minutes\n\n"
        "Call volume grew 21% (4,200 \u2192 5,100) while wait times doubled "
        "(2.1 \u2192 4.2 min). Staffing has not scaled with demand. "
        "Additional agents or IVR deflection is recommended."
    ))

    # === TIMELINE / SEQUENCE (10) ===

    examples.append(_c(
        "[DOC-1: Requirements Doc]\nApproved: 2025-01-15. Signed by: VP Engineering.\n"
        "[DOC-2: Design Doc]\nCompleted: 2025-02-10. Reviewed by: Architecture Board.\n"
        "[DOC-3: Go-Live Report]\nDeployed: 2025-03-28. Approved by: Change Advisory Board.\n\n"
        "Reconstruct the project timeline from these documents.",
        "## Project Timeline\n\n"
        "1. **2025-01-15** \u2014 Requirements approved (VP Engineering)\n"
        "2. **2025-02-10** \u2014 Design completed (Architecture Board)\n"
        "3. **2025-03-28** \u2014 Go-live deployed (Change Advisory Board)\n\n"
        "Total duration: **72 days**. Requirements-to-design took 26 days; "
        "design-to-deployment took 46 days. The build/deploy phase consumed "
        "64% of the timeline."
    ))

    examples.append(_c(
        "[DOC-1: Incident Report]\nDetected: 2025-06-12 09:14 UTC. Service: payment-api.\n"
        "[DOC-2: RCA Document]\nRoot cause identified: 2025-06-12 11:42 UTC. Cause: DB failover.\n"
        "[DOC-3: Post-Incident Review]\nResolved: 2025-06-12 13:05 UTC. Customer impact: 3h 51m.\n\n"
        "Build the incident timeline.",
        "## Incident Timeline\n\n"
        "1. **09:14 UTC** \u2014 Incident detected in payment-api\n"
        "2. **11:42 UTC** \u2014 Root cause identified (DB failover) \u2014 2h 28m\n"
        "3. **13:05 UTC** \u2014 Incident resolved \u2014 1h 23m after RCA\n\n"
        "**Total duration:** 3h 51m. Detection-to-diagnosis took 63% of "
        "the resolution time, suggesting monitoring gaps."
    ))

    examples.append(_c(
        "[DOC-1: Patent Filing]\nFiled: 2024-03-20. Patent #: US-2024-88712.\n"
        "[DOC-2: Patent Office Response]\nExaminer action: 2024-09-14. Status: objections raised.\n"
        "[DOC-3: Amendment Filing]\nAmendment submitted: 2024-12-01.\n\n"
        "Reconstruct the patent prosecution timeline.",
        "## Patent Timeline\n\n"
        "1. **2024-03-20** \u2014 Initial filing (US-2024-88712)\n"
        "2. **2024-09-14** \u2014 Examiner objections raised (~6 months)\n"
        "3. **2024-12-01** \u2014 Amendment submitted (~2.5 months response)\n\n"
        "The response to examiner objections took 78 days. Awaiting next "
        "office action. Total prosecution time so far: ~8.5 months."
    ))

    examples.append(_c(
        "[DOC-1: Employee Onboarding]\nHire date: 2025-04-01. Orientation completed: 2025-04-03.\n"
        "[DOC-2: Training Record]\nCore training completed: 2025-04-18.\n"
        "[DOC-3: Probation Review]\n90-day review: 2025-07-01. Outcome: passed.\n\n"
        "Map the employee onboarding sequence.",
        "## Onboarding Timeline\n\n"
        "1. **Apr 1** \u2014 Hire date\n"
        "2. **Apr 3** \u2014 Orientation completed (2 days)\n"
        "3. **Apr 18** \u2014 Core training completed (15 days)\n"
        "4. **Jul 1** \u2014 90-day probation review passed\n\n"
        "Onboarding to full productivity took 17 days. Probation period "
        "of 91 days completed successfully."
    ))

    examples.append(_c(
        "[DOC-1: RFP Published]\nDate: 2025-02-01. Deadline: 2025-03-01.\n"
        "[DOC-2: Vendor Shortlist]\nDate: 2025-03-15. Vendors: 3 shortlisted.\n"
        "[DOC-3: Contract Award]\nDate: 2025-04-10. Awarded to: Vendor B.\n\n"
        "Outline the procurement timeline.",
        "## Procurement Timeline\n\n"
        "1. **Feb 1** \u2014 RFP published (28-day response window)\n"
        "2. **Mar 1** \u2014 Submission deadline\n"
        "3. **Mar 15** \u2014 Shortlist finalized (14-day evaluation)\n"
        "4. **Apr 10** \u2014 Contract awarded to Vendor B (26-day selection)\n\n"
        "Total cycle: **68 days** from RFP to award."
    ))

    examples.append(_c(
        "[DOC-1: FDA Submission]\nSubmitted: 2025-01-10. Product: MedDevice X.\n"
        "[DOC-2: FDA Review Letter]\nDate: 2025-04-22. Status: additional data requested.\n"
        "[DOC-3: Supplemental Filing]\nSubmitted: 2025-06-15.\n\n"
        "Reconstruct the regulatory timeline.",
        "## Regulatory Timeline\n\n"
        "1. **Jan 10** \u2014 Initial FDA submission (MedDevice X)\n"
        "2. **Apr 22** \u2014 FDA requests additional data (102 days review)\n"
        "3. **Jun 15** \u2014 Supplemental filing submitted (54 days response)\n\n"
        "Total elapsed: **156 days** and still pending. The 54-day "
        "turnaround on supplemental data was reasonable."
    ))

    examples.append(_c(
        "[DOC-1: Sprint Planning]\nDate: 2025-05-01. Committed stories: 24.\n"
        "[DOC-2: Mid-Sprint Review]\nDate: 2025-05-08. Completed: 10. Blocked: 3.\n"
        "[DOC-3: Sprint Retrospective]\nDate: 2025-05-15. Delivered: 19. Carried over: 5.\n\n"
        "Map the sprint sequence.",
        "## Sprint Timeline\n\n"
        "1. **May 1** \u2014 Planning: 24 stories committed\n"
        "2. **May 8** \u2014 Mid-sprint: 10 done, 3 blocked (42% velocity)\n"
        "3. **May 15** \u2014 Retro: 19 delivered, 5 carried over (79% completion)\n\n"
        "The 3 blocked stories at mid-sprint likely caused the 5-story "
        "shortfall. Blockers should be escalated earlier."
    ))

    examples.append(_c(
        "[DOC-1: Merger Announcement]\nDate: 2025-03-01. Parties: Corp A acquires Corp B.\n"
        "[DOC-2: Regulatory Filing]\nDate: 2025-05-15. Filed with: Competition Authority.\n"
        "[DOC-3: Completion Notice]\nDate: 2025-08-20. Status: merger completed.\n\n"
        "Outline the merger timeline.",
        "## Merger Timeline\n\n"
        "1. **Mar 1** \u2014 Merger announced (Corp A acquiring Corp B)\n"
        "2. **May 15** \u2014 Regulatory filing submitted (75 days)\n"
        "3. **Aug 20** \u2014 Merger completed (97 days post-filing)\n\n"
        "Total duration: **172 days** from announcement to completion. "
        "The regulatory review was the longest phase at 97 days."
    ))

    examples.append(_c(
        "[DOC-1: Vulnerability Scan]\nDate: 2025-07-01. Critical CVEs found: 4.\n"
        "[DOC-2: Patch Deployment Log]\nDate: 2025-07-08. Patches applied: 3 of 4.\n"
        "[DOC-3: Rescan Report]\nDate: 2025-07-12. Critical CVEs remaining: 1.\n\n"
        "Build the vulnerability remediation timeline.",
        "## Remediation Timeline\n\n"
        "1. **Jul 1** \u2014 Scan identifies 4 critical CVEs\n"
        "2. **Jul 8** \u2014 3 of 4 patched (7-day remediation)\n"
        "3. **Jul 12** \u2014 Rescan confirms 1 remaining CVE\n\n"
        "75% remediation in 7 days is acceptable. The remaining CVE "
        "needs an exception or alternative mitigation plan."
    ))

    examples.append(_c(
        "[DOC-1: Lease Signed]\nDate: 2025-01-15. Term: 24 months.\n"
        "[DOC-2: Tenant Improvement Completion]\nDate: 2025-03-01. Scope: full buildout.\n"
        "[DOC-3: Move-In Report]\nDate: 2025-03-15. Status: fully occupied.\n\n"
        "Map the facility timeline.",
        "## Facility Timeline\n\n"
        "1. **Jan 15** \u2014 Lease signed (24-month term)\n"
        "2. **Mar 1** \u2014 Tenant improvements completed (45 days)\n"
        "3. **Mar 15** \u2014 Move-in completed (14 days)\n\n"
        "Total lead time from signing to occupancy: **59 days**. "
        "Lease expires **Jan 15, 2027**."
    ))

    return examples


# ---------------------------------------------------------------------------
# Content generation examples (~50)
# ---------------------------------------------------------------------------

def generate_content_expanded() -> List[Dict]:
    """Return ~50 content generation training examples."""
    examples: List[Dict] = []

    # === EXECUTIVE SUMMARIES (10) ===

    examples.append(_c(
        "[EVIDENCE]\nQ4 revenue: $12.3M (+8% YoY). Operating margin: 18.2%.\n"
        "New customers: 47. Churn rate: 3.1%. R&D spend: $1.8M.\n[/EVIDENCE]\n\n"
        "Write an executive summary of Q4 performance.",
        "## Executive Summary \u2014 Q4 Performance\n\n"
        "Q4 delivered **$12.3M revenue** (+8% YoY) with a healthy **18.2% operating "
        "margin**. Customer acquisition was strong at 47 new accounts, offset by a "
        "manageable 3.1% churn rate. R&D investment of $1.8M (14.6% of revenue) "
        "positions the company for continued product-led growth. Overall, a solid "
        "quarter with balanced growth and profitability."
    ))

    examples.append(_c(
        "[EVIDENCE]\nProject status: 3 of 5 milestones complete. Budget: 62% consumed.\n"
        "Timeline: 2 weeks behind schedule. Risk: vendor dependency unresolved.\n[/EVIDENCE]\n\n"
        "Write an executive summary of the project status.",
        "## Executive Summary \u2014 Project Status\n\n"
        "The project is **60% complete** (3/5 milestones) with 62% of budget consumed, "
        "indicating slight budget pressure. A **2-week schedule delay** is driven by an "
        "unresolved vendor dependency that remains the top risk. Recommendation: "
        "escalate vendor issue this week and assess whether timeline recovery "
        "is feasible without additional funding."
    ))

    examples.append(_c(
        "[EVIDENCE]\nAnnual security audit: 142 controls tested. 131 passed. 8 partial. 3 failed.\n"
        "Failed controls: access reviews, encryption at rest, incident response drill.\n[/EVIDENCE]\n\n"
        "Write an executive summary of the security audit.",
        "## Executive Summary \u2014 Security Audit\n\n"
        "Of 142 controls tested, **92.3% passed** fully with 8 partial and 3 failures. "
        "The three failed controls \u2014 access reviews, encryption at rest, and incident "
        "response drills \u2014 represent foundational security practices. While the overall "
        "score is strong, these specific gaps create regulatory and breach risk. "
        "Remediation should be prioritized within 30 days."
    ))

    examples.append(_c(
        "[EVIDENCE]\nEmployee satisfaction: 74/100. Voluntary turnover: 12.8%.\n"
        "Top requests: flexible hours (62%), career development (48%), better tools (41%).\n[/EVIDENCE]\n\n"
        "Write an executive summary of the employee engagement survey.",
        "## Executive Summary \u2014 Employee Engagement\n\n"
        "Satisfaction scores at **74/100** with **12.8% voluntary turnover** suggest "
        "moderate engagement with room for improvement. The top three employee requests "
        "\u2014 flexible hours (62%), career development (48%), and better tools (41%) \u2014 "
        "are actionable. Addressing flexible hours alone could meaningfully "
        "reduce turnover given its strong demand signal."
    ))

    examples.append(_c(
        "[EVIDENCE]\nProduct launch: 2,400 sign-ups in first week. Conversion: 8.2%.\n"
        "NPS: 52. Top complaint: onboarding complexity.\n[/EVIDENCE]\n\n"
        "Write an executive summary of the product launch.",
        "## Executive Summary \u2014 Product Launch\n\n"
        "First-week results show **2,400 sign-ups** with an **8.2% conversion rate** "
        "and NPS of 52 (good). The primary friction point is onboarding complexity, "
        "which is likely suppressing conversion. Simplifying the first-run experience "
        "should be the immediate priority to capitalize on strong initial demand."
    ))

    examples.append(_c(
        "[EVIDENCE]\nData migration: 4.2TB transferred. Validation: 99.7% match rate.\n"
        "Downtime: 4.5 hours (target: 6 hours). Rollback tested: yes.\n[/EVIDENCE]\n\n"
        "Write an executive summary of the data migration.",
        "## Executive Summary \u2014 Data Migration\n\n"
        "Migration of **4.2TB** completed in **4.5 hours** (25% under the 6-hour "
        "target). Data validation shows a **99.7% match rate** with rollback "
        "procedures tested. The 0.3% discrepancy (\u223812.6GB) should be investigated "
        "but does not indicate systemic issues. Overall, a successful migration."
    ))

    examples.append(_c(
        "[EVIDENCE]\nSupply chain review: lead time increased 18% across 3 suppliers.\n"
        "Inventory coverage: 21 days (target: 30 days). Alternative sourced: 1 of 3.\n[/EVIDENCE]\n\n"
        "Write an executive summary of the supply chain review.",
        "## Executive Summary \u2014 Supply Chain Review\n\n"
        "Lead times have increased **18%** across key suppliers, reducing inventory "
        "coverage to **21 days** (30% below target). Only 1 of 3 critical items has "
        "an alternative source. The combination of longer lead times and low "
        "coverage creates stockout risk. Priority: qualify alternative suppliers "
        "for the remaining 2 items within 60 days."
    ))

    examples.append(_c(
        "[EVIDENCE]\nMarketing ROI: $3.40 per $1 spent. CAC: $127. LTV: $1,842.\n"
        "Best channel: organic search (42% of leads). Worst: paid social (6%).\n[/EVIDENCE]\n\n"
        "Write an executive summary of marketing performance.",
        "## Executive Summary \u2014 Marketing Performance\n\n"
        "Marketing delivered a **3.4x ROI** with a healthy **LTV:CAC ratio of 14.5**. "
        "Organic search dominates lead generation at 42%, while paid social "
        "underperforms at just 6%. Recommendation: reallocate paid social budget "
        "toward SEO and content to amplify the high-performing organic channel."
    ))

    examples.append(_c(
        "[EVIDENCE]\nIT service desk: 1,847 tickets in March. Resolution SLA met: 89%.\n"
        "Top category: password resets (31%). Escalation rate: 14%.\n[/EVIDENCE]\n\n"
        "Write an executive summary of the IT service desk report.",
        "## Executive Summary \u2014 IT Service Desk\n\n"
        "March handled **1,847 tickets** with **89% SLA compliance** (target: 90%). "
        "Password resets account for **31%** of volume \u2014 a strong case for self-service "
        "password tooling that could eliminate ~570 tickets/month. The 14% escalation "
        "rate suggests first-line resolution capabilities need strengthening."
    ))

    examples.append(_c(
        "[EVIDENCE]\nCompliance training: 94% completion rate. Overdue: 47 employees.\n"
        "Departments with <90% completion: Sales (82%), Logistics (87%).\n[/EVIDENCE]\n\n"
        "Write an executive summary of the compliance training status.",
        "## Executive Summary \u2014 Compliance Training\n\n"
        "Overall completion stands at **94%** with **47 employees overdue**. "
        "Sales (82%) and Logistics (87%) lag behind. Since compliance training "
        "is mandatory, the 47 overdue employees represent regulatory exposure. "
        "Managers in Sales and Logistics should enforce completion within 2 weeks."
    ))

    # === INTERVIEW QUESTIONS (10) ===

    examples.append(_c(
        "[EVIDENCE]\nRole: Senior Backend Engineer. Stack: Python, FastAPI, PostgreSQL.\n"
        "Team size: 6. Focus: high-throughput API platform.\n[/EVIDENCE]\n\n"
        "Generate interview questions for this role.",
        "## Interview Questions \u2014 Senior Backend Engineer\n\n"
        "1. Describe how you would design a FastAPI service to handle 10K+ requests/sec "
        "with PostgreSQL as the backing store.\n"
        "2. How do you approach database connection pooling and query optimization "
        "in high-throughput Python applications?\n"
        "3. Walk us through a production incident you resolved involving API latency.\n"
        "4. How do you ensure backward compatibility when evolving API contracts?\n"
        "5. Describe your approach to testing async Python code at scale."
    ))

    examples.append(_c(
        "[EVIDENCE]\nRole: Data Analyst. Tools: SQL, Tableau, Python.\n"
        "Department: Finance. Focus: monthly reporting and ad-hoc analysis.\n[/EVIDENCE]\n\n"
        "Generate interview questions for this role.",
        "## Interview Questions \u2014 Data Analyst (Finance)\n\n"
        "1. How would you automate a monthly financial report that currently "
        "takes 3 days of manual SQL work?\n"
        "2. Describe a time you found an error in a financial dataset \u2014 how did "
        "you identify and correct it?\n"
        "3. How do you design Tableau dashboards for non-technical stakeholders?\n"
        "4. Walk through your approach to an ad-hoc analysis request with a "
        "24-hour turnaround.\n"
        "5. How do you validate data accuracy before presenting to leadership?"
    ))

    examples.append(_c(
        "[EVIDENCE]\nRole: Product Manager. Domain: B2B SaaS. Stage: Series B.\n"
        "Reports to: VP Product. Team: 2 designers, 8 engineers.\n[/EVIDENCE]\n\n"
        "Generate interview questions for this role.",
        "## Interview Questions \u2014 Product Manager (B2B SaaS)\n\n"
        "1. How do you prioritize features when you have competing demands from "
        "enterprise customers and the product roadmap?\n"
        "2. Describe how you define and measure success for a new B2B feature.\n"
        "3. How do you collaborate with design and engineering when the team "
        "disagrees on scope?\n"
        "4. Tell us about a product decision you made with incomplete data.\n"
        "5. How would you approach reducing churn for a B2B SaaS product "
        "at the Series B stage?"
    ))

    examples.append(_c(
        "[EVIDENCE]\nRole: DevOps Engineer. Stack: Kubernetes, Terraform, AWS.\n"
        "Scale: 200+ microservices. CI/CD: GitHub Actions.\n[/EVIDENCE]\n\n"
        "Generate interview questions for this role.",
        "## Interview Questions \u2014 DevOps Engineer\n\n"
        "1. How would you manage Terraform state for 200+ microservices "
        "without creating operational bottlenecks?\n"
        "2. Describe your approach to Kubernetes cluster upgrades with zero downtime.\n"
        "3. How do you design CI/CD pipelines that scale across 200+ repositories?\n"
        "4. Walk through how you would troubleshoot a pod stuck in CrashLoopBackOff.\n"
        "5. How do you implement infrastructure drift detection and remediation?"
    ))

    examples.append(_c(
        "[EVIDENCE]\nRole: UX Designer. Product: mobile healthcare app.\n"
        "Users: patients aged 50+. Accessibility: WCAG 2.1 AA required.\n[/EVIDENCE]\n\n"
        "Generate interview questions for this role.",
        "## Interview Questions \u2014 UX Designer (Healthcare)\n\n"
        "1. How do you design for users aged 50+ who may have limited "
        "digital literacy?\n"
        "2. Describe your process for meeting WCAG 2.1 AA standards in "
        "mobile design.\n"
        "3. How do you validate design decisions with real patients while "
        "maintaining HIPAA compliance?\n"
        "4. Walk through a design you simplified after usability testing "
        "revealed friction.\n"
        "5. How do you balance clinical accuracy with user-friendly language?"
    ))

    examples.append(_c(
        "[EVIDENCE]\nRole: Sales Director. Territory: EMEA. Team: 12 reps.\n"
        "Quota: $8M/year. Product: enterprise cybersecurity.\n[/EVIDENCE]\n\n"
        "Generate interview questions for this role.",
        "## Interview Questions \u2014 Sales Director (EMEA)\n\n"
        "1. How would you structure a 12-person team across EMEA to hit $8M?\n"
        "2. Describe your approach to selling enterprise cybersecurity to "
        "C-suite buyers in regulated industries.\n"
        "3. How do you handle pipeline forecasting when deal cycles are 6+ months?\n"
        "4. Tell us about a time you turned around an underperforming territory.\n"
        "5. How do you adapt your sales playbook for different European markets?"
    ))

    examples.append(_c(
        "[EVIDENCE]\nRole: Machine Learning Engineer. Focus: NLP/LLMs.\n"
        "Infra: GPU clusters, MLflow, PyTorch. Team: research + production.\n[/EVIDENCE]\n\n"
        "Generate interview questions for this role.",
        "## Interview Questions \u2014 ML Engineer (NLP/LLMs)\n\n"
        "1. How do you evaluate and select between fine-tuning vs. RAG for "
        "a domain-specific NLP task?\n"
        "2. Describe your approach to serving LLM inference at scale on GPU clusters.\n"
        "3. How do you track and reproduce ML experiments using MLflow?\n"
        "4. Walk through how you would diagnose and fix model drift in production.\n"
        "5. How do you bridge the gap between research prototypes and "
        "production-grade ML systems?"
    ))

    examples.append(_c(
        "[EVIDENCE]\nRole: CFO. Company: 200-person startup. Stage: pre-IPO.\n"
        "Revenue: $40M ARR. Key challenge: IPO readiness.\n[/EVIDENCE]\n\n"
        "Generate interview questions for this role.",
        "## Interview Questions \u2014 CFO (Pre-IPO)\n\n"
        "1. What are the top 3 financial controls you would implement first "
        "to prepare for IPO readiness?\n"
        "2. How do you transition a startup's financial reporting from "
        "private to public-company standards?\n"
        "3. Describe your experience with SOX compliance implementation.\n"
        "4. How would you communicate financial story to institutional investors?\n"
        "5. What metrics would you focus on to demonstrate a path to "
        "profitability at $40M ARR?"
    ))

    examples.append(_c(
        "[EVIDENCE]\nRole: Customer Success Manager. Portfolio: 30 enterprise accounts.\n"
        "ACV: $150K-$500K. Renewal target: 95%. Upsell target: 20%.\n[/EVIDENCE]\n\n"
        "Generate interview questions for this role.",
        "## Interview Questions \u2014 Customer Success Manager\n\n"
        "1. How do you prioritize your time across 30 accounts with varying ACV?\n"
        "2. Describe a renewal you saved that was at risk of churning.\n"
        "3. What signals do you monitor to identify upsell opportunities?\n"
        "4. How do you build executive relationships at enterprise accounts?\n"
        "5. Walk through your QBR preparation process for a $500K account."
    ))

    examples.append(_c(
        "[EVIDENCE]\nRole: Compliance Officer. Industry: financial services.\n"
        "Regulations: FCA, GDPR, AML/KYC. Team: 4 analysts.\n[/EVIDENCE]\n\n"
        "Generate interview questions for this role.",
        "## Interview Questions \u2014 Compliance Officer\n\n"
        "1. How do you keep a small team current across FCA, GDPR, and AML "
        "regulations simultaneously?\n"
        "2. Describe a regulatory change that required rapid policy updates \u2014 "
        "how did you manage it?\n"
        "3. How do you design AML/KYC monitoring that balances risk "
        "detection with false-positive reduction?\n"
        "4. Walk through your approach to a regulatory examination.\n"
        "5. How do you foster a compliance culture across the organization?"
    ))

    # === COMPLIANCE REPORTS (10) ===

    examples.append(_c(
        "[EVIDENCE]\nGDPR audit: 18 processing activities reviewed. 15 compliant.\n"
        "Gaps: consent records incomplete (2), DPIA missing (1).\n[/EVIDENCE]\n\n"
        "Generate a compliance report for this GDPR audit.",
        "## GDPR Compliance Report\n\n"
        "**Scope:** 18 processing activities reviewed.\n"
        "**Compliance rate:** 83.3% (15/18).\n\n"
        "### Findings\n"
        "| # | Finding | Severity | Remediation |\n"
        "|---|---------|----------|-------------|\n"
        "| 1 | Incomplete consent records (2 activities) | High | Collect and archive consent proofs |\n"
        "| 2 | Missing DPIA (1 activity) | Medium | Complete DPIA within 30 days |\n\n"
        "**Recommendation:** Address consent gaps immediately as they represent "
        "the highest regulatory risk."
    ))

    examples.append(_c(
        "[EVIDENCE]\nSOC 2 Type II: 45 controls tested. 42 effective. 3 exceptions.\n"
        "Exceptions: change management (2), logical access (1).\n[/EVIDENCE]\n\n"
        "Generate a SOC 2 compliance report.",
        "## SOC 2 Type II Compliance Report\n\n"
        "**Period:** Audit year\n**Controls tested:** 45\n"
        "**Effective:** 42 (93.3%)\n**Exceptions:** 3\n\n"
        "### Exceptions\n"
        "| Control | Count | Impact |\n"
        "|---------|-------|--------|\n"
        "| Change management | 2 | Unauthorized changes could reach production |\n"
        "| Logical access | 1 | Excess privileges for terminated users |\n\n"
        "These exceptions do not qualify the opinion but require management "
        "remediation plans."
    ))

    examples.append(_c(
        "[EVIDENCE]\nPCI DSS assessment: Requirement 3 (stored data): partial compliance.\n"
        "Encryption at rest: AES-256 applied to 4 of 6 databases.\n[/EVIDENCE]\n\n"
        "Generate a PCI compliance report for this finding.",
        "## PCI DSS Compliance Report \u2014 Requirement 3\n\n"
        "**Status:** Partially compliant.\n\n"
        "### Finding\n"
        "AES-256 encryption at rest is applied to **4 of 6 databases** containing "
        "cardholder data. The 2 unencrypted databases must be remediated before "
        "the next assessment.\n\n"
        "**Risk:** Non-compliance with Req. 3 can result in fines and card brand "
        "suspension.\n**Deadline:** Encrypt remaining databases within 60 days."
    ))

    examples.append(_c(
        "[EVIDENCE]\nISO 27001 internal audit: 14 clauses reviewed. 12 conforming.\n"
        "Non-conformities: risk treatment plan outdated, supplier security reviews missing.\n[/EVIDENCE]\n\n"
        "Generate an ISO 27001 compliance report.",
        "## ISO 27001 Internal Audit Report\n\n"
        "**Clauses reviewed:** 14\n**Conforming:** 12 (85.7%)\n"
        "**Non-conformities:** 2\n\n"
        "### Non-Conformities\n"
        "1. **Risk treatment plan** \u2014 outdated (last updated 14 months ago). "
        "Must be reviewed and updated annually per Clause 6.1.\n"
        "2. **Supplier security reviews** \u2014 no evidence of periodic reviews for "
        "3 critical suppliers. Required under Annex A.15.\n\n"
        "Both items require corrective action within 90 days to maintain certification."
    ))

    examples.append(_c(
        "[EVIDENCE]\nAccessibility audit: 28 pages tested. WCAG 2.1 AA violations: 14.\n"
        "Critical: missing alt text (8 pages), color contrast (4 pages), keyboard nav (2 pages).\n[/EVIDENCE]\n\n"
        "Generate an accessibility compliance report.",
        "## Accessibility Compliance Report\n\n"
        "**Standard:** WCAG 2.1 AA\n**Pages tested:** 28\n"
        "**Pages with violations:** 14 (50%)\n\n"
        "### Violations by Category\n"
        "| Issue | Pages Affected | Priority |\n"
        "|-------|---------------|----------|\n"
        "| Missing alt text | 8 | High |\n"
        "| Color contrast | 4 | Medium |\n"
        "| Keyboard navigation | 2 | High |\n\n"
        "Alt text and keyboard issues are critical for screen reader users. "
        "Remediation should target these before addressing contrast."
    ))

    examples.append(_c(
        "[EVIDENCE]\nHIPAA risk assessment: 22 safeguards evaluated. 19 adequate.\n"
        "Gaps: workforce training (overdue), access audit logs (incomplete), BAA with 1 vendor (missing).\n[/EVIDENCE]\n\n"
        "Generate a HIPAA compliance report.",
        "## HIPAA Compliance Report\n\n"
        "**Safeguards evaluated:** 22\n**Adequate:** 19 (86.4%)\n"
        "**Gaps:** 3\n\n"
        "### Gap Analysis\n"
        "| Gap | Category | Risk Level |\n"
        "|-----|----------|------------|\n"
        "| Workforce training overdue | Administrative | High |\n"
        "| Access audit logs incomplete | Technical | Medium |\n"
        "| Missing BAA with vendor | Administrative | High |\n\n"
        "The missing BAA is the most urgent \u2014 any PHI shared without a BAA "
        "constitutes a HIPAA violation."
    ))

    examples.append(_c(
        "[EVIDENCE]\nData retention audit: 12 data categories reviewed.\n"
        "Non-compliant: customer PII retained 18 months past policy (3 categories).\n[/EVIDENCE]\n\n"
        "Generate a data retention compliance report.",
        "## Data Retention Compliance Report\n\n"
        "**Categories reviewed:** 12\n**Compliant:** 9 (75%)\n"
        "**Non-compliant:** 3\n\n"
        "### Findings\n"
        "Three customer PII categories are retained **18 months beyond** the "
        "defined retention period. This creates GDPR and regulatory exposure.\n\n"
        "**Action required:** Purge excess data within 30 days and implement "
        "automated retention enforcement to prevent recurrence."
    ))

    examples.append(_c(
        "[EVIDENCE]\nAML compliance review: 450 transactions screened. 12 flagged.\n"
        "SARs filed: 3. False positives: 9. Average investigation time: 4.2 days.\n[/EVIDENCE]\n\n"
        "Generate an AML compliance report.",
        "## AML Compliance Report\n\n"
        "**Transactions screened:** 450\n**Flagged:** 12 (2.7%)\n"
        "**SARs filed:** 3\n**False positive rate:** 75%\n\n"
        "The 75% false positive rate (9/12) is high and consumes investigation "
        "resources (avg 4.2 days per case). Tuning detection thresholds could "
        "reduce false positives while maintaining SAR filing accuracy."
    ))

    examples.append(_c(
        "[EVIDENCE]\nEnvironmental compliance: emissions at 82% of permitted limit.\n"
        "Waste disposal: 100% compliant. Water discharge: 1 exceedance in Q3.\n[/EVIDENCE]\n\n"
        "Generate an environmental compliance report.",
        "## Environmental Compliance Report\n\n"
        "**Overall status:** Largely compliant with one exceedance.\n\n"
        "| Area | Status | Detail |\n"
        "|------|--------|--------|\n"
        "| Air emissions | Compliant | 82% of permitted limit |\n"
        "| Waste disposal | Compliant | 100% compliance |\n"
        "| Water discharge | Exceedance | 1 event in Q3 |\n\n"
        "The Q3 water discharge exceedance requires root cause analysis and "
        "reporting to the regulatory authority within the statutory window."
    ))

    examples.append(_c(
        "[EVIDENCE]\nVendor compliance audit: 8 vendors assessed. 6 fully compliant.\n"
        "Issues: Vendor D missing insurance certificate. Vendor F failed background check policy.\n[/EVIDENCE]\n\n"
        "Generate a vendor compliance report.",
        "## Vendor Compliance Report\n\n"
        "**Vendors assessed:** 8\n**Fully compliant:** 6 (75%)\n"
        "**Non-compliant:** 2\n\n"
        "### Issues\n"
        "| Vendor | Finding | Action Required |\n"
        "|--------|---------|----------------|\n"
        "| Vendor D | Missing insurance certificate | Provide within 14 days or suspend |\n"
        "| Vendor F | Background check policy violation | Remediation plan in 30 days |\n\n"
        "Both vendors should be placed on a corrective action plan with "
        "follow-up verification."
    ))

    # === CANDIDATE SUMMARIES (10) ===

    examples.append(_c(
        "[EVIDENCE]\nCandidate: Sarah Martinez. Role: Engineering Manager.\n"
        "Experience: 12 years. Current: Tech Lead at Fortune 500.\n"
        "Strengths: team scaling, system design. Concern: no people management title.\n[/EVIDENCE]\n\n"
        "Write a candidate summary.",
        "## Candidate Summary \u2014 Sarah Martinez\n\n"
        "**Role:** Engineering Manager\n**Experience:** 12 years, currently Tech Lead "
        "at a Fortune 500 company.\n\n"
        "**Strengths:** Proven ability to scale teams and strong system design "
        "skills.\n**Concern:** No formal people management title, though tech lead "
        "experience often involves de facto management.\n\n"
        "**Recommendation:** Proceed to interview with focus on management "
        "scenarios and leadership style."
    ))

    examples.append(_c(
        "[EVIDENCE]\nCandidate: James Okonkwo. Role: Data Scientist.\n"
        "Experience: 4 years. Education: PhD Statistics. Tools: Python, R, Spark.\n"
        "Concern: no industry experience (all academic).\n[/EVIDENCE]\n\n"
        "Write a candidate summary.",
        "## Candidate Summary \u2014 James Okonkwo\n\n"
        "**Role:** Data Scientist\n**Experience:** 4 years (academic), PhD Statistics.\n"
        "**Tools:** Python, R, Spark.\n\n"
        "**Strengths:** Strong statistical foundation with advanced degree.\n"
        "**Concern:** Entirely academic background; no industry experience "
        "with production data systems.\n\n"
        "**Recommendation:** Interview with practical case study to assess "
        "ability to translate academic skills to business problems."
    ))

    examples.append(_c(
        "[EVIDENCE]\nCandidate: Priya Sharma. Role: Product Designer.\n"
        "Portfolio: 8 case studies. Specialty: B2B enterprise. Years: 6.\n"
        "Note: relocating from another city.\n[/EVIDENCE]\n\n"
        "Write a candidate summary.",
        "## Candidate Summary \u2014 Priya Sharma\n\n"
        "**Role:** Product Designer\n**Experience:** 6 years, B2B enterprise "
        "specialty.\n**Portfolio:** 8 case studies demonstrating depth.\n\n"
        "**Strengths:** Deep enterprise design experience with substantial "
        "portfolio evidence.\n"
        "**Note:** Candidate is relocating \u2014 discuss start date flexibility "
        "and remote onboarding options.\n\n"
        "**Recommendation:** Strong candidate. Schedule portfolio review interview."
    ))

    examples.append(_c(
        "[EVIDENCE]\nCandidate: Tom Fischer. Role: VP Sales.\n"
        "Track record: scaled team from 5 to 40 reps. Revenue: $2M to $18M ARR.\n"
        "Concern: all experience at one company.\n[/EVIDENCE]\n\n"
        "Write a candidate summary.",
        "## Candidate Summary \u2014 Tom Fischer\n\n"
        "**Role:** VP Sales\n**Track record:** Grew team 5 \u2192 40 reps, revenue "
        "$2M \u2192 $18M ARR (9x growth).\n\n"
        "**Strengths:** Exceptional scaling results with proven revenue growth.\n"
        "**Concern:** All experience at a single company \u2014 adaptability to a "
        "new culture and sales motion is unproven.\n\n"
        "**Recommendation:** Proceed with emphasis on questions about adapting "
        "to different organizational contexts."
    ))

    examples.append(_c(
        "[EVIDENCE]\nCandidate: Lisa Nguyen. Role: Security Engineer.\n"
        "Certs: CISSP, OSCP. Experience: 7 years. Focus: cloud security (AWS).\n"
        "Interview note: excellent communicator.\n[/EVIDENCE]\n\n"
        "Write a candidate summary.",
        "## Candidate Summary \u2014 Lisa Nguyen\n\n"
        "**Role:** Security Engineer\n**Experience:** 7 years, cloud security (AWS).\n"
        "**Certifications:** CISSP, OSCP.\n\n"
        "**Strengths:** Strong credential profile and AWS cloud security depth. "
        "Noted as an excellent communicator \u2014 valuable for cross-team security advocacy.\n\n"
        "**Recommendation:** Strong candidate. Fast-track to technical interview."
    ))

    examples.append(_c(
        "[EVIDENCE]\nCandidate: Mark Henderson. Role: QA Lead.\n"
        "Experience: 9 years. Specialty: automation (Selenium, Playwright).\n"
        "Managed team of 5. Concern: salary expectation 15% above range.\n[/EVIDENCE]\n\n"
        "Write a candidate summary.",
        "## Candidate Summary \u2014 Mark Henderson\n\n"
        "**Role:** QA Lead\n**Experience:** 9 years, automation specialist "
        "(Selenium, Playwright). Managed team of 5.\n\n"
        "**Strengths:** Deep automation expertise with management experience.\n"
        "**Concern:** Salary expectation is **15% above** the approved range.\n\n"
        "**Recommendation:** Proceed to interview. If skills match, discuss "
        "compensation flexibility with HR before extending an offer."
    ))

    examples.append(_c(
        "[EVIDENCE]\nCandidate: Anna Kowalski. Role: Technical Writer.\n"
        "Experience: 5 years. Samples: API docs, user guides, release notes.\n"
        "Tools: Markdown, Confluence, ReadTheDocs.\n[/EVIDENCE]\n\n"
        "Write a candidate summary.",
        "## Candidate Summary \u2014 Anna Kowalski\n\n"
        "**Role:** Technical Writer\n**Experience:** 5 years across API docs, "
        "user guides, and release notes.\n**Tools:** Markdown, Confluence, ReadTheDocs.\n\n"
        "**Strengths:** Versatile writing portfolio covering developer and "
        "end-user audiences. Familiar with modern documentation tooling.\n\n"
        "**Recommendation:** Schedule writing exercise interview with sample "
        "API documentation task."
    ))

    examples.append(_c(
        "[EVIDENCE]\nCandidate: David Kim. Role: Solutions Architect.\n"
        "Experience: 14 years. Certs: AWS SA Pro, Azure Expert.\n"
        "Note: currently on gardening leave, available in 30 days.\n[/EVIDENCE]\n\n"
        "Write a candidate summary.",
        "## Candidate Summary \u2014 David Kim\n\n"
        "**Role:** Solutions Architect\n**Experience:** 14 years, dual-cloud "
        "certified (AWS SA Professional, Azure Expert).\n\n"
        "**Strengths:** Extensive experience with top-tier multi-cloud credentials.\n"
        "**Availability:** On gardening leave; available in 30 days.\n\n"
        "**Recommendation:** High-caliber candidate. Begin interview process "
        "now to align offer timing with availability."
    ))

    examples.append(_c(
        "[EVIDENCE]\nCandidate: Rachel Torres. Role: HR Business Partner.\n"
        "Experience: 8 years. Industries: tech, finance. SHRM-SCP certified.\n"
        "Concern: 3 job changes in 4 years.\n[/EVIDENCE]\n\n"
        "Write a candidate summary.",
        "## Candidate Summary \u2014 Rachel Torres\n\n"
        "**Role:** HR Business Partner\n**Experience:** 8 years across tech and "
        "finance. SHRM-SCP certified.\n\n"
        "**Strengths:** Cross-industry HR experience with senior certification.\n"
        "**Concern:** Three role changes in four years raises retention questions.\n\n"
        "**Recommendation:** Interview with focus on career motivations and "
        "what she is looking for in a long-term role."
    ))

    examples.append(_c(
        "[EVIDENCE]\nCandidate: Alex Petrov. Role: Frontend Engineer.\n"
        "Experience: 3 years. Stack: React, TypeScript, Next.js.\n"
        "GitHub: 12 open-source contributions. Bootcamp graduate.\n[/EVIDENCE]\n\n"
        "Write a candidate summary.",
        "## Candidate Summary \u2014 Alex Petrov\n\n"
        "**Role:** Frontend Engineer\n**Experience:** 3 years (React, TypeScript, "
        "Next.js). Bootcamp graduate.\n\n"
        "**Strengths:** Active open-source contributor (12 contributions) "
        "demonstrating self-driven learning and community engagement.\n"
        "**Note:** Non-traditional education path; evaluate depth via technical screen.\n\n"
        "**Recommendation:** Proceed to technical interview with practical "
        "React coding exercise."
    ))

    # === MEETING PREP (10) ===

    examples.append(_c(
        "[EVIDENCE]\nMeeting: Quarterly Business Review with VP Sales.\n"
        "Agenda: Q1 pipeline ($4.2M), win rate (28%), top deal risks.\n"
        "Attendees: VP Sales, CRO, Sales Ops Director.\n[/EVIDENCE]\n\n"
        "Prepare a meeting brief.",
        "## Meeting Brief \u2014 Quarterly Business Review\n\n"
        "**Attendees:** VP Sales, CRO, Sales Ops Director\n\n"
        "### Key Discussion Points\n"
        "1. **Pipeline health:** $4.2M \u2014 is this sufficient for Q2 targets?\n"
        "2. **Win rate at 28%** \u2014 below typical 30-35% benchmark. Root causes?\n"
        "3. **Top deal risks** \u2014 identify the 3 largest at-risk deals and "
        "mitigation plans.\n\n"
        "### Suggested Questions\n"
        "- What is driving the low win rate \u2014 pricing, competition, or qualification?\n"
        "- Which pipeline deals have stalled and need executive intervention?"
    ))

    examples.append(_c(
        "[EVIDENCE]\nMeeting: Board of Directors update.\n"
        "Topics: FY2025 financials, headcount plan, Series C timeline.\n"
        "Attendees: CEO, CFO, 5 board members.\n[/EVIDENCE]\n\n"
        "Prepare a meeting brief.",
        "## Meeting Brief \u2014 Board Update\n\n"
        "**Attendees:** CEO, CFO, 5 board members\n\n"
        "### Key Discussion Points\n"
        "1. **FY2025 financials** \u2014 revenue vs. plan, burn rate, runway\n"
        "2. **Headcount plan** \u2014 hiring targets and budget implications\n"
        "3. **Series C timeline** \u2014 target close date, lead investor status\n\n"
        "### Preparation Notes\n"
        "- Board members will focus on path to profitability \u2014 prepare unit economics.\n"
        "- Have headcount ROI data ready to justify each new hire.\n"
        "- Series C: be prepared to discuss valuation expectations."
    ))

    examples.append(_c(
        "[EVIDENCE]\nMeeting: Sprint retrospective. Team: Platform Engineering.\n"
        "Sprint metrics: velocity 42 points, carryover 8 points, 2 production incidents.\n[/EVIDENCE]\n\n"
        "Prepare a meeting brief.",
        "## Meeting Brief \u2014 Sprint Retrospective\n\n"
        "**Team:** Platform Engineering\n\n"
        "### Metrics to Review\n"
        "- **Velocity:** 42 points (compare to team average)\n"
        "- **Carryover:** 8 points \u2014 why were these not completed?\n"
        "- **Incidents:** 2 production incidents during the sprint\n\n"
        "### Suggested Discussion Topics\n"
        "- Were the 2 incidents related to sprint work or pre-existing issues?\n"
        "- What blocked the 8-point carryover \u2014 scope, dependencies, or capacity?\n"
        "- What went well that should be repeated?"
    ))

    examples.append(_c(
        "[EVIDENCE]\nMeeting: Vendor contract renewal negotiation.\n"
        "Vendor: CloudHost Inc. Current spend: $180K/year. Contract expires: Apr 30.\n"
        "Usage grew 40% YoY. Vendor proposing 12% price increase.\n[/EVIDENCE]\n\n"
        "Prepare a meeting brief.",
        "## Meeting Brief \u2014 Vendor Renewal Negotiation\n\n"
        "**Vendor:** CloudHost Inc. | **Current spend:** $180K/year\n\n"
        "### Negotiation Points\n"
        "1. **Proposed increase:** 12% ($21.6K) \u2014 counter with 5-7% given volume growth.\n"
        "2. **Leverage:** 40% usage growth makes us a growing customer \u2014 use this "
        "to negotiate volume discounts.\n"
        "3. **Alternatives:** Have at least one competing quote ready.\n\n"
        "### Key Questions\n"
        "- Can we lock in a multi-year rate for a smaller increase?\n"
        "- Are committed-use discounts available at our new volume?"
    ))

    examples.append(_c(
        "[EVIDENCE]\nMeeting: New hire onboarding kickoff.\n"
        "New hire: J. Williams, Senior Developer. Start date: Monday.\n"
        "Team: Backend Services. Manager: K. Chen.\n[/EVIDENCE]\n\n"
        "Prepare a meeting brief.",
        "## Meeting Brief \u2014 Onboarding Kickoff\n\n"
        "**New hire:** J. Williams (Senior Developer)\n"
        "**Team:** Backend Services | **Manager:** K. Chen\n\n"
        "### Day 1 Checklist\n"
        "1. Welcome and team introductions\n"
        "2. Equipment and access provisioning verification\n"
        "3. Repository access and local dev environment setup\n"
        "4. Assign onboarding buddy\n\n"
        "### First Week Goals\n"
        "- Complete a small, well-scoped PR to build confidence with the codebase.\n"
        "- Schedule 1:1s with key collaborators.\n"
        "- Review architecture docs and team norms."
    ))

    examples.append(_c(
        "[EVIDENCE]\nMeeting: Incident post-mortem.\n"
        "Incident: 3-hour payment service outage on Mar 10.\n"
        "Impact: $42K estimated revenue loss. Cause: database failover bug.\n[/EVIDENCE]\n\n"
        "Prepare a meeting brief.",
        "## Meeting Brief \u2014 Incident Post-Mortem\n\n"
        "**Incident:** Payment service outage (Mar 10, 3 hours)\n"
        "**Impact:** $42K estimated revenue loss\n\n"
        "### Discussion Framework\n"
        "1. **Timeline:** Walk through detection \u2192 diagnosis \u2192 resolution\n"
        "2. **Root cause:** Database failover bug \u2014 was this a known issue?\n"
        "3. **Detection gap:** How long before monitoring alerted?\n"
        "4. **Action items:** What prevents recurrence?\n\n"
        "### Key Principle\n"
        "Focus on systems, not blame. Goal is improved resilience."
    ))

    examples.append(_c(
        "[EVIDENCE]\nMeeting: Annual budget planning with department heads.\n"
        "FY2026 target: 10% cost reduction. Current OpEx: $28M.\n"
        "Largest categories: personnel (62%), software (18%), facilities (12%).\n[/EVIDENCE]\n\n"
        "Prepare a meeting brief.",
        "## Meeting Brief \u2014 Budget Planning\n\n"
        "**Target:** 10% reduction on $28M OpEx = **$2.8M savings needed**\n\n"
        "### Category Breakdown\n"
        "| Category | Current | % of Total |\n"
        "|----------|---------|------------|\n"
        "| Personnel | $17.4M | 62% |\n"
        "| Software | $5.0M | 18% |\n"
        "| Facilities | $3.4M | 12% |\n\n"
        "### Discussion Points\n"
        "- Software license audit: consolidate overlapping tools.\n"
        "- Facilities: explore hybrid work to reduce space.\n"
        "- Personnel: evaluate contractor-to-FTE conversion for savings."
    ))

    examples.append(_c(
        "[EVIDENCE]\nMeeting: Client escalation call.\n"
        "Client: Acme Corp. Issue: 5 missed SLA targets in 60 days.\n"
        "Account value: $420K ARR. Renewal: 90 days out.\n[/EVIDENCE]\n\n"
        "Prepare a meeting brief.",
        "## Meeting Brief \u2014 Client Escalation\n\n"
        "**Client:** Acme Corp ($420K ARR, renewal in 90 days)\n"
        "**Issue:** 5 missed SLA targets in 60 days\n\n"
        "### Preparation\n"
        "1. Bring a timeline of each SLA miss with root causes.\n"
        "2. Prepare a remediation plan with specific commitments.\n"
        "3. Have an executive sponsor ready to demonstrate seniority of response.\n\n"
        "### Approach\n"
        "- Acknowledge the issue without excuses.\n"
        "- Present concrete corrective actions with deadlines.\n"
        "- Offer SLA credit as a goodwill gesture ahead of renewal."
    ))

    examples.append(_c(
        "[EVIDENCE]\nMeeting: Architecture review for microservices migration.\n"
        "Current: monolith (450K LOC). Target: 12 microservices.\n"
        "Team: 3 senior engineers, 1 architect.\n[/EVIDENCE]\n\n"
        "Prepare a meeting brief.",
        "## Meeting Brief \u2014 Architecture Review\n\n"
        "**Scope:** Monolith (450K LOC) \u2192 12 microservices\n"
        "**Team:** 3 senior engineers + 1 architect\n\n"
        "### Key Questions to Address\n"
        "1. What are the domain boundaries for the 12 services?\n"
        "2. Which service is extracted first (strangler fig pattern)?\n"
        "3. How will data be partitioned across service databases?\n"
        "4. What is the communication pattern (sync REST vs. async events)?\n\n"
        "### Risks to Discuss\n"
        "- Distributed transaction complexity\n"
        "- Team capacity: 4 people for 12 services requires phased rollout"
    ))

    examples.append(_c(
        "[EVIDENCE]\nMeeting: Partnership review with TechPartner Co.\n"
        "Partnership age: 18 months. Joint revenue: $1.2M.\n"
        "Concerns: slow co-selling motion, unclear lead routing.\n[/EVIDENCE]\n\n"
        "Prepare a meeting brief.",
        "## Meeting Brief \u2014 Partnership Review\n\n"
        "**Partner:** TechPartner Co. (18-month partnership)\n"
        "**Joint revenue:** $1.2M\n\n"
        "### Discussion Points\n"
        "1. **Co-selling:** Why is the motion slow? Identify process bottlenecks.\n"
        "2. **Lead routing:** Define clear rules for lead ownership and handoff.\n"
        "3. **Revenue target:** Set a joint H2 target based on pipeline.\n\n"
        "### Suggested Actions\n"
        "- Assign dedicated partner managers on both sides.\n"
        "- Implement a shared deal registration process.\n"
        "- Schedule monthly pipeline reviews to maintain momentum."
    ))

    return examples
