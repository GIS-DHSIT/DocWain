"""Domain-specific and intent understanding expansion."""
from __future__ import annotations
from typing import Any, Dict, List

SYSTEM_PROMPT = (
    "You are DocWain — Document Wise AI Node — an intelligent document intelligence "
    "platform engineered by DHS IT Solutions Ltd. You process enterprise documents and "
    "provide grounded, evidence-backed answers. You never hallucinate. Every claim "
    "traces to uploaded documents. You work within profile-isolated scopes and never "
    "leak data across users. You analyze documents through metadata reasoning, "
    "structural understanding, entity recognition, and cross-document intelligence."
)

def _c(user: str, assistant: str) -> Dict[str, Any]:
    return {"messages": [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]}


# ---------------------------------------------------------------------------
# Domain-specific examples (~80 total)
# ---------------------------------------------------------------------------

def _generate_hr() -> List[Dict]:
    """HR domain examples (15)."""
    return [
        _c("[EVIDENCE]\nEmployee: Sarah Chen | ID: EMP-4421\nDepartment: Engineering | Title: Senior Software Engineer\nStart Date: 2022-03-14 | Status: Active\nBase Salary: $142,000 | Bonus Target: 15%\n[/EVIDENCE]\n\nWhat is Sarah Chen's compensation package?",
           "## Compensation — Sarah Chen (EMP-4421)\n\n- **Base Salary:** $142,000\n- **Bonus Target:** 15% ($21,300 at target)\n- **Department:** Engineering\n- **Title:** Senior Software Engineer\n- **Employment Start:** 2022-03-14\n- **Status:** Active"),

        _c("[EVIDENCE]\nPolicy: Annual Leave Entitlement\nFull-time employees: 25 days per annum\nPart-time: pro-rata based on contracted hours\nCarry-over: max 5 days into next fiscal year\nBuy/sell: up to 3 days at daily rate\n[/EVIDENCE]\n\nHow many holiday days can I carry over?",
           "## Annual Leave Carry-Over\n\nPer the Annual Leave Entitlement policy:\n\n- **Maximum carry-over:** 5 days into the next fiscal year\n- Full-time entitlement is **25 days** per annum\n- You may also **buy or sell** up to **3 days** at your daily rate\n\nPart-time employees receive entitlement on a pro-rata basis."),

        _c("[EVIDENCE]\nGrievance Procedure — Stage 1: Informal resolution with line manager within 5 working days.\nStage 2: Formal written grievance to HR within 10 working days.\nStage 3: Appeal panel hearing within 15 working days of Stage 2 outcome.\n[/EVIDENCE]\n\nWhat is the grievance process?",
           "## Grievance Procedure\n\n| Stage | Action | Timeframe |\n|-------|--------|-----------|\n| **Stage 1** | Informal resolution with line manager | 5 working days |\n| **Stage 2** | Formal written grievance submitted to HR | 10 working days |\n| **Stage 3** | Appeal panel hearing | 15 working days after Stage 2 outcome |\n\nThe process is designed to escalate progressively from informal to formal resolution."),

        _c("[EVIDENCE]\nProbation Review — Employee: James Okafor\nProbation Period: 6 months (ends 2025-09-01)\nManager Assessment: Exceeds expectations\nRecommendation: Confirm permanent employment\nAreas of strength: Technical delivery, team collaboration\n[/EVIDENCE]\n\nWhat is the outcome of James Okafor's probation review?",
           "## Probation Review — James Okafor\n\n- **Probation Period:** 6 months (ending 2025-09-01)\n- **Manager Assessment:** Exceeds expectations\n- **Recommendation:** Confirm permanent employment\n- **Strengths:** Technical delivery, team collaboration\n\nThe review recommends **permanent confirmation** based on strong performance."),

        _c("[EVIDENCE]\nExpense Policy v3.2\nMeals: up to $75/day domestic, $100/day international\nMileage: $0.67/mile (IRS 2025 rate)\nHotel: pre-approved cap of $250/night domestic\nReceipts required for all claims over $25\n[/EVIDENCE]\n\nWhat is the meal allowance for business travel?",
           "## Business Travel Meal Allowance\n\nPer Expense Policy v3.2:\n\n| Travel Type | Daily Meal Limit |\n|-------------|------------------|\n| **Domestic** | $75/day |\n| **International** | $100/day |\n\nReceipts are required for all claims over **$25**. Hotel stays are capped at **$250/night** domestic (pre-approval required)."),

        _c("[EVIDENCE]\nOffer Letter — Candidate: Priya Sharma\nPosition: Data Analyst | Department: Business Intelligence\nStart Date: 2025-11-01 | Salary: $95,000\nReporting to: Director of Analytics\nBenefits: Medical, dental, vision, 401(k) match 4%\n[/EVIDENCE]\n\nSummarise Priya Sharma's offer letter.",
           "## Offer Summary — Priya Sharma\n\n- **Position:** Data Analyst\n- **Department:** Business Intelligence\n- **Start Date:** 2025-11-01\n- **Annual Salary:** $95,000\n- **Reports To:** Director of Analytics\n\n## Benefits\n\n- Medical, dental, and vision coverage\n- 401(k) with **4% employer match**"),

        _c("[EVIDENCE]\nPerformance Review — Q3 2025\nEmployee: Marcus Williams | Rating: 3.8/5.0\nGoals Met: 4 of 5 | Stretch Goal: Completed\nDevelopment Areas: Public speaking, stakeholder management\nManager Notes: Strong technical contributor, needs visibility improvement\n[/EVIDENCE]\n\nWhat areas does Marcus Williams need to develop?",
           "## Development Areas — Marcus Williams (Q3 2025)\n\nPer the performance review (overall rating **3.8/5.0**):\n\n- **Public speaking** — identified as a growth area\n- **Stakeholder management** — needs improvement\n- Manager notes that Marcus is a strong technical contributor but should work on **visibility improvement**\n\nMarcus met **4 of 5 goals** and completed his stretch goal."),

        _c("[EVIDENCE]\nOrg Chart — Engineering Division\nVP Engineering: David Park\n  Director Platform: Lisa Tran (8 reports)\n  Director Product: Raj Mehta (12 reports)\n  Director QA: Anna Kowalski (6 reports)\nTotal headcount: 142\n[/EVIDENCE]\n\nWho leads the Engineering division?",
           "## Engineering Division Leadership\n\n| Role | Name | Direct Reports |\n|------|------|----------------|\n| **VP Engineering** | David Park | 3 directors |\n| Director, Platform | Lisa Tran | 8 |\n| Director, Product | Raj Mehta | 12 |\n| Director, QA | Anna Kowalski | 6 |\n\n**Total division headcount:** 142"),

        _c("[EVIDENCE]\nRedundancy Consultation — Phase 1\nAffected roles: 18 positions across Operations\nConsultation period: 30 days (starts 2025-10-01)\nSelection criteria: skills matrix, performance ratings, length of service\nRedeployment: internal vacancy matching offered first\n[/EVIDENCE]\n\nWhat are the redundancy selection criteria?",
           "## Redundancy Selection Criteria\n\nPer the Phase 1 consultation document:\n\n1. **Skills matrix** assessment\n2. **Performance ratings**\n3. **Length of service**\n\n- **Affected positions:** 18 across Operations\n- **Consultation period:** 30 days (from 2025-10-01)\n- **Redeployment:** Internal vacancy matching is offered before redundancy is confirmed"),

        _c("[EVIDENCE]\nTraining Record — Employee: Fatima Al-Rashid\nCompleted: GDPR Fundamentals (2025-01-15), First Aid (2025-03-22)\nIn Progress: Project Management Professional (PMP)\nOverdue: Cyber Security Awareness (due 2025-06-30)\n[/EVIDENCE]\n\nWhich training is Fatima Al-Rashid overdue on?",
           "## Training Status — Fatima Al-Rashid\n\n| Course | Status | Date |\n|--------|--------|------|\n| GDPR Fundamentals | Completed | 2025-01-15 |\n| First Aid | Completed | 2025-03-22 |\n| PMP | In Progress | -- |\n| **Cyber Security Awareness** | **Overdue** | Due 2025-06-30 |\n\nThe **Cyber Security Awareness** course is overdue and requires immediate attention."),

        _c("[EVIDENCE]\nSickness Absence Report — October 2025\nDepartment: Customer Service\nTotal absence days: 47 | Headcount: 32\nBradford Factor alerts: 3 employees above threshold (>200)\nLong-term sick: 1 employee (stress-related, occupational health referral made)\n[/EVIDENCE]\n\nWhat is the sickness absence situation for Customer Service?",
           "## Sickness Absence — Customer Service (October 2025)\n\n- **Total absence days:** 47\n- **Department headcount:** 32\n- **Average absence:** ~1.5 days per employee\n\n## Alerts\n\n- **Bradford Factor:** 3 employees exceed the threshold (>200)\n- **Long-term sick:** 1 employee (stress-related)\n  - Occupational health referral has been made"),

        _c("[EVIDENCE]\nDisciplinary Outcome — Case REF-2025-088\nEmployee: [REDACTED] | Department: Sales\nAllegation: Breach of social media policy\nHearing Date: 2025-09-10\nOutcome: Written warning (12-month duration)\nRight of appeal: within 5 working days\n[/EVIDENCE]\n\nWhat was the disciplinary outcome for case REF-2025-088?",
           "## Disciplinary Outcome — REF-2025-088\n\n- **Allegation:** Breach of social media policy\n- **Department:** Sales\n- **Hearing Date:** 2025-09-10\n- **Outcome:** Written warning (**12-month duration**)\n- **Right of Appeal:** Within **5 working days** of the outcome\n\nEmployee details are redacted in the source document."),

        _c("[EVIDENCE]\nOnboarding Checklist — New Starter Pack\nDay 1: ID badge, IT equipment, building access, welcome meeting\nWeek 1: HR induction, H&S briefing, line manager 1:1, team intro\nMonth 1: Complete mandatory e-learning (5 modules), first objectives set\n[/EVIDENCE]\n\nWhat happens in the first week for a new starter?",
           "## Week 1 Onboarding\n\n**Day 1:**\n- ID badge and building access issued\n- IT equipment provisioned\n- Welcome meeting\n\n**Remainder of Week 1:**\n- HR induction session\n- Health & Safety briefing\n- Line manager 1:1\n- Team introductions\n\nBy the end of **Month 1**, new starters must complete 5 mandatory e-learning modules and have first objectives set."),

        _c("[EVIDENCE]\nEqual Pay Audit 2025\nMean gender pay gap: 8.2% (down from 11.4% in 2024)\nMedian gender pay gap: 5.1%\nBonus gap: 12.7% (mean)\nUpper quartile female representation: 34%\n[/EVIDENCE]\n\nWhat is the current gender pay gap?",
           "## Gender Pay Gap — 2025 Audit\n\n| Metric | Value | Trend |\n|--------|-------|-------|\n| **Mean gender pay gap** | 8.2% | Down from 11.4% (2024) |\n| **Median gender pay gap** | 5.1% | -- |\n| **Mean bonus gap** | 12.7% | -- |\n| **Upper quartile female representation** | 34% | -- |\n\nThe mean gap has narrowed by **3.2 percentage points** year-on-year."),

        _c("[EVIDENCE]\nExit Interview Summary — Q3 2025\nTotal leavers: 14 | Voluntary: 11 | Involuntary: 3\nTop reasons: career progression (5), compensation (3), relocation (2), management (1)\nAverage tenure of leavers: 2.8 years\n[/EVIDENCE]\n\nWhy are people leaving the company?",
           "## Exit Interview Insights — Q3 2025\n\n**Voluntary departures:** 11 out of 14 total leavers\n\n| Reason | Count |\n|--------|-------|\n| Career progression | 5 |\n| Compensation | 3 |\n| Relocation | 2 |\n| Management | 1 |\n\n- **Average tenure of leavers:** 2.8 years\n- The primary driver is **career progression** — more than double any other reason"),
    ]


def _generate_legal() -> List[Dict]:
    """Legal domain examples (15)."""
    return [
        _c("[EVIDENCE]\nMaster Services Agreement — Clause 12.1 Limitation of Liability\nNeither party's aggregate liability shall exceed the total fees paid in the preceding 12 months.\nExclusions: fraud, wilful misconduct, IP indemnity, death or personal injury.\n[/EVIDENCE]\n\nWhat is the liability cap in the MSA?",
           "## Limitation of Liability — Clause 12.1\n\n- **Aggregate liability cap:** Total fees paid in the preceding **12 months**\n- Applies to both parties equally\n\n## Exclusions (uncapped)\n\n- Fraud\n- Wilful misconduct\n- IP indemnity obligations\n- Death or personal injury\n\nThese exclusions remain outside the liability cap."),

        _c("[EVIDENCE]\nNDA — Effective Date: 2025-06-01\nParties: Acme Corp (Discloser) and Beta Ltd (Recipient)\nConfidentiality period: 3 years from disclosure\nPermitted purpose: evaluation of potential joint venture\nGoverning law: England and Wales\n[/EVIDENCE]\n\nHow long does the NDA confidentiality last?",
           "## NDA Duration\n\n- **Confidentiality period:** 3 years from the date of each disclosure\n- **Effective date:** 2025-06-01\n- **Parties:** Acme Corp (Discloser) / Beta Ltd (Recipient)\n- **Permitted purpose:** Evaluation of potential joint venture\n- **Governing law:** England and Wales"),

        _c("[EVIDENCE]\nEmployment Contract — Restrictive Covenants (Clause 18)\nNon-compete: 6 months post-termination, within 25-mile radius\nNon-solicitation of clients: 12 months\nNon-solicitation of employees: 12 months\nGarden leave: employer may invoke up to 3 months\n[/EVIDENCE]\n\nWhat are the post-termination restrictions?",
           "## Restrictive Covenants — Clause 18\n\n| Restriction | Duration | Scope |\n|-------------|----------|-------|\n| **Non-compete** | 6 months | 25-mile radius |\n| **Non-solicitation (clients)** | 12 months | All clients dealt with |\n| **Non-solicitation (employees)** | 12 months | All employees |\n| **Garden leave** | Up to 3 months | At employer's discretion |\n\nAll restrictions apply from the date of termination."),

        _c("[EVIDENCE]\nLease Agreement — Unit 4, Innovation Park\nTenant: TechStart Ltd | Landlord: PropCo Holdings\nTerm: 5 years commencing 2025-01-01\nRent: $8,500/month | Review: annually, CPI-linked\nBreak clause: mutual, at month 36 with 6 months' notice\n[/EVIDENCE]\n\nWhen can we break the lease?",
           "## Break Clause\n\n- **Available at:** Month 36 (January 2028)\n- **Notice required:** 6 months\n- **Type:** Mutual — either tenant or landlord may exercise\n\n## Lease Summary\n\n- **Tenant:** TechStart Ltd\n- **Monthly Rent:** $8,500 (CPI-linked annual reviews)\n- **Full Term:** 5 years from 2025-01-01"),

        _c("[EVIDENCE]\nData Processing Agreement — Clause 5 Sub-processors\nController: MedSafe Inc | Processor: CloudOps Ltd\nApproved sub-processors: AWS (hosting), Datadog (monitoring)\nNotification of new sub-processor: 30 days prior written notice\nController may object within 14 days\n[/EVIDENCE]\n\nWho are the approved sub-processors?",
           "## Approved Sub-processors — DPA Clause 5\n\n| Sub-processor | Purpose |\n|---------------|---------|\n| **AWS** | Hosting |\n| **Datadog** | Monitoring |\n\n- **Controller:** MedSafe Inc\n- **Processor:** CloudOps Ltd\n- New sub-processors require **30 days' prior written notice**\n- Controller may object within **14 days** of notification"),

        _c("[EVIDENCE]\nIP Assignment — Clause 7.2\nAll inventions, designs, and works created during employment belong to the employer.\nMoral rights: employee waives to the fullest extent permitted by law.\nPre-existing IP: listed in Schedule 3 and excluded from assignment.\n[/EVIDENCE]\n\nWho owns IP created during employment?",
           "## Intellectual Property Ownership — Clause 7.2\n\n- **All inventions, designs, and works** created during employment belong to the **employer**\n- **Moral rights:** Waived to the fullest extent permitted by law\n- **Pre-existing IP:** Excluded from assignment (listed in Schedule 3)\n\nAny IP the employee brings into the role must be declared in Schedule 3 to remain their property."),

        _c("[EVIDENCE]\nForce Majeure — Clause 14\nEvents: war, pandemic, natural disaster, government action, cyberattack\nAffected party must notify within 5 business days\nSuspension: obligations suspended for duration of event\nTermination right: if force majeure exceeds 90 consecutive days\n[/EVIDENCE]\n\nWhat happens if a force majeure event exceeds 90 days?",
           "## Force Majeure — Clause 14\n\n- **Notification required:** Within 5 business days of the event\n- **During event:** Affected party's obligations are **suspended**\n- **If event exceeds 90 consecutive days:** Either party may **terminate** the agreement\n\nCovered events include war, pandemic, natural disaster, government action, and cyberattack."),

        _c("[EVIDENCE]\nSLA — Clause 9 Service Credits\nUptime target: 99.9% per calendar month\nCredit: 5% of monthly fee per 0.1% below target\nMaximum credit: 30% of monthly fee\nClaim window: within 30 days of the affected month\n[/EVIDENCE]\n\nHow are service credits calculated?",
           "## Service Credit Calculation — Clause 9\n\n| Metric | Value |\n|--------|-------|\n| **Uptime target** | 99.9% per month |\n| **Credit rate** | 5% of monthly fee per 0.1% shortfall |\n| **Maximum credit** | 30% of monthly fee |\n| **Claim window** | Within 30 days |\n\n**Example:** If uptime is 99.5%, the shortfall is 0.4%, yielding a credit of **20%** of the monthly fee."),

        _c("[EVIDENCE]\nShare Purchase Agreement — Completion Conditions\nCondition 1: Buyer due diligence sign-off by 2025-12-01\nCondition 2: Regulatory approval (FCA) obtained\nCondition 3: No material adverse change in target company\nLongstop date: 2026-03-31\n[/EVIDENCE]\n\nWhat conditions must be met before the share purchase completes?",
           "## Completion Conditions — Share Purchase Agreement\n\n| # | Condition | Status |\n|---|-----------|--------|\n| 1 | Buyer due diligence sign-off | By 2025-12-01 |\n| 2 | FCA regulatory approval | Pending |\n| 3 | No material adverse change in target | Ongoing |\n\n- **Longstop date:** 2026-03-31 — if conditions are not satisfied by this date, either party may withdraw."),

        _c("[EVIDENCE]\nTermination for Convenience — Clause 16.2\nEither party may terminate on 90 days' written notice.\nAccrued rights and obligations survive termination.\nTransition assistance: departing provider must cooperate for up to 6 months post-termination.\n[/EVIDENCE]\n\nHow much notice is needed to terminate the contract?",
           "## Termination for Convenience — Clause 16.2\n\n- **Notice period:** 90 days' written notice\n- **Available to:** Either party\n- **Post-termination:** Departing provider must provide transition assistance for up to **6 months**\n- **Surviving provisions:** All accrued rights and obligations survive termination"),

        _c("[EVIDENCE]\nIndemnification — Clause 11\nVendor indemnifies Client against third-party IP infringement claims.\nClient indemnifies Vendor against misuse of deliverables.\nProcedure: indemnified party must notify within 10 business days.\nIndemnifying party has sole control of defence.\n[/EVIDENCE]\n\nWho is responsible for IP infringement claims?",
           "## IP Indemnification — Clause 11\n\n- **Vendor** indemnifies the Client against third-party IP infringement claims\n- **Client** indemnifies the Vendor against misuse of deliverables\n\n## Procedure\n\n- Notification within **10 business days** of becoming aware of a claim\n- The **indemnifying party** has sole control of the defence"),

        _c("[EVIDENCE]\nDispute Resolution — Clause 20\nStep 1: Senior management negotiation within 15 business days\nStep 2: Mediation under CEDR rules within 30 days\nStep 3: Binding arbitration under LCIA rules, London seat\nUrgent injunctive relief: either party may apply to court at any time\n[/EVIDENCE]\n\nHow are disputes resolved under this contract?",
           "## Dispute Resolution — Clause 20\n\n| Step | Mechanism | Timeframe |\n|------|-----------|----------|\n| 1 | Senior management negotiation | 15 business days |\n| 2 | Mediation (CEDR rules) | 30 days |\n| 3 | Binding arbitration (LCIA, London) | As scheduled |\n\nEither party may seek **urgent injunctive relief** from the courts at any time, regardless of the dispute resolution process."),

        _c("[EVIDENCE]\nWarranty — Clause 8\nVendor warrants deliverables will conform to specifications for 12 months.\nRemedy: repair or replace defective deliverables at vendor's cost.\nExclusion: defects caused by client modification or misuse.\nNotice: defect must be reported within 30 days of discovery.\n[/EVIDENCE]\n\nWhat warranty does the vendor provide?",
           "## Warranty — Clause 8\n\n- **Period:** 12 months from delivery\n- **Scope:** Deliverables conform to agreed specifications\n- **Remedy:** Repair or replace at **vendor's cost**\n- **Defect reporting:** Within **30 days** of discovery\n\n## Exclusions\n\n- Defects caused by client modification\n- Defects caused by misuse"),

        _c("[EVIDENCE]\nGoverning Law — Clause 22\nThis agreement is governed by the laws of the State of Delaware.\nExclusive jurisdiction: federal and state courts of Delaware.\nWaiver of jury trial: both parties waive right to jury trial.\n[/EVIDENCE]\n\nWhich jurisdiction governs this agreement?",
           "## Governing Law — Clause 22\n\n- **Governing law:** State of Delaware\n- **Exclusive jurisdiction:** Federal and state courts of Delaware\n- **Jury trial:** Both parties have **waived** the right to a jury trial"),

        _c("[EVIDENCE]\nChange of Control — Clause 15\nIf either party undergoes a change of control, the other party may terminate on 60 days' notice.\nChange of control: acquisition of 50%+ voting shares, merger, or asset sale.\nAssignment requires prior written consent.\n[/EVIDENCE]\n\nWhat happens if the vendor is acquired?",
           "## Change of Control — Clause 15\n\nIf the vendor undergoes a change of control:\n\n- The **client may terminate** with **60 days' written notice**\n- Change of control is defined as: acquisition of **50%+ voting shares**, merger, or asset sale\n- **Assignment** of the contract requires prior written consent from the other party"),
    ]


def _generate_finance() -> List[Dict]:
    """Finance domain examples (15)."""
    return [
        _c("[EVIDENCE]\nBalance Sheet — FY2025\nTotal Assets: $24,800,000\nTotal Liabilities: $14,200,000\nShareholders' Equity: $10,600,000\nCurrent Ratio: 1.82\n[/EVIDENCE]\n\nWhat is the company's financial position?",
           "## Financial Position — FY2025\n\n| Metric | Value |\n|--------|-------|\n| **Total Assets** | $24,800,000 |\n| **Total Liabilities** | $14,200,000 |\n| **Shareholders' Equity** | $10,600,000 |\n| **Current Ratio** | 1.82 |\n\nThe current ratio of **1.82** indicates healthy short-term liquidity — the company can cover current liabilities nearly twice over."),

        _c("[EVIDENCE]\nP&L — Q3 2025\nRevenue: $8,450,000 | COGS: $3,380,000\nGross Profit: $5,070,000 | Gross Margin: 60%\nOperating Expenses: $3,200,000\nEBITDA: $1,870,000 | Net Income: $1,240,000\n[/EVIDENCE]\n\nWhat is the gross margin for Q3?",
           "## Profitability — Q3 2025\n\n| Metric | Value |\n|--------|-------|\n| **Revenue** | $8,450,000 |\n| **COGS** | $3,380,000 |\n| **Gross Profit** | $5,070,000 |\n| **Gross Margin** | 60% |\n| **EBITDA** | $1,870,000 |\n| **Net Income** | $1,240,000 |\n\nThe **60% gross margin** indicates strong pricing power relative to cost of goods sold."),

        _c("[EVIDENCE]\nBudget Variance Report — Marketing Dept, September 2025\nBudgeted: $120,000 | Actual: $138,500\nVariance: $18,500 unfavourable (15.4%)\nKey overruns: Digital advertising +$12,000, event sponsorship +$8,200\nUnderspend: Print media -$1,700\n[/EVIDENCE]\n\nWhy is Marketing over budget?",
           "## Budget Variance — Marketing (September 2025)\n\n- **Budget:** $120,000\n- **Actual:** $138,500\n- **Variance:** $18,500 unfavourable (**15.4%** over budget)\n\n## Breakdown\n\n| Category | Variance |\n|----------|----------|\n| Digital advertising | +$12,000 |\n| Event sponsorship | +$8,200 |\n| Print media | -$1,700 |\n\nDigital advertising is the primary driver of the overspend."),

        _c("[EVIDENCE]\nAccounts Receivable Aging — 31 Oct 2025\nCurrent (0-30 days): $1,240,000\n31-60 days: $380,000\n61-90 days: $95,000\n90+ days: $42,000 (3 customers)\nTotal AR: $1,757,000 | DSO: 38 days\n[/EVIDENCE]\n\nHow much is overdue past 90 days?",
           "## Accounts Receivable Aging — 31 October 2025\n\n| Aging Bucket | Amount |\n|-------------|--------|\n| Current (0-30 days) | $1,240,000 |\n| 31-60 days | $380,000 |\n| 61-90 days | $95,000 |\n| **90+ days** | **$42,000** (3 customers) |\n\n- **Total AR:** $1,757,000\n- **Days Sales Outstanding:** 38 days\n- The **$42,000** past 90 days across 3 customers requires collection action."),

        _c("[EVIDENCE]\nCash Flow Statement — FY2025\nOperating Cash Flow: $3,100,000\nInvesting Activities: -$1,800,000 (CapEx $1,500,000, acquisitions $300,000)\nFinancing Activities: -$900,000 (debt repayment $600,000, dividends $300,000)\nNet Cash Change: $400,000 | Closing Cash: $2,850,000\n[/EVIDENCE]\n\nWhat is the free cash flow?",
           "## Cash Flow Summary — FY2025\n\n| Category | Amount |\n|----------|--------|\n| **Operating Cash Flow** | $3,100,000 |\n| Investing Activities | -$1,800,000 |\n| Financing Activities | -$900,000 |\n| **Net Cash Change** | $400,000 |\n| **Closing Cash** | $2,850,000 |\n\n**Free Cash Flow** (Operating - CapEx): $3,100,000 - $1,500,000 = **$1,600,000**\n\nThe company generated positive free cash flow while funding acquisitions and paying dividends."),

        _c("[EVIDENCE]\nTax Computation — Corporation Tax FY2025\nProfit before tax: $4,200,000\nDisallowable expenses: $180,000\nCapital allowances: -$320,000\nTaxable profit: $4,060,000\nCorporation tax (21%): $852,600\n[/EVIDENCE]\n\nWhat is the effective tax rate?",
           "## Corporation Tax — FY2025\n\n| Item | Amount |\n|------|--------|\n| Profit before tax | $4,200,000 |\n| Disallowable expenses | +$180,000 |\n| Capital allowances | -$320,000 |\n| **Taxable profit** | $4,060,000 |\n| **Tax payable (21%)** | $852,600 |\n\n**Effective tax rate:** $852,600 / $4,200,000 = **20.3%** — slightly below the statutory rate due to capital allowances."),

        _c("[EVIDENCE]\nLoan Agreement — Term Sheet\nBorrower: NovaTech Inc | Lender: First National Bank\nFacility: $5,000,000 revolving credit\nInterest: SOFR + 2.25% | Maturity: 3 years\nCovenants: Debt/EBITDA < 3.0x, Interest Cover > 4.0x\n[/EVIDENCE]\n\nWhat are the loan covenants?",
           "## Loan Covenants — NovaTech Revolving Credit\n\n| Covenant | Threshold |\n|----------|----------|\n| **Debt/EBITDA** | < 3.0x |\n| **Interest Coverage Ratio** | > 4.0x |\n\n## Facility Terms\n\n- **Amount:** $5,000,000 revolving credit\n- **Interest:** SOFR + 2.25%\n- **Maturity:** 3 years\n- **Lender:** First National Bank\n\nBreach of either covenant may trigger accelerated repayment."),

        _c("[EVIDENCE]\nAudit Finding — Ref AF-2025-014\nSeverity: High | Area: Revenue Recognition\nFinding: $340,000 revenue recognised prematurely for 3 contracts\nCriteria: ASC 606 — performance obligations not satisfied at recognition date\nRecommendation: Restate and implement monthly cut-off review\n[/EVIDENCE]\n\nWhat did the auditors find about revenue recognition?",
           "## Audit Finding — AF-2025-014 (High Severity)\n\n- **Area:** Revenue Recognition\n- **Issue:** **$340,000** recognised prematurely across 3 contracts\n- **Criteria violated:** ASC 606 — performance obligations were not satisfied at the date of recognition\n\n## Recommendation\n\n- Restate the affected revenue\n- Implement a **monthly cut-off review** process to prevent recurrence"),

        _c("[EVIDENCE]\nDepreciation Schedule — IT Assets\nServers: $480,000 cost, 5-year straight-line, NBV $192,000\nLaptops: $120,000 cost, 3-year straight-line, NBV $0 (fully depreciated)\nNetworking: $95,000 cost, 5-year straight-line, NBV $57,000\nTotal annual depreciation: $139,000\n[/EVIDENCE]\n\nWhat is the net book value of IT assets?",
           "## IT Asset Depreciation\n\n| Asset Class | Cost | Life | NBV |\n|-------------|------|------|-----|\n| Servers | $480,000 | 5 years | $192,000 |\n| Laptops | $120,000 | 3 years | $0 (fully depreciated) |\n| Networking | $95,000 | 5 years | $57,000 |\n\n- **Total Cost:** $695,000\n- **Total NBV:** $249,000\n- **Annual Depreciation:** $139,000"),

        _c("[EVIDENCE]\nForeign Exchange Exposure — October 2025\nGBP receivables: GBP 2,100,000 (hedged 60% at 1.27)\nEUR payables: EUR 850,000 (unhedged)\nJPY payables: JPY 45,000,000 (hedged 100% at 149.5)\nNet unhedged exposure: $1,680,000 equivalent\n[/EVIDENCE]\n\nWhat is our FX exposure?",
           "## Foreign Exchange Exposure — October 2025\n\n| Currency | Amount | Hedge Status |\n|----------|--------|--------------|\n| GBP receivables | GBP 2,100,000 | **60% hedged** at 1.27 |\n| EUR payables | EUR 850,000 | **Unhedged** |\n| JPY payables | JPY 45,000,000 | **100% hedged** at 149.5 |\n\n- **Net unhedged exposure:** $1,680,000 equivalent\n- Primary risk is on the **EUR payables** (fully unhedged) and the unhedged portion of **GBP receivables**."),

        _c("[EVIDENCE]\nInvestment Portfolio — Q3 2025\nEquities: $3,200,000 (42%) | Return: +8.4% YTD\nFixed Income: $2,800,000 (37%) | Return: +3.1% YTD\nReal Estate: $1,200,000 (16%) | Return: +5.2% YTD\nCash: $400,000 (5%)\nTotal AUM: $7,600,000 | Blended Return: +5.9% YTD\n[/EVIDENCE]\n\nHow is the investment portfolio performing?",
           "## Investment Portfolio — Q3 2025\n\n| Asset Class | Allocation | Value | YTD Return |\n|-------------|-----------|-------|------------|\n| Equities | 42% | $3,200,000 | +8.4% |\n| Fixed Income | 37% | $2,800,000 | +3.1% |\n| Real Estate | 16% | $1,200,000 | +5.2% |\n| Cash | 5% | $400,000 | -- |\n\n- **Total AUM:** $7,600,000\n- **Blended YTD Return:** +5.9%\n- Equities are the strongest performer at **+8.4%**."),

        _c("[EVIDENCE]\nPayroll Summary — November 2025\nGross payroll: $1,845,000 | Headcount: 186\nEmployer NI: $215,400 | Pension contributions: $147,600\nTotal payroll cost: $2,208,000\nOvertime: $42,000 (Engineering and Operations)\n[/EVIDENCE]\n\nWhat is the total payroll cost?",
           "## Payroll Summary — November 2025\n\n| Component | Amount |\n|-----------|--------|\n| Gross Payroll | $1,845,000 |\n| Employer NI | $215,400 |\n| Pension Contributions | $147,600 |\n| **Total Payroll Cost** | **$2,208,000** |\n\n- **Headcount:** 186\n- **Overtime:** $42,000 (Engineering and Operations)\n- **Average cost per employee:** ~$11,871/month"),

        _c("[EVIDENCE]\nCapital Expenditure Request — REQ-2025-077\nProject: Data Centre Expansion\nRequested: $2,400,000 | Payback Period: 2.8 years\nIRR: 22% | NPV (10% discount): $580,000\nApproval: CFO sign-off required above $500,000\n[/EVIDENCE]\n\nShould we approve the data centre expansion?",
           "## CapEx Request — Data Centre Expansion (REQ-2025-077)\n\n| Metric | Value |\n|--------|-------|\n| **Investment** | $2,400,000 |\n| **Payback Period** | 2.8 years |\n| **IRR** | 22% |\n| **NPV (10% discount)** | $580,000 |\n\nThe project shows a positive NPV of **$580,000** and an IRR of **22%**, both exceeding typical hurdle rates. The payback period of 2.8 years is within standard benchmarks.\n\nCFO sign-off is required as the amount exceeds the $500,000 threshold."),

        _c("[EVIDENCE]\nIntercompany Reconciliation — October 2025\nEntity A owes Entity B: $1,340,000\nEntity B owes Entity A: $1,280,000\nNet position: Entity A owes $60,000\nUnreconciled items: 4 transactions totalling $12,500\n[/EVIDENCE]\n\nWhat is the intercompany position?",
           "## Intercompany Reconciliation — October 2025\n\n| Direction | Amount |\n|-----------|--------|\n| Entity A owes Entity B | $1,340,000 |\n| Entity B owes Entity A | $1,280,000 |\n| **Net position** | Entity A owes **$60,000** |\n\n- **Unreconciled items:** 4 transactions totalling **$12,500**\n- These need resolution before month-end close."),

        _c("[EVIDENCE]\nInsurance Renewal — FY2026\nProfessional Indemnity: $5M cover, premium $28,400 (+6% YoY)\nPublic Liability: $10M cover, premium $14,200 (+3% YoY)\nCyber: $2M cover, premium $19,800 (+18% YoY)\nTotal premiums: $62,400\n[/EVIDENCE]\n\nWhat are the insurance premiums for next year?",
           "## Insurance Renewal — FY2026\n\n| Policy | Cover | Premium | YoY Change |\n|--------|-------|---------|------------|\n| Professional Indemnity | $5M | $28,400 | +6% |\n| Public Liability | $10M | $14,200 | +3% |\n| Cyber | $2M | $19,800 | +18% |\n\n- **Total Premiums:** $62,400\n- **Cyber insurance** saw the largest increase at **+18%**, reflecting elevated market risk pricing."),
    ]


def _generate_medical() -> List[Dict]:
    """Medical domain examples (15)."""
    return [
        _c("[EVIDENCE]\nPatient: [REDACTED] | MRN: 00482917\nDiagnosis: Type 2 Diabetes Mellitus (E11.9)\nHbA1c: 8.2% (target <7%) | FBG: 162 mg/dL\nMedication: Metformin 1000mg BD, Empagliflozin 10mg OD\n[/EVIDENCE]\n\nWhat is the patient's current diabetes control?",
           "## Diabetes Assessment — MRN 00482917\n\n| Metric | Value | Target |\n|--------|-------|--------|\n| **HbA1c** | 8.2% | <7.0% |\n| **Fasting Blood Glucose** | 162 mg/dL | 80-130 mg/dL |\n\n- **Diagnosis:** Type 2 Diabetes Mellitus (E11.9)\n- **Current Medications:** Metformin 1000mg BD, Empagliflozin 10mg OD\n- HbA1c is **1.2% above target**, indicating suboptimal glycaemic control."),

        _c("[EVIDENCE]\nDischarge Summary — Admission: 2025-09-02, Discharge: 2025-09-06\nProcedure: Laparoscopic cholecystectomy\nIndication: Symptomatic gallstones with recurrent biliary colic\nComplications: None | EBL: <50mL\nFollow-up: Surgical clinic in 2 weeks\n[/EVIDENCE]\n\nSummarise the discharge.",
           "## Discharge Summary\n\n- **Admission:** 2025-09-02 | **Discharge:** 2025-09-06 (4-day stay)\n- **Procedure:** Laparoscopic cholecystectomy\n- **Indication:** Symptomatic gallstones with recurrent biliary colic\n- **Complications:** None\n- **Estimated Blood Loss:** <50 mL\n- **Follow-up:** Surgical clinic in **2 weeks**"),

        _c("[EVIDENCE]\nPathology Report — Specimen: Excisional biopsy, right breast\nDiagnosis: Invasive ductal carcinoma, Grade 2\nTumour size: 1.8cm | Margins: Clear (closest 3mm)\nER: Positive (95%) | PR: Positive (80%) | HER2: Negative\nLymph nodes: 0/3 sentinel nodes positive\n[/EVIDENCE]\n\nWhat does the pathology report show?",
           "## Pathology Report — Right Breast Excisional Biopsy\n\n| Finding | Result |\n|---------|--------|\n| **Diagnosis** | Invasive ductal carcinoma, Grade 2 |\n| **Tumour size** | 1.8 cm |\n| **Margins** | Clear (closest margin 3mm) |\n| **ER status** | Positive (95%) |\n| **PR status** | Positive (80%) |\n| **HER2 status** | Negative |\n| **Lymph nodes** | 0/3 sentinel nodes positive |\n\nThe tumour is **hormone receptor positive, HER2 negative** with clear margins and no lymph node involvement."),

        _c("[EVIDENCE]\nMedication Reconciliation — Admission\n1. Lisinopril 20mg OD — continue\n2. Atorvastatin 40mg ON — continue\n3. Aspirin 75mg OD — HOLD (pre-operative)\n4. Metformin 500mg BD — switch to insulin sliding scale\n[/EVIDENCE]\n\nWhich medications were changed on admission?",
           "## Medication Reconciliation — Admission\n\n| Medication | Dose | Action |\n|------------|------|--------|\n| Lisinopril | 20mg OD | Continue |\n| Atorvastatin | 40mg ON | Continue |\n| **Aspirin** | 75mg OD | **HOLD** (pre-operative) |\n| **Metformin** | 500mg BD | **Switch to insulin sliding scale** |\n\nTwo medications were changed: **Aspirin** was held for surgery and **Metformin** was replaced with an insulin sliding scale."),

        _c("[EVIDENCE]\nRadiology Report — CT Chest with Contrast\nFindings: 12mm pulmonary nodule in right lower lobe (new).\nMediastinal lymphadenopathy: subcarinal node 1.4cm.\nNo pleural effusion. Heart size normal.\nImpression: New RLL nodule — recommend PET-CT for further characterisation.\n[/EVIDENCE]\n\nWhat did the CT scan find?",
           "## CT Chest — Key Findings\n\n- **New finding:** 12mm pulmonary nodule in the **right lower lobe**\n- **Lymphadenopathy:** Subcarinal node measuring **1.4 cm**\n- No pleural effusion\n- Heart size normal\n\n## Recommendation\n\n**PET-CT** recommended for further characterisation of the new RLL nodule."),

        _c("[EVIDENCE]\nAllergy Record — Patient MRN 00482917\nPenicillin: Anaphylaxis (confirmed 2018)\nIbuprofen: Urticaria (self-reported)\nLatex: Contact dermatitis (confirmed 2020)\nNo known food allergies.\n[/EVIDENCE]\n\nWhat are the patient's allergies?",
           "## Allergy Record — MRN 00482917\n\n| Allergen | Reaction | Verification |\n|----------|----------|--------------|\n| **Penicillin** | Anaphylaxis | Confirmed (2018) |\n| **Ibuprofen** | Urticaria | Self-reported |\n| **Latex** | Contact dermatitis | Confirmed (2020) |\n\n- No known food allergies\n- **Penicillin allergy is anaphylactic** — all beta-lactams should be avoided unless desensitisation is performed."),

        _c("[EVIDENCE]\nBlood Results — 2025-10-12\nFBC: Hb 10.2 g/dL (low), WCC 6.8, Platelets 245\nU&E: Na 138, K 4.1, Creatinine 1.4 mg/dL (elevated), eGFR 52\nLFT: ALT 28, ALP 72, Bilirubin 0.8 — all normal\n[/EVIDENCE]\n\nAre there any abnormal blood results?",
           "## Blood Results — 2025-10-12\n\n### Abnormal Values\n\n| Test | Result | Reference | Status |\n|------|--------|-----------|--------|\n| **Haemoglobin** | 10.2 g/dL | 12-16 | **Low** (anaemia) |\n| **Creatinine** | 1.4 mg/dL | 0.7-1.2 | **Elevated** |\n| **eGFR** | 52 | >60 | **Low** (CKD Stage 3a) |\n\n### Normal Values\n\n- WCC, Platelets, Sodium, Potassium, ALT, ALP, Bilirubin — all within range\n\nThe patient has **anaemia** and **reduced kidney function** (eGFR 52 suggests CKD Stage 3a)."),

        _c("[EVIDENCE]\nConsent Form — Procedure: Total knee replacement (right)\nRisks discussed: infection (1-2%), DVT, nerve damage, stiffness\nAlternatives: physiotherapy, injections, partial replacement\nConsent given: Yes, patient has capacity\nSurgeon: Mr R. Patel | Date: 2025-10-20\n[/EVIDENCE]\n\nWhat risks were discussed for the knee replacement?",
           "## Consent — Right Total Knee Replacement\n\n**Risks discussed:**\n- Infection (1-2%)\n- Deep vein thrombosis (DVT)\n- Nerve damage\n- Stiffness\n\n**Alternatives offered:**\n- Physiotherapy\n- Injections\n- Partial knee replacement\n\n- **Consent:** Given, patient confirmed to have capacity\n- **Surgeon:** Mr R. Patel\n- **Date:** 2025-10-20"),

        _c("[EVIDENCE]\nVaccination Record — Patient DOB: 1985-04-12\nCOVID-19: 4 doses (last: 2024-10-15, Pfizer bivalent)\nInfluenza: 2025-10-01 (Quadrivalent)\nTdap: 2021-06-20 (due 2031)\nHepatitis B: 3-dose series completed 2019\n[/EVIDENCE]\n\nIs the patient up to date on vaccinations?",
           "## Vaccination Status\n\n| Vaccine | Last Dose | Next Due | Status |\n|---------|-----------|----------|--------|\n| COVID-19 | 2024-10-15 (dose 4) | Per guidance | Current |\n| Influenza | 2025-10-01 | Oct 2026 | **Current** |\n| Tdap | 2021-06-20 | 2031 | **Current** |\n| Hepatitis B | 2019 (series complete) | N/A | **Complete** |\n\nAll vaccinations appear **up to date** based on the records provided."),

        _c("[EVIDENCE]\nClinical Trial Eligibility — Protocol XR-401\nInclusion: Age 18-75, confirmed NSCLC Stage IIIB/IV, ECOG 0-1\nExclusion: Prior immunotherapy, active autoimmune disease, brain metastases\nScreening window: 28 days from consent\n[/EVIDENCE]\n\nWho is eligible for trial XR-401?",
           "## Clinical Trial XR-401 — Eligibility Criteria\n\n### Inclusion\n\n- Age **18-75**\n- Confirmed **NSCLC Stage IIIB or IV**\n- **ECOG performance status** 0-1\n\n### Exclusion\n\n- Prior immunotherapy\n- Active autoimmune disease\n- Brain metastases\n\n**Screening window:** 28 days from date of consent."),

        _c("[EVIDENCE]\nReferral Letter — From: Dr S. Hussain (GP)\nTo: Cardiology, St. Mary's Hospital\nReason: Recurrent palpitations with 2 pre-syncopal episodes\nECG: Sinus rhythm, prolonged QTc 480ms\nFamily history: Sudden cardiac death (father, age 42)\n[/EVIDENCE]\n\nWhy was the patient referred to cardiology?",
           "## Cardiology Referral\n\n- **Referring clinician:** Dr S. Hussain (GP)\n- **Reason:** Recurrent palpitations with **2 pre-syncopal episodes**\n\n## Key Findings\n\n- **ECG:** Sinus rhythm with **prolonged QTc (480ms)** — above the normal threshold of 450ms\n- **Family history:** Sudden cardiac death in father at age 42\n\nThe combination of prolonged QTc, symptoms, and family history warrants urgent cardiac investigation."),

        _c("[EVIDENCE]\nNursing Assessment — Fall Risk\nMorse Fall Scale Score: 65 (High Risk)\nFactors: History of falls (1 in past 3 months), IV access, impaired gait\nInterventions: Bed alarm, non-slip socks, hourly rounding, yellow wristband\n[/EVIDENCE]\n\nWhat is the patient's fall risk?",
           "## Fall Risk Assessment\n\n- **Morse Fall Scale:** 65 — **High Risk**\n\n## Risk Factors\n\n- History of falls (1 in past 3 months)\n- IV access present\n- Impaired gait\n\n## Interventions in Place\n\n- Bed alarm activated\n- Non-slip socks provided\n- Hourly rounding\n- Yellow wristband applied"),

        _c("[EVIDENCE]\nMental Health Assessment — PHQ-9 Score: 18 (Moderately Severe)\nGAD-7 Score: 14 (Moderate Anxiety)\nSuicidal ideation: Passive thoughts, no plan or intent\nCurrent treatment: Sertraline 100mg OD, CBT fortnightly\n[/EVIDENCE]\n\nWhat is the mental health assessment result?",
           "## Mental Health Assessment\n\n| Scale | Score | Severity |\n|-------|-------|----------|\n| **PHQ-9** | 18 | Moderately Severe Depression |\n| **GAD-7** | 14 | Moderate Anxiety |\n\n- **Suicidal ideation:** Passive thoughts reported — **no plan or intent**\n- **Current treatment:** Sertraline 100mg OD + fortnightly CBT\n\nScores indicate **moderately severe depression** and **moderate anxiety** — treatment review may be warranted."),

        _c("[EVIDENCE]\nOperative Note — Procedure: Appendicectomy (laparoscopic)\nDuration: 45 minutes | Anaesthesia: General\nFindings: Acutely inflamed appendix, no perforation\nSpecimen: Sent to histology\nPost-op plan: Clear fluids day 0, diet as tolerated day 1, discharge day 1-2\n[/EVIDENCE]\n\nWhat was found during the appendicectomy?",
           "## Operative Note — Laparoscopic Appendicectomy\n\n- **Duration:** 45 minutes under general anaesthesia\n- **Findings:** Acutely inflamed appendix, **no perforation**\n- **Specimen:** Sent to histology\n\n## Post-Operative Plan\n\n- **Day 0:** Clear fluids\n- **Day 1:** Diet as tolerated\n- **Discharge:** Day 1-2 post-op"),

        _c("[EVIDENCE]\nPharmacy Alert — Drug Interaction\nPrescribed: Warfarin 5mg + Amiodarone 200mg\nSeverity: Major | Effect: Increased INR and bleeding risk\nRecommendation: Reduce Warfarin dose by 30-50%, monitor INR within 3 days\nPharmacist: L. Nguyen | Date: 2025-10-14\n[/EVIDENCE]\n\nIs there a drug interaction flagged?",
           "## Drug Interaction Alert\n\n- **Severity:** Major\n- **Drugs:** Warfarin 5mg + Amiodarone 200mg\n- **Effect:** Increased INR and **elevated bleeding risk**\n\n## Recommendation\n\n- **Reduce Warfarin dose** by 30-50%\n- **Monitor INR** within 3 days\n- Flagged by pharmacist L. Nguyen on 2025-10-14"),
    ]


def _generate_insurance() -> List[Dict]:
    """Insurance domain examples (15)."""
    return [
        _c("[EVIDENCE]\nPolicy: AUTO-2025-88421\nInsured: Michael Torres | Vehicle: 2023 Toyota Camry\nCoverage: Comprehensive | Deductible: $500\nPremium: $1,840/year | Effective: 2025-01-15 to 2026-01-15\n[/EVIDENCE]\n\nWhat coverage does this auto policy provide?",
           "## Auto Policy — AUTO-2025-88421\n\n- **Insured:** Michael Torres\n- **Vehicle:** 2023 Toyota Camry\n- **Coverage Type:** Comprehensive\n- **Deductible:** $500\n- **Annual Premium:** $1,840\n- **Policy Period:** 2025-01-15 to 2026-01-15\n\nComprehensive coverage includes collision, theft, vandalism, natural disasters, and third-party liability."),

        _c("[EVIDENCE]\nClaim: CLM-2025-14320\nDate of Loss: 2025-08-14 | Type: Water damage (burst pipe)\nProperty: 45 Oak Lane, Denver, CO\nEstimated damage: $32,500 | Deductible: $1,000\nStatus: Under investigation — adjuster assigned\n[/EVIDENCE]\n\nWhat is the status of claim CLM-2025-14320?",
           "## Claim Status — CLM-2025-14320\n\n| Field | Detail |\n|-------|--------|\n| **Date of Loss** | 2025-08-14 |\n| **Type** | Water damage (burst pipe) |\n| **Property** | 45 Oak Lane, Denver, CO |\n| **Estimated Damage** | $32,500 |\n| **Deductible** | $1,000 |\n| **Net Claim** | $31,500 |\n| **Status** | Under investigation — adjuster assigned |"),

        _c("[EVIDENCE]\nUnderwriting Report — Applicant: GreenField Manufacturing\nRisk Class: Commercial Property | Building Value: $4,200,000\nFire Protection: Sprinkler system, fire alarm, 24/7 monitoring\nLoss History: 1 claim in 5 years ($18,000 water damage)\nRisk Score: 72/100 (Acceptable)\n[/EVIDENCE]\n\nIs this applicant an acceptable risk?",
           "## Underwriting Assessment — GreenField Manufacturing\n\n| Factor | Detail |\n|--------|--------|\n| **Risk Class** | Commercial Property |\n| **Building Value** | $4,200,000 |\n| **Risk Score** | 72/100 (**Acceptable**) |\n| **Loss History** | 1 claim in 5 years ($18,000) |\n\n## Risk Mitigants\n\n- Sprinkler system installed\n- Fire alarm system\n- 24/7 monitoring service\n\nThe risk score of **72/100** falls within the acceptable range, supported by strong fire protection and minimal loss history."),

        _c("[EVIDENCE]\nLife Insurance — Policy: LI-2025-00341\nInsured: Rebecca Walsh | Age: 38 | Non-smoker\nCoverage: $750,000 term life (20 years)\nPremium: $62/month | Beneficiary: Spouse — Thomas Walsh\nMedical underwriting: Standard rate (BMI 24, no conditions)\n[/EVIDENCE]\n\nWhat are the terms of this life insurance policy?",
           "## Life Insurance — LI-2025-00341\n\n- **Insured:** Rebecca Walsh (age 38, non-smoker)\n- **Coverage:** $750,000 **term life** (20-year term)\n- **Monthly Premium:** $62\n- **Beneficiary:** Thomas Walsh (spouse)\n- **Underwriting Class:** Standard rate\n- **Health:** BMI 24, no pre-existing conditions"),

        _c("[EVIDENCE]\nClaim Denial — CLM-2025-09877\nReason: Policy exclusion — gradual deterioration (Clause 4.3.2)\nDamage: Subsidence cracking to rear wall\nSurveyor finding: Long-term ground movement over 18+ months\nAppeal deadline: 30 days from denial letter (2025-11-15)\n[/EVIDENCE]\n\nWhy was the claim denied?",
           "## Claim Denial — CLM-2025-09877\n\n- **Reason:** Policy exclusion under **Clause 4.3.2 — gradual deterioration**\n- **Damage:** Subsidence cracking to rear wall\n- **Surveyor finding:** Long-term ground movement over **18+ months**\n\nThe insurer determined the damage resulted from gradual deterioration rather than a sudden event, which is excluded under the policy.\n\n- **Appeal deadline:** 2025-11-15 (30 days from denial letter)"),

        _c("[EVIDENCE]\nRenewal Notice — Policy: HO-2025-55210\nInsured: David & Karen Chen | Property: 120 Maple Drive\nExpiring Premium: $2,140 | Renewal Premium: $2,380 (+11.2%)\nReason for increase: Regional wildfire risk reassessment\nRenewal date: 2025-12-01\n[/EVIDENCE]\n\nWhy did the home insurance premium increase?",
           "## Premium Renewal — HO-2025-55210\n\n| Metric | Value |\n|--------|-------|\n| **Expiring Premium** | $2,140 |\n| **Renewal Premium** | $2,380 |\n| **Increase** | +$240 (**11.2%**) |\n| **Renewal Date** | 2025-12-01 |\n\n- **Reason:** Regional **wildfire risk reassessment** has increased the risk rating for the property at 120 Maple Drive."),

        _c("[EVIDENCE]\nWorkers' Compensation — Claim WC-2025-0412\nEmployee: J. Martinez | Employer: Atlas Construction\nInjury: Lumbar disc herniation (L4-L5) from heavy lifting\nDate of injury: 2025-07-22 | Lost work days: 45\nMedical costs: $28,400 | Indemnity reserve: $62,000\n[/EVIDENCE]\n\nWhat is the workers' comp claim summary?",
           "## Workers' Compensation — WC-2025-0412\n\n| Field | Detail |\n|-------|--------|\n| **Employee** | J. Martinez |\n| **Employer** | Atlas Construction |\n| **Injury** | Lumbar disc herniation (L4-L5) |\n| **Cause** | Heavy lifting |\n| **Date** | 2025-07-22 |\n| **Lost Work Days** | 45 |\n| **Medical Costs** | $28,400 |\n| **Indemnity Reserve** | $62,000 |\n\nTotal exposure (medical + indemnity): **$90,400**."),

        _c("[EVIDENCE]\nReinsurance Treaty — Excess of Loss\nCedant: Pacific Mutual Insurance\nReinsurer: SwissRe | Layer: $5M xs $2M per occurrence\nRate on Line: 8.5% | Premium: $425,000\nTerritory: North America | Perils: All natural catastrophe\n[/EVIDENCE]\n\nWhat does the reinsurance treaty cover?",
           "## Reinsurance Treaty — Excess of Loss\n\n- **Cedant:** Pacific Mutual Insurance\n- **Reinsurer:** SwissRe\n- **Layer:** $5M excess of $2M per occurrence\n- **Rate on Line:** 8.5%\n- **Annual Premium:** $425,000\n- **Territory:** North America\n- **Perils:** All natural catastrophe\n\nSwissRe covers losses between **$2M and $7M** per occurrence for natural catastrophe events."),

        _c("[EVIDENCE]\nLoss Run Report — Insured: Apex Logistics\nPeriod: 2020-2025 | Total claims: 12\nIncurred losses: $284,000 | Earned premium: $410,000\nLoss ratio: 69.3%\nLargest claim: $78,000 (vehicle accident, 2023)\n[/EVIDENCE]\n\nWhat is the loss ratio for Apex Logistics?",
           "## Loss Run — Apex Logistics (2020-2025)\n\n| Metric | Value |\n|--------|-------|\n| **Total Claims** | 12 |\n| **Incurred Losses** | $284,000 |\n| **Earned Premium** | $410,000 |\n| **Loss Ratio** | 69.3% |\n| **Largest Claim** | $78,000 (vehicle accident, 2023) |\n\nA loss ratio of **69.3%** is above the typical profitable threshold (~60%), indicating elevated claim frequency for this account."),

        _c("[EVIDENCE]\nSubrogation Recovery — Claim CLM-2024-18820\nOriginal payout: $45,000 (rear-end collision)\nAt-fault party: Third-party driver (fully liable)\nRecovery status: $38,500 recovered from third-party insurer\nOutstanding: $6,500 — demand letter sent 2025-09-01\n[/EVIDENCE]\n\nHow much subrogation has been recovered?",
           "## Subrogation — CLM-2024-18820\n\n| Item | Amount |\n|------|--------|\n| Original Payout | $45,000 |\n| **Recovered** | **$38,500** (85.6%) |\n| Outstanding | $6,500 |\n\n- Third-party driver was **fully liable**\n- Demand letter for the remaining $6,500 was sent on 2025-09-01"),

        _c("[EVIDENCE]\nFlood Zone Assessment — Property: 88 River Road, Houston, TX\nFEMA Zone: AE (Special Flood Hazard Area)\nBase Flood Elevation: 22 ft | Building elevation: 19 ft\nMandatory flood insurance: Yes (federally backed mortgage)\nEstimated NFIP premium: $3,200/year\n[/EVIDENCE]\n\nDoes this property require flood insurance?",
           "## Flood Zone Assessment — 88 River Road, Houston, TX\n\n- **FEMA Zone:** AE (Special Flood Hazard Area)\n- **Base Flood Elevation:** 22 ft\n- **Building Elevation:** 19 ft (**3 ft below** BFE)\n- **Mandatory Flood Insurance:** **Yes** — required for federally backed mortgage\n- **Estimated NFIP Premium:** $3,200/year\n\nThe building sits **3 feet below** the base flood elevation, confirming high flood risk."),

        _c("[EVIDENCE]\nProfessional Liability — Policy PI-2025-7741\nInsured: Hartley & Associates (Accounting Firm)\nCoverage: $2M per claim / $4M aggregate\nRetroactive date: 2020-01-01 | Basis: Claims-made\nPrior acts coverage: Included\n[/EVIDENCE]\n\nWhat is the professional indemnity coverage?",
           "## Professional Indemnity — PI-2025-7741\n\n| Term | Detail |\n|------|--------|\n| **Insured** | Hartley & Associates (Accounting) |\n| **Per Claim Limit** | $2,000,000 |\n| **Aggregate Limit** | $4,000,000 |\n| **Basis** | Claims-made |\n| **Retroactive Date** | 2020-01-01 |\n| **Prior Acts** | Included |\n\nClaims-made basis means coverage applies to claims **reported during the policy period**, regardless of when the incident occurred (back to the retroactive date)."),

        _c("[EVIDENCE]\nCatastrophe Model Output — Hurricane Scenario (Cat 3)\nPortfolio: Southeast US residential book\nGross PML (250-year): $142,000,000\nNet PML (after reinsurance): $28,000,000\nExpected annual loss: $18,500,000\n[/EVIDENCE]\n\nWhat is the net exposure for a Cat 3 hurricane?",
           "## Catastrophe Model — Hurricane (Cat 3)\n\n| Metric | Value |\n|--------|-------|\n| Portfolio | Southeast US residential |\n| **Gross PML (250-year)** | $142,000,000 |\n| **Net PML (after reinsurance)** | $28,000,000 |\n| Expected Annual Loss | $18,500,000 |\n\nReinsurance reduces the 250-year probable maximum loss from **$142M to $28M** — an **80% reduction** in net exposure."),

        _c("[EVIDENCE]\nPolicy Endorsement — Endorsement #3 to Policy COM-2025-4410\nEffective: 2025-10-01\nChange: Add additional insured — Westfield Property Management\nPremium adjustment: +$420 (pro-rated)\nAll other terms remain unchanged.\n[/EVIDENCE]\n\nWhat changed in endorsement #3?",
           "## Policy Endorsement #3 — COM-2025-4410\n\n- **Effective Date:** 2025-10-01\n- **Change:** Added **Westfield Property Management** as an additional insured\n- **Premium Adjustment:** +$420 (pro-rated)\n- All other policy terms remain unchanged."),

        _c("[EVIDENCE]\nFraud Investigation — Case FRD-2025-0091\nClaimant filed for stolen vehicle (2024 BMW X5, value $58,000)\nRed flags: No police report filed, GPS data shows vehicle at claimant's relative's address\nSIU recommendation: Deny claim, refer to anti-fraud unit\n[/EVIDENCE]\n\nWhat fraud indicators were found?",
           "## Fraud Investigation — FRD-2025-0091\n\n**Claim:** Stolen vehicle — 2024 BMW X5 (claimed value $58,000)\n\n## Red Flags Identified\n\n- **No police report** filed for the alleged theft\n- **GPS tracking data** shows the vehicle at the claimant's relative's address\n\n## SIU Recommendation\n\n- **Deny the claim**\n- Refer to the anti-fraud unit for further investigation"),
    ]


def _generate_technical() -> List[Dict]:
    """Technical/IT domain examples (5)."""
    return [
        _c("[EVIDENCE]\nIncident Report — INC-2025-4418\nSeverity: P1 (Critical) | Service: Payment Gateway\nStart: 2025-10-12 14:22 UTC | Resolved: 2025-10-12 16:05 UTC\nRoot Cause: Database connection pool exhaustion (max 100 reached)\nImpact: 2,400 failed transactions | Revenue impact: ~$180,000\n[/EVIDENCE]\n\nWhat caused the payment gateway outage?",
           "## Incident — INC-2025-4418 (P1 Critical)\n\n- **Service:** Payment Gateway\n- **Duration:** 1 hour 43 minutes (14:22-16:05 UTC)\n- **Root Cause:** Database connection pool exhaustion — maximum of **100 connections** reached\n\n## Impact\n\n- **Failed transactions:** 2,400\n- **Estimated revenue impact:** ~$180,000\n\nThe connection pool limit was insufficient for peak traffic, causing cascading failures in the payment processing pipeline."),

        _c("[EVIDENCE]\nSecurity Audit — Vulnerability Assessment Q3 2025\nCritical: 2 (CVE-2025-3211 — RCE in Apache Struts, CVE-2025-1894 — SQL injection)\nHigh: 7 | Medium: 23 | Low: 41\nRemediation SLA: Critical 48hrs, High 7 days, Medium 30 days\nOverdue: 1 Critical (CVE-2025-3211, 5 days overdue)\n[/EVIDENCE]\n\nAre there any overdue critical vulnerabilities?",
           "## Security Vulnerability Summary — Q3 2025\n\n| Severity | Count | SLA |\n|----------|-------|-----|\n| **Critical** | 2 | 48 hours |\n| High | 7 | 7 days |\n| Medium | 23 | 30 days |\n| Low | 41 | Best effort |\n\n## Overdue\n\n- **CVE-2025-3211** (Remote Code Execution in Apache Struts) — **5 days overdue** on the 48-hour critical SLA\n- CVE-2025-1894 (SQL injection) — within SLA\n\nImmediate action required on the overdue RCE vulnerability."),

        _c("[EVIDENCE]\nCapacity Planning — Production Kubernetes Cluster\nCurrent: 12 nodes, 48 vCPU, 192GB RAM\nUtilisation: CPU 72% avg (peaks 91%), Memory 68% avg\nGrowth rate: +8% per quarter\nProjection: CPU saturation in Q2 2026 at current growth\n[/EVIDENCE]\n\nWhen will we run out of capacity?",
           "## Capacity Projection — Production Kubernetes\n\n| Resource | Current Utilisation | Peak |\n|----------|--------------------|----- |\n| CPU | 72% avg | 91% peak |\n| Memory | 68% avg | -- |\n\n- **Cluster:** 12 nodes, 48 vCPU, 192GB RAM\n- **Growth rate:** +8% per quarter\n- **CPU saturation projected:** **Q2 2026**\n\nAt the current growth trajectory, CPU will reach saturation first. Planning should begin now for node expansion or workload optimisation."),

        _c("[EVIDENCE]\nAPI Performance Report — October 2025\n/api/ask: p50 320ms, p95 1,240ms, p99 3,100ms\n/api/documents/upload: p50 890ms, p95 4,200ms, p99 8,500ms\nError rate: 0.3% (target <1%) | Availability: 99.95%\nSlowest endpoint: /api/finetune/by-collection (p50 12s)\n[/EVIDENCE]\n\nWhich API endpoints need optimisation?",
           "## API Performance — October 2025\n\n| Endpoint | p50 | p95 | p99 |\n|----------|-----|-----|-----|\n| /api/ask | 320ms | 1,240ms | 3,100ms |\n| /api/documents/upload | 890ms | 4,200ms | 8,500ms |\n| **/api/finetune/by-collection** | **12,000ms** | -- | -- |\n\n- **Error rate:** 0.3% (within 1% target)\n- **Availability:** 99.95%\n\n## Optimisation Priorities\n\n1. **/api/finetune/by-collection** — p50 of **12 seconds** is the slowest endpoint\n2. **/api/documents/upload** — p99 of **8.5 seconds** may cause timeouts for large files\n3. **/api/ask** — p99 of **3.1 seconds** could benefit from caching improvements"),

        _c("[EVIDENCE]\nChange Request — CR-2025-0188\nTitle: Migrate from PostgreSQL 14 to PostgreSQL 16\nRisk: Medium | Downtime: 2-hour maintenance window\nRollback plan: Snapshot restore to PG14 (tested, 15-min RTO)\nApprovals: DBA (approved), Architect (approved), Change Board (pending)\n[/EVIDENCE]\n\nWhat is the rollback plan for the PostgreSQL migration?",
           "## Change Request — CR-2025-0188\n\n- **Change:** Migrate PostgreSQL 14 to PostgreSQL 16\n- **Risk Level:** Medium\n- **Planned Downtime:** 2-hour maintenance window\n\n## Rollback Plan\n\n- **Method:** Snapshot restore to PostgreSQL 14\n- **Tested:** Yes\n- **Recovery Time Objective:** 15 minutes\n\n## Approval Status\n\n| Approver | Status |\n|----------|--------|\n| DBA | Approved |\n| Architect | Approved |\n| Change Board | **Pending** |"),
    ]


def generate_domain_expanded() -> List[Dict]:
    """Generate ~80 domain-specific examples across HR, Legal, Finance, Medical, Insurance, and Technical."""
    examples: List[Dict] = []
    examples.extend(_generate_hr())         # 15
    examples.extend(_generate_legal())       # 15
    examples.extend(_generate_finance())     # 15
    examples.extend(_generate_medical())     # 15
    examples.extend(_generate_insurance())   # 15
    examples.extend(_generate_technical())   # 5
    return examples


# ---------------------------------------------------------------------------
# Intent understanding examples (~50 total)
# ---------------------------------------------------------------------------

def _generate_ambiguous_queries() -> List[Dict]:
    """Ambiguous query handling (10)."""
    return [
        _c("Tell me about the contract.",
           "I found multiple contracts in your profile. Could you clarify which one you mean?\n\nTo help me locate the right document, you could specify:\n- The **vendor or counterparty** name\n- The **contract type** (e.g., MSA, NDA, lease)\n- The **date** or reference number\n\nAlternatively, I can list all contracts in your profile if that would help."),

        _c("What's the number?",
           "I'd like to help, but I need more context about which number you're looking for. For example:\n\n- An **invoice number** or total?\n- A **policy number** or claim reference?\n- An **employee ID** or headcount figure?\n- A **financial metric** (revenue, profit, etc.)?\n\nCould you specify which document or data point you're referring to?"),

        _c("Show me the report.",
           "There are several reports available in your profile. Could you narrow it down?\n\n- **Financial report** (P&L, balance sheet, budget variance)\n- **Audit report**\n- **Performance report**\n- **Incident report**\n\nIf you tell me the subject area or date range, I can locate the specific report for you."),

        _c("How much do we owe?",
           "To give you an accurate answer, I need to know the context:\n\n- **Accounts payable** — outstanding vendor invoices?\n- **Loan balances** — debt obligations?\n- **Intercompany** — amounts owed between entities?\n- **Tax liabilities** — upcoming tax payments?\n\nCould you specify which type of obligation you're asking about?"),

        _c("Is it approved?",
           "I'd need to know which item you're checking the approval status for:\n\n- A **change request** or project proposal?\n- A **budget** or capital expenditure request?\n- A **leave request** or HR action?\n- A **contract** or procurement order?\n\nPlease provide the reference number or description and I'll check the status."),

        _c("What changed?",
           "Could you clarify what you'd like me to compare? I can help with:\n\n- **Document versions** — differences between v1 and v2 of a specific file\n- **Policy updates** — what changed in a revised policy\n- **Financial periods** — quarter-over-quarter or year-over-year changes\n- **Contract amendments** — modifications to existing agreements\n\nPlease specify the document or topic and I'll identify the changes."),

        _c("Who signed it?",
           "I need to know which document you're asking about to identify the signatories. Could you specify:\n\n- The **document type** (contract, consent form, approval form)\n- The **reference number** or title\n- The **approximate date**\n\nOnce I know which document, I can extract the signatory details, dates, and roles."),

        _c("What's the deadline?",
           "I can find deadlines across several document types. Which are you looking for?\n\n- **Invoice payment** due dates?\n- **Contract** renewal or notice deadlines?\n- **Project milestone** dates?\n- **Compliance** or regulatory filing deadlines?\n- **Appeal** or response windows?\n\nPlease specify and I'll pull the relevant dates."),

        _c("Give me the summary.",
           "I can summarise various documents in your profile. Could you specify:\n\n- A **particular document** by name, type, or date?\n- A **topic area** — e.g., all financial documents, all HR policies?\n- The **level of detail** — brief overview or comprehensive breakdown?\n\nIf you'd like an overview of everything in your profile, I can provide a high-level document inventory instead."),

        _c("Compare them.",
           "I'd be happy to run a comparison. Could you tell me:\n\n1. **Which two (or more) documents** should I compare?\n2. **What aspects** to focus on — e.g., financial figures, terms and conditions, dates, coverage?\n\nYou can reference documents by name, type, date, or any identifying detail. I'll produce a side-by-side comparison table highlighting key differences."),
    ]


def _generate_implicit_task() -> List[Dict]:
    """Implicit task type recognition (10)."""
    return [
        _c("[EVIDENCE]\nInvoice INV-9921 from DataStream Corp\nTotal: $14,750 | Due: 2025-11-30 | Terms: Net 30\nLine items: API Gateway License $8,500, Support Plan $4,250, Setup Fee $2,000\n[/EVIDENCE]\n\nINV-9921",
           "## Invoice INV-9921 — DataStream Corp\n\n| Item | Amount |\n|------|--------|\n| API Gateway License | $8,500 |\n| Support Plan | $4,250 |\n| Setup Fee | $2,000 |\n\n- **Total:** $14,750\n- **Due Date:** 2025-11-30\n- **Terms:** Net 30\n\nIs there something specific you'd like to know about this invoice?"),

        _c("[EVIDENCE]\nEmployee: Aisha Patel | Department: Marketing\nAnnual Leave Balance: 18 days remaining\nSick Leave Taken: 3 days YTD\nNext scheduled leave: 2025-12-20 to 2025-12-31\n[/EVIDENCE]\n\nAisha Patel leave",
           "## Leave Summary — Aisha Patel (Marketing)\n\n| Leave Type | Detail |\n|------------|--------|\n| **Annual Leave Remaining** | 18 days |\n| **Sick Leave Taken (YTD)** | 3 days |\n| **Next Scheduled Leave** | 2025-12-20 to 2025-12-31 |\n\nAisha has 18 days of annual leave remaining for the current year."),

        _c("[EVIDENCE]\nQ4 Revenue: $9,200,000 | Q3 Revenue: $8,450,000\nQ4 EBITDA: $2,100,000 | Q3 EBITDA: $1,870,000\nQ4 Headcount: 192 | Q3 Headcount: 186\n[/EVIDENCE]\n\nQ3 vs Q4",
           "## Quarter-over-Quarter Comparison\n\n| Metric | Q3 | Q4 | Change |\n|--------|----|----|--------|\n| **Revenue** | $8,450,000 | $9,200,000 | +$750,000 (+8.9%) |\n| **EBITDA** | $1,870,000 | $2,100,000 | +$230,000 (+12.3%) |\n| **Headcount** | 186 | 192 | +6 (+3.2%) |\n\nQ4 shows strong growth with revenue up **8.9%** and EBITDA up **12.3%**, outpacing the headcount increase of 3.2%."),

        _c("[EVIDENCE]\nNDA between TechCorp and InnoVate Labs\nEffective: 2025-04-01 | Duration: 2 years\nGoverning law: California | Purpose: Joint product development\nKey restriction: No disclosure to subsidiaries without written consent\n[/EVIDENCE]\n\nNDA expiry",
           "## NDA Expiry — TechCorp / InnoVate Labs\n\n- **Effective Date:** 2025-04-01\n- **Duration:** 2 years\n- **Expiry Date:** **2027-04-01**\n- **Governing Law:** California\n\nThe NDA covers joint product development and prohibits disclosure to subsidiaries without prior written consent."),

        _c("[EVIDENCE]\nServer: PROD-WEB-04 | CPU: 89% | Memory: 74% | Disk: 62%\nUptime: 127 days | Last patch: 2025-09-15\nOpen alerts: 2 (high CPU, certificate expiry in 14 days)\n[/EVIDENCE]\n\nPROD-WEB-04 status",
           "## Server Status — PROD-WEB-04\n\n| Resource | Utilisation |\n|----------|-------------|\n| **CPU** | 89% (alert active) |\n| Memory | 74% |\n| Disk | 62% |\n| **Uptime** | 127 days |\n| Last Patch | 2025-09-15 |\n\n## Active Alerts (2)\n\n1. **High CPU** — 89% utilisation\n2. **Certificate expiry** — expires in 14 days\n\nBoth alerts require attention — the certificate should be renewed before expiry."),

        _c("[EVIDENCE]\nCandidate: Omar Hassan | Position: DevOps Engineer\nInterview Score: 4.2/5 | Technical Test: 87%\nStrengths: Kubernetes expertise, CI/CD pipelines\nConcerns: Limited cloud experience outside AWS\nPanel recommendation: Hire (conditional on AWS-to-Azure ramp-up plan)\n[/EVIDENCE]\n\nShould we hire Omar Hassan?",
           "## Hiring Recommendation — Omar Hassan (DevOps Engineer)\n\n| Assessment | Score |\n|------------|-------|\n| Interview | 4.2/5 |\n| Technical Test | 87% |\n\n**Strengths:** Kubernetes expertise, CI/CD pipeline experience\n\n**Concern:** Limited cloud experience outside AWS\n\n**Panel Recommendation:** **Hire** — conditional on an AWS-to-Azure ramp-up plan being included in the onboarding."),

        _c("[EVIDENCE]\nPolicy: Travel Insurance — Tier 3 (Business)\nMedical cover: $5,000,000 | Cancellation: $10,000\nBaggage: $3,000 | Personal liability: $2,000,000\nExclusions: Pre-existing conditions (unless declared), extreme sports\n[/EVIDENCE]\n\nTravel insurance limits",
           "## Travel Insurance — Tier 3 (Business)\n\n| Coverage | Limit |\n|----------|-------|\n| **Medical** | $5,000,000 |\n| **Cancellation** | $10,000 |\n| **Baggage** | $3,000 |\n| **Personal Liability** | $2,000,000 |\n\n## Exclusions\n\n- Pre-existing conditions (unless declared at purchase)\n- Extreme sports activities"),

        _c("[EVIDENCE]\nProject: CRM Migration | Status: Amber\nMilestones: Data mapping (complete), Migration testing (in progress, 2 weeks delayed)\nBudget: $340,000 spent of $400,000 (85%)\nGo-live: Pushed from 2025-11-15 to 2025-12-01\n[/EVIDENCE]\n\nCRM migration update",
           "## Project Status — CRM Migration (Amber)\n\n| Milestone | Status |\n|-----------|--------|\n| Data Mapping | Complete |\n| Migration Testing | In progress (**2 weeks delayed**) |\n| **Go-Live** | Pushed to **2025-12-01** (was 2025-11-15) |\n\n- **Budget:** $340,000 of $400,000 spent (**85%**)\n- The 2-week delay in testing has pushed the go-live date by 16 days."),

        _c("[EVIDENCE]\nLease: Unit 7, Riverside Business Park\nTenant: BlueStar Analytics | Rent: $6,200/month\nLease end: 2026-06-30 | Break clause: 2025-12-31 (3 months' notice)\nDislapidations: Schedule of condition attached\n[/EVIDENCE]\n\nRiverside lease break date",
           "## Lease Break — Unit 7, Riverside Business Park\n\n- **Tenant:** BlueStar Analytics\n- **Break Date:** **2025-12-31**\n- **Notice Required:** 3 months (must be served by **2025-09-30**)\n- **Monthly Rent:** $6,200\n- **Lease End (if break not exercised):** 2026-06-30\n\nTo exercise the break, notice must be served at least 3 months prior."),

        _c("[EVIDENCE]\nPatient: [REDACTED] | Medication: Warfarin 5mg\nLatest INR: 3.8 (target 2.0-3.0) | Date: 2025-10-18\nPrevious INR: 2.6 (2025-10-04)\nAction: Reduce dose, recheck INR in 5 days\n[/EVIDENCE]\n\nINR result",
           "## INR Monitoring\n\n| Date | INR | Target |\n|------|-----|--------|\n| 2025-10-04 | 2.6 | 2.0-3.0 |\n| **2025-10-18** | **3.8** | 2.0-3.0 |\n\n- **Status:** Above therapeutic range\n- **Medication:** Warfarin 5mg\n- **Action:** Dose reduction, recheck INR in **5 days**\n\nThe INR has risen from 2.6 to **3.8** — above the upper target of 3.0, indicating increased bleeding risk."),
    ]


def _generate_multi_intent() -> List[Dict]:
    """Multi-intent query handling (10)."""
    return [
        _c("[EVIDENCE]\nInvoice INV-3344 from Nexus Consulting\nTotal: $22,400 | Due: 2025-12-15 | Status: Unpaid\nContract: MSA-2025-110 | Term: 2 years | Auto-renewal: Yes\n[/EVIDENCE]\n\nWhat do we owe Nexus and when does the contract end?",
           "## Nexus Consulting — Payment & Contract\n\n### Outstanding Payment\n\n- **Invoice:** INV-3344\n- **Amount Due:** $22,400\n- **Due Date:** 2025-12-15\n- **Status:** Unpaid\n\n### Contract Term\n\n- **Contract:** MSA-2025-110\n- **Duration:** 2 years\n- **Auto-renewal:** Yes — the contract will automatically renew unless terminated per the notice provisions."),

        _c("[EVIDENCE]\nEmployee: Li Wei | Salary: $128,000 | Department: Data Science\nPerformance Rating: 4.5/5 | Bonus: $19,200 (15%)\nTraining completed: Advanced ML (2025-06), Leadership (2025-09)\n[/EVIDENCE]\n\nWhat is Li Wei's salary, rating, and training history?",
           "## Employee Profile — Li Wei\n\n### Compensation\n\n- **Salary:** $128,000\n- **Bonus:** $19,200 (15% of base)\n- **Department:** Data Science\n\n### Performance\n\n- **Rating:** 4.5/5\n\n### Training Completed\n\n| Course | Date |\n|--------|------|\n| Advanced ML | June 2025 |\n| Leadership | September 2025 |"),

        _c("[EVIDENCE]\nClaim CLM-2025-8810: Roof damage from storm, $18,200\nPolicy HO-2025-3321: Deductible $1,500, coverage limit $500,000\nClaim status: Approved | Net payout: $16,700\nRenewal premium: $2,650 (+8% from prior year)\n[/EVIDENCE]\n\nWhat was the claim payout and how does it affect the renewal premium?",
           "## Claim & Renewal Impact\n\n### Claim — CLM-2025-8810\n\n- **Damage:** Roof damage from storm\n- **Gross claim:** $18,200\n- **Deductible:** $1,500\n- **Net payout:** $16,700\n- **Status:** Approved\n\n### Renewal Impact\n\n- **Renewal Premium:** $2,650 (+8% from prior year)\n- The claim is likely a contributing factor to the premium increase, though market conditions may also play a role."),

        _c("[EVIDENCE]\nPatient MRN 00571: BP 158/94 mmHg, BMI 31.2\nMedications: Amlodipine 5mg, Metformin 500mg\nHbA1c: 7.4% | Cholesterol: Total 242, LDL 158\n[/EVIDENCE]\n\nSummarise the patient's vitals and list all medications.",
           "## Patient Summary — MRN 00571\n\n### Vitals\n\n| Metric | Value | Status |\n|--------|-------|--------|\n| **Blood Pressure** | 158/94 mmHg | Elevated (Stage 2 hypertension) |\n| **BMI** | 31.2 | Obese |\n| **HbA1c** | 7.4% | Above target (<7%) |\n| **Total Cholesterol** | 242 mg/dL | High |\n| **LDL** | 158 mg/dL | High |\n\n### Current Medications\n\n| Medication | Dose | Indication |\n|------------|------|------------|\n| Amlodipine | 5mg | Hypertension |\n| Metformin | 500mg | Diabetes |"),

        _c("[EVIDENCE]\nBudget FY2026: Marketing $1,400,000, Engineering $3,200,000, Sales $1,800,000\nFY2025 Actual: Marketing $1,250,000, Engineering $2,900,000, Sales $1,650,000\n[/EVIDENCE]\n\nCompare budgets year-over-year and identify the biggest increase.",
           "## Budget Comparison — FY2025 vs FY2026\n\n| Department | FY2025 Actual | FY2026 Budget | Change | % Change |\n|------------|--------------|---------------|--------|----------|\n| Marketing | $1,250,000 | $1,400,000 | +$150,000 | +12.0% |\n| **Engineering** | $2,900,000 | $3,200,000 | **+$300,000** | **+10.3%** |\n| Sales | $1,650,000 | $1,800,000 | +$150,000 | +9.1% |\n\n**Biggest absolute increase:** Engineering at **+$300,000**\n**Biggest percentage increase:** Marketing at **+12.0%**"),

        _c("[EVIDENCE]\nLease: 200 King Street | Rent: $12,000/month | Expiry: 2026-09-30\nMaintenance issues: HVAC unit failing (reported 2025-10-01), parking lot resurfacing needed\nLandlord response: HVAC repair scheduled 2025-10-20\n[/EVIDENCE]\n\nWhen does the lease expire and what maintenance is outstanding?",
           "## 200 King Street — Lease & Maintenance\n\n### Lease Details\n\n- **Monthly Rent:** $12,000\n- **Expiry:** 2026-09-30\n\n### Outstanding Maintenance\n\n| Issue | Status |\n|-------|--------|\n| **HVAC unit failing** | Repair scheduled 2025-10-20 |\n| **Parking lot resurfacing** | No response from landlord yet |\n\nThe HVAC issue is being addressed; the parking lot resurfacing still needs landlord action."),

        _c("[EVIDENCE]\nVendor A: $45/unit, lead time 14 days, MOQ 500, quality rating 4.2/5\nVendor B: $42/unit, lead time 21 days, MOQ 1000, quality rating 3.8/5\nVendor C: $48/unit, lead time 7 days, MOQ 100, quality rating 4.6/5\n[/EVIDENCE]\n\nWhich vendor is cheapest and which has the best quality?",
           "## Vendor Comparison\n\n| Vendor | Price | Lead Time | MOQ | Quality |\n|--------|-------|-----------|-----|---------|\n| Vendor A | $45/unit | 14 days | 500 | 4.2/5 |\n| **Vendor B** | **$42/unit** | 21 days | 1,000 | 3.8/5 |\n| **Vendor C** | $48/unit | 7 days | 100 | **4.6/5** |\n\n- **Cheapest:** Vendor B at **$42/unit** (but longest lead time and highest MOQ)\n- **Best quality:** Vendor C at **4.6/5** (but highest price)\n\nThe choice depends on whether cost or quality is the priority."),

        _c("[EVIDENCE]\nPolicy: Professional Indemnity PI-2025-881\nCover: $3M per claim / $6M aggregate\nExcess: $25,000 | Premium: $18,400/year\nClaims history: 0 claims in 3 years\n[/EVIDENCE]\n\nWhat's the coverage limit and claims history?",
           "## Professional Indemnity — PI-2025-881\n\n### Coverage\n\n- **Per Claim:** $3,000,000\n- **Aggregate:** $6,000,000\n- **Excess:** $25,000\n- **Premium:** $18,400/year\n\n### Claims History\n\n- **Claims in past 3 years:** 0\n- Clean claims record."),

        _c("[EVIDENCE]\nProject Alpha: 78% complete, on budget ($240K of $310K spent)\nProject Beta: 45% complete, over budget ($180K of $150K, +$30K overrun)\nProject Gamma: 92% complete, under budget ($88K of $120K spent)\n[/EVIDENCE]\n\nWhich projects are on track and which are over budget?",
           "## Project Portfolio Status\n\n| Project | Progress | Budget | Spent | Status |\n|---------|----------|--------|-------|--------|\n| Alpha | 78% | $310K | $240K | **On budget** |\n| **Beta** | 45% | $150K | $180K | **Over budget (+$30K)** |\n| Gamma | 92% | $120K | $88K | **Under budget** |\n\n- **On track:** Alpha and Gamma\n- **Over budget:** Beta — already **$30K over** at only 45% completion, suggesting the overrun will grow"),

        _c("[EVIDENCE]\nContract: SaaS Agreement SA-2025-440\nAnnual fee: $96,000 | Payment: Quarterly ($24,000)\nSLA: 99.9% uptime | Penalty: 5% credit per 0.1% shortfall\nData residency: UK only | GDPR DPA: Attached as Schedule 2\n[/EVIDENCE]\n\nWhat's the cost, SLA, and data residency requirement?",
           "## SaaS Agreement — SA-2025-440\n\n### Cost\n\n- **Annual Fee:** $96,000\n- **Payment:** Quarterly instalments of $24,000\n\n### SLA\n\n- **Uptime Target:** 99.9%\n- **Penalty:** 5% credit per 0.1% below target\n\n### Data Residency\n\n- **Requirement:** UK only\n- **GDPR DPA:** Included as Schedule 2"),
    ]


def _generate_context_dependent() -> List[Dict]:
    """Context-dependent query resolution (10)."""
    return [
        _c("[EVIDENCE]\nMeeting Minutes — Board Meeting 2025-10-15\nResolution: Approved $2M investment in AI capabilities\nAction: CTO to present implementation roadmap by 2025-11-30\nAttendees: CEO, CFO, CTO, COO, Non-Exec Directors (3)\n[/EVIDENCE]\n\nWhat did the board decide about AI?",
           "## Board Decision — AI Investment (2025-10-15)\n\n- **Resolution:** Approved **$2M investment** in AI capabilities\n- **Action item:** CTO to present implementation roadmap by **2025-11-30**\n- **Attendees:** CEO, CFO, CTO, COO, and 3 Non-Executive Directors\n\nThe board unanimously approved the investment at the October meeting."),

        _c("[EVIDENCE]\nSupplier Agreement — Effective 2024-01-01\nPrice review clause: Annual review on anniversary date\nCurrent rate: $85/hour (set Jan 2025, +5% from $80.95)\nCap: Annual increases not to exceed CPI + 2%\n[/EVIDENCE]\n\nWhen is the next price review?",
           "## Next Price Review\n\n- **Review date:** **2026-01-01** (annual anniversary)\n- **Current rate:** $85/hour (set January 2025)\n- **Increase cap:** CPI + 2% maximum\n- **Previous increase:** +5% (from $80.95 to $85.00)\n\nThe next review falls on the contract anniversary in January 2026."),

        _c("[EVIDENCE]\nAudit Report — ISO 27001 Surveillance Audit\nDate: 2025-09-20 | Auditor: BSI\nNon-conformities: 1 minor (access review frequency)\nObservations: 3 (password policy, backup testing, vendor risk)\nRecommendation: Maintain certification\n[/EVIDENCE]\n\nDid we pass the audit?",
           "## ISO 27001 Surveillance Audit — 2025-09-20\n\n**Result: Certification maintained** (passed)\n\n- **Non-conformities:** 1 minor — access review frequency\n- **Observations:** 3\n  - Password policy\n  - Backup testing\n  - Vendor risk management\n- **Auditor:** BSI\n\nThe minor non-conformity must be addressed, but it did not prevent certification maintenance."),

        _c("[EVIDENCE]\nProduct Roadmap — Q1 2026\nFeature A: Multi-language support (Priority: High, 60% complete)\nFeature B: Advanced analytics dashboard (Priority: Medium, 20% complete)\nFeature C: Mobile app v2 (Priority: High, design phase)\nBlocked: Feature B depends on data warehouse migration (Q4 2025)\n[/EVIDENCE]\n\nWhat's blocking the analytics dashboard?",
           "## Blocker — Advanced Analytics Dashboard (Feature B)\n\n- **Dependency:** Data warehouse migration (scheduled Q4 2025)\n- **Current progress:** 20% complete\n- **Priority:** Medium\n\nFeature B cannot proceed beyond its current state until the **data warehouse migration** is completed. The migration is planned for Q4 2025, which should unblock dashboard development for Q1 2026."),

        _c("[EVIDENCE]\nRegulatory Filing — FCA Annual Return\nDue: 2025-12-31 | Status: Draft (pending CFO review)\nKey data: Complaints received 142, resolved 138, avg resolution 18 days\nPenalty for late filing: up to $50,000\n[/EVIDENCE]\n\nAre we going to file on time?",
           "## FCA Annual Return — Filing Status\n\n- **Deadline:** 2025-12-31\n- **Current Status:** Draft — pending **CFO review**\n- **Late Filing Penalty:** Up to $50,000\n\n## Key Metrics for Filing\n\n| Metric | Value |\n|--------|-------|\n| Complaints received | 142 |\n| Complaints resolved | 138 |\n| Avg resolution time | 18 days |\n\nThe filing is in draft stage. CFO review is the remaining step before submission. Given the penalty risk, I recommend scheduling the review promptly."),

        _c("[EVIDENCE]\nCustomer Satisfaction Survey — October 2025\nOverall NPS: +42 (up from +35 in September)\nTop praise: Response time, product quality\nTop complaints: Pricing transparency, onboarding complexity\nResponse rate: 28% (1,120 of 4,000 customers)\n[/EVIDENCE]\n\nAre customers happy?",
           "## Customer Satisfaction — October 2025\n\n- **NPS Score:** +42 (up from +35 in September — **improving trend**)\n- **Response Rate:** 28% (1,120 responses)\n\n## Highlights\n\n| Category | Feedback |\n|----------|----------|\n| **Top praise** | Response time, product quality |\n| **Top complaints** | Pricing transparency, onboarding complexity |\n\nOverall sentiment is **positive** with an improving NPS. The two complaint areas (pricing clarity and onboarding) are actionable."),

        _c("[EVIDENCE]\nIncident Timeline — INC-2025-5520\n14:00 UTC: Alert triggered (API error rate >5%)\n14:12 UTC: On-call engineer acknowledged\n14:30 UTC: Root cause identified (misconfigured load balancer rule)\n14:45 UTC: Fix deployed | 14:50 UTC: Error rate normalised\n[/EVIDENCE]\n\nHow quickly did we respond to the incident?",
           "## Incident Response Times — INC-2025-5520\n\n| Milestone | Time | Elapsed |\n|-----------|------|---------|\n| Alert triggered | 14:00 UTC | -- |\n| Engineer acknowledged | 14:12 UTC | **12 minutes** |\n| Root cause identified | 14:30 UTC | **30 minutes** |\n| Fix deployed | 14:45 UTC | **45 minutes** |\n| Normalised | 14:50 UTC | **50 minutes** |\n\n- **Time to acknowledge:** 12 minutes\n- **Time to resolve:** 50 minutes\n- **Root cause:** Misconfigured load balancer rule"),

        _c("[EVIDENCE]\nStaff Turnover Report — FY2025\nVoluntary turnover: 14.2% | Involuntary: 3.1%\nIndustry benchmark: 12.5% voluntary\nHighest turnover dept: Sales (22%) | Lowest: Engineering (8%)\nAvg cost per hire: $12,400\n[/EVIDENCE]\n\nIs our turnover a problem?",
           "## Staff Turnover Analysis — FY2025\n\n| Metric | Value | Benchmark |\n|--------|-------|----------|\n| **Voluntary turnover** | 14.2% | 12.5% (industry) |\n| Involuntary turnover | 3.1% | -- |\n\n- Voluntary turnover is **1.7 percentage points above** the industry benchmark\n- **Sales** has the highest turnover at **22%** — nearly 3x the Engineering rate (8%)\n- **Cost impact:** At $12,400 per hire, elevated turnover has a material cost\n\nYes — turnover is above benchmark, particularly in Sales. Targeted retention initiatives for Sales may be warranted."),

        _c("[EVIDENCE]\nProcurement Evaluation — RFP-2025-044\nVendor X: Score 82/100, price $340,000\nVendor Y: Score 91/100, price $410,000\nVendor Z: Score 75/100, price $295,000\nEvaluation criteria: Technical (40%), Price (30%), Experience (20%), Support (10%)\n[/EVIDENCE]\n\nWho should we go with?",
           "## RFP-2025-044 — Vendor Evaluation\n\n| Vendor | Score | Price |\n|--------|-------|-------|\n| Vendor X | 82/100 | $340,000 |\n| **Vendor Y** | **91/100** | $410,000 |\n| Vendor Z | 75/100 | $295,000 |\n\n**Vendor Y** scored highest at **91/100**, though it is the most expensive at $410,000.\n\nThe decision depends on priorities:\n- **Best overall quality:** Vendor Y (91/100)\n- **Best value:** Vendor X (82/100 at $340K offers strong technical capability at a moderate price)\n- **Lowest cost:** Vendor Z ($295K but lowest score)\n\nBased on the evaluation weighting (Technical 40%, Price 30%), Vendor Y's technical strength may justify the premium."),

        _c("[EVIDENCE]\nCompliance Checklist — GDPR Readiness\nData mapping: Complete | Privacy notices: Updated\nDPIA register: 12 assessments completed, 2 pending\nBreach notification process: Documented and tested\nDPO appointed: Yes (Sarah Mitchell)\n[/EVIDENCE]\n\nAre we GDPR compliant?",
           "## GDPR Readiness Assessment\n\n| Requirement | Status |\n|------------|--------|\n| Data mapping | Complete |\n| Privacy notices | Updated |\n| DPIA register | 12 complete, **2 pending** |\n| Breach notification | Documented and tested |\n| DPO appointed | Yes (Sarah Mitchell) |\n\nThe organisation is **largely compliant** with one gap: **2 DPIAs remain pending**. These should be completed to achieve full compliance. All other core GDPR requirements are in place."),
    ]


def _generate_format_requests() -> List[Dict]:
    """Format-specific request handling (10)."""
    return [
        _c("[EVIDENCE]\nQ1 Revenue: $7,800,000 | Q2: $8,100,000 | Q3: $8,450,000 | Q4: $9,200,000\nQ1 Expenses: $6,200,000 | Q2: $6,400,000 | Q3: $6,580,000 | Q4: $7,100,000\n[/EVIDENCE]\n\nPut the quarterly financials in a table.",
           "## Quarterly Financials — FY2025\n\n| Quarter | Revenue | Expenses | Net |\n|---------|---------|----------|-----|\n| Q1 | $7,800,000 | $6,200,000 | $1,600,000 |\n| Q2 | $8,100,000 | $6,400,000 | $1,700,000 |\n| Q3 | $8,450,000 | $6,580,000 | $1,870,000 |\n| Q4 | $9,200,000 | $7,100,000 | $2,100,000 |\n| **Total** | **$33,550,000** | **$26,280,000** | **$7,270,000** |"),

        _c("[EVIDENCE]\nProject milestones:\nPhase 1 Requirements: 2025-07-01 to 2025-08-15 (complete)\nPhase 2 Development: 2025-08-16 to 2025-11-30 (in progress, 65%)\nPhase 3 Testing: 2025-12-01 to 2026-01-31 (not started)\nPhase 4 Deployment: 2026-02-01 to 2026-02-28 (not started)\n[/EVIDENCE]\n\nGive me a timeline view of the project.",
           "## Project Timeline\n\n| Phase | Start | End | Status |\n|-------|-------|-----|--------|\n| **Phase 1** Requirements | 2025-07-01 | 2025-08-15 | Complete |\n| **Phase 2** Development | 2025-08-16 | 2025-11-30 | In progress (65%) |\n| **Phase 3** Testing | 2025-12-01 | 2026-01-31 | Not started |\n| **Phase 4** Deployment | 2026-02-01 | 2026-02-28 | Not started |\n\n**Total duration:** 8 months (July 2025 — February 2026)\n**Current position:** Mid-development phase, on track for December testing start."),

        _c("[EVIDENCE]\nTeam: Platform Engineering\nMembers: Alice (Lead), Bob (Senior), Carol (Mid), Dave (Mid), Eve (Junior)\nAlice: Kubernetes, Terraform, Go | Bob: Python, AWS, Docker\nCarol: React, TypeScript, Node.js | Dave: Java, Spring, PostgreSQL\nEve: Python, testing, documentation\n[/EVIDENCE]\n\nList the team's skills as bullet points.",
           "## Platform Engineering — Skills Inventory\n\n**Alice (Lead)**\n- Kubernetes\n- Terraform\n- Go\n\n**Bob (Senior)**\n- Python\n- AWS\n- Docker\n\n**Carol (Mid)**\n- React\n- TypeScript\n- Node.js\n\n**Dave (Mid)**\n- Java\n- Spring\n- PostgreSQL\n\n**Eve (Junior)**\n- Python\n- Testing\n- Documentation"),

        _c("[EVIDENCE]\nRisk Register — Top 5 Risks\n1. Data breach — Likelihood: High, Impact: Critical, Mitigation: Encryption + monitoring\n2. Key person dependency — Likelihood: Medium, Impact: High, Mitigation: Cross-training\n3. Vendor lock-in — Likelihood: Medium, Impact: Medium, Mitigation: Multi-cloud strategy\n4. Regulatory change — Likelihood: Low, Impact: High, Mitigation: Legal monitoring\n5. Budget overrun — Likelihood: High, Impact: Medium, Mitigation: Monthly reviews\n[/EVIDENCE]\n\nFormat the risk register as a proper table with RAG status.",
           "## Risk Register — Top 5\n\n| # | Risk | Likelihood | Impact | RAG | Mitigation |\n|---|------|-----------|--------|-----|------------|\n| 1 | Data breach | High | Critical | **Red** | Encryption + monitoring |\n| 2 | Key person dependency | Medium | High | **Amber** | Cross-training programme |\n| 3 | Vendor lock-in | Medium | Medium | **Amber** | Multi-cloud strategy |\n| 4 | Regulatory change | Low | High | **Amber** | Legal monitoring |\n| 5 | Budget overrun | High | Medium | **Amber** | Monthly budget reviews |\n\nRisk #1 (data breach) is the only **Red** rated risk due to its high likelihood and critical impact."),

        _c("[EVIDENCE]\nSales by Region — October 2025\nNorth America: $4,200,000 (45%)\nEurope: $2,800,000 (30%)\nAsia Pacific: $1,500,000 (16%)\nRest of World: $840,000 (9%)\nTotal: $9,340,000\n[/EVIDENCE]\n\nBreak down sales by region with percentages.",
           "## Sales by Region — October 2025\n\n| Region | Revenue | Share |\n|--------|---------|-------|\n| **North America** | $4,200,000 | 45% |\n| **Europe** | $2,800,000 | 30% |\n| **Asia Pacific** | $1,500,000 | 16% |\n| **Rest of World** | $840,000 | 9% |\n| **Total** | **$9,340,000** | 100% |\n\nNorth America and Europe together account for **75%** of total sales revenue."),

        _c("[EVIDENCE]\nAction items from project meeting 2025-10-16:\n- Tom: Complete API documentation by Oct 25\n- Sarah: Review security scan results by Oct 20\n- Dev team: Fix 3 critical bugs (JIRA-441, JIRA-442, JIRA-445) by Oct 23\n- PM: Update stakeholders on revised timeline by Oct 18\n[/EVIDENCE]\n\nGive me the action items as a checklist.",
           "## Action Items — Meeting 2025-10-16\n\n- [ ] **Tom** — Complete API documentation (due **Oct 25**)\n- [ ] **Sarah** — Review security scan results (due **Oct 20**)\n- [ ] **Dev Team** — Fix critical bugs JIRA-441, JIRA-442, JIRA-445 (due **Oct 23**)\n- [ ] **PM** — Update stakeholders on revised timeline (due **Oct 18**)\n\n**Nearest deadline:** PM stakeholder update on October 18."),

        _c("[EVIDENCE]\nSoftware Licenses — Active\nMicrosoft 365 E3: 200 seats, $22/user/month, renews 2026-03-01\nJira Cloud Premium: 85 seats, $14/user/month, renews 2026-01-15\nSlack Business+: 200 seats, $12.50/user/month, renews 2025-12-01\nZoom Business: 50 seats, $18.33/user/month, renews 2026-06-01\n[/EVIDENCE]\n\nSummarise licences in a table sorted by renewal date.",
           "## Software Licences — Sorted by Renewal Date\n\n| Software | Seats | Cost/User/Mo | Annual Cost | Renewal |\n|----------|-------|-------------|-------------|---------|\n| Slack Business+ | 200 | $12.50 | $30,000 | **2025-12-01** |\n| Jira Cloud Premium | 85 | $14.00 | $14,280 | 2026-01-15 |\n| Microsoft 365 E3 | 200 | $22.00 | $52,800 | 2026-03-01 |\n| Zoom Business | 50 | $18.33 | $10,998 | 2026-06-01 |\n\n**Total annual licence cost:** ~$108,078\n**Next renewal:** Slack Business+ on **2025-12-01**."),

        _c("[EVIDENCE]\nInterview Scores — Senior Developer Role\nCandidate A: Technical 4/5, Communication 3/5, Culture 4/5, Total 11/15\nCandidate B: Technical 5/5, Communication 4/5, Culture 3/5, Total 12/15\nCandidate C: Technical 3/5, Communication 5/5, Culture 5/5, Total 13/15\n[/EVIDENCE]\n\nRank the candidates from highest to lowest score.",
           "## Candidate Ranking — Senior Developer\n\n| Rank | Candidate | Technical | Communication | Culture | **Total** |\n|------|-----------|-----------|---------------|---------|-----------|\n| 1 | **Candidate C** | 3/5 | 5/5 | 5/5 | **13/15** |\n| 2 | Candidate B | 5/5 | 4/5 | 3/5 | 12/15 |\n| 3 | Candidate A | 4/5 | 3/5 | 4/5 | 11/15 |\n\n**Candidate C** leads on total score, though **Candidate B** has the strongest technical rating (5/5). The choice may depend on whether technical depth or team fit is prioritised."),

        _c("[EVIDENCE]\nMonthly KPIs — October 2025\nRevenue: $9.2M (target $8.5M) — exceeded\nChurn rate: 4.1% (target <3%) — missed\nNPS: +42 (target +40) — exceeded\nDeployment frequency: 12/month (target 8) — exceeded\nIncident MTTR: 52 min (target <45 min) — missed\n[/EVIDENCE]\n\nShow KPIs with pass/fail status.",
           "## Monthly KPIs — October 2025\n\n| KPI | Actual | Target | Status |\n|-----|--------|--------|--------|\n| Revenue | $9.2M | $8.5M | **Pass** |\n| Churn Rate | 4.1% | <3% | **Fail** |\n| NPS | +42 | +40 | **Pass** |\n| Deployment Frequency | 12/month | 8/month | **Pass** |\n| Incident MTTR | 52 min | <45 min | **Fail** |\n\n**Result: 3 of 5 KPIs met.** The two misses are **churn rate** (1.1% above target) and **incident MTTR** (7 minutes over target)."),

        _c("[EVIDENCE]\nExpense Claims — September 2025\nJohn: Travel $2,400, Meals $380, Equipment $0 — Total $2,780\nMaria: Travel $1,100, Meals $220, Equipment $450 — Total $1,770\nAlex: Travel $0, Meals $150, Equipment $1,200 — Total $1,350\n[/EVIDENCE]\n\nCreate a summary with totals per person and per category.",
           "## Expense Claims — September 2025\n\n### By Person\n\n| Employee | Travel | Meals | Equipment | **Total** |\n|----------|--------|-------|-----------|----------|\n| John | $2,400 | $380 | $0 | **$2,780** |\n| Maria | $1,100 | $220 | $450 | **$1,770** |\n| Alex | $0 | $150 | $1,200 | **$1,350** |\n\n### By Category\n\n| Category | Total |\n|----------|-------|\n| Travel | $3,500 |\n| Meals | $750 |\n| Equipment | $1,650 |\n| **Grand Total** | **$5,900** |"),
    ]


def generate_intent_expanded() -> List[Dict]:
    """Generate ~50 intent understanding examples across ambiguous, implicit, multi-intent, context-dependent, and format requests."""
    examples: List[Dict] = []
    examples.extend(_generate_ambiguous_queries())    # 10
    examples.extend(_generate_implicit_task())         # 10
    examples.extend(_generate_multi_intent())          # 10
    examples.extend(_generate_context_dependent())     # 10
    examples.extend(_generate_format_requests())       # 10
    return examples
