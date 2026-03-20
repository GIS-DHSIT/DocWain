"""Visualization SFT training data generator for DocWain.

Generates synthetic chat-format training examples that teach the model
to emit <!--DOCWAIN_VIZ {JSON} --> directives when the user's question
contains chartable data, use flow-analysis arrows for procedural content,
and respond with plain text for simple factual questions.

Three output modes:
1. Text + VIZ directive  (chartable data in evidence)
2. Text + flow arrows    (procedural / step-by-step content)
3. Text-only             (simple factual answers)
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

SYSTEM_PROMPT = (
    "You are DocWain -- Document Wise AI Node -- an intelligent document intelligence "
    "platform engineered by DHS IT Solutions Ltd. You analyze enterprise documents and "
    "provide grounded, evidence-backed answers. When the user's question involves "
    "chartable numeric data, you include a <!--DOCWAIN_VIZ {JSON} --> directive so "
    "the visualization pipeline can render an appropriate chart. You choose the best "
    "chart type based on the data shape and question intent."
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _viz(
    chart_type: str,
    title: str,
    labels: List[str],
    values: List[float],
    unit: Optional[str] = None,
    secondary_values: Optional[List[float]] = None,
    secondary_name: Optional[str] = None,
) -> str:
    """Build a <!--DOCWAIN_VIZ ... --> directive string."""
    payload: Dict[str, Any] = {
        "chart_type": chart_type,
        "title": title,
        "labels": labels,
        "values": values,
    }
    if unit is not None:
        payload["unit"] = unit
    if secondary_values is not None:
        payload["secondary_values"] = secondary_values
    if secondary_name is not None:
        payload["secondary_name"] = secondary_name
    return f"<!--DOCWAIN_VIZ\n{json.dumps(payload, indent=2)}\n-->"


def _msg(user: str, assistant: str) -> Dict[str, Any]:
    """Build a chat-format training example with system/user/assistant."""
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]
    }


# ---------------------------------------------------------------------------
# Chart examples
# ---------------------------------------------------------------------------

def _bar_examples() -> List[Dict]:
    return [
        _msg(
            "Show revenue by quarter.\n\n[EVIDENCE]\nQ1: $1.2M, Q2: $1.5M, Q3: $1.8M, Q4: $2.1M.",
            "## Quarterly Revenue\n\n| Quarter | Revenue |\n|---------|--------|\n| Q1 | **$1.2M** |\n| Q2 | **$1.5M** |\n| Q3 | **$1.8M** |\n| Q4 | **$2.1M** |\n\nRevenue grew steadily each quarter, reaching **$2.1M** in Q4.\n\n"
            + _viz("bar", "Quarterly Revenue", ["Q1", "Q2", "Q3", "Q4"], [1.2, 1.5, 1.8, 2.1], unit="$M"),
        ),
        _msg(
            "What are the invoice totals per vendor?\n\n[EVIDENCE]\nAcme Corp: $45,000. GlobalTech: $32,000. NovaSys: $28,000. PrimeLine: $51,000.",
            "## Invoice Totals by Vendor\n\n| Vendor | Total |\n|--------|------|\n| PrimeLine | **$51,000** |\n| Acme Corp | **$45,000** |\n| GlobalTech | **$32,000** |\n| NovaSys | **$28,000** |\n\n**PrimeLine** has the highest invoice total at **$51,000**.\n\n"
            + _viz("bar", "Invoice Totals by Vendor", ["PrimeLine", "Acme Corp", "GlobalTech", "NovaSys"], [51000, 45000, 32000, 28000], unit="$"),
        ),
        _msg(
            "How many support tickets were raised each month?\n\n[EVIDENCE]\nJan: 120, Feb: 95, Mar: 140, Apr: 110, May: 85.",
            "## Monthly Support Tickets\n\n| Month | Tickets |\n|-------|--------|\n| Mar | **140** |\n| Jan | **120** |\n| Apr | **110** |\n| Feb | **95** |\n| May | **85** |\n\nMarch had the highest ticket volume at **140**.\n\n"
            + _viz("bar", "Monthly Support Tickets", ["Jan", "Feb", "Mar", "Apr", "May"], [120, 95, 140, 110, 85]),
        ),
        _msg(
            "Show training hours per team.\n\n[EVIDENCE]\nAlpha team: 240 hrs, Bravo: 180 hrs, Charlie: 310 hrs, Delta: 160 hrs.",
            "## Training Hours by Team\n\n| Team | Hours |\n|------|------|\n| Charlie | **310** |\n| Alpha | **240** |\n| Bravo | **180** |\n| Delta | **160** |\n\nTeam Charlie invested the most in training at **310 hours**.\n\n"
            + _viz("bar", "Training Hours by Team", ["Alpha", "Bravo", "Charlie", "Delta"], [240, 180, 310, 160], unit="hrs"),
        ),
    ]


def _horizontal_bar_examples() -> List[Dict]:
    return [
        _msg(
            "Show department headcount.\n\n[EVIDENCE]\nEngineering: 85, Sales: 42, Marketing: 28, HR: 15, Finance: 22, Legal: 10.",
            "## Department Headcount\n\n| Department | Headcount |\n|------------|----------|\n| Engineering | **85** |\n| Sales | **42** |\n| Marketing | **28** |\n| Finance | **22** |\n| HR | **15** |\n| Legal | **10** |\n\nEngineering is the largest department with **85** employees.\n\n"
            + _viz("horizontal_bar", "Department Headcount", ["Engineering", "Sales", "Marketing", "Finance", "HR", "Legal"], [85, 42, 28, 22, 15, 10]),
        ),
        _msg(
            "Compare document page counts.\n\n[EVIDENCE]\nAnnual Report: 124 pages. Employee Handbook: 78 pages. Policy Manual: 56 pages. Onboarding Guide: 34 pages.",
            "## Document Page Counts\n\n| Document | Pages |\n|----------|------|\n| Annual Report | **124** |\n| Employee Handbook | **78** |\n| Policy Manual | **56** |\n| Onboarding Guide | **34** |\n\n"
            + _viz("horizontal_bar", "Document Page Counts", ["Annual Report", "Employee Handbook", "Policy Manual", "Onboarding Guide"], [124, 78, 56, 34], unit="pages"),
        ),
    ]


def _grouped_bar_examples() -> List[Dict]:
    return [
        _msg(
            "Compare Q1 and Q2 expenses by category.\n\n[EVIDENCE]\nSalaries Q1: $800K, Q2: $820K. Cloud Q1: $120K, Q2: $150K. Travel Q1: $45K, Q2: $60K. Marketing Q1: $95K, Q2: $110K.",
            "## Q1 vs Q2 Expenses\n\n| Category | Q1 | Q2 |\n|----------|-----|----|\n| Salaries | **$800K** | **$820K** |\n| Cloud | **$120K** | **$150K** |\n| Marketing | **$95K** | **$110K** |\n| Travel | **$45K** | **$60K** |\n\nAll categories increased in Q2, with cloud costs rising **25%**.\n\n"
            + _viz("grouped_bar", "Q1 vs Q2 Expenses", ["Salaries", "Cloud", "Marketing", "Travel"], [800, 120, 95, 45], unit="$K", secondary_values=[820, 150, 110, 60], secondary_name="Q2"),
        ),
        _msg(
            "Show planned vs actual budget.\n\n[EVIDENCE]\nIT planned $500K actual $480K. HR planned $200K actual $210K. Ops planned $350K actual $360K.",
            "## Planned vs Actual Budget\n\n| Dept | Planned | Actual |\n|------|---------|--------|\n| IT | **$500K** | **$480K** |\n| Ops | **$350K** | **$360K** |\n| HR | **$200K** | **$210K** |\n\nIT came in under budget while HR and Ops slightly exceeded targets.\n\n"
            + _viz("grouped_bar", "Planned vs Actual Budget", ["IT", "HR", "Ops"], [500, 200, 350], unit="$K", secondary_values=[480, 210, 360], secondary_name="Actual"),
        ),
    ]


def _stacked_bar_examples() -> List[Dict]:
    return [
        _msg(
            "Break down project costs by phase.\n\n[EVIDENCE]\nProject Alpha: Design $50K, Development $120K, Testing $30K. Project Beta: Design $40K, Development $90K, Testing $25K.",
            "## Project Cost Breakdown\n\n| Project | Design | Development | Testing | Total |\n|---------|--------|-------------|---------|-------|\n| Alpha | $50K | $120K | $30K | **$200K** |\n| Beta | $40K | $90K | $25K | **$155K** |\n\n"
            + _viz("stacked_bar", "Project Cost Breakdown by Phase", ["Alpha", "Beta"], [50, 40], unit="$K", secondary_values=[120, 90], secondary_name="Development"),
        ),
        _msg(
            "Show hiring sources by department.\n\n[EVIDENCE]\nEng referral 20 agency 10 direct 15. Sales referral 8 agency 12 direct 5.",
            "## Hiring Sources\n\n| Dept | Referral | Agency | Direct | Total |\n|------|----------|--------|--------|-------|\n| Engineering | 20 | 10 | 15 | **45** |\n| Sales | 8 | 12 | 5 | **25** |\n\n"
            + _viz("stacked_bar", "Hiring Sources by Department", ["Engineering", "Sales"], [20, 8], secondary_values=[10, 12], secondary_name="Agency"),
        ),
    ]


def _donut_examples() -> List[Dict]:
    return [
        _msg(
            "What is the expense breakdown?\n\n[EVIDENCE]\nSalaries: 55%, Cloud Infrastructure: 20%, Marketing: 12%, Office: 8%, Travel: 5%.",
            "## Expense Breakdown\n\n| Category | Share |\n|----------|------|\n| Salaries | **55%** |\n| Cloud Infrastructure | **20%** |\n| Marketing | **12%** |\n| Office | **8%** |\n| Travel | **5%** |\n\n**Salaries** account for more than half of total expenses.\n\n"
            + _viz("donut", "Expense Breakdown", ["Salaries", "Cloud Infrastructure", "Marketing", "Office", "Travel"], [55, 20, 12, 8, 5], unit="%"),
        ),
        _msg(
            "Show document types distribution.\n\n[EVIDENCE]\nPDFs: 340, Word Docs: 180, Spreadsheets: 95, Images: 45, Other: 20.",
            "## Document Type Distribution\n\n| Type | Count |\n|------|------|\n| PDFs | **340** |\n| Word Docs | **180** |\n| Spreadsheets | **95** |\n| Images | **45** |\n| Other | **20** |\n\n"
            + _viz("donut", "Document Type Distribution", ["PDFs", "Word Docs", "Spreadsheets", "Images", "Other"], [340, 180, 95, 45, 20]),
        ),
        _msg(
            "What share of revenue comes from each region?\n\n[EVIDENCE]\nNorth America: 48%, Europe: 28%, Asia-Pacific: 16%, Latin America: 8%.",
            "## Revenue by Region\n\n| Region | Share |\n|--------|------|\n| North America | **48%** |\n| Europe | **28%** |\n| Asia-Pacific | **16%** |\n| Latin America | **8%** |\n\nNorth America contributes nearly half of total revenue.\n\n"
            + _viz("donut", "Revenue by Region", ["North America", "Europe", "Asia-Pacific", "Latin America"], [48, 28, 16, 8], unit="%"),
        ),
        _msg(
            "Show the customer satisfaction survey results.\n\n[EVIDENCE]\nVery satisfied: 42%, Satisfied: 31%, Neutral: 15%, Dissatisfied: 8%, Very dissatisfied: 4%.",
            "## Customer Satisfaction Distribution\n\n| Rating | Share |\n|--------|------|\n| Very Satisfied | **42%** |\n| Satisfied | **31%** |\n| Neutral | **15%** |\n| Dissatisfied | **8%** |\n| Very Dissatisfied | **4%** |\n\n**73%** of customers are satisfied or very satisfied.\n\n"
            + _viz("donut", "Customer Satisfaction", ["Very Satisfied", "Satisfied", "Neutral", "Dissatisfied", "Very Dissatisfied"], [42, 31, 15, 8, 4], unit="%"),
        ),
    ]


def _line_examples() -> List[Dict]:
    return [
        _msg(
            "Show the monthly revenue trend.\n\n[EVIDENCE]\nJan $400K, Feb $420K, Mar $390K, Apr $450K, May $480K, Jun $510K.",
            "## Monthly Revenue Trend\n\n| Month | Revenue |\n|-------|--------|\n| Jan | **$400K** |\n| Feb | **$420K** |\n| Mar | **$390K** |\n| Apr | **$450K** |\n| May | **$480K** |\n| Jun | **$510K** |\n\nRevenue trended upward, reaching **$510K** in June after a small dip in March.\n\n"
            + _viz("line", "Monthly Revenue Trend", ["Jan", "Feb", "Mar", "Apr", "May", "Jun"], [400, 420, 390, 450, 480, 510], unit="$K"),
        ),
        _msg(
            "Plot document upload counts over the last 6 weeks.\n\n[EVIDENCE]\nWeek 1: 45, Week 2: 62, Week 3: 58, Week 4: 71, Week 5: 89, Week 6: 94.",
            "## Weekly Document Uploads\n\n| Week | Uploads |\n|------|--------|\n| Week 1 | **45** |\n| Week 2 | **62** |\n| Week 3 | **58** |\n| Week 4 | **71** |\n| Week 5 | **89** |\n| Week 6 | **94** |\n\nUpload activity has been steadily increasing, doubling from Week 1 to Week 6.\n\n"
            + _viz("line", "Weekly Document Uploads", ["W1", "W2", "W3", "W4", "W5", "W6"], [45, 62, 58, 71, 89, 94]),
        ),
        _msg(
            "How has average response time changed?\n\n[EVIDENCE]\nJan: 4.2s, Feb: 3.8s, Mar: 3.5s, Apr: 3.1s, May: 2.8s, Jun: 2.4s.",
            "## Response Time Trend\n\n| Month | Avg Response Time |\n|-------|------------------|\n| Jan | **4.2s** |\n| Feb | **3.8s** |\n| Mar | **3.5s** |\n| Apr | **3.1s** |\n| May | **2.8s** |\n| Jun | **2.4s** |\n\nResponse time improved by **43%** over 6 months, from 4.2s to 2.4s.\n\n"
            + _viz("line", "Average Response Time", ["Jan", "Feb", "Mar", "Apr", "May", "Jun"], [4.2, 3.8, 3.5, 3.1, 2.8, 2.4], unit="seconds"),
        ),
    ]


def _multi_line_examples() -> List[Dict]:
    return [
        _msg(
            "Compare revenue and costs over quarters.\n\n[EVIDENCE]\nQ1 revenue $1.2M cost $0.9M. Q2 revenue $1.5M cost $1.0M. Q3 revenue $1.8M cost $1.1M. Q4 revenue $2.1M cost $1.2M.",
            "## Revenue vs Costs\n\n| Quarter | Revenue | Cost |\n|---------|---------|------|\n| Q1 | **$1.2M** | $0.9M |\n| Q2 | **$1.5M** | $1.0M |\n| Q3 | **$1.8M** | $1.1M |\n| Q4 | **$2.1M** | $1.2M |\n\nMargins are widening -- revenue growth outpaces cost increases.\n\n"
            + _viz("multi_line", "Revenue vs Costs", ["Q1", "Q2", "Q3", "Q4"], [1.2, 1.5, 1.8, 2.1], unit="$M", secondary_values=[0.9, 1.0, 1.1, 1.2], secondary_name="Costs"),
        ),
        _msg(
            "Show planned vs actual headcount by quarter.\n\n[EVIDENCE]\nQ1 planned 200 actual 195. Q2 planned 220 actual 218. Q3 planned 250 actual 240. Q4 planned 280 actual 265.",
            "## Planned vs Actual Headcount\n\n| Quarter | Planned | Actual |\n|---------|---------|--------|\n| Q1 | 200 | **195** |\n| Q2 | 220 | **218** |\n| Q3 | 250 | **240** |\n| Q4 | 280 | **265** |\n\nActual headcount consistently lagged behind plan, with the gap widening in later quarters.\n\n"
            + _viz("multi_line", "Planned vs Actual Headcount", ["Q1", "Q2", "Q3", "Q4"], [200, 220, 250, 280], secondary_values=[195, 218, 240, 265], secondary_name="Actual"),
        ),
    ]


def _area_examples() -> List[Dict]:
    return [
        _msg(
            "Show cumulative spending over the year.\n\n[EVIDENCE]\nJan $50K, Feb $110K, Mar $175K, Apr $245K, May $320K, Jun $400K.",
            "## Cumulative Spending\n\n| Month | Cumulative |\n|-------|----------|\n| Jan | **$50K** |\n| Feb | **$110K** |\n| Mar | **$175K** |\n| Apr | **$245K** |\n| May | **$320K** |\n| Jun | **$400K** |\n\nSpending reached **$400K** by mid-year.\n\n"
            + _viz("area", "Cumulative Spending", ["Jan", "Feb", "Mar", "Apr", "May", "Jun"], [50, 110, 175, 245, 320, 400], unit="$K"),
        ),
        _msg(
            "Plot active user growth.\n\n[EVIDENCE]\nMonth 1: 500, Month 2: 850, Month 3: 1400, Month 4: 2100, Month 5: 3200, Month 6: 4800.",
            "## Active User Growth\n\n| Month | Users |\n|-------|------|\n| Month 1 | **500** |\n| Month 2 | **850** |\n| Month 3 | **1,400** |\n| Month 4 | **2,100** |\n| Month 5 | **3,200** |\n| Month 6 | **4,800** |\n\nUser base is growing exponentially, nearly **10x** over 6 months.\n\n"
            + _viz("area", "Active User Growth", ["M1", "M2", "M3", "M4", "M5", "M6"], [500, 850, 1400, 2100, 3200, 4800]),
        ),
    ]


def _scatter_examples() -> List[Dict]:
    return [
        _msg(
            "Plot deal size vs close time.\n\n[EVIDENCE]\nDeal A: $50K, 15 days. Deal B: $120K, 45 days. Deal C: $30K, 8 days. Deal D: $200K, 60 days. Deal E: $80K, 30 days.",
            "## Deal Size vs Close Time\n\n| Deal | Size | Days to Close |\n|------|------|---------------|\n| D | **$200K** | 60 |\n| B | **$120K** | 45 |\n| E | **$80K** | 30 |\n| A | **$50K** | 15 |\n| C | **$30K** | 8 |\n\nLarger deals take longer to close. There is a positive correlation between deal size and closing time.\n\n"
            + _viz("scatter", "Deal Size vs Close Time", ["A", "B", "C", "D", "E"], [50, 120, 30, 200, 80], unit="$K", secondary_values=[15, 45, 8, 60, 30], secondary_name="Days to Close"),
        ),
        _msg(
            "Show team productivity vs hours worked.\n\n[EVIDENCE]\nAlpha: 40hrs 92 tasks. Bravo: 45hrs 88 tasks. Charlie: 38hrs 95 tasks. Delta: 50hrs 80 tasks.",
            "## Team Productivity vs Hours\n\n| Team | Hours | Tasks |\n|------|-------|------|\n| Charlie | 38 | **95** |\n| Alpha | 40 | **92** |\n| Bravo | 45 | **88** |\n| Delta | 50 | **80** |\n\nCharlie is the most efficient team -- highest output with fewest hours.\n\n"
            + _viz("scatter", "Productivity vs Hours Worked", ["Alpha", "Bravo", "Charlie", "Delta"], [40, 45, 38, 50], unit="hrs", secondary_values=[92, 88, 95, 80], secondary_name="Tasks Completed"),
        ),
    ]


def _radar_examples() -> List[Dict]:
    return [
        _msg(
            "Rate the vendor across evaluation criteria.\n\n[EVIDENCE]\nReliability: 9/10, Cost: 7/10, Support: 8/10, Innovation: 6/10, Compliance: 9/10.",
            "## Vendor Evaluation\n\n| Criterion | Score |\n|-----------|------|\n| Reliability | **9/10** |\n| Compliance | **9/10** |\n| Support | **8/10** |\n| Cost | **7/10** |\n| Innovation | **6/10** |\n\nStrong on reliability and compliance; innovation could improve.\n\n"
            + _viz("radar", "Vendor Evaluation", ["Reliability", "Cost", "Support", "Innovation", "Compliance"], [9, 7, 8, 6, 9]),
        ),
        _msg(
            "Compare employee competencies.\n\n[EVIDENCE]\nCommunication: 8, Technical: 9, Leadership: 7, Teamwork: 9, Problem Solving: 8.",
            "## Employee Competency Profile\n\n| Competency | Rating |\n|------------|-------|\n| Technical | **9** |\n| Teamwork | **9** |\n| Communication | **8** |\n| Problem Solving | **8** |\n| Leadership | **7** |\n\nStrong technical and teamwork skills. Leadership development recommended.\n\n"
            + _viz("radar", "Employee Competency Profile", ["Communication", "Technical", "Leadership", "Teamwork", "Problem Solving"], [8, 9, 7, 9, 8]),
        ),
    ]


def _waterfall_examples() -> List[Dict]:
    return [
        _msg(
            "Show the profit and loss waterfall.\n\n[EVIDENCE]\nRevenue: $5.0M. COGS: -$2.0M. Gross Profit: $3.0M. Opex: -$1.5M. Tax: -$0.4M. Net Profit: $1.1M.",
            "## Profit & Loss Waterfall\n\n| Item | Amount |\n|------|-------|\n| Revenue | **$5.0M** |\n| COGS | -$2.0M |\n| Gross Profit | **$3.0M** |\n| Opex | -$1.5M |\n| Tax | -$0.4M |\n| **Net Profit** | **$1.1M** |\n\n"
            + _viz("waterfall", "Profit & Loss Waterfall", ["Revenue", "COGS", "Gross Profit", "Opex", "Tax", "Net Profit"], [5.0, -2.0, 3.0, -1.5, -0.4, 1.1], unit="$M"),
        ),
        _msg(
            "Break down budget changes.\n\n[EVIDENCE]\nOriginal budget: $2M. Headcount increase: +$300K. Cloud savings: -$150K. New tools: +$80K. Revised budget: $2.23M.",
            "## Budget Change Waterfall\n\n| Item | Amount |\n|------|-------|\n| Original | **$2.0M** |\n| Headcount | +$300K |\n| Cloud Savings | -$150K |\n| New Tools | +$80K |\n| **Revised** | **$2.23M** |\n\n"
            + _viz("waterfall", "Budget Changes", ["Original", "Headcount", "Cloud Savings", "New Tools", "Revised"], [2000, 300, -150, 80, 2230], unit="$K"),
        ),
    ]


def _gauge_examples() -> List[Dict]:
    return [
        _msg(
            "What is the SLA compliance rate?\n\n[EVIDENCE]\nCurrent SLA compliance: 97.3%. Target: 99.5%.",
            "## SLA Compliance\n\n| Metric | Value |\n|--------|------|\n| SLA Compliance | **97.3%** |\n| Target | **99.5%** |\n\nCurrent SLA compliance is **97.3%** against a target of **99.5%**. The gap of **2.2pp** needs attention.\n\n"
            + _viz("gauge", "SLA Compliance", ["SLA Compliance"], [97.3], unit="%"),
        ),
        _msg(
            "Show the project completion percentage.\n\n[EVIDENCE]\nProject Omega: 73% complete. 22 of 30 milestones achieved.",
            "## Project Omega Progress\n\n| Metric | Value |\n|--------|------|\n| Completion | **73%** |\n| Milestones Achieved | **22 of 30** |\n\n**73%** complete with **22 of 30** milestones achieved. **8 milestones** remain.\n\n"
            + _viz("gauge", "Project Omega Completion", ["Completion"], [73], unit="%"),
        ),
        _msg(
            "What is our current customer retention rate?\n\n[EVIDENCE]\nCustomer retention rate for Q4: 91.5%. Industry average: 85%.",
            "## Customer Retention\n\n| Metric | Value |\n|--------|------|\n| Retention Rate | **91.5%** |\n| Industry Average | **85%** |\n\nCurrent retention is **91.5%**, which is **6.5pp above** the industry average of 85%.\n\n"
            + _viz("gauge", "Customer Retention Rate", ["Retention"], [91.5], unit="%"),
        ),
    ]


def _treemap_examples() -> List[Dict]:
    return [
        _msg(
            "Show budget allocation by department.\n\n[EVIDENCE]\nEngineering: $4.2M, Sales: $2.8M, Marketing: $1.5M, HR: $0.8M, Legal: $0.5M, IT Ops: $1.2M.",
            "## Budget Allocation\n\n| Department | Budget |\n|------------|-------|\n| Engineering | **$4.2M** |\n| Sales | **$2.8M** |\n| Marketing | **$1.5M** |\n| IT Ops | **$1.2M** |\n| HR | **$0.8M** |\n| Legal | **$0.5M** |\n\nEngineering takes the largest share at **$4.2M**.\n\n"
            + _viz("treemap", "Budget Allocation by Department", ["Engineering", "Sales", "Marketing", "IT Ops", "HR", "Legal"], [4.2, 2.8, 1.5, 1.2, 0.8, 0.5], unit="$M"),
        ),
        _msg(
            "Show storage usage by file type.\n\n[EVIDENCE]\nPDFs: 12GB, Images: 8GB, Spreadsheets: 5GB, Videos: 15GB, Other: 3GB.",
            "## Storage Usage by File Type\n\n| Type | Size |\n|------|------|\n| Videos | **15GB** |\n| PDFs | **12GB** |\n| Images | **8GB** |\n| Spreadsheets | **5GB** |\n| Other | **3GB** |\n\nVideos and PDFs together account for **63%** of storage.\n\n"
            + _viz("treemap", "Storage Usage by File Type", ["PDFs", "Images", "Spreadsheets", "Videos", "Other"], [12, 8, 5, 15, 3], unit="GB"),
        ),
    ]


# ---------------------------------------------------------------------------
# Flow analysis examples
# ---------------------------------------------------------------------------

def _flow_examples() -> List[Dict]:
    return [
        _msg(
            "Describe the invoice approval process.\n\n[EVIDENCE]\nInvoice submitted by vendor. Finance team validates amounts. Manager reviews and approves if under $10K. Director approval required above $10K. Payment issued within Net 30.",
            "## Invoice Approval Process\n\n1. Vendor submits invoice\n2. Finance team validates line items and totals\n3. Routing decision based on amount:\n   - Under $10K \u2192 Manager approval\n   - Over $10K \u2192 Director approval\n4. Approved invoice queued for payment\n5. Payment issued within **Net 30** terms\n\n**Key threshold:** $10K determines the approval authority level.",
        ),
        _msg(
            "How does the document onboarding pipeline work?\n\n[EVIDENCE]\nUser uploads file. System extracts text via OCR if needed. Content chunked into 300-token segments. Embeddings generated. Vectors stored in database. Profile index updated.",
            "## Document Onboarding Pipeline\n\n1. User uploads document\n2. System detects file type \u2192 applies appropriate extractor\n3. OCR applied if the document is image-based\n4. Extracted text split into ~300-token chunks\n5. Embedding model generates vector for each chunk\n6. Vectors stored in vector database \u2192 profile index updated\n\nThe entire pipeline runs asynchronously after upload.",
        ),
        _msg(
            "What is the employee offboarding procedure?\n\n[EVIDENCE]\nManager initiates separation. HR schedules exit interview. IT revokes access within 24 hours. Final paycheck processed. Equipment returned. Knowledge transfer completed.",
            "## Employee Offboarding Procedure\n\n1. Manager initiates separation request in HR system\n2. HR schedules exit interview \u2192 collects feedback\n3. IT revokes system access within **24 hours**\n4. Employee returns company equipment\n5. Knowledge transfer sessions completed with team\n6. Final paycheck processed by payroll\n\n**Critical SLA:** IT access revocation must happen within 24 hours of notification.",
        ),
        _msg(
            "Explain the contract renewal workflow.\n\n[EVIDENCE]\nSystem alerts 90 days before expiry. Account manager reviews terms. Legal reviews amendments. Negotiation with vendor. Final approval by procurement. Contract signed and filed.",
            "## Contract Renewal Workflow\n\n1. Automated alert triggered **90 days** before expiry\n2. Account manager reviews current terms and performance\n3. Legal reviews proposed amendments \u2192 flags risks\n4. Negotiation round with vendor\n5. Procurement gives final approval\n6. Contract signed \u2192 filed in document management system\n\nThe 90-day lead time ensures adequate review and negotiation runway.",
        ),
        _msg(
            "How are security incidents handled?\n\n[EVIDENCE]\nIncident detected by monitoring. Triage team assesses severity. P1 incidents escalated to CISO within 1 hour. Containment actions taken. Root cause analysis performed. Post-mortem report filed.",
            "## Security Incident Response\n\n1. Monitoring system detects anomaly \u2192 alert raised\n2. Triage team assesses severity (P1 through P4)\n3. P1 incidents \u2192 escalated to CISO within **1 hour**\n4. Containment actions executed immediately\n5. Root cause analysis performed\n6. Post-mortem report filed \u2192 lessons learned distributed\n\n**SLA:** P1 incidents require CISO notification within 1 hour.",
        ),
        _msg(
            "Describe the data backup process.\n\n[EVIDENCE]\nDaily incremental backups at 2 AM. Weekly full backups on Sunday. Backups replicated to secondary region. 90-day retention policy. Monthly restore drills.",
            "## Data Backup Process\n\n1. Daily incremental backups run at **2:00 AM**\n2. Weekly full backup every Sunday\n3. Backups replicated \u2192 secondary geographic region\n4. **90-day** retention policy applied\n5. Monthly restore drills validate backup integrity\n\nThe combination of incremental and full backups provides both efficiency and reliability.",
        ),
    ]


# ---------------------------------------------------------------------------
# Text-only examples
# ---------------------------------------------------------------------------

def _text_only_examples() -> List[Dict]:
    return [
        _msg(
            "When does the contract expire?\n\n[EVIDENCE]\nContract effective January 1, 2025, term 24 months.",
            "The contract expires **January 1, 2027** (24-month term from January 1, 2025).",
        ),
        _msg(
            "Who is the landlord?\n\n[EVIDENCE]\nLease: Landlord is Riverside Properties LLC, 500 Harbor Drive, Miami FL.",
            "The landlord is **Riverside Properties LLC**, located at **500 Harbor Drive, Miami, FL**.",
        ),
        _msg(
            "What is the payment amount?\n\n[EVIDENCE]\nPayment: $15,000 to Johnson & Associates for consulting.",
            "**$15,000** to **Johnson & Associates** for consulting services.",
        ),
        _msg(
            "How many pages is the report?\n\n[EVIDENCE]\nDocument: Compliance_Report_2025.pdf, 63 pages.",
            "**Compliance_Report_2025.pdf** is **63 pages**.",
        ),
        _msg(
            "What is the warranty period?\n\n[EVIDENCE]\nProduct warranty: 36 months from date of delivery.",
            "The warranty period is **36 months** from the date of delivery.",
        ),
        _msg(
            "Who signed the agreement?\n\n[EVIDENCE]\nSigned by: Maria Torres (CEO, ClientCo) and James Park (VP Sales, VendorInc) on March 15, 2025.",
            "The agreement was signed by:\n- **Maria Torres**, CEO of ClientCo\n- **James Park**, VP Sales of VendorInc\n\nSigned on **March 15, 2025**.",
        ),
        _msg(
            "What is the notice period?\n\n[EVIDENCE]\nTermination clause: Either party may terminate with 60 days written notice.",
            "The notice period is **60 days** written notice by either party.",
        ),
        _msg(
            "What currency is used in the contract?\n\n[EVIDENCE]\nAll amounts in United States Dollars (USD). Governing law: State of Delaware.",
            "All contract amounts are in **United States Dollars (USD)**. The governing law is the **State of Delaware**.",
        ),
        _msg(
            "Is there a non-compete clause?\n\n[EVIDENCE]\nSection 8.2: Non-compete period of 12 months within 50-mile radius after termination.",
            "Yes. **Section 8.2** includes a non-compete clause: **12 months** within a **50-mile radius** after termination.",
        ),
        _msg(
            "What is the insurance requirement?\n\n[EVIDENCE]\nContractor must maintain general liability insurance of at least $2M per occurrence.",
            "The contractor must maintain general liability insurance of at least **$2M per occurrence**.",
        ),
        _msg(
            "What is the total headcount?\n\n[EVIDENCE]\nCompany roster shows 342 active employees across all departments.",
            "The total active headcount is **342 employees**.",
        ),
        _msg(
            "When was the last audit completed?\n\n[EVIDENCE]\nInternal audit completed February 28, 2025. Next audit scheduled August 2025.",
            "The last internal audit was completed **February 28, 2025**. The next audit is scheduled for **August 2025**.",
        ),
    ]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_viz_sft_examples(seed: int = 42) -> List[Dict[str, Any]]:
    """Generate all visualization SFT training examples.

    Returns a list of chat-format dicts, each containing:
        {"messages": [system_msg, user_msg, assistant_msg]}
    """
    rng = random.Random(seed)

    all_examples: List[Dict[str, Any]] = []

    # Chart examples
    chart_generators = [
        _bar_examples,
        _horizontal_bar_examples,
        _grouped_bar_examples,
        _stacked_bar_examples,
        _donut_examples,
        _line_examples,
        _multi_line_examples,
        _area_examples,
        _scatter_examples,
        _radar_examples,
        _waterfall_examples,
        _gauge_examples,
        _treemap_examples,
    ]

    for gen_fn in chart_generators:
        examples = gen_fn()
        all_examples.extend(examples)

    # Flow analysis examples
    all_examples.extend(_flow_examples())

    # Text-only examples
    all_examples.extend(_text_only_examples())

    rng.shuffle(all_examples)

    logger.info(
        "Generated %d viz SFT examples (charts: %d, flow: %d, text: %d)",
        len(all_examples),
        sum(1 for e in all_examples if "DOCWAIN_VIZ" in e["messages"][2]["content"]),
        sum(1 for e in all_examples if "\u2192" in e["messages"][2]["content"] and "DOCWAIN_VIZ" not in e["messages"][2]["content"]),
        sum(1 for e in all_examples if "\u2192" not in e["messages"][2]["content"] and "DOCWAIN_VIZ" not in e["messages"][2]["content"]),
    )

    return all_examples


def write_viz_sft_dataset(
    output_dir: Path | str = "finetune_data",
    seed: int = 42,
) -> Dict[str, Any]:
    """Write train/eval JSONL files for visualization SFT.

    Args:
        output_dir: Directory to write the JSONL files into.
        seed: Random seed for reproducibility.

    Returns:
        Dict with train_count, eval_count, total, train_path, eval_path.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    examples = generate_viz_sft_examples(seed=seed)

    split_idx = int(len(examples) * 0.9)
    train = examples[:split_idx]
    eval_set = examples[split_idx:]

    train_path = output_dir / "viz_sft_train.jsonl"
    eval_path = output_dir / "viz_sft_eval.jsonl"

    for path, data in [(train_path, train), (eval_path, eval_set)]:
        with open(path, "w", encoding="utf-8") as f:
            for example in data:
                f.write(json.dumps(example, ensure_ascii=False) + "\n")

    total = len(examples)
    logger.info(
        "Viz SFT dataset: %d total, %d train, %d eval -> %s",
        total, len(train), len(eval_set), output_dir,
    )

    return {
        "train_count": len(train),
        "eval_count": len(eval_set),
        "total": total,
        "train_path": str(train_path),
        "eval_path": str(eval_path),
    }
