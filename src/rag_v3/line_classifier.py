"""ML-based line role classifier replacing regex KV extraction.

Multi-head NumPy MLP that classifies document lines by **role**
(kv_pair / heading / bullet / narrative / skip) and **domain-specific
category** (e.g. diagnoses, medications, coverage, clauses, etc.)
using semantic embeddings + layout features.

Architecture::

    Input: embedding(1024) + layout_features(8) = 1032-dim
    Shared:       Linear(1032 → 128) → ReLU
    Role head:    Linear(128 → 5)   → Softmax
    Medical head: Linear(128 → 7)   → Softmax
    Policy head:  Linear(128 → 6)   → Softmax
    Invoice head: Linear(128 → 5)   → Softmax
    Legal head:   Linear(128 → 4)   → Softmax

Usage::

    from src.rag_v3.line_classifier import classify_lines, ensure_line_classifier

    ensure_line_classifier(embedder)
    results = classify_lines(lines, domain="medical", embedder=embedder)
"""

from __future__ import annotations

import pickle
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.utils.logging_utils import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Head label taxonomies
# ---------------------------------------------------------------------------
ROLE_NAMES: List[str] = ["kv_pair", "heading", "bullet", "narrative", "skip"]
MEDICAL_NAMES: List[str] = [
    "patient_info", "diagnoses", "medications", "procedures",
    "lab_results", "vitals", "other",
]
POLICY_NAMES: List[str] = [
    "policy_info", "coverage", "premiums", "exclusions", "terms", "other",
]
INVOICE_NAMES: List[str] = ["items", "totals", "parties", "terms", "invoice_metadata", "other"]
LEGAL_NAMES: List[str] = ["clauses", "parties", "obligations", "other"]

HEAD_NAMES: Dict[str, List[str]] = {
    "role": ROLE_NAMES,
    "medical": MEDICAL_NAMES,
    "policy": POLICY_NAMES,
    "invoice": INVOICE_NAMES,
    "legal": LEGAL_NAMES,
}

# Which head to use per domain
_DOMAIN_HEAD: Dict[str, str] = {
    "medical": "medical",
    "policy": "policy",
    "invoice": "invoice",
    "legal": "legal",
    "contract": "legal",
}

# ---------------------------------------------------------------------------
# Layout features (8-dim, no regex)
# ---------------------------------------------------------------------------
_BULLET_CHARS = frozenset("•-*●▪■►‣⁃")


def _layout_features(line: str) -> np.ndarray:
    """Compute 8 structural features for a line of text."""
    feat = np.zeros(8, dtype=np.float32)
    if not line:
        return feat

    # 1. has_colon
    colon_idx = line.find(":")
    feat[0] = 1.0 if colon_idx > 0 else 0.0

    # 2. colon_position_ratio
    feat[1] = colon_idx / len(line) if colon_idx > 0 else 0.0

    # 3. starts_with_bullet
    stripped = line.lstrip()
    feat[2] = 1.0 if stripped and stripped[0] in _BULLET_CHARS else 0.0

    # 4. starts_with_number (numbered list: "1.", "2)", "3 ")
    if stripped:
        i = 0
        while i < len(stripped) and stripped[i].isdigit():
            i += 1
        if i > 0 and i < len(stripped) and stripped[i] in ".):":
            feat[3] = 1.0

    # 5. word_count_norm
    words = line.split()
    feat[4] = min(len(words) / 20.0, 1.0)

    # 6. uppercase_ratio
    alpha_chars = [c for c in line if c.isalpha()]
    if alpha_chars:
        feat[5] = sum(1 for c in alpha_chars if c.isupper()) / len(alpha_chars)

    # 7. has_value_after_colon
    if colon_idx > 0 and colon_idx < len(line) - 1:
        after = line[colon_idx + 1:].strip()
        feat[6] = 1.0 if len(after) >= 2 else 0.0

    # 8. line_length_norm
    feat[7] = min(len(line) / 200.0, 1.0)

    return feat


# ---------------------------------------------------------------------------
# Colon-based split (no regex)
# ---------------------------------------------------------------------------

def _split_at_colon(line: str) -> Tuple[str, str]:
    """Split a line at the first colon into (label, value).

    Returns ("", line.strip()) when no valid split point is found.
    No regex used — simple string find.
    """
    idx = line.find(":")
    if idx > 0 and idx < len(line) - 1:
        label = line[:idx].strip()
        value = line[idx + 1:].strip()
        # Sanity: label should look like a label (not a URL or time)
        if label and len(label) <= 60 and value:
            return label, value
    return "", line.strip()


# ---------------------------------------------------------------------------
# LineClassification result
# ---------------------------------------------------------------------------

@dataclass
class LineClassification:
    """Classification result for a single line."""
    role: str               # kv_pair | heading | bullet | narrative | skip
    role_confidence: float
    category: str           # domain-specific category (e.g. diagnoses, coverage)
    category_confidence: float
    label: str              # KV label (empty for non-KV roles)
    value: str              # KV value or full line text


# ---------------------------------------------------------------------------
# Training templates (~250 self-supervised samples)
# ---------------------------------------------------------------------------
# Format: (line_text, role, medical, policy, invoice, legal)

TRAINING_TEMPLATES: List[Tuple[str, str, str, str, str, str]] = [
    # ── KV pairs: medical / patient info ──
    ("Patient Name: John Doe", "kv_pair", "patient_info", "other", "other", "other"),
    ("Age: 45 years", "kv_pair", "patient_info", "other", "other", "other"),
    ("Date of Birth: 15/03/1980", "kv_pair", "patient_info", "other", "other", "other"),
    ("Gender: Male", "kv_pair", "patient_info", "other", "other", "other"),
    ("MRN: 123456789", "kv_pair", "patient_info", "other", "other", "other"),
    ("Medical Record Number: MR-2024-001", "kv_pair", "patient_info", "other", "other", "other"),
    ("Blood Group: O Positive", "kv_pair", "patient_info", "other", "other", "other"),
    ("Admission Date: 2024-01-15", "kv_pair", "patient_info", "other", "other", "other"),
    ("Allergies: Penicillin, Sulfa drugs", "kv_pair", "patient_info", "other", "other", "other"),
    ("Emergency Contact: Jane Doe (555) 123-4567", "kv_pair", "patient_info", "other", "other", "other"),

    # ── KV pairs: medical / diagnoses ──
    ("Diagnosis: Type 2 Diabetes Mellitus", "kv_pair", "diagnoses", "other", "other", "other"),
    ("Primary Diagnosis: Acute Myocardial Infarction", "kv_pair", "diagnoses", "other", "other", "other"),
    ("Assessment: Hypertension Stage II", "kv_pair", "diagnoses", "other", "other", "other"),
    ("Clinical Impression: Pneumonia, community acquired", "kv_pair", "diagnoses", "other", "other", "other"),
    ("ICD-10: E11.9 Type 2 diabetes without complications", "kv_pair", "diagnoses", "other", "other", "other"),
    ("Chief Complaint: Chest pain radiating to left arm", "kv_pair", "diagnoses", "other", "other", "other"),
    ("Condition: Chronic Kidney Disease Stage 3", "kv_pair", "diagnoses", "other", "other", "other"),
    ("Secondary Diagnosis: Hyperlipidemia", "kv_pair", "diagnoses", "other", "other", "other"),

    # ── KV pairs: medical / medications ──
    ("Medication: Metformin 500mg twice daily", "kv_pair", "medications", "other", "other", "other"),
    ("Prescription: Lisinopril 10mg once daily", "kv_pair", "medications", "other", "other", "other"),
    ("Drug: Atorvastatin 20mg at bedtime", "kv_pair", "medications", "other", "other", "other"),
    ("Dosage: Amoxicillin 500mg three times a day", "kv_pair", "medications", "other", "other", "other"),
    ("Treatment: Insulin Glargine 20 units subcutaneous", "kv_pair", "medications", "other", "other", "other"),
    ("Rx: Omeprazole 20mg daily before breakfast", "kv_pair", "medications", "other", "other", "other"),

    # ── KV pairs: medical / procedures ──
    ("Procedure: Cardiac catheterization", "kv_pair", "procedures", "other", "other", "other"),
    ("Surgery: Laparoscopic cholecystectomy", "kv_pair", "procedures", "other", "other", "other"),
    ("Imaging: Chest X-ray PA and lateral", "kv_pair", "procedures", "other", "other", "other"),
    ("Treatment Plan: Physical therapy 3x weekly for 6 weeks", "kv_pair", "procedures", "other", "other", "other"),
    ("MRI: Brain with and without contrast", "kv_pair", "procedures", "other", "other", "other"),

    # ── KV pairs: medical / lab results ──
    ("Hemoglobin: 12.5 g/dL", "kv_pair", "lab_results", "other", "other", "other"),
    ("Blood Glucose: 126 mg/dL (fasting)", "kv_pair", "lab_results", "other", "other", "other"),
    ("HbA1c: 7.2%", "kv_pair", "lab_results", "other", "other", "other"),
    ("Creatinine: 1.2 mg/dL", "kv_pair", "lab_results", "other", "other", "other"),
    ("WBC Count: 8,500/mcL", "kv_pair", "lab_results", "other", "other", "other"),
    ("Cholesterol Total: 220 mg/dL", "kv_pair", "lab_results", "other", "other", "other"),
    ("TSH: 2.5 mIU/L", "kv_pair", "lab_results", "other", "other", "other"),
    ("Platelet Count: 250,000/mcL", "kv_pair", "lab_results", "other", "other", "other"),

    # ── KV pairs: medical / vitals ──
    ("Blood Pressure: 130/85 mmHg", "kv_pair", "vitals", "other", "other", "other"),
    ("Heart Rate: 72 bpm", "kv_pair", "vitals", "other", "other", "other"),
    ("Temperature: 98.6°F", "kv_pair", "vitals", "other", "other", "other"),
    ("Respiratory Rate: 16 breaths/min", "kv_pair", "vitals", "other", "other", "other"),
    ("Oxygen Saturation: 98% on room air", "kv_pair", "vitals", "other", "other", "other"),
    ("SpO2: 97%", "kv_pair", "vitals", "other", "other", "other"),
    ("BMI: 24.5", "kv_pair", "vitals", "other", "other", "other"),
    ("Weight: 75 kg", "kv_pair", "vitals", "other", "other", "other"),

    # ── KV pairs: policy / policy_info ──
    ("Policy Number: INS-2024-00451", "kv_pair", "other", "policy_info", "other", "other"),
    ("Insured Name: Rajesh Kumar", "kv_pair", "other", "policy_info", "other", "other"),
    ("Policyholder: ABC Corporation", "kv_pair", "other", "policy_info", "other", "other"),
    ("Effective Date: 01/04/2024", "kv_pair", "other", "policy_info", "other", "other"),
    ("Expiry Date: 31/03/2025", "kv_pair", "other", "policy_info", "other", "other"),
    ("Sum Insured: Rs. 50,00,000", "kv_pair", "other", "policy_info", "other", "other"),
    ("Vehicle Registration: KA-01-AB-1234", "kv_pair", "other", "policy_info", "other", "other"),
    ("Type of Cover: Comprehensive", "kv_pair", "other", "policy_info", "other", "other"),
    ("IDV: Rs. 4,50,000", "kv_pair", "other", "policy_info", "other", "other"),
    ("Plan: Gold Health Insurance", "kv_pair", "other", "policy_info", "other", "other"),

    # ── KV pairs: policy / coverage ──
    ("Coverage: Own Damage and Third Party Liability", "kv_pair", "other", "coverage", "other", "other"),
    ("Personal Accident Cover: Rs. 15,00,000", "kv_pair", "other", "coverage", "other", "other"),
    ("Third Party Liability: Unlimited", "kv_pair", "other", "coverage", "other", "other"),
    ("Scope of Coverage: Inpatient hospitalization", "kv_pair", "other", "coverage", "other", "other"),

    # ── KV pairs: policy / premiums ──
    ("Net Premium: Rs. 12,500", "kv_pair", "other", "premiums", "other", "other"),
    ("Total Premium: Rs. 14,750 including GST", "kv_pair", "other", "premiums", "other", "other"),
    ("GST (18%): Rs. 2,250", "kv_pair", "other", "premiums", "other", "other"),
    ("OD Premium: Rs. 8,500", "kv_pair", "other", "premiums", "other", "other"),
    ("NCB Discount: 25%", "kv_pair", "other", "premiums", "other", "other"),
    ("Basic Premium: Rs. 10,000", "kv_pair", "other", "premiums", "other", "other"),

    # ── KV pairs: policy / exclusions ──
    ("Exclusion: Pre-existing conditions for first 4 years", "kv_pair", "other", "exclusions", "other", "other"),
    ("Deductible: Rs. 5,000 per claim", "kv_pair", "other", "exclusions", "other", "other"),
    ("Waiting Period: 30 days for initial claims", "kv_pair", "other", "exclusions", "other", "other"),
    ("Not Covered: Cosmetic surgery and dental treatment", "kv_pair", "other", "exclusions", "other", "other"),

    # ── KV pairs: policy / terms ──
    ("Renewal: Annual with 15-day grace period", "kv_pair", "other", "terms", "other", "other"),
    ("Claims Procedure: Notify insurer within 24 hours", "kv_pair", "other", "terms", "other", "other"),
    ("Cancellation: 15 days notice required", "kv_pair", "other", "terms", "other", "other"),
    ("Dispute Resolution: Arbitration as per Indian Arbitration Act", "kv_pair", "other", "terms", "other", "other"),

    # ── KV pairs: invoice / items ──
    ("Item: Professional consulting services", "kv_pair", "other", "other", "items", "other"),
    ("Product: Wireless Bluetooth Headphones", "kv_pair", "other", "other", "items", "other"),
    ("Description: Software license renewal (annual)", "kv_pair", "other", "other", "items", "other"),
    ("Service: Cloud hosting - 12 month subscription", "kv_pair", "other", "other", "items", "other"),
    ("Line Item: Web development Phase 2", "kv_pair", "other", "other", "items", "other"),

    # ── KV pairs: invoice / totals ──
    ("Total Amount Due: $2,450.00", "kv_pair", "other", "other", "totals", "other"),
    ("Subtotal: $2,100.00", "kv_pair", "other", "other", "totals", "other"),
    ("Balance Due: $1,225.00", "kv_pair", "other", "other", "totals", "other"),
    ("Grand Total: Rs. 1,18,000", "kv_pair", "other", "other", "totals", "other"),
    ("Tax (GST 18%): Rs. 18,000", "kv_pair", "other", "other", "totals", "other"),
    ("Amount: USD 5,000.00", "kv_pair", "other", "other", "totals", "other"),

    # ── KV pairs: invoice / parties ──
    ("Bill To: Acme Industries Pvt. Ltd.", "kv_pair", "other", "other", "parties", "other"),
    ("From: TechSoft Solutions", "kv_pair", "other", "other", "parties", "other"),
    ("Vendor: Global Supply Co.", "kv_pair", "other", "other", "parties", "other"),
    ("Customer: Beta Corp", "kv_pair", "other", "other", "parties", "other"),
    ("Invoice To: Finance Department", "kv_pair", "other", "other", "parties", "other"),

    # ── KV pairs: invoice / terms ──
    ("Payment Terms: Net 30", "kv_pair", "other", "other", "terms", "other"),
    ("Due Date: February 28, 2025", "kv_pair", "other", "other", "terms", "other"),
    ("Terms: Payment due upon receipt", "kv_pair", "other", "other", "terms", "other"),
    ("Invoice Number: INV-2025-0042", "kv_pair", "other", "other", "invoice_metadata", "other"),
    ("Invoice Date: January 15, 2025", "kv_pair", "other", "other", "invoice_metadata", "other"),

    # ── KV pairs: legal / clauses ──
    ("Governing Law: State of Delaware", "kv_pair", "other", "other", "other", "clauses"),
    ("Effective Date: January 1, 2025", "kv_pair", "other", "other", "other", "clauses"),
    ("Term: 24 months from the effective date", "kv_pair", "other", "other", "other", "clauses"),
    ("Jurisdiction: Courts of Mumbai, India", "kv_pair", "other", "other", "other", "clauses"),
    ("Confidentiality: All information exchanged shall remain confidential", "kv_pair", "other", "other", "other", "clauses"),
    ("Termination: Either party may terminate with 30 days written notice", "kv_pair", "other", "other", "other", "clauses"),

    # ── KV pairs: legal / parties ──
    ("Party A: TechCorp Inc., a Delaware corporation", "kv_pair", "other", "other", "other", "parties"),
    ("Contractor: Freelance Solutions LLC", "kv_pair", "other", "other", "other", "parties"),
    ("Licensor: Patent Holdings Corp", "kv_pair", "other", "other", "other", "parties"),
    ("Employer: Global Enterprises Ltd.", "kv_pair", "other", "other", "other", "parties"),
    ("Vendor: CloudSoft Technologies", "kv_pair", "other", "other", "other", "parties"),

    # ── KV pairs: legal / obligations ──
    ("Liability Cap: Not to exceed $500,000", "kv_pair", "other", "other", "other", "obligations"),
    ("Payment: Client shall pay within 30 days of invoice", "kv_pair", "other", "other", "other", "obligations"),
    ("Warranty: Contractor warrants services for 12 months", "kv_pair", "other", "other", "other", "obligations"),

    # ── Headings (all domains) ──
    ("Patient Information:", "heading", "patient_info", "other", "other", "other"),
    ("Diagnoses:", "heading", "diagnoses", "other", "other", "other"),
    ("Medications:", "heading", "medications", "other", "other", "other"),
    ("Procedures:", "heading", "procedures", "other", "other", "other"),
    ("Lab Results:", "heading", "lab_results", "other", "other", "other"),
    ("Laboratory Results (Jan 2024):", "heading", "lab_results", "other", "other", "other"),
    ("Vital Signs:", "heading", "vitals", "other", "other", "other"),
    ("Assessment and Plan:", "heading", "diagnoses", "other", "other", "other"),
    ("Treatment Plan:", "heading", "procedures", "other", "other", "other"),
    ("History of Present Illness:", "heading", "diagnoses", "other", "other", "other"),
    ("Policy Details:", "heading", "other", "policy_info", "other", "other"),
    ("Coverage Details:", "heading", "other", "coverage", "other", "other"),
    ("Premium Summary:", "heading", "other", "premiums", "other", "other"),
    ("Exclusions and Limitations:", "heading", "other", "exclusions", "other", "other"),
    ("Terms and Conditions:", "heading", "other", "terms", "other", "other"),
    ("Invoice Items:", "heading", "other", "other", "items", "other"),
    ("Payment Details:", "heading", "other", "other", "totals", "other"),
    ("Billing Information:", "heading", "other", "other", "parties", "other"),
    ("Definitions:", "heading", "other", "other", "other", "clauses"),
    ("RECITALS", "heading", "other", "other", "other", "clauses"),
    ("Obligations of the Parties:", "heading", "other", "other", "other", "obligations"),

    # ── Bullets ──
    ("• Aspirin 81mg daily oral", "bullet", "medications", "other", "other", "other"),
    ("- Diabetes Mellitus Type 2, controlled", "bullet", "diagnoses", "other", "other", "other"),
    ("• Physical therapy twice weekly", "bullet", "procedures", "other", "other", "other"),
    ("- Hemoglobin A1c 6.8% (target <7%)", "bullet", "lab_results", "other", "other", "other"),
    ("• Fire and natural calamity coverage included", "bullet", "other", "coverage", "other", "other"),
    ("- Pre-existing conditions excluded for 48 months", "bullet", "other", "exclusions", "other", "other"),
    ("• Cashless hospitalization at network hospitals", "bullet", "other", "coverage", "other", "other"),
    ("- Premium payable annually or semi-annually", "bullet", "other", "premiums", "other", "other"),
    ("• Cloud server hosting (12 months)", "bullet", "other", "other", "items", "other"),
    ("- Software maintenance and support", "bullet", "other", "other", "items", "other"),
    ("• The Contractor shall deliver within 90 days", "bullet", "other", "other", "other", "obligations"),
    ("- Neither party shall assign this agreement", "bullet", "other", "other", "other", "clauses"),

    # ── Narrative (clinical false-positive cases — NOT KV pairs) ──
    ("Plan Activate stroke protocol - Not a candidate for thrombolysis", "narrative", "procedures", "other", "other", "other"),
    ("Patient presented with acute chest pain - transferred to ICU", "narrative", "diagnoses", "other", "other", "other"),
    ("Diabetes is well controlled with current medication regimen", "narrative", "diagnoses", "other", "other", "other"),
    ("Continue current medications and monitor blood pressure weekly", "narrative", "medications", "other", "other", "other"),
    ("The patient was discharged in stable condition with follow-up in 2 weeks", "narrative", "procedures", "other", "other", "other"),
    ("CT scan shows no acute intracranial pathology - recommend follow up MRI", "narrative", "procedures", "other", "other", "other"),
    ("Blood cultures pending - started empiric antibiotics", "narrative", "medications", "other", "other", "other"),
    ("Left knee replacement surgery went well - patient tolerating diet", "narrative", "procedures", "other", "other", "other"),
    ("The policy covers loss or damage due to fire, explosion, lightning, and earthquake", "narrative", "other", "coverage", "other", "other"),
    ("Fire and natural calamity coverage included for residential properties", "narrative", "other", "coverage", "other", "other"),
    ("This agreement shall be governed by and construed in accordance with the laws of California", "narrative", "other", "other", "other", "clauses"),
    ("The contractor agrees to indemnify and hold harmless the client from any claims", "narrative", "other", "other", "other", "obligations"),
    ("Both parties shall maintain confidentiality of all proprietary information", "narrative", "other", "other", "other", "obligations"),
    ("Insurance does not cover loss arising from war, invasion, or nuclear hazard", "narrative", "other", "exclusions", "other", "other"),
    ("Services rendered for the period January through March 2025", "narrative", "other", "other", "items", "other"),
    ("Payment shall be made within thirty calendar days of receipt of invoice", "narrative", "other", "other", "other", "obligations"),

    # ── Skip lines ──
    ("---", "skip", "other", "other", "other", "other"),
    ("Page 1 of 5", "skip", "other", "other", "other", "other"),
    ("Confidential", "skip", "other", "other", "other", "other"),
    ("", "skip", "other", "other", "other", "other"),
    ("____________________", "skip", "other", "other", "other", "other"),
    ("================", "skip", "other", "other", "other", "other"),
    ("...", "skip", "other", "other", "other", "other"),
    ("*  *  *", "skip", "other", "other", "other", "other"),
    ("Table of Contents", "skip", "other", "other", "other", "other"),
    ("END OF DOCUMENT", "skip", "other", "other", "other", "other"),

    # ── Additional cross-domain KV pairs ──
    ("Doctor: Dr. Smith, Cardiology", "kv_pair", "patient_info", "other", "other", "other"),
    ("Attending Physician: Dr. Sarah Johnson", "kv_pair", "patient_info", "other", "other", "other"),
    ("Insurer: National Insurance Company Ltd.", "kv_pair", "other", "policy_info", "other", "other"),
    ("Underwriter: Star Health Insurance", "kv_pair", "other", "policy_info", "other", "other"),
    ("Add-on Cover: Zero Depreciation", "kv_pair", "other", "coverage", "other", "other"),
    ("Rider: Critical Illness Benefit", "kv_pair", "other", "coverage", "other", "other"),
    ("Discount: 10% for online purchase", "kv_pair", "other", "premiums", "other", "other"),
    ("Sub-limit: Rs. 5,000 per day for room rent", "kv_pair", "other", "exclusions", "other", "other"),
    ("Invoice Date: 2025-01-15", "kv_pair", "other", "other", "invoice_metadata", "other"),
    ("PO Number: PO-2025-0123", "kv_pair", "other", "other", "invoice_metadata", "other"),

    # ── KV pairs: invoice / invoice_metadata ──
    ("Invoice No: 12345", "kv_pair", "other", "other", "invoice_metadata", "other"),
    ("Invoice #: INV-2024-0789", "kv_pair", "other", "other", "invoice_metadata", "other"),
    ("Date: 15 February 2025", "kv_pair", "other", "other", "invoice_metadata", "other"),
    ("Reference Number: REF-2025-001", "kv_pair", "other", "other", "invoice_metadata", "other"),
    ("Order Number: ORD-456", "kv_pair", "other", "other", "invoice_metadata", "other"),
    ("Purchase Order: PO-2025-789", "kv_pair", "other", "other", "invoice_metadata", "other"),
    ("Bill Number: BILL-2025-042", "kv_pair", "other", "other", "invoice_metadata", "other"),
    ("Receipt No: REC-78901", "kv_pair", "other", "other", "invoice_metadata", "other"),
    ("Clause 5.1: Limitation of Liability", "kv_pair", "other", "other", "other", "clauses"),
    ("Section 3: Term and Termination", "kv_pair", "other", "other", "other", "clauses"),
    ("Licensee: DataAnalytics Corp", "kv_pair", "other", "other", "other", "parties"),
    ("Between: Alpha Solutions AND Beta Enterprises", "kv_pair", "other", "other", "other", "parties"),

    # ── Narrative: ambiguous lines that look like KV but are not ──
    ("Diabetes Mellitus Type 2 with peripheral neuropathy", "narrative", "diagnoses", "other", "other", "other"),
    ("Hypertension - well controlled on current regimen", "narrative", "diagnoses", "other", "other", "other"),
    ("The sum insured represents the maximum liability of the insurer", "narrative", "other", "terms", "other", "other"),
    ("All claims must be filed within 90 days of the event", "narrative", "other", "terms", "other", "other"),
    ("Aspirin 81mg daily for cardiovascular protection", "bullet", "medications", "other", "other", "other"),
    ("Metformin 500mg twice daily with meals", "bullet", "medications", "other", "other", "other"),
    ("Complete blood count within normal limits", "narrative", "lab_results", "other", "other", "other"),
    ("The total consideration for services shall not exceed fifty thousand dollars", "narrative", "other", "other", "totals", "other"),
    ("Items purchased include office supplies and IT equipment", "narrative", "other", "other", "items", "other"),

    # ── More headings ──
    ("SECTION 1 - DEFINITIONS AND INTERPRETATION", "heading", "other", "other", "other", "clauses"),
    ("ARTICLE III: REPRESENTATIONS AND WARRANTIES", "heading", "other", "other", "other", "clauses"),
    ("Clinical Notes:", "heading", "diagnoses", "other", "other", "other"),
    ("Discharge Summary:", "heading", "procedures", "other", "other", "other"),
    ("Claims History:", "heading", "other", "terms", "other", "other"),
    ("Schedule of Benefits:", "heading", "other", "coverage", "other", "other"),

    # ── Additional bullets ──
    ("1. Metformin 500mg oral twice daily", "bullet", "medications", "other", "other", "other"),
    ("2. Blood glucose monitoring before meals", "bullet", "procedures", "other", "other", "other"),
    ("3. Follow up with endocrinology in 3 months", "bullet", "procedures", "other", "other", "other"),
    ("1. Third party liability as per Motor Vehicles Act", "bullet", "other", "coverage", "other", "other"),
    ("2. Personal accident cover for owner-driver", "bullet", "other", "coverage", "other", "other"),
    ("a) Web development services", "bullet", "other", "other", "items", "other"),
    ("b) Server maintenance and monitoring", "bullet", "other", "other", "items", "other"),
    ("i) Licensee shall not sublicense without written consent", "bullet", "other", "other", "other", "obligations"),
    ("ii) Licensor reserves all intellectual property rights", "bullet", "other", "other", "other", "clauses"),
]


# ---------------------------------------------------------------------------
# Data augmentation: word-dropout
# ---------------------------------------------------------------------------

def _word_dropout(text: str, rng: np.random.RandomState, drop_rate: float = 0.15) -> str:
    """Randomly drop words from a text string for augmentation."""
    words = text.split()
    if len(words) <= 2:
        return text
    keep = [w for w in words if rng.random() > drop_rate]
    return " ".join(keep) if keep else text


# ---------------------------------------------------------------------------
# LineRoleClassifier — multi-head NumPy MLP
# ---------------------------------------------------------------------------

class LineRoleClassifier:
    """Multi-head MLP for line role + domain category classification.

    Architecture:
        Shared: Linear(input_dim → 128) → ReLU
        5 heads: role(5), medical(7), policy(6), invoice(5), legal(4)
    """

    def __init__(self, input_dim: int = 1032):
        self.input_dim = input_dim
        self.hidden_dim = 128
        self._trained = False

        # Shared layer: (input_dim → 128)
        self.W_shared = np.random.randn(input_dim, self.hidden_dim).astype(np.float32) * np.sqrt(2.0 / input_dim)
        self.b_shared = np.zeros(self.hidden_dim, dtype=np.float32)

        # Per-head output layers
        self.heads: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        for name, labels in HEAD_NAMES.items():
            n_out = len(labels)
            W = np.random.randn(self.hidden_dim, n_out).astype(np.float32) * np.sqrt(2.0 / self.hidden_dim)
            b = np.zeros(n_out, dtype=np.float32)
            self.heads[name] = (W, b)

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        """Row-wise numerically stable softmax."""
        if x.ndim == 1:
            x = x.reshape(1, -1)
        shifted = x - x.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        return exp / exp.sum(axis=1, keepdims=True)

    def _forward_shared(self, X: np.ndarray) -> np.ndarray:
        """Shared layer forward pass → ReLU activations."""
        return np.maximum(0, X @ self.W_shared + self.b_shared)

    def _forward_head(self, hidden: np.ndarray, head_name: str) -> np.ndarray:
        """Single-head forward pass → softmax probabilities."""
        W, b = self.heads[head_name]
        logits = hidden @ W + b
        return self._softmax(logits)

    def _forward_all(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Forward pass through shared layer + all heads."""
        hidden = self._forward_shared(X)
        return {name: self._forward_head(hidden, name) for name in self.heads}

    # --- Training ---

    def train(
        self,
        embedder: Any,
        epochs: int = 500,
        lr: float = 0.3,
        augment_factor: int = 3,
    ) -> List[float]:
        """Train on self-supervised templates with word-dropout augmentation."""
        rng = np.random.RandomState(42)

        # Build augmented dataset
        texts: List[str] = []
        labels: Dict[str, List[int]] = {name: [] for name in HEAD_NAMES}

        # Build label indices
        label_to_idx = {
            name: {lbl: i for i, lbl in enumerate(lbls)}
            for name, lbls in HEAD_NAMES.items()
        }

        for text, role, med, pol, inv, leg in TRAINING_TEMPLATES:
            if not text.strip():
                continue
            # Original + augmented copies
            variants = [text]
            for _ in range(augment_factor - 1):
                variants.append(_word_dropout(text, rng))

            for v in variants:
                texts.append(v)
                labels["role"].append(label_to_idx["role"][role])
                labels["medical"].append(label_to_idx["medical"][med])
                labels["policy"].append(label_to_idx["policy"][pol])
                labels["invoice"].append(label_to_idx["invoice"][inv])
                labels["legal"].append(label_to_idx["legal"][leg])

        if not texts:
            log.warning("No training texts after filtering")
            return []

        # Encode all texts
        try:
            embeddings = embedder.encode(texts, normalize_embeddings=True)
            if not isinstance(embeddings, np.ndarray):
                embeddings = np.array(embeddings)
        except Exception:
            log.error("Failed to encode training texts for line classifier", exc_info=True)
            return []

        # Compute layout features for each text
        layout = np.array([_layout_features(t) for t in texts], dtype=np.float32)
        X = np.hstack([embeddings, layout]).astype(np.float32)

        # Update input_dim if needed
        if X.shape[1] != self.input_dim:
            old_dim = self.input_dim
            self.input_dim = X.shape[1]
            self.W_shared = np.random.randn(self.input_dim, self.hidden_dim).astype(np.float32) * np.sqrt(2.0 / self.input_dim)
            log.info("Input dim adjusted from %d to %d", old_dim, self.input_dim)

        # Build one-hot label matrices
        Y: Dict[str, np.ndarray] = {}
        for name, idxs in labels.items():
            n_classes = len(HEAD_NAMES[name])
            mat = np.zeros((len(idxs), n_classes), dtype=np.float32)
            for i, idx in enumerate(idxs):
                mat[i, idx] = 1.0
            Y[name] = mat

        n = len(X)
        losses: List[float] = []

        for epoch in range(epochs):
            # Forward shared
            hidden = self._forward_shared(X)

            total_loss = 0.0
            # Gradient accumulators for shared layer
            dW_shared = np.zeros_like(self.W_shared)
            db_shared = np.zeros_like(self.b_shared)

            for name in HEAD_NAMES:
                W_h, b_h = self.heads[name]
                logits = hidden @ W_h + b_h
                probs = self._softmax(logits)

                # Cross-entropy loss
                eps = 1e-7
                loss = -np.mean(np.sum(Y[name] * np.log(probs + eps), axis=1))
                total_loss += loss

                # Backward through head
                dlogits = (probs - Y[name]) / n
                dW_h = hidden.T @ dlogits
                db_h = np.sum(dlogits, axis=0)

                # Gradient into shared hidden
                dhidden = dlogits @ W_h.T
                dhidden[hidden <= 0] = 0  # ReLU gradient

                dW_shared += X.T @ dhidden
                db_shared += np.sum(dhidden, axis=0)

                # Update head weights
                self.heads[name] = (W_h - lr * dW_h, b_h - lr * db_h)

            # Update shared weights (sum of gradients from all heads)
            self.W_shared -= lr * dW_shared
            self.b_shared -= lr * db_shared

            losses.append(float(total_loss))

        self._trained = True
        log.info(
            "Line classifier trained: %d samples, %d epochs, final_loss=%.4f",
            n, epochs, losses[-1] if losses else float("nan"),
        )
        return losses

    # --- Prediction ---

    def predict_single(self, x: np.ndarray, domain: str = "medical") -> LineClassification:
        """Predict role + category for a single feature vector."""
        x = x.reshape(1, -1)
        if x.shape[1] != self.input_dim:
            return _heuristic_classify_from_features(x[0], domain)

        hidden = self._forward_shared(x)

        # Role head
        role_probs = self._forward_head(hidden, "role")[0]
        role_idx = int(np.argmax(role_probs))
        role = ROLE_NAMES[role_idx]
        role_conf = float(role_probs[role_idx])

        # Domain head
        head_name = _DOMAIN_HEAD.get(domain, "medical")
        cat_probs = self._forward_head(hidden, head_name)[0]
        cat_idx = int(np.argmax(cat_probs))
        cat_names = HEAD_NAMES[head_name]
        category = cat_names[cat_idx]
        cat_conf = float(cat_probs[cat_idx])

        return LineClassification(
            role=role,
            role_confidence=role_conf,
            category=category,
            category_confidence=cat_conf,
            label="",
            value="",
        )

    def predict_batch(
        self, X: np.ndarray, domain: str = "medical", lines: Optional[List[str]] = None,
    ) -> List[LineClassification]:
        """Batch predict for multiple feature vectors."""
        if X.shape[1] != self.input_dim:
            return [_heuristic_classify_from_features(X[i], domain) for i in range(len(X))]

        hidden = self._forward_shared(X)

        role_probs = self._forward_head(hidden, "role")
        role_idxs = np.argmax(role_probs, axis=1)
        role_confs = np.max(role_probs, axis=1)

        head_name = _DOMAIN_HEAD.get(domain, "medical")
        cat_probs = self._forward_head(hidden, head_name)
        cat_idxs = np.argmax(cat_probs, axis=1)
        cat_confs = np.max(cat_probs, axis=1)
        cat_names = HEAD_NAMES[head_name]

        results = []
        for i in range(len(X)):
            role = ROLE_NAMES[int(role_idxs[i])]
            role_conf = float(role_confs[i])
            category = cat_names[int(cat_idxs[i])]
            cat_conf = float(cat_confs[i])

            # Determine label/value based on role
            label, value = "", ""
            if lines is not None and i < len(lines):
                line = lines[i]
                if role == "kv_pair" and role_conf >= 0.50:
                    label, value = _split_at_colon(line)
                    if not label:
                        value = line
                elif role == "heading":
                    value = line.rstrip(":")
                else:
                    value = line

                # Confidence gate: fall back to heuristic for low-confidence predictions
                if role_conf < 0.50:
                    heuristic = _heuristic_classify(line, domain)
                    role = heuristic.role
                    role_conf = heuristic.role_confidence
                    category = heuristic.category
                    cat_conf = heuristic.category_confidence
                    label = heuristic.label
                    value = heuristic.value

            results.append(LineClassification(
                role=role,
                role_confidence=role_conf,
                category=category,
                category_confidence=cat_conf,
                label=label,
                value=value,
            ))

        return results

    # --- Persistence ---

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "W_shared": self.W_shared,
            "b_shared": self.b_shared,
            "heads": {name: (W.copy(), b.copy()) for name, (W, b) in self.heads.items()},
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "_trained": self._trained,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        log.info("Line classifier saved to %s", path)

    def load(self, path: str | Path) -> None:
        path = Path(path)
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.W_shared = data["W_shared"]
        self.b_shared = data["b_shared"]
        self.heads = data["heads"]
        self.input_dim = data.get("input_dim", 1032)
        self.hidden_dim = data.get("hidden_dim", 128)
        self._trained = data.get("_trained", True)
        log.info("Line classifier loaded from %s", path)


# ---------------------------------------------------------------------------
# Heuristic fallback (no regex, no embedder needed)
# ---------------------------------------------------------------------------

def _heuristic_classify(line: str, domain: str) -> LineClassification:
    """Classify a line using simple heuristics (colon split, bullet detection).

    When embedder is unavailable, this provides keyword-based category routing
    to preserve the same extraction quality as the old regex-based extractors.
    """
    stripped = line.strip()
    if not stripped or len(stripped) < 3:
        return LineClassification("skip", 1.0, "other", 1.0, "", "")

    # Check for heading (line ends with colon, short, no value)
    if stripped.endswith(":") and len(stripped) < 60:
        inner = stripped[:-1].strip()
        if inner and not inner[-1].isdigit():
            cat = _keyword_category(inner, domain)
            return LineClassification("heading", 0.8, cat, 0.6, "", inner)

    # Check for bullet
    if stripped[0] in _BULLET_CHARS:
        content = stripped[1:].strip()
        cat = _keyword_category(content, domain)
        return LineClassification("bullet", 0.8, cat, 0.5, "", content)

    # Check for numbered list
    i = 0
    while i < len(stripped) and stripped[i].isdigit():
        i += 1
    if i > 0 and i < len(stripped) and stripped[i] in ".):":
        content = stripped[i + 1:].strip()
        cat = _keyword_category(content, domain)
        return LineClassification("bullet", 0.7, cat, 0.5, "", content)

    # Check for KV pair (colon-based split)
    label, value = _split_at_colon(stripped)
    if label and value and len(value) >= 2:
        cat = _keyword_category(label, domain)
        return LineClassification("kv_pair", 0.7, cat, 0.6, label, value)

    # Default: narrative
    cat = _keyword_category(stripped, domain)
    return LineClassification("narrative", 0.6, cat, 0.4, "", stripped)


# ---------------------------------------------------------------------------
# Keyword-based category routing (heuristic fallback)
# ---------------------------------------------------------------------------
_MEDICAL_KEYWORDS: Dict[str, set] = {
    "patient_info": {
        "patient", "name", "age", "dob", "date of birth", "gender", "sex", "mrn",
        "medical record", "patient id", "admission", "discharge", "blood group",
        "blood type", "address", "phone", "emergency contact", "insurance",
        "allergies", "allergy", "doctor", "physician", "attending",
    },
    "diagnoses": {
        "diagnosis", "diagnoses", "condition", "assessment", "impression",
        "chief complaint", "presenting complaint", "history of present illness",
        "clinical finding", "clinical impression", "primary diagnosis",
        "secondary diagnosis", "comorbidities", "icd",
    },
    "medications": {
        "medication", "medications", "drug", "drugs", "prescription", "dosage",
        "dose", "medicine", "treatment", "therapy", "prescribed", "rx",
        "frequency", "route", "prn",
    },
    "procedures": {
        "procedure", "procedures", "surgery", "surgical", "operation",
        "intervention", "treatment plan", "plan of care", "care plan",
        "imaging", "radiology", "x-ray", "mri", "ct scan", "ultrasound",
    },
    "lab_results": {
        "lab", "laboratory", "test", "result", "results", "blood test",
        "cbc", "bmp", "cmp", "hba1c", "hemoglobin", "wbc", "rbc", "platelet",
        "creatinine", "glucose", "cholesterol", "triglyceride", "bilirubin",
        "alt", "ast", "tsh", "esr", "crp", "urinalysis",
    },
    "vitals": {
        "vital", "vitals", "blood pressure", "bp", "heart rate", "pulse",
        "temperature", "temp", "respiratory rate", "oxygen saturation", "spo2",
        "weight", "height", "bmi",
    },
}

_POLICY_KEYWORDS: Dict[str, set] = {
    "policy_info": {
        "policy", "policy number", "insured", "policyholder", "proposer",
        "insurer", "underwriter", "effective date", "inception", "expiry",
        "period", "vehicle", "registration", "make", "model", "vin",
        "chassis", "engine", "type of cover", "plan", "sum insured",
        "sum assured", "idv", "insured declared value",
    },
    "coverage": {
        "coverage", "cover", "covered", "benefit", "protection", "scope",
        "personal accident", "third party", "own damage", "comprehensive",
        "fire", "theft", "natural calamity", "flood", "earthquake",
        "liability", "bodily injury", "property damage",
    },
    "premiums": {
        "premium", "amount", "total", "net premium", "gross premium",
        "gst", "tax", "cgst", "sgst", "igst", "cess", "discount",
        "ncb", "no claim bonus", "loading", "od premium", "tp premium",
        "basic premium", "add-on", "addon", "rider", "payable",
    },
    "exclusions": {
        "exclusion", "excluded", "not covered", "exception", "limitation",
        "deductible", "excess", "waiting period", "sub-limit",
    },
    "terms": {
        "term", "condition", "clause", "provision", "warranty",
        "endorsement", "renewal", "cancellation", "claim", "claims procedure",
        "grievance", "dispute", "arbitration", "jurisdiction",
    },
}

_INVOICE_KEYWORDS: Dict[str, set] = {
    "items": {
        "item", "product", "service", "description", "line item",
    },
    "totals": {
        "total", "amount due", "subtotal", "balance due", "grand total", "tax",
    },
    "parties": {
        "bill to", "billed to", "invoice to", "from", "vendor", "customer",
    },
    "terms": {
        "payment terms", "due date", "terms", "invoice number", "invoice date", "po number",
    },
}

_LEGAL_KEYWORDS: Dict[str, set] = {
    "clauses": {
        "clause", "section", "article", "paragraph", "governing law",
        "effective date", "term", "jurisdiction", "confidentiality",
        "termination", "definitions",
    },
    "parties": {
        "party", "parties", "between", "licensor", "licensee", "vendor",
        "client", "employer", "employee", "landlord", "tenant", "buyer",
        "seller", "contractor", "subcontractor",
    },
    "obligations": {
        "shall", "must", "agrees to", "obligated", "undertakes", "covenants",
        "liability", "indemnify", "warrant",
    },
}

_DOMAIN_KEYWORD_MAP: Dict[str, Dict[str, set]] = {
    "medical": _MEDICAL_KEYWORDS,
    "policy": _POLICY_KEYWORDS,
    "invoice": _INVOICE_KEYWORDS,
    "legal": _LEGAL_KEYWORDS,
    "contract": _LEGAL_KEYWORDS,
}


def _keyword_category(text: str, domain: str) -> str:
    """Match text against domain-specific keyword sets. Returns category or 'other'."""
    kw_map = _DOMAIN_KEYWORD_MAP.get(domain)
    if not kw_map:
        return "other"
    text_lower = text.lower()
    for category, keywords in kw_map.items():
        if any(kw in text_lower for kw in keywords):
            return category
    return "other"


def _heuristic_classify_from_features(feat_vec: np.ndarray, domain: str) -> LineClassification:
    """Fallback when classifier dimensions mismatch."""
    return LineClassification("narrative", 0.3, "other", 0.3, "", "")


# ---------------------------------------------------------------------------
# Singleton management
# ---------------------------------------------------------------------------
_classifier_instance: Optional[LineRoleClassifier] = None
_classifier_lock = threading.Lock()
_DEFAULT_MODEL_PATH = Path("models/line_role_classifier.pkl")


def get_line_classifier() -> Optional[LineRoleClassifier]:
    """Return the singleton line classifier, or None if not initialized."""
    return _classifier_instance


def set_line_classifier(clf: Optional[LineRoleClassifier]) -> None:
    """Set the singleton (for testing)."""
    global _classifier_instance
    _classifier_instance = clf


def ensure_line_classifier(
    embedder: Any,
    model_path: Optional[Path] = None,
) -> LineRoleClassifier:
    """Load or train the line classifier singleton. Thread-safe."""
    global _classifier_instance
    with _classifier_lock:
        if _classifier_instance is not None:
            return _classifier_instance

        path = model_path or _DEFAULT_MODEL_PATH
        clf = LineRoleClassifier()

        if path.exists():
            try:
                clf.load(path)
                _classifier_instance = clf
                return clf
            except Exception:
                log.warning("Failed to load line classifier from %s, retraining", path, exc_info=True)

        clf.train(embedder)
        try:
            clf.save(path)
        except Exception:
            log.warning("Failed to save line classifier to %s", path, exc_info=True)

        _classifier_instance = clf
        return clf


# ---------------------------------------------------------------------------
# High-level API: classify_lines()
# ---------------------------------------------------------------------------

def classify_lines(
    lines: List[str],
    domain: str,
    embedder: Any = None,
) -> List[LineClassification]:
    """Batch-classify lines using trained MLP, falling back to heuristic.

    Parameters
    ----------
    lines : list of str
        Raw text lines to classify.
    domain : str
        Document domain (medical, policy, invoice, legal).
    embedder : Any, optional
        Sentence-transformer embedder for semantic features.
        When None, falls back to heuristic classification.

    Returns
    -------
    list of LineClassification
    """
    clf = get_line_classifier()
    if clf is None or not clf._trained or embedder is None:
        return [_heuristic_classify(line, domain) for line in lines]

    if not lines:
        return []

    try:
        embeddings = embedder.encode(lines, normalize_embeddings=True)
        if not isinstance(embeddings, np.ndarray):
            embeddings = np.array(embeddings)
        # Validate shape: must be (n_lines, embed_dim)
        if embeddings.ndim != 2 or embeddings.shape[0] != len(lines):
            log.debug("Embedder returned shape %s for %d lines, falling back to heuristic", embeddings.shape, len(lines))
            return [_heuristic_classify(line, domain) for line in lines]
    except Exception:
        log.warning("Embedder failed in classify_lines, falling back to heuristic", exc_info=True)
        return [_heuristic_classify(line, domain) for line in lines]

    layout = np.array([_layout_features(line) for line in lines], dtype=np.float32)
    X = np.hstack([embeddings, layout]).astype(np.float32)

    return clf.predict_batch(X, domain=domain, lines=lines)
