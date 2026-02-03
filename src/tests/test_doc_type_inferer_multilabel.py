from src.docwain_intel.doc_type_inferer import infer_doc_type


def test_infer_invoice():
    text = "Invoice\nBill To: Acme Corp\nAmount Due: $1,200"
    result = infer_doc_type(text)
    assert result["doc_domain"] == "finance"
    assert result["doc_kind"] == "invoice"


def test_infer_medical():
    text = "Patient Name: John Doe\nDiagnosis: Hypertension\nMedication: Lisinopril"
    result = infer_doc_type(text)
    assert result["doc_domain"] == "medical"
    assert result["doc_kind"] == "medical_document"


def test_infer_linkedin():
    text = "LinkedIn Profile\nlinkedin.com/in/jane-doe\nConnections: 500+"
    result = infer_doc_type(text)
    assert result["doc_domain"] == "hr"
    assert result["doc_kind"] == "linkedin_profile"
