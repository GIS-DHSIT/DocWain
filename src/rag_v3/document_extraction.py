"""
Document-Level HR Extraction from Complete Resume Data

This module provides intelligent HR extraction that works at the document level,
leveraging complete extracted resume data instead of relying on fragmented vector chunks.

This approach ensures:
1. Full context is available (no truncation)
2. All candidate information is properly extracted
3. No reliance on incomplete chunks from Qdrant
"""

from typing import Any, Dict, List, Optional
import re
from src.utils.logging_utils import get_logger
from datetime import datetime

logger = get_logger(__name__)

_REPR_PREFIX_RE = re.compile(r"^(?:Extracted\s*Document|ExtractedDocument)\s*\(\s*full_text='", re.IGNORECASE)

def _unwrap_extracted_document_repr(text: str) -> str:
    """Strip ExtractedDocument repr wrapper if present, returning the actual full_text content."""
    if not text or not _REPR_PREFIX_RE.match(text):
        return text
    stripped = _REPR_PREFIX_RE.sub("", text)
    # Remove trailing ') or ',field=...) at the end
    stripped = re.sub(r"(?:',\s*\w+=.*|'\s*\)\s*)$", "", stripped, flags=re.DOTALL)
    return stripped.replace("\\n", "\n").strip()

def extract_hr_from_complete_document(document_data: Any) -> Dict[str, Any]:
    """
    Extract HR/candidate information from COMPLETE document structures.

    Handles:
    1. ResumeScreeningDetailedResponse objects
    2. ResumeProfile objects
    3. StructuredDocument dataclass (converted to dict)
    4. Dictionary formats with raw_text/sections

    Args:
        document_data: Complete extracted document from pickle

    Returns:
        Extracted candidate fields dictionary
    """

    if not document_data:
        return _empty_candidate()

    try:
        # If it has candidate_profile attribute, it's a ResumeScreeningDetailedResponse
        if hasattr(document_data, 'candidate_profile'):
            candidate_profile = getattr(document_data, 'candidate_profile', None)
            if candidate_profile and hasattr(candidate_profile, 'extracted'):
                return _extract_from_resume_profile(candidate_profile.extracted)

        # If it has name, experience, skills attrs, it's a ResumeProfile
        if hasattr(document_data, 'name') and hasattr(document_data, 'skills'):
            return _extract_from_resume_profile(document_data)

        # If it's a dictionary
        if isinstance(document_data, dict):
            # CRITICAL: Check if this is a StructuredDocument format (from structured_extraction.py)
            # It has 'raw_text', 'sections', 'document_classification' instead of direct fields
            if 'raw_text' in document_data or 'sections' in document_data:
                logger.info("Processing StructuredDocument format for HR extraction")
                return _extract_from_structured_document(document_data)

            # Legacy format: direct field names like 'name', 'skills', etc.
            return _extract_from_dict(document_data)

    except Exception as e:
        logger.warning(f"Error in extract_hr_from_complete_document: {e}")

    return _empty_candidate()

def _extract_from_resume_profile(resume_profile: Any) -> Dict[str, Any]:
    """Extract from ResumeProfile object with correct Pydantic field names."""

    result = _empty_candidate()

    if not resume_profile:
        return result

    try:
        # Name - from name or headline
        name = getattr(resume_profile, 'name', None)
        if not name:
            headline = getattr(resume_profile, 'headline', None)
            if headline:
                name = headline.split(' - ')[0] if ' - ' in headline else headline[:50]
        if name:
            result["name"] = name

        # Skills (called 'skills' in ResumeProfile, not 'technical_skills')
        skills = getattr(resume_profile, 'skills', []) or []
        if skills:
            result["technical_skills"] = [str(s) for s in skills if s]

        # Experience
        experiences = getattr(resume_profile, 'experience', []) or []
        if experiences:
            # Try to calculate years
            exp_summary = _build_experience_summary(experiences)
            if exp_summary.get('summary'):
                result["experience_summary"] = exp_summary['summary']
            if exp_summary.get('years'):
                result["total_years_experience"] = exp_summary['years']

        # Education (note: field is 'field' not 'field_of_study')
        educations = getattr(resume_profile, 'education', []) or []
        edu_list = []
        for edu in educations:
            degree = getattr(edu, 'degree', None)
            institution = getattr(edu, 'institution', None)
            field = getattr(edu, 'field', None)  # Note: 'field' not 'field_of_study'

            if degree and institution:
                edu_str = f"{degree} from {institution}"
                if field:
                    edu_str += f" in {field}"
                edu_list.append(edu_str)
            elif institution:
                edu_list.append(str(institution))

        if edu_list:
            result["education"] = edu_list

        # Certifications
        certs = getattr(resume_profile, 'certifications', []) or []
        cert_list = []
        for cert in certs:
            name = getattr(cert, 'name', None)
            issuer = getattr(cert, 'issuer', None)

            if name:
                cert_str = str(name)
                if issuer:
                    cert_str += f" ({issuer})"
                cert_list.append(cert_str)

        if cert_list:
            result["certifications"] = cert_list

        # Summary
        summary = getattr(resume_profile, 'summary', None)
        if summary and not result.get("experience_summary"):
            result["experience_summary"] = summary[:500]

        # Extract contact info from links
        links = getattr(resume_profile, 'links', []) or []
        for link in links:
            if not link:
                continue
            link_str = str(link)
            link_lower = link_str.lower()

            # Email detection
            if '@' in link_str:
                if link_str not in result["email"]:
                    result["email"].append(link_str)
            # LinkedIn detection
            elif 'linkedin' in link_lower:
                if link_str not in result["linkedin"]:
                    result["linkedin"].append(link_str)
            # Phone detection (starts with + or multiple digits)
            elif link_str.startswith('+') or (len(link_str) > 5 and sum(c.isdigit() for c in link_str) > 4):
                if link_str not in result["phone"]:
                    result["phone"].append(link_str)

        result["source_type"] = "Resume"

    except Exception as e:
        logger.debug(f"Error extracting from ResumeProfile: {e}")

    return result

def _build_experience_summary(experiences: List[Any]) -> Dict[str, Any]:
    """Build experience summary and calculate years from experience list."""
    result = {}

    try:
        if not experiences:
            return result

        # Get most recent/first experience for summary
        exp = experiences[0]
        title = getattr(exp, 'title', None)
        company = getattr(exp, 'company', None)
        description = getattr(exp, 'description', None)

        if title and company:
            summary = f"{title} at {company}"
            if description:
                desc_part = str(description)[:150]
                summary += f": {desc_part}"
            result["summary"] = summary[:500]

        # Calculate total years
        total_months = 0
        for exp in experiences:
            start = getattr(exp, 'start_date', None)
            end = getattr(exp, 'end_date', None)

            if start and end:
                try:
                    if isinstance(start, str):
                        from datetime import datetime
                        start = datetime.fromisoformat(start[:10])  # Take only YYYY-MM-DD
                    if isinstance(end, str):
                        from datetime import datetime
                        end = datetime.fromisoformat(end[:10])

                    delta = (end - start).days
                    total_months += delta / 30.5
                except Exception as exc:
                    logger.debug("Failed to calculate experience duration from dates", exc_info=True)

        if total_months >= 12:
            years = total_months / 12
            if years == int(years):
                result["years"] = f"{int(years)} years"
            else:
                result["years"] = f"{years:.1f} years"

    except Exception as e:
        logger.debug(f"Error building experience summary: {e}")

    return result

def _extract_from_structured_document(document_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract HR/candidate information from document data.

    Handles multiple formats:
    1. ExtractedDocument format: full_text, sections with {title, text}
    2. StructuredDocument format: raw_text, sections with {section_type, content}
    3. canonical_json format: sections with title/text
    """
    result = _empty_candidate()

    # Get full text - check both field names, unwrap ExtractedDocument repr if needed
    full_text = document_data.get('full_text', '') or document_data.get('raw_text', '') or ''
    full_text = _unwrap_extracted_document_repr(full_text)

    # Get sections - could be list of Section objects or dicts
    sections = document_data.get('sections', []) or []

    # Also check canonical_json for structured section data
    canonical_json = document_data.get('canonical_json', {}) or {}
    if canonical_json and not sections:
        sections = canonical_json.get('sections', []) or []

    logger.info(f"Document extraction: {len(sections)} sections, {len(full_text)} chars text")

    # STEP 1: Extract from sections by TITLE (ExtractedDocument format)
    # Section titles like 'ABINAYA P', 'KEY SKILLS', 'CERTIFICATION', etc.
    for section in sections:
        # Handle both dict and object formats
        if hasattr(section, 'title'):
            title = getattr(section, 'title', '') or ''
            text = getattr(section, 'text', '') or ''
        else:
            title = section.get('title', '') or section.get('section_title', '') or ''
            text = section.get('text', '') or section.get('content', '') or ''

        text = _unwrap_extracted_document_repr(text)
        if not text:
            continue

        title_lower = title.lower().strip()

        # Name section - usually first section with person's name as title
        # Check if title looks like a name (2-4 capitalized words, no common headers)
        if not result["name"] and _title_looks_like_name(title):
            result["name"] = title.strip()
            # Also extract contact from the text below the name
            contact = _extract_contact_from_text(text)
            if contact.get('emails'):
                result["email"] = contact['emails']
            if contact.get('phones'):
                result["phone"] = contact['phones']

        # Objective/Summary section
        elif any(kw in title_lower for kw in ['objective', 'summary', 'profile', 'about']):
            if not result["experience_summary"]:
                result["experience_summary"] = _truncate(_clean_text(text), 300)

        # Professional Summary / Experience section
        elif any(kw in title_lower for kw in ['professional summary', 'experience', 'employment', 'work history']):
            if not result["experience_summary"]:
                result["experience_summary"] = _truncate(_clean_text(text), 300)
            # Try to extract years from experience section
            if not result["total_years_experience"]:
                result["total_years_experience"] = _extract_years_of_experience(text)

        # Skills section
        elif any(kw in title_lower for kw in ['skill', 'competenc', 'expertise', 'technologies']):
            extracted_skills = _extract_skills_from_section_text(text)
            result["technical_skills"] = _merge_lists(result["technical_skills"], extracted_skills)

        # Education section
        elif any(kw in title_lower for kw in ['education', 'academic', 'qualification', 'degree']):
            extracted_edu = _extract_education_items(text)
            result["education"] = _merge_lists(result["education"], extracted_edu)

        # Certification section - may also contain education items
        elif any(kw in title_lower for kw in ['certification', 'certificate', 'credential', 'license']):
            extracted_certs = _extract_certification_items(text)
            result["certifications"] = _merge_lists(result["certifications"], extracted_certs)
            # Also extract any education items that might be mixed in
            extracted_edu = _extract_education_from_cert_section(text)
            result["education"] = _merge_lists(result["education"], extracted_edu)

        # Achievements/Awards section
        elif any(kw in title_lower for kw in ['achievement', 'award', 'recognition', 'honor', 'accomplishment']):
            extracted_achievements = _extract_achievement_items(text)
            result["achievements"] = _merge_lists(result["achievements"], extracted_achievements)

        # Personal Details - extract contact info
        elif any(kw in title_lower for kw in ['personal', 'contact', 'details']):
            contact = _extract_contact_from_text(text)
            if contact.get('emails') and not result["email"]:
                result["email"] = contact['emails']
            if contact.get('phones') and not result["phone"]:
                result["phone"] = contact['phones']
            if contact.get('linkedins') and not result["linkedin"]:
                result["linkedin"] = contact['linkedins']

    # STEP 2: Fallback to full_text extraction for missing fields
    if full_text:
        # Extract name from first few lines if still missing
        if not result["name"]:
            result["name"] = _extract_name_from_full_text(full_text)

        # Extract contact info if missing
        if not result["email"]:
            result["email"] = _extract_emails(full_text)
        if not result["phone"]:
            result["phone"] = _extract_phones(full_text)
        if not result["linkedin"]:
            result["linkedin"] = _extract_linkedin(full_text)

        # Extract skills if missing
        if not result["technical_skills"]:
            result["technical_skills"] = _extract_technical_skills(full_text)

        if not result["functional_skills"]:
            result["functional_skills"] = _extract_functional_skills(full_text)

        # Extract education if missing
        if not result["education"]:
            result["education"] = _extract_education(full_text)

        # Extract certifications if missing
        if not result["certifications"]:
            result["certifications"] = _extract_certifications(full_text)

        # Extract experience summary if missing
        if not result["experience_summary"]:
            result["experience_summary"] = _extract_experience_summary(full_text)

        # Extract years if missing
        if not result["total_years_experience"]:
            result["total_years_experience"] = _extract_years_of_experience(full_text)

    result["source_type"] = "Resume"

    # Log extraction results
    filled_fields = [k for k, v in result.items() if v and v != []]
    logger.info(f"Extracted HR fields: {filled_fields}")

    return result

def _title_looks_like_name(title: str) -> bool:
    """Check if a section title looks like a person's name."""
    if not title or len(title) > 50:
        return False

    title_lower = title.lower()

    # Skip common section headers
    skip_keywords = [
        'objective', 'summary', 'profile', 'experience', 'education',
        'skill', 'certification', 'achievement', 'award', 'contact',
        'personal', 'details', 'reference', 'project', 'hobby',
        'interest', 'language', 'declaration', 'responsibility', 'role',
        'content', 'document', 'section', 'untitled', 'introduction',
    ]
    if any(kw in title_lower for kw in skip_keywords):
        return False

    # Check if it looks like a name (2-4 words, mostly letters, capitalized)
    words = title.split()
    if len(words) < 1 or len(words) > 4:
        return False

    # Most characters should be letters
    letter_ratio = sum(1 for c in title if c.isalpha()) / max(len(title), 1)
    if letter_ratio < 0.8:
        return False

    return True

def _clean_text(text: str, preserve_newlines: bool = False) -> str:
    """Clean text by removing special characters and normalizing whitespace."""
    if not text:
        return ""
    # Remove unicode bullets and special chars
    text = re.sub(r'[\uf0b7\uf0d8\uf0a7]', '', text)
    text = re.sub(r'\\u[0-9a-fA-F]{4}', '', text)

    if preserve_newlines:
        # Normalize horizontal whitespace only, preserve newlines
        text = re.sub(r'[^\S\n]+', ' ', text)
        text = re.sub(r'\n\s*\n+', '\n', text)  # Collapse multiple newlines
    else:
        # Normalize all whitespace
        text = re.sub(r'\s+', ' ', text)

    return text.strip()

def _extract_skills_from_section_text(text: str) -> List[str]:
    """Extract skills from a skills section text."""
    skills = []

    if not text:
        return skills

    # Clean text but preserve newlines for splitting
    text = _clean_text(text, preserve_newlines=True)

    # Split by newlines, bullets, or other delimiters
    lines = text.split('\n')

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Remove leading bullet characters
        line = re.sub(r'^[\-\*•·◦▪►]\s*', '', line)
        line = line.strip()

        if not line or len(line) < 3:
            continue

        # If line contains comma-separated items, split them
        if ',' in line and len(line) < 150:
            parts = [p.strip() for p in line.split(',')]
            for part in parts:
                if part and len(part) > 2 and len(part) < 60:
                    skills.append(part)
        elif len(line) < 80:
            # Single skill per line
            skills.append(line)

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for s in skills:
        key = s.lower()
        if key not in seen:
            seen.add(key)
            unique.append(s)

    return unique[:25]

def _extract_education_items(text: str) -> List[str]:
    """Extract education items from education section text."""
    items = []

    if not text:
        return items

    # Clean text preserving newlines
    text = _clean_text(text, preserve_newlines=True)

    # Split by lines
    lines = text.split('\n')

    for line in lines:
        line = line.strip()
        if not line or len(line) < 5:
            continue

        # Remove bullet characters
        line = re.sub(r'^[\-\*•·]\s*', '', line)
        line = line.strip()

        # Skip if too short or too long
        if len(line) < 10 or len(line) > 200:
            continue

        # Skip if it looks like personal details
        if any(kw in line.lower() for kw in ['date of birth', 'gender', 'nationality', 'language']):
            continue

        items.append(line)

    return items[:5]

def _extract_certification_items(text: str) -> List[str]:
    """Extract certification items from certification section text."""
    items = []

    # Clean text
    text = _clean_text(text)

    # Split by lines or bullets
    lines = re.split(r'[\n•\uf0b7]', text)

    for line in lines:
        line = _clean_text(line)
        if not line or len(line) < 3:
            continue

        # Skip if it looks like education info mixed in
        if any(kw in line.lower() for kw in ['university', 'college', 'school', 'degree']):
            # But keep if it's clearly a certification
            if 'certified' not in line.lower() and 'certification' not in line.lower():
                continue

        items.append(line)

    return items[:10]

def _extract_achievement_items(text: str) -> List[str]:
    """Extract achievement items from achievements section text."""
    items = []

    # Clean text
    text = _clean_text(text)

    # Split by lines or bullets
    lines = re.split(r'[\n•\uf0b7]', text)

    for line in lines:
        line = _clean_text(line)
        if line and len(line) > 10 and len(line) < 200:
            items.append(line)

    return items[:5]

def _extract_education_from_cert_section(text: str) -> List[str]:
    """Extract education items that may be mixed into certification sections."""
    items = []

    if not text:
        return items

    # Clean text preserving newlines
    text = _clean_text(text, preserve_newlines=True)

    # Education keywords that indicate education rather than certification
    edu_keywords = ['university', 'college', 'school', 'institute', 'b.tech', 'btech',
                    'm.tech', 'mtech', 'bsc', 'b.sc', 'msc', 'm.sc', 'mba', 'bba',
                    'bachelor', 'master', 'ph.d', 'phd', 'b.a', 'b.e', 'm.a', 'm.e',
                    'diploma', 'degree']

    lines = text.split('\n')

    for line in lines:
        line = line.strip()
        if not line or len(line) < 10:
            continue

        line_lower = line.lower()

        # Check if line contains education keywords
        if any(kw in line_lower for kw in edu_keywords):
            # Skip if it's clearly a certification line
            if 'certified' in line_lower or 'certification' in line_lower:
                continue
            items.append(line)

    return items[:5]

def _extract_name_from_full_text(text: str) -> Optional[str]:
    """Extract candidate name from full document text."""
    if not text:
        return None

    # Clean and get first few lines
    lines = text.split('\n')[:15]

    for line in lines:
        line = _clean_text(line)
        if not line or len(line) < 3 or len(line) > 50:
            continue

        # Skip lines with contact info or section headers
        if any(skip in line.lower() for skip in [
            '@', 'http', 'phone', 'email', 'objective', 'summary',
            'skill', 'education', 'experience', 'address', '+91', '+1'
        ]):
            continue

        # Check if it looks like a name
        words = line.split()
        if 2 <= len(words) <= 4:
            # Check if words are capitalized
            if all(w[0].isupper() for w in words if w and w[0].isalpha()):
                # Check letter ratio
                letter_ratio = sum(1 for c in line if c.isalpha()) / max(len(line), 1)
                if letter_ratio > 0.85:
                    return line

    return None

def _extract_name_from_section(content: str) -> Optional[str]:
    """Extract name from contact/header section."""
    if not content:
        return None

    lines = content.split('\n')
    for line in lines[:5]:  # Check first 5 lines
        cleaned = line.strip()
        if not cleaned or len(cleaned) > 60:
            continue

        # Skip if it looks like a header or contains contact info
        if any(skip in cleaned.lower() for skip in ['email', 'phone', 'linkedin', 'contact', '@', 'http']):
            continue

        # Check if it looks like a name (2-4 capitalized words)
        words = cleaned.split()
        if 2 <= len(words) <= 4:
            if all(w[0].isupper() for w in words if w):
                # Filter out common non-name patterns
                if not any(skip in cleaned.lower() for skip in ['skills', 'experience', 'education', 'objective', 'summary']):
                    return cleaned

    return None

def _extract_contact_from_text(text: str) -> Dict[str, List[str]]:
    """Extract contact info from text."""
    emails = _extract_emails(text)
    phones = _extract_phones(text)
    linkedins = _extract_linkedin(text)
    return {"emails": emails, "phones": phones, "linkedins": linkedins}

def _extract_skills_from_content(content: str) -> List[str]:
    """Extract skills from section content."""
    skills = []

    # Split by common delimiters
    for line in content.split('\n'):
        cleaned = line.strip()
        if not cleaned:
            continue

        # Remove bullet points
        cleaned = re.sub(r'^[\-\*•·]\s*', '', cleaned)

        # Split by commas if multiple skills on one line
        if ',' in cleaned:
            parts = [p.strip() for p in cleaned.split(',')]
            skills.extend([p for p in parts if p and len(p) > 2 and len(p) < 50])
        elif len(cleaned) > 2 and len(cleaned) < 50:
            skills.append(cleaned)

    # Deduplicate
    seen = set()
    unique = []
    for s in skills:
        lower = s.lower()
        if lower not in seen:
            seen.add(lower)
            unique.append(s)

    return unique[:30]  # Return top 30

def _extract_name_from_text(text: str) -> Optional[str]:
    """Extract name from raw text."""
    if not text:
        return None

    lines = text.split('\n')[:15]

    for line in lines:
        cleaned = line.strip()
        if not cleaned or len(cleaned) > 60 or len(cleaned) < 5:
            continue

        # Skip obvious non-name lines
        if any(skip in cleaned.lower() for skip in [
            'email', 'phone', 'linkedin', 'contact', '@', 'http',
            'skills', 'experience', 'education', 'objective', 'summary',
            'certification', 'project', 'award', 'resume', 'cv'
        ]):
            continue

        # Check if looks like a name (2-4 capitalized words, mostly letters)
        words = cleaned.split()
        if 2 <= len(words) <= 4:
            if all(w[0].isupper() for w in words if w and w[0].isalpha()):
                letter_ratio = sum(1 for c in cleaned if c.isalpha()) / len(cleaned)
                if letter_ratio > 0.8:
                    return cleaned

    return None

def _merge_lists(existing: List[str], new_items: List[str]) -> List[str]:
    """Merge two lists, avoiding duplicates."""
    result = list(existing or [])
    seen = {item.lower() for item in result}
    for item in new_items or []:
        if item and item.lower() not in seen:
            result.append(item)
            seen.add(item.lower())
    return result

def _truncate(text: str, max_len: int) -> str:
    """Truncate text to max length."""
    if not text:
        return ""
    text = ' '.join(text.split())  # Normalize whitespace
    if len(text) <= max_len:
        return text
    return text[:max_len].rsplit(' ', 1)[0] + '...'

def _extract_from_dict(document_data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract from dictionary format."""

    result = _empty_candidate()

    # Extract name
    for key in ['name', 'candidate_name', 'full_name']:
        if key in document_data and document_data[key]:
            result["name"] = document_data[key]
            break

    # Extract contact info
    if 'email' in document_data and document_data['email']:
        result["email"] = [document_data['email']] if isinstance(document_data['email'], str) else document_data['email']

    if 'phone' in document_data and document_data['phone']:
        result["phone"] = [document_data['phone']] if isinstance(document_data['phone'], str) else document_data['phone']

    if 'linkedin' in document_data and document_data['linkedin']:
        result["linkedin"] = [document_data['linkedin']] if isinstance(document_data['linkedin'], str) else document_data['linkedin']

    # Extract skills
    if 'technical_skills' in document_data:
        result["technical_skills"] = document_data['technical_skills'] or []
    elif 'skills' in document_data:
        result["technical_skills"] = document_data['skills'] or []

    if 'functional_skills' in document_data:
        result["functional_skills"] = document_data['functional_skills'] or []

    # Extract education
    if 'education' in document_data:
        result["education"] = document_data['education'] or []

    # Extract certifications
    if 'certifications' in document_data:
        result["certifications"] = document_data['certifications'] or []

    # Extract experience
    if 'total_years_experience' in document_data:
        result["total_years_experience"] = document_data['total_years_experience']

    if 'experience_summary' in document_data:
        result["experience_summary"] = document_data['experience_summary']
    elif 'summary' in document_data:
        result["experience_summary"] = document_data['summary']

    # Extract achievements
    if 'achievements' in document_data:
        result["achievements"] = document_data['achievements'] or []

    return result

def _extract_content(document_data: Dict[str, Any]) -> str:
    """Extract text content from various document structures."""
    if not document_data:
        return ""

    # Try different content field names
    for key in ["content", "text", "raw_text", "full_text", "body"]:
        if key in document_data and document_data[key]:
            return str(document_data[key])

    # Try to reconstruct from sections
    sections = document_data.get("sections", {})
    if isinstance(sections, dict):
        section_texts = []
        for section_name, section_content in sections.items():
            if isinstance(section_content, str):
                section_texts.append(section_content)
            elif isinstance(section_content, list):
                section_texts.extend(str(item) for item in section_content)
        if section_texts:
            return "\n\n".join(section_texts)

    # Try concatenating all string values
    all_text = []
    for value in document_data.values():
        if isinstance(value, str):
            all_text.append(value)
        elif isinstance(value, list):
            all_text.extend(str(item) for item in value if isinstance(item, str))

    return "\n".join(all_text)

def _extract_name(content: str, document_data: Dict[str, Any]) -> Optional[str]:
    """Extract candidate name from content or metadata."""
    # Try metadata first
    if "name" in document_data:
        name = document_data["name"]
        if name and isinstance(name, str) and len(name.strip()) > 0:
            return name.strip()

    # Look for name patterns in content
    lines = content.split('\n')
    for line in lines[:10]:  # Check first 10 lines
        cleaned = line.strip()
        if not cleaned or len(cleaned) > 100:
            continue

        # Check if line looks like a name
        if _looks_like_name(cleaned):
            return cleaned

    return None

def _looks_like_name(text: str) -> bool:
    """Check if text looks like a person's name."""
    if not text or len(text) < 3:
        return False

    # Remove common patterns
    text_lower = text.lower()

    # Filter out section headers and common patterns
    skip_patterns = [
        "summary", "objective", "experience", "education", "skill",
        "certification", "contact", "achievement", "award", "project",
        "portfolio", "profile", "resume", "cv", "technical", "functional"
    ]

    for pattern in skip_patterns:
        if pattern in text_lower:
            return False

    # Check if it looks like a name (multiple words, mostly letters)
    words = text.split()
    if len(words) < 2:
        return False

    # Most words should start with capital and contain mostly letters
    capital_words = sum(1 for word in words if word and word[0].isupper())
    letter_ratio = sum(1 for char in text if char.isalpha()) / len(text)

    return capital_words >= 2 and letter_ratio > 0.8

def _extract_emails(content: str) -> List[str]:
    """Extract email addresses from content."""
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    matches = re.findall(email_pattern, content)
    return list(dict.fromkeys(matches))  # Remove duplicates while preserving order

def _extract_phones(content: str) -> List[str]:
    """Extract phone numbers from content."""
    # Patterns for various phone formats
    patterns = [
        r'\+91[0-9]{10}',  # Indian format +91
        r'\+[0-9]{1,3}\s?[0-9]{6,}',  # International format
        r'\b[0-9]{10}\b',  # 10 digit numbers
        r'\b[0-9]{3}[-.]?[0-9]{3}[-.]?[0-9]{4}\b',  # US format
    ]

    all_matches = []
    for pattern in patterns:
        matches = re.findall(pattern, content)
        all_matches.extend(matches)

    return list(dict.fromkeys(all_matches))  # Remove duplicates

def _extract_linkedin(content: str) -> List[str]:
    """Extract LinkedIn profile URLs from content."""
    linkedin_pattern = r'(?:https?://)?(?:www\.)?linkedin\.com/in/[A-Za-z0-9-]+'
    matches = re.findall(linkedin_pattern, content, re.IGNORECASE)
    return list(dict.fromkeys(matches))

def _extract_technical_skills(content: str) -> List[str]:
    """Extract technical skills from content."""
    # Common technical skill keywords
    tech_keywords = {
        "python", "java", "javascript", "typescript", "c++", "c#", "golang", "rust", "kotlin",
        "react", "angular", "vue", "node.js", "express", "django", "flask", "spring", "fastapi",
        "sql", "mongodb", "postgresql", "mysql", "redis", "elasticsearch", "kafka",
        "aws", "azure", "gcp", "docker", "kubernetes", "jenkins", "git", "ci/cd",
        "rest", "graphql", "api", "microservices", "devops", "terraform", "ansible",
        "machine learning", "tensorflow", "pytorch", "keras", "scikit-learn", "nlp", "cv",
        "html", "css", "sass", "webpack", "npm", "yarn", "typescript", "babel"
    }

    content_lower = content.lower()
    found_skills = set()

    for skill in tech_keywords:
        if skill in content_lower:
            found_skills.add(skill)

    # Also look for explicit skill sections
    skill_section_pattern = r'(?:technical\s+skills?|technologies?|tech\s+stack|programming)\s*[:\-]?\s*([^\n]+(?:\n(?!.*:)[^\n]+)*)'
    matches = re.findall(skill_section_pattern, content, re.IGNORECASE)

    for match in matches:
        # Split by common delimiters and clean up
        items = re.split(r'[,;•/]', match)
        for item in items:
            cleaned = item.strip().lower()
            if cleaned and len(cleaned) > 2:
                found_skills.add(cleaned)

    return sorted(list(found_skills))

def _extract_functional_skills(content: str) -> List[str]:
    """Extract functional/soft skills from content."""
    func_keywords = {
        "communication", "leadership", "teamwork", "project management", "collaboration",
        "problem solving", "critical thinking", "analytical", "planning", "organization",
        "negotiation", "stakeholder management", "agile", "scrum", "kanban",
        "documentation", "presentation", "mentoring", "training", "quality assurance",
        "process improvement", "business analysis", "requirement gathering"
    }

    content_lower = content.lower()
    found_skills = set()

    for skill in func_keywords:
        if skill in content_lower:
            found_skills.add(skill)

    # Look for functional skills sections
    skill_section_pattern = r'(?:functional\s+skills?|soft\s+skills?|competencies?|expertise)\s*[:\-]?\s*([^\n]+(?:\n(?!.*:)[^\n]+)*)'
    matches = re.findall(skill_section_pattern, content, re.IGNORECASE)

    for match in matches:
        items = re.split(r'[,;•/]', match)
        for item in items:
            cleaned = item.strip().lower()
            if cleaned and len(cleaned) > 2:
                found_skills.add(cleaned)

    return sorted(list(found_skills))

def _extract_certifications(content: str) -> List[str]:
    """Extract certifications from content."""
    # Explicit certification patterns
    cert_pattern = r'(?:certification|certified|credential|certificate|license)[:\-]?\s*([^\n]+)'
    matches = re.findall(cert_pattern, content, re.IGNORECASE)

    certs = []
    for match in matches:
        items = re.split(r'[,;•]', match)
        certs.extend(item.strip() for item in items if item.strip())

    # Look for known certification names
    known_certs = [
        "aws", "gcp", "azure", "oracle", "salesforce", "scrum", "pmp",
        "cissp", "ccna", "rhce", "comptia", "microsoft", "google", "certified"
    ]

    content_lower = content.lower()
    for cert in known_certs:
        if cert in content_lower and cert not in [c.lower() for c in certs]:
            certs.append(cert)

    return certs

def _extract_education(content: str) -> List[str]:
    """Extract education information from content."""
    edu_list = []
    seen = set()

    # Education keywords that must appear as whole words or at word boundaries
    edu_keywords = [
        r'\bB\.?Tech\b', r'\bBTech\b', r'\bB\.?E\b', r'\bBE\b',
        r'\bM\.?Tech\b', r'\bMTech\b', r'\bM\.?E\b', r'\bME\b',
        r'\bB\.?Sc\b', r'\bBSc\b', r'\bB\.?S\b',
        r'\bM\.?Sc\b', r'\bMSc\b', r'\bM\.?S\b',
        r'\bB\.?A\b', r'\bBA\b', r'\bM\.?A\b', r'\bMA\b',
        r'\bMBA\b', r'\bBBA\b', r'\bPh\.?D\b', r'\bPhD\b',
        r'\bBachelor\b', r'\bMaster\b', r'\bDiploma\b',
        r'\bDegree\b.*(?:in|from)',
    ]

    # Also look for university/college/institute mentions
    institution_pattern = r'[^\n]*(?:University|College|Institute|School)[^\n]*'
    institution_matches = re.findall(institution_pattern, content, re.IGNORECASE)

    for match in institution_matches:
        cleaned = ' '.join(match.split())[:150]
        # Skip if it's a certification or short
        if len(cleaned) < 15:
            continue
        lower = cleaned.lower()
        if 'certified' in lower or 'certification' in lower:
            continue
        if lower not in seen:
            seen.add(lower)
            edu_list.append(cleaned)

    # Look for degree patterns
    for pattern in edu_keywords:
        full_pattern = pattern + r'[^\n]*'
        matches = re.findall(full_pattern, content, re.IGNORECASE)
        for match in matches:
            cleaned = ' '.join(match.split())[:150]
            if len(cleaned) < 10:
                continue
            lower = cleaned.lower()
            if lower not in seen:
                seen.add(lower)
                edu_list.append(cleaned)

    return edu_list[:5]

def _extract_experience_summary(content: str) -> Optional[str]:
    """Extract professional summary or experience overview."""
    # Look for explicit summary sections
    summary_pattern = r'(?:summary|objective|profile|professional\s+summary)[:\-]?\s*([^\n]+(?:\n(?!.*[:\-])[^\n]+)*)'
    matches = re.findall(summary_pattern, content, re.IGNORECASE | re.MULTILINE)

    if matches:
        # Return the longest summary
        summary = max(matches, key=len)
        return ' '.join(summary.split())[:500]

    # Fallback: return first meaningful paragraph
    lines = content.split('\n')
    for i, line in enumerate(lines[:20]):
        stripped = line.strip()
        if len(stripped) > 50 and not any(skip in stripped.lower() for skip in ["contact", "email", "phone"]):
            return stripped[:500]

    return None

def _extract_years_of_experience(content: str) -> Optional[str]:
    """Extract total years of experience."""
    # Look for explicit years patterns
    patterns = [
        r'(\d+)\s*\+?\s*(?:years?|yrs?)\s+(?:of\s+)?experience',
        r'(?:experience|total|years)[:\-]?\s*(\d+)\s*\+?\s*(?:years?|yrs?)',
    ]

    for pattern in patterns:
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            years = match.group(1)
            return f"{years} years"

    return None

def _extract_achievements(content: str) -> List[str]:
    """Extract achievements, awards, and accomplishments."""
    # Look for achievement sections
    achieve_pattern = r'(?:achievement|award|accomplishment|honor|recognition)[:\-]?\s*([^\n]+)'
    matches = re.findall(achieve_pattern, content, re.IGNORECASE)

    achievements = []
    for match in matches:
        items = re.split(r'[,;•]', match)
        achievements.extend(item.strip() for item in items if item.strip())

    return achievements

def _infer_source_type(document_data: Dict[str, Any]) -> str:
    """Infer document source type from metadata or content."""
    # Check metadata
    if "source" in document_data:
        return str(document_data["source"])

    if "type" in document_data:
        doc_type = str(document_data["type"]).lower()
        if "resume" in doc_type or "cv" in doc_type:
            return "Resume"
        elif "linkedin" in doc_type:
            return "LinkedIn Profile"

    if "filename" in document_data:
        filename = str(document_data["filename"]).lower()
        if "linkedin" in filename:
            return "LinkedIn Profile"
        elif "resume" in filename or "cv" in filename:
            return "Resume"

    return "Resume"  # Default

def _empty_candidate() -> Dict[str, Any]:
    """Return empty candidate structure."""
    return {
        "name": None,
        "email": [],
        "phone": [],
        "linkedin": [],
        "technical_skills": [],
        "functional_skills": [],
        "certifications": [],
        "education": [],
        "experience_summary": None,
        "total_years_experience": None,
        "achievements": [],
        "source_type": "Resume",
    }

