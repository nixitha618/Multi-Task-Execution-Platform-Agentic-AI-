import re
from pypdf import PdfReader

def extract_sections_from_text(text: str):
    # Common resume and document section headers
    section_patterns = [
        r"(?:^|\n)\s*(PROJECTS|ACADEMIC PROJECTS|PERSONAL PROJECTS|KEY PROJECTS)\b",
        r"(?:^|\n)\s*(SKILLS SUMMARY|TECHNICAL SKILLS|SKILLS|CORE COMPETENCIES)\b",
        r"(?:^|\n)\s*(INTERNSHIP EXPERIENCE|WORK EXPERIENCE|EXPERIENCE|PROFESSIONAL EXPERIENCE)\b",
        r"(?:^|\n)\s*(EDUCATION|ACADEMIC BACKGROUND)\b",
        r"(?:^|\n)\s*(CERTIFICATIONS|COURSES & CERTIFICATIONS|LICENSES & CERTIFICATIONS)\b",
        r"(?:^|\n)\s*(ABOUT ME|SUMMARY|PROFILE|PROFESSIONAL SUMMARY|OBJECTIVE)\b",
        r"(?:^|\n)\s*(ACHIEVEMENTS|AWARDS|HONORS)\b",
        r"(?:^|\n)\s*(PUBLICATIONS|RESEARCH)\b",
    ]
    combined_pattern = "|".join(section_patterns)
    
    # Find all header positions
    matches = list(re.finditer(combined_pattern, text, flags=re.IGNORECASE))
    if not matches:
        return [{"section": "GENERAL", "text": text}]
    
    sections = []
    # Add intro/header section before first match
    if matches[0].start() > 0:
        intro_text = text[:matches[0].start()].strip()
        if intro_text:
            sections.append({"section": "HEADER", "text": intro_text})
            
    for i, m in enumerate(matches):
        sec_name = m.group(0).strip().upper()
        start = m.end()
        end = matches[i+1].start() if i + 1 < len(matches) else len(text)
        sec_content = text[start:end].strip()
        sections.append({
            "section": sec_name,
            "text": f"{sec_name}\n{sec_content}"
        })
    return sections

reader = PdfReader('data/uploads/Nixitha_Resume (1).pdf')
raw = '\n\n'.join(p.extract_text() for p in reader.pages if p.extract_text())
secs = extract_sections_from_text(raw)
print(f"Extracted {len(secs)} semantic sections:")
for s in secs:
    preview = s['text'][:80].replace('\n', ' ').encode('ascii', errors='replace').decode()
    print(f"- Section [{s['section']}]: len={len(s['text'])} | {preview}...")
