import re
import spacy
from typing import Dict, List, Any
import pdfplumber
import docx2txt
import PyPDF2
import io

class ResumeParser:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.sections = [
            'experience', 'education', 'skills', 'projects', 
            'certifications', 'summary', 'objective'
        ]
    
    def parse_resume(self, file_content: bytes, filename: str) -> Dict[str, Any]:
        """Parse resume file and extract structured information"""
        # Extract text based on file type
        if filename.lower().endswith('.pdf'):
            text = self._extract_text_from_pdf(file_content)
        elif filename.lower().endswith('.docx'):
            text = self._extract_text_from_docx(file_content)
        else:
            raise ValueError("Unsupported file format. Please upload PDF or DOCX.")
        
        # Clean and structure the text
        cleaned_text = self._clean_text(text)
        structured_data = self._structure_resume(cleaned_text)
        
        return {
            "raw_text": text,
            "cleaned_text": cleaned_text,
            "structured_data": structured_data,
            "filename": filename
        }
    
    def _extract_text_from_pdf(self, file_content: bytes) -> str:
        """Extract text from PDF using pdfplumber (more accurate)"""
        text = ""
        try:
            with pdfplumber.open(io.BytesIO(file_content)) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        except Exception as e:
            # Fallback to PyPDF2 if pdfplumber fails
            try:
                pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_content))
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
            except Exception as e2:
                raise Exception(f"Failed to extract text from PDF: {str(e2)}")
        
        return text
    
    def _extract_text_from_docx(self, file_content: bytes) -> str:
        """Extract text from DOCX file"""
        try:
            # Save to temporary file and process
            with io.BytesIO(file_content) as file:
                text = docx2txt.process(file)
            return text
        except Exception as e:
            raise Exception(f"Failed to extract text from DOCX: {str(e)}")
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize resume text"""
        # Remove special characters but keep relevant symbols
        text = re.sub(r'[^\w\s@\+\.\-\(\)]', ' ', text)
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove header/footer-like content (repeated text)
        lines = text.split('\n')
        unique_lines = []
        seen_lines = set()
        
        for line in lines:
            stripped = line.strip()
            if stripped and stripped not in seen_lines:
                unique_lines.append(stripped)
                seen_lines.add(stripped)
        
        return ' '.join(unique_lines)
    
    def _structure_resume(self, text: str) -> Dict[str, List[str]]:
        """Structure resume text into sections"""
        structured_data = {section: [] for section in self.sections}
        
        # Split text into lines for section detection
        lines = text.split('\n')
        current_section = None
        
        for line in lines:
            line_lower = line.lower().strip()
            
            # Check if this line indicates a new section
            section_found = False
            for section in self.sections:
                if section in line_lower and len(line_lower.split()) < 5:
                    current_section = section
                    section_found = True
                    break
            
            # If no section header found, add content to current section
            if not section_found and current_section and line.strip():
                structured_data[current_section].append(line.strip())
        
        # If no sections were detected, try to identify them with NLP
        if not any(structured_data.values()):
            return self._structure_with_nlp(text)
        
        # Join the lines for each section
        for section in structured_data:
            structured_data[section] = ' '.join(structured_data[section])
        
        return structured_data
    
    def _structure_with_nlp(self, text: str) -> Dict[str, str]:
        """Fallback method to structure resume using NLP"""
        structured_data = {section: "" for section in self.sections}
        doc = self.nlp(text)
        
        # Extract entities and classify them
        skills = []
        organizations = []
        dates = []
        degrees = []
        
        for ent in doc.ents:
            if ent.label_ in ["ORG", "COMPANY"]:
                organizations.append(ent.text)
            elif ent.label_ == "DATE":
                dates.append(ent.text)
            elif ent.label_ in ["DEGREE", "EDUCATION"]:
                degrees.append(ent.text)
        
        # Extract skills based on common patterns
        skill_patterns = [
            r'\b(?:python|java|javascript|sql|html|css|react|angular|vue|node|express|django|flask)\b',
            r'\b(?:machine learning|deep learning|data analysis|tableau|power bi|aws|azure|google cloud)\b',
            r'\b(?:docker|kubernetes|ci/cd|devops|agile|scrum)\b'
        ]
        
        for pattern in skill_patterns:
            skills.extend(re.findall(pattern, text, re.IGNORECASE))
        
        # Assign extracted information to sections
        structured_data['skills'] = ' '.join(set(skills))
        structured_data['education'] = ' '.join(set(degrees))
        
        # For experience, look for date patterns and organizations
        experience_text = ""
        for i, sent in enumerate(doc.sents):
            if any(org in sent.text for org in organizations) or any(date in sent.text for date in dates):
                experience_text += sent.text + " "
        
        structured_data['experience'] = experience_text
        
        return structured_data