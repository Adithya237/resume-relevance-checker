import re
import spacy
from typing import Dict, List, Tuple
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords', quiet=True)

class JDParser:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.stop_words = set(stopwords.words('english'))
        
    def parse_jd(self, jd_text: str) -> Dict:
        """Parse job description text and extract key components"""
        # Clean text
        cleaned_text = self._clean_text(jd_text)
        
        # Extract components
        role_title = self._extract_role_title(cleaned_text)
        must_have_skills = self._extract_skills(cleaned_text, "must")
        good_to_have_skills = self._extract_skills(cleaned_text, "good")
        qualifications = self._extract_qualifications(cleaned_text)
        experience = self._extract_experience(cleaned_text)
        
        return {
            "role_title": role_title,
            "must_have_skills": must_have_skills,
            "good_to_have_skills": good_to_have_skills,
            "qualifications": qualifications,
            "experience": experience,
            "raw_text": jd_text,
            "cleaned_text": cleaned_text
        }
    
    def _clean_text(self, text: str) -> str:
        """Clean and preprocess text"""
        # Remove special characters and extra whitespace
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        text = text.lower().strip()
        return text
    
    def _extract_role_title(self, text: str) -> str:
        """Extract job role title from JD"""
        # Look for common role patterns
        patterns = [
            r'position:\s*([^\n]+)',
            r'role:\s*([^\n]+)',
            r'job title:\s*([^\n]+)',
            r'we are looking for a\s+([^\n\.]+)',
            r'looking for a\s+([^\n\.]+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        # Fallback: use the first line or sentence
        return text.split('.')[0][:100]
    
    def _extract_skills(self, text: str, skill_type: str) -> List[str]:
        """Extract skills from JD based on type (must/good)"""
        skills = []
        
        # Define patterns for must-have and good-to-have skills
        if skill_type == "must":
            patterns = [
                r'must have[^\.]*?([^\.]+)',
                r'required[^\.]*?([^\.]+)',
                r'essential[^\.]*?([^\.]+)',
                r'must possess[^\.]*?([^\.]+)'
            ]
        else:
            patterns = [
                r'good to have[^\.]*?([^\.]+)',
                r'preferred[^\.]*?([^\.]+)',
                r'nice to have[^\.]*?([^\.]+)',
                r'desired[^\.]*?([^\.]+)'
            ]
        
        # Extract text using patterns
        extracted_text = ""
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                extracted_text += " " + " ".join(matches)
        
        if not extracted_text:
            return skills
        
        # Use spaCy to extract noun phrases which often contain skills
        doc = self.nlp(extracted_text)
        
        # Common skill indicators
        skill_indicators = {'skill', 'experience', 'knowledge', 'ability', 'familiarity', 'proficiency'}
        
        for chunk in doc.noun_chunks:
            # Check if the noun chunk contains skill indicators
            if any(indicator in chunk.text.lower() for indicator in skill_indicators):
                # Extract the actual skill from the phrase
                skill_text = chunk.text.lower()
                # Remove common filler words
                skill_text = re.sub(r'\b(and|or|with|in|of|the|a|an)\b', '', skill_text)
                skill_text = re.sub(r'\s+', ' ', skill_text).strip()
                
                if skill_text and len(skill_text) > 3:
                    skills.append(skill_text)
        
        # If no skills found with NLP, fallback to keyword matching
        if not skills:
            # Common technical skills
            technical_skills = [
                'python', 'java', 'javascript', 'sql', 'html', 'css', 'react', 'angular', 
                'vue', 'node', 'express', 'django', 'flask', 'machine learning', 'deep learning',
                'data analysis', 'tableau', 'power bi', 'excel', 'aws', 'azure', 'google cloud',
                'docker', 'kubernetes', 'ci/cd', 'devops'
            ]
            
            for skill in technical_skills:
                if skill in extracted_text:
                    skills.append(skill)
        
        return list(set(skills))[:15]  # Return unique skills, limit to 15
    
    def _extract_qualifications(self, text: str) -> List[str]:
        """Extract qualifications/education requirements"""
        qualifications = []
        
        # Patterns for education requirements
        patterns = [
            r'bachelor[^\.]*?([^\.]+)',
            r'master[^\.]*?([^\.]+)',
            r'phd[^\.]*?([^\.]+)',
            r'degree[^\.]*?([^\.]+)',
            r'education[^\.]*?([^\.]+)',
            r'diploma[^\.]*?([^\.]+)'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                # Clean up the qualification text
                qual = re.sub(r'\b(in|of|or|and)\b', '', match, flags=re.IGNORECASE)
                qual = re.sub(r'\s+', ' ', qual).strip()
                if qual and len(qual) > 5:
                    qualifications.append(qual)
        
        return list(set(qualifications))[:10]  # Return unique qualifications, limit to 10
    
    def _extract_experience(self, text: str) -> str:
        """Extract experience requirements"""
        patterns = [
            r'experience[^\.]*?(\d+[^\.]*years?)',
            r'(\d+[^\.]*years?)[^\.]*experience',
            r'minimum[^\.]*?(\d+[^\.]*years?)',
            r'at least[^\.]*?(\d+[^\.]*years?)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return "Not specified"