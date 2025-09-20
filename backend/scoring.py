from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import fuzz, process
from typing import Dict, List, Tuple
import numpy as np
import re

class RelevanceScorer:
    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    
    def calculate_hard_match_score(self, resume_text: str, jd_data: Dict) -> Dict:
        """Calculate hard matching score based on keywords and skills"""
        jd_text = jd_data['cleaned_text']
        must_have_skills = jd_data['must_have_skills']
        good_to_have_skills = jd_data['good_to_have_skills']
        qualifications = jd_data['qualifications']
        
        # Keyword matching using TF-IDF
        tfidf_matrix = self.tfidf_vectorizer.fit_transform([resume_text, jd_text])
        keyword_similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        
        # Skill matching
        must_have_match = self._match_skills(resume_text, must_have_skills)
        good_to_have_match = self._match_skills(resume_text, good_to_have_skills)
        
        # Qualification matching
        qualification_match = self._match_qualifications(resume_text, qualifications)
        
        # Calculate scores
        keyword_score = keyword_similarity * 30  # Weighted 30%
        must_have_score = (must_have_match / max(1, len(must_have_skills))) * 40  # Weighted 40%
        good_to_have_score = (good_to_have_match / max(1, len(good_to_have_skills))) * 20  # Weighted 20%
        qualification_score = (qualification_match / max(1, len(qualifications))) * 10  # Weighted 10%
        
        hard_match_score = keyword_score + must_have_score + good_to_have_score + qualification_score
        
        return {
            "hard_match_score": hard_match_score,
            "keyword_score": keyword_score,
            "must_have_score": must_have_score,
            "good_to_have_score": good_to_have_score,
            "qualification_score": qualification_score,
            "must_have_matched": must_have_match,
            "good_to_have_matched": good_to_have_match,
            "qualification_matched": qualification_match,
            "must_have_total": len(must_have_skills),
            "good_to_have_total": len(good_to_have_skills),
            "qualification_total": len(qualifications)
        }
    
    def _match_skills(self, resume_text: str, skills: List[str]) -> int:
        """Match skills using fuzzy matching"""
        matched_count = 0
        
        for skill in skills:
            # Use fuzzy matching with partial ratio
            match_score = fuzz.partial_ratio(skill, resume_text.lower())
            if match_score >= 80:  # 80% similarity threshold
                matched_count += 1
        
        return matched_count
    
    def _match_qualifications(self, resume_text: str, qualifications: List[str]) -> int:
        """Match qualifications using fuzzy matching"""
        matched_count = 0
        
        for qualification in qualifications:
            # Use token set ratio for better qualification matching
            match_score = fuzz.token_set_ratio(qualification, resume_text.lower())
            if match_score >= 70:  # 70% similarity threshold
                matched_count += 1
        
        return matched_count
    
    def calculate_final_score(self, hard_match_score: float, semantic_score: float) -> Dict:
        """Calculate final relevance score combining hard and semantic matching"""
        # Weighted average: 60% hard match, 40% semantic match
        final_score = (hard_match_score * 0.6) + (semantic_score * 0.4)
        
        # Determine verdict
        if final_score >= 80:
            verdict = "High"
        elif final_score >= 60:
            verdict = "Medium"
        elif final_score >= 40:
            verdict = "Low"
        else:
            verdict = "Poor"
        
        return {
            "relevance_score": final_score,
            "verdict": verdict,
            "hard_match_component": hard_match_score,
            "semantic_match_component": semantic_score
        }
    
    def generate_feedback(self, resume_data: Dict, jd_data: Dict, 
                         hard_match_results: Dict, missing_elements: List[str]) -> List[str]:
        """Generate personalized feedback for improvement"""
        feedback = []
        
        # Feedback on missing must-have skills
        if hard_match_results['must_have_matched'] < hard_match_results['must_have_total']:
            feedback.append(
                f"Missing {hard_match_results['must_have_total'] - hard_match_results['must_have_matched']} "
                f"out of {hard_match_results['must_have_total']} must-have skills"
            )
        
        # Feedback on missing good-to-have skills
        if hard_match_results['good_to_have_matched'] < hard_match_results['good_to_have_total']:
            feedback.append(
                f"Missing {hard_match_results['good_to_have_total'] - hard_match_results['good_to_have_matched']} "
                f"out of {hard_match_results['good_to_have_total']} good-to-have skills"
            )
        
        # Feedback on missing qualifications
        if hard_match_results['qualification_matched'] < hard_match_results['qualification_total']:
            feedback.append(
                f"Missing {hard_match_results['qualification_total'] - hard_match_results['qualification_matched']} "
                f"out of {hard_match_results['qualification_total']} required qualifications"
            )
        
        # Add specific missing elements
        for element in missing_elements[:3]:  # Limit to top 3 missing elements
            feedback.append(f"Consider adding: {element}")
        
        # General advice based on score
        final_score = self.calculate_final_score(
            hard_match_results['hard_match_score'], 
            0  # Semantic score not available here
        )['relevance_score']
        
        if final_score < 60:
            feedback.append("Tailor your resume to include more keywords from the job description")
            feedback.append("Quantify your achievements with specific metrics and results")
        
        return feedback
    
    def identify_missing_elements(self, resume_data: Dict, jd_data: Dict) -> List[str]:
        """Identify missing skills, qualifications, and other elements"""
        missing_elements = []
        
        resume_text = resume_data['cleaned_text'].lower()
        jd_must_have = jd_data['must_have_skills']
        jd_good_to_have = jd_data['good_to_have_skills']
        jd_qualifications = jd_data['qualifications']
        
        # Check for missing must-have skills
        for skill in jd_must_have:
            if fuzz.partial_ratio(skill, resume_text) < 80:
                missing_elements.append(f"Must-have skill: {skill}")
        
        # Check for missing good-to-have skills
        for skill in jd_good_to_have:
            if fuzz.partial_ratio(skill, resume_text) < 80:
                missing_elements.append(f"Good-to-have skill: {skill}")
        
        # Check for missing qualifications
        for qualification in jd_qualifications:
            if fuzz.token_set_ratio(qualification, resume_text) < 70:
                missing_elements.append(f"Qualification: {qualification}")
        
        return missing_elements