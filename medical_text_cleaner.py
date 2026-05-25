#!/usr/bin/env python
"""
Medical text cleaner and summarizer
Cleans raw extracted text and creates concise medical summaries
"""

import re
from typing import Tuple, List


class MedicalTextCleaner:
    """Clean and summarize medical text from PDF extractions"""
    
    # Medical conditions to look for
    MEDICAL_CONDITIONS = {
        'glucose': 'glucose level',
        'blood pressure': 'blood pressure',
        'cholesterol': 'cholesterol level',
        'hemoglobin': 'hemoglobin',
        'white blood': 'white blood cell',
        'creatinine': 'kidney function',
        'sugar': 'blood sugar',
        'fever': 'fever',
        'dengue': 'dengue fever',
        'diabetes': 'diabetes',
        'thyroid': 'thyroid function',
        'tsh': 'thyroid',
        'kidney': 'kidney function',
        'liver': 'liver function',
        'bilirubin': 'liver',
        'platelet': 'platelet count',
    }
    
    @staticmethod
    def extract_key_findings(text: str) -> str:
        """
        Extract key medical findings from raw text
        
        Args:
            text: Raw extracted text from PDF/image
            
        Returns:
            Cleaned summary with key findings
        """
        if not text or len(text) < 30:
            return "Medical assessment pending."
        
        # Remove page markers and metadata
        text = re.sub(r'--- Page \d+ ---', ' ', text)
        text = re.sub(r'\*{5,}', '', text)
        
        # Clean whitespace
        text = ' '.join(text.split())
        
        # Split into sentences
        sentences = re.split(r'[.!?]+', text)
        
        # Find sentences with medical content
        medical_keywords = [
            'glucose', 'blood', 'positive', 'negative', 'high', 'low', 'normal',
            'elevated', 'result', 'finding', 'diagnosis', 'patient', 'test',
            'antibody', 'fever', 'infection', 'level', 'value', 'range',
            'mg/dl', 'mg/l', 'igg', 'igm', 'detected', 'confirmed'
        ]
        
        good_sentences = []
        for sent in sentences:
            sent = sent.strip()
            sent_lower = sent.lower()
            
            # Skip metadata and headers and very long sentences
            if any(skip in sent_lower for skip in ['page', 'centre', 'lpl', 'tab', 'unit', 'result', 'name', 'date', 'age', 'sex', 'reported', 'received', 'collected']):
                continue
            
            # Check if sentence has medical keywords and is reasonable length
            if (len(sent) > 20 and len(sent) < 200 and 
                any(kw in sent_lower for kw in medical_keywords)):
                # Skip very long interpretation tables
                if '|' not in sent or len(sent.split('|')) < 4:
                    good_sentences.append(sent)
        
        # Build summary - take only first 1-2 good findings
        if good_sentences:
            # Filter for most relevant: those with actual values
            findings_with_values = [s for s in good_sentences if any(c.isdigit() for c in s)]
            
            if findings_with_values:
                summary = findings_with_values[0]
            else:
                # Fall back to any good sentence
                summary = good_sentences[0]
        else:
            summary = "Medical test report processed - detailed findings require clinical review."
        
        # Final cleanup
        summary = re.sub(r'\s+', ' ', summary).strip()
        
        # Capitalize
        if summary and not summary[0].isupper():
            summary = summary[0].upper() + summary[1:]
        
        return summary if len(summary) > 15 else "Medical assessment completed."
    
    @staticmethod
    def get_recommended_action(text: str) -> str:
        """
        Get recommended medical action based on findings
        
        Args:
            text: Medical text
            
        Returns:
            Recommended action
        """
        text_lower = text.lower()
        
        # Check for specific conditions and return recommendations
        recommendations = {
            'glucose': 'Glucose management and dietary intervention recommended',
            'diabetes': 'Diabetes management plan and specialist consultation advised',
            'blood pressure': 'Blood pressure monitoring and lifestyle modification recommended',
            'fever': 'Fever management and hydration therapy recommended',
            'dengue': 'Dengue monitoring and symptomatic treatment advised',
            'igg positive': 'Follow-up testing and clinical correlation advised',
            'igm positive': 'Active infection suspected - appropriate clinical management required',
            'positive': 'Positive result confirmed - clinical assessment and management advised',
            'cholesterol': 'Cholesterol management and dietary changes recommended',
            'kidney': 'Kidney function monitoring and specialist consultation advised',
            'liver': 'Liver function assessment and specialist consultation recommended',
            'thyroid': 'Thyroid function monitoring and endocrinology consultation advised',
            'hemoglobin': 'Hemoglobin assessment and nutritional supplementation advised',
            'infection': 'Infection control measures and appropriate treatment advised',
            'abnormal': 'Abnormal findings - specialist consultation recommended',
        }
        
        for keyword, recommendation in recommendations.items():
            if keyword in text_lower:
                return recommendation
        
        # Default recommendation
        return 'Standard medical care and follow-up consultation recommended'
    
    @staticmethod
    def clean_summary(text: str) -> Tuple[str, str]:
        """
        Clean text and generate summary with recommendation
        
        Args:
            text: Raw extracted text
            
        Returns:
            Tuple of (cleaned_summary, recommendation)
        """
        # Extract key findings
        summary = MedicalTextCleaner.extract_key_findings(text)
        
        # Get recommendation
        recommendation = MedicalTextCleaner.get_recommended_action(text)
        
        return summary, recommendation


# Example usage
if __name__ == "__main__":
    # Test with sample text
    sample_text = """
    DENGUE FEVER PANEL Investigation Result. DENGUE FEVER ANTIBODY, IgG 3.40 Positive. DENGUE FEVER ANTIBODY, IgM 2.60 Positive. Interpretation: IgG Positive indicates patient has been exposed to dengue virus. IgM Positive suggests Primary or Secondary dengue infection.
    """
    
    cleaner = MedicalTextCleaner()
    summary, recommendation = cleaner.clean_summary(sample_text)
    
    print(f"Summary: {summary}")
    print(f"Recommendation: {recommendation}")
    
    @staticmethod
    def get_recommended_action(text: str) -> str:
        """
        Get recommended medical action based on findings
        
        Args:
            text: Medical text
            
        Returns:
            Recommended action
        """
        text_lower = text.lower()
        
        # Check for specific conditions and return recommendations
        recommendations = {
            'glucose': 'Glucose management and dietary intervention recommended',
            'diabetes': 'Diabetes management plan and specialist consultation advised',
            'blood pressure': 'Blood pressure monitoring and lifestyle modification recommended',
            'fever': 'Fever management and hydration therapy recommended',
            'dengue': 'Dengue monitoring and symptomatic treatment advised',
            'cholesterol': 'Cholesterol management and dietary changes recommended',
            'kidney': 'Kidney function monitoring and specialist consultation advised',
            'liver': 'Liver function assessment and hepatology consultation recommended',
            'thyroid': 'Thyroid function monitoring and endocrinology consultation advised',
            'hemoglobin': 'Hemoglobin assessment and nutritional supplementation advised',
            'infection': 'Infection control measures and appropriate antibiotic therapy advised',
            'abnormal': 'Abnormal findings - specialist consultation recommended',
            'positive': 'Positive result confirmed - appropriate clinical management advised',
        }
        
        for keyword, recommendation in recommendations.items():
            if keyword in text_lower:
                return recommendation
        
        # Default recommendation
        return 'Standard medical care and follow-up consultation recommended'
    
    @staticmethod
    def clean_summary(text: str) -> Tuple[str, str]:
        """
        Clean text and generate summary with recommendation
        
        Args:
            text: Raw extracted text
            
        Returns:
            Tuple of (cleaned_summary, recommendation)
        """
        # Extract key findings
        summary = MedicalTextCleaner.extract_key_findings(text)
        
        # Get recommendation
        recommendation = MedicalTextCleaner.get_recommended_action(text)
        
        return summary, recommendation


# Example usage
if __name__ == "__main__":
    # Test with sample text
    sample_text = """
    Patient presents with elevated blood glucose levels. Fasting glucose 145 mg/dL (normal: 70-100).
    Random glucose 210 mg/dL (normal: <140). Diabetes risk detected.
    """
    
    cleaner = MedicalTextCleaner()
    summary, recommendation = cleaner.clean_summary(sample_text)
    
    print(f"Summary: {summary}")
    print(f"Recommendation: {recommendation}")
