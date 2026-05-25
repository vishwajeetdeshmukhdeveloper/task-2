"""
Data Processor - Extract and structure medical data from extracted text files
Converts unstructured medical reports to structured format:
Date - Summary, Medicine Recommendation
"""

import os
import json
import re
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import pandas as pd


class MedicalDataProcessor:
    """Process extracted medical reports and create structured dataset"""
    
    # Common patterns to extract dates
    DATE_PATTERNS = [
        r'\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}',  # MM/DD/YYYY or DD-MM-YYYY
        r'\d{4}[/\-]\d{1,2}[/\-]\d{1,2}',    # YYYY/MM/DD
        r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}',  # Month DD, YYYY
    ]
    
    # Common medicine keywords
    MEDICINE_KEYWORDS = [
        'medication', 'medicine', 'prescription', 'prescribed', 'drug', 'tablet',
        'capsule', 'injection', 'dose', 'treatment', 'therapy', 'antibiotic',
        'paracetamol', 'ibuprofen', 'aspirin', 'amoxicillin', 'vitamin',
        'recommend', 'suggested', 'advised', 'mg', 'ml', 'tablets'
    ]
    
    # Common summary keywords
    SUMMARY_KEYWORDS = [
        'result', 'finding', 'diagnosis', 'abnormal', 'normal', 'positive', 'negative',
        'elevated', 'decreased', 'within range', 'critical', 'report', 'summary'
    ]
    
    def __init__(self, output_dir: str = 'output', data_dir: str = 'data'):
        """
        Initialize processor
        
        Args:
            output_dir: Directory containing extracted text files
            data_dir: Directory to save processed data
        """
        self.output_dir = output_dir
        self.data_dir = data_dir
        self.extracted_data: List[Dict] = []
        
        os.makedirs(data_dir, exist_ok=True)
    
    def extract_date(self, text: str) -> Optional[str]:
        """
        Extract date from text
        
        Args:
            text: Text to search for dates
            
        Returns:
            Extracted date string or None
        """
        for pattern in self.DATE_PATTERNS:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                return matches[0]
        return None
    
    def extract_medicine_recommendations(self, text: str) -> str:
        """
        Extract medicine recommendations from text
        
        Args:
            text: Text to search for medicines
            
        Returns:
            Medicine recommendations as string
        """
        lines = text.split('\n')
        medicine_lines = []
        
        for i, line in enumerate(lines):
            line_lower = line.lower()
            
            # Check if line contains medicine-related keywords
            if any(keyword in line_lower for keyword in self.MEDICINE_KEYWORDS):
                medicine_lines.append(line.strip())
        
        if medicine_lines:
            return ' | '.join(medicine_lines)
        
        # If no specific lines found, look for patterns like "X mg" or "X tablets"
        medicine_pattern = r'\d+\s*(?:mg|ml|tablets?|capsules?|doses?)'
        matches = re.findall(medicine_pattern, text, re.IGNORECASE)
        
        if matches:
            return ' | '.join(matches)
        
        return "Standard care and follow-up recommended"
    
    def extract_summary(self, text: str, max_length: int = 200) -> str:
        """
        Extract summary from text
        
        Args:
            text: Text to summarize
            max_length: Maximum summary length
            
        Returns:
            Summary text
        """
        lines = text.split('\n')
        summary_lines = []
        
        # Look for lines with summary keywords
        for line in lines:
            line_lower = line.lower()
            if any(keyword in line_lower for keyword in self.SUMMARY_KEYWORDS):
                summary_lines.append(line.strip())
        
        if summary_lines:
            summary = ' '.join(summary_lines)
        else:
            # Use first meaningful paragraph
            summary = ' '.join([l.strip() for l in lines if len(l.strip()) > 10])
        
        # Truncate to max length
        if len(summary) > max_length:
            summary = summary[:max_length] + "..."
        
        return summary.strip()
    
    def process_file(self, file_path: str) -> Optional[Dict]:
        """
        Process a single extracted text file
        
        Args:
            file_path: Path to text file
            
        Returns:
            Dictionary with structured data or None
        """
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return None
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract date
            date = self.extract_date(content)
            if not date:
                date = datetime.now().strftime('%m/%d/%Y')
            
            # Extract summary
            summary = self.extract_summary(content)
            
            # Extract medicine recommendations
            medicine = self.extract_medicine_recommendations(content)
            
            data_record = {
                'source_file': os.path.basename(file_path),
                'date': date,
                'summary': summary,
                'medicine_recommendation': medicine,
                'full_text': content[:500],  # First 500 chars for reference
                'processed_at': datetime.now().isoformat()
            }
            
            return data_record
        
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            return None
    
    def process_directory(self, directory: Optional[str] = None, recursive: bool = True) -> List[Dict]:
        """
        Process all text files in a directory
        
        Args:
            directory: Directory path (uses self.output_dir if None)
            recursive: Whether to search subdirectories
            
        Returns:
            List of processed data records
        """
        if directory is None:
            directory = self.output_dir
        
        if not os.path.exists(directory):
            print(f"Directory not found: {directory}")
            return []
        
        processed_data = []
        pattern = '**/*.txt' if recursive else '*.txt'
        
        print(f"Processing files from: {directory}\n")
        
        for file_path in Path(directory).glob(pattern):
            print(f"Processing: {file_path}")
            record = self.process_file(str(file_path))
            
            if record:
                processed_data.append(record)
                self.extracted_data.append(record)
                print(f"  ✓ Extracted: {record['date']} - {record['summary'][:50]}...")
        
        print(f"\nTotal files processed: {len(processed_data)}")
        return processed_data
    
    def save_to_csv(self, data: Optional[List[Dict]] = None, output_file: Optional[str] = None) -> bool:
        """
        Save processed data to CSV
        
        Args:
            data: Data to save (uses self.extracted_data if None)
            output_file: Output file path
            
        Returns:
            True if successful
        """
        if data is None:
            data = self.extracted_data
        
        if not data:
            print("No data to save")
            return False
        
        if output_file is None:
            output_file = os.path.join(self.data_dir, 'medical_data.csv')
        
        try:
            df = pd.DataFrame(data)
            
            # Select key columns for CSV
            columns_to_save = ['date', 'summary', 'medicine_recommendation', 'source_file']
            df_save = df[columns_to_save]
            
            df_save.to_csv(output_file, index=False, encoding='utf-8')
            
            print(f"\n✓ Data saved to CSV: {output_file}")
            print(f"Records: {len(df_save)}")
            return True
        
        except Exception as e:
            print(f"Error saving CSV: {str(e)}")
            return False
    
    def save_to_json(self, data: Optional[List[Dict]] = None, output_file: Optional[str] = None) -> bool:
        """
        Save processed data to JSON
        
        Args:
            data: Data to save (uses self.extracted_data if None)
            output_file: Output file path
            
        Returns:
            True if successful
        """
        if data is None:
            data = self.extracted_data
        
        if not data:
            print("No data to save")
            return False
        
        if output_file is None:
            output_file = os.path.join(self.data_dir, 'medical_data.json')
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            print(f"✓ Data saved to JSON: {output_file}")
            print(f"Records: {len(data)}")
            return True
        
        except Exception as e:
            print(f"Error saving JSON: {str(e)}")
            return False
    
    def create_training_dataset(self, output_file: Optional[str] = None) -> pd.DataFrame:
        """
        Create training dataset in structured format
        
        Args:
            output_file: Optional output file path
            
        Returns:
            DataFrame with training data
        """
        if not self.extracted_data:
            print("No extracted data. Run process_directory() first.")
            return pd.DataFrame()
        
        df = pd.DataFrame(self.extracted_data)
        
        # Create feature: combined text for NLP
        df['combined_text'] = df['summary'] + ' ' + df['medicine_recommendation']
        
        # Create target: medicine recommendation (for classification/regression)
        df['target'] = df['medicine_recommendation']
        
        if output_file:
            df.to_csv(output_file, index=False, encoding='utf-8')
            print(f"Training dataset saved to: {output_file}")
        
        return df
    
    def get_statistics(self) -> Dict:
        """Get statistics about processed data"""
        if not self.extracted_data:
            return {}
        
        df = pd.DataFrame(self.extracted_data)
        
        stats = {
            'total_records': len(df),
            'dates_count': df['date'].nunique(),
            'avg_summary_length': df['summary'].str.len().mean(),
            'avg_medicine_length': df['medicine_recommendation'].str.len().mean(),
            'sources': df['source_file'].unique().tolist(),
        }
        
        return stats
    
    def print_summary(self) -> None:
        """Print summary of processed data"""
        if not self.extracted_data:
            print("No data processed yet")
            return
        
        print("\n" + "=" * 80)
        print("DATA PROCESSING SUMMARY")
        print("=" * 80)
        
        stats = self.get_statistics()
        
        print(f"Total Records: {stats['total_records']}")
        print(f"Unique Dates: {stats['dates_count']}")
        print(f"Avg Summary Length: {stats['avg_summary_length']:.1f} chars")
        print(f"Avg Medicine Recommendation Length: {stats['avg_medicine_length']:.1f} chars")
        print(f"\nSample Records:")
        
        for i, record in enumerate(self.extracted_data[:3], 1):
            print(f"\n{i}. Date: {record['date']}")
            print(f"   Summary: {record['summary']}")
            print(f"   Medicine: {record['medicine_recommendation']}")
        
        print("\n" + "=" * 80)
