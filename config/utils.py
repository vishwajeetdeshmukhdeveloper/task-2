"""
Utility functions for extraction system
"""

import os
from pathlib import Path
from typing import List, Optional
import json
from datetime import datetime


def get_supported_files(directory: str, recursive: bool = True) -> List[str]:
    """
    Get list of supported files in directory
    
    Args:
        directory: Directory path
        recursive: Whether to search subdirectories
        
    Returns:
        List of supported file paths
    """
    supported_extensions = {'.pdf', '.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp', '.gif'}
    files = []
    
    pattern = '**/*' if recursive else '*'
    
    for ext in supported_extensions:
        for file_path in Path(directory).glob(f"{pattern}{ext}"):
            files.append(str(file_path))
    
    return sorted(files)


def validate_file(file_path: str) -> tuple[bool, str]:
    """
    Validate if file exists and is supported
    
    Args:
        file_path: Path to file
        
    Returns:
        Tuple of (is_valid: bool, message: str)
    """
    if not os.path.exists(file_path):
        return False, f"File not found: {file_path}"
    
    supported_extensions = {'.pdf', '.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp', '.gif'}
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext not in supported_extensions:
        return False, f"Unsupported file type: {ext}"
    
    return True, "File is valid"


def create_output_filename(input_file: str, output_dir: str = 'output') -> str:
    """
    Create output filename based on input filename
    
    Args:
        input_file: Input file path
        output_dir: Output directory
        
    Returns:
        Output file path
    """
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    return os.path.join(output_dir, f"{base_name}_extracted.txt")


def save_json_log(data: dict, file_path: str) -> bool:
    """
    Save dictionary as JSON log file
    
    Args:
        data: Dictionary to save
        file_path: Output file path
        
    Returns:
        True if successful
    """
    try:
        os.makedirs(os.path.dirname(file_path) if os.path.dirname(file_path) else '.', exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        return True
    except Exception as e:
        print(f"Error saving JSON log: {str(e)}")
        return False


def get_file_stats(file_path: str) -> dict:
    """
    Get file statistics
    
    Args:
        file_path: Path to file
        
    Returns:
        Dictionary with file stats
    """
    if not os.path.exists(file_path):
        return {}
    
    stat = os.stat(file_path)
    
    return {
        "path": file_path,
        "size_bytes": stat.st_size,
        "size_mb": stat.st_size / (1024 * 1024),
        "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
    }


def cleanup_output_dir(output_dir: str, keep_days: int = 7) -> int:
    """
    Clean up old files in output directory
    
    Args:
        output_dir: Output directory path
        keep_days: Number of days to keep files
        
    Returns:
        Number of files deleted
    """
    from datetime import timedelta
    import time
    
    if not os.path.exists(output_dir):
        return 0
    
    cutoff_time = time.time() - (keep_days * 24 * 60 * 60)
    deleted_count = 0
    
    for file_path in Path(output_dir).glob('*_extracted.txt'):
        if os.path.getmtime(file_path) < cutoff_time:
            try:
                os.remove(file_path)
                deleted_count += 1
            except Exception as e:
                print(f"Error deleting {file_path}: {str(e)}")
    
    return deleted_count


def print_file_preview(file_path: str, lines: int = 10) -> None:
    """
    Print preview of extracted text file
    
    Args:
        file_path: Path to text file
        lines: Number of lines to preview
    """
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.readlines()[:lines]
            print(f"\n--- Preview of {os.path.basename(file_path)} ---")
            print("".join(content))
            if len(content) >= lines:
                print(f"... (showing first {lines} lines)")
    except Exception as e:
        print(f"Error reading file: {str(e)}")
