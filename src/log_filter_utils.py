#!/usr/bin/env python3
"""
Log Field Extraction Utilities for log-experiment-v5

This module provides field extraction capabilities that allow parsing structured
log data based on preprocessed patterns. It enables targeted extraction of 
specific "columns" of data from logs using whitespace and X-masked field indices.

Key Functions:
- extract_fields(): Extract all fields from raw log using pattern
- categorize_extracted_fields(): Separate fields into whitespace vs X-masked categories
- select_fields_by_indices(): Select specific fields using provided indices

Usage:
    from log_filter_utils import extract_fields, categorize_extracted_fields
    
    fields = extract_fields(raw_log, pattern)
    whitespace_fields, x_fields = categorize_extracted_fields(raw_log, pattern)
"""

import re
from typing import List, Dict, Tuple, Optional


def extract_fields(raw_log: str, preprocessed_pattern: str) -> List[str]:
    """
    Extract variable content from raw logs using preprocessed structural pattern as template.
    
    Args:
        raw_log: Original log line
        preprocessed_pattern: Structural symbols (e.g., '. - - [X] "X"')
    
    Returns:
        Array of extracted variable content
        
    Extraction Rules:
        - Whitespace between symbols -> extract alphanumeric content
        - 'X' between symbols -> extract alphanumeric + symbol content as whole  
        - Content before first symbol -> extract as whole
        - Content after last symbol -> extract as whole
        
    Example:
        >>> extract_fields('192.168.1.1 - - [22/Jan/2019:14:18:06] "GET /test" 200', '. - - [X] "X"')
        ['192.168.1.1', '22/Jan/2019:14:18:06', 'GET /test', '200']
    """
    if not raw_log or not preprocessed_pattern:
        return []
    
    # Parse the preprocessed pattern to identify extraction points
    pattern_tokens = preprocessed_pattern.split()
    extraction_map = _parse_pattern_tokens(pattern_tokens)
    
    # Extract fields from raw log based on the pattern
    extracted_fields = _extract_content_by_pattern(raw_log, extraction_map)
    
    return extracted_fields


def categorize_extracted_fields(raw_log: str, preprocessed_pattern: str) -> Tuple[List[str], List[str]]:
    """
    Extract fields from raw log and categorize them into whitespace and X-masked fields.
    
    This function performs field extraction and then categorizes the results based on
    how they were extracted from the pattern structure.
    
    Args:
        raw_log: Original log line
        preprocessed_pattern: Structural symbols pattern
        
    Returns:
        Tuple of (whitespace_fields, x_masked_fields)
        - whitespace_fields: Content extracted from gaps between symbols
        - x_masked_fields: Content extracted from X-marked positions (enclosed content)
        
    Example:
        >>> whitespace, x_fields = categorize_extracted_fields(
        ...     '192.168.1.1 - - [22/Jan/2019:14:18:06] "GET /test" 200', 
        ...     '. - - [X] "X"'
        ... )
        >>> print(whitespace)  # ['192.168.1.1', '200'] 
        >>> print(x_fields)    # ['22/Jan/2019:14:18:06', 'GET /test']
    """
    if not raw_log or not preprocessed_pattern:
        return [], []
        
    # Parse pattern and extract with categorization
    pattern_tokens = preprocessed_pattern.split()
    extraction_map = _parse_pattern_tokens(pattern_tokens)
    
    whitespace_fields, x_fields = _extract_categorized_content(raw_log, extraction_map)
    
    return whitespace_fields, x_fields


def select_fields_by_indices(whitespace_fields: List[str], x_fields: List[str], 
                           whitespace_index: List[int], x_index: List[int]) -> Tuple[List[str], List[str]]:
    """
    Select specific fields using provided indices.
    
    Args:
        whitespace_fields: Available whitespace-extracted fields
        x_fields: Available X-masked fields
        whitespace_index: Indices of whitespace fields to select
        x_index: Indices of X-masked fields to select
        
    Returns:
        Tuple of (selected_whitespace_fields, selected_x_fields)
        
    Note:
        Invalid indices are silently ignored to prevent errors.
        
    Example:
        >>> select_fields_by_indices(['ip', 'user', 'code'], ['timestamp', 'request'], [0, 2], [1])
        (['ip', 'code'], ['request'])
    """
    selected_whitespace = []
    selected_x = []
    
    # Select whitespace fields by indices
    for idx in whitespace_index:
        if 0 <= idx < len(whitespace_fields):
            selected_whitespace.append(whitespace_fields[idx])
    
    # Select X-masked fields by indices  
    for idx in x_index:
        if 0 <= idx < len(x_fields):
            selected_x.append(x_fields[idx])
            
    return selected_whitespace, selected_x


def select_fields_by_unified_indices(all_fields: List[str], field_indices: List[int]) -> List[str]:
    """
    Select specific fields using unified field indices.
    
    This function provides a unified indexing system where all extracted fields
    (both whitespace and X-masked) are treated as a single ordered list.
    
    Args:
        all_fields: All extracted fields in order (from extract_fields)
        field_indices: List of indices to select from the unified field list
        
    Returns:
        List of selected fields in the same order as the indices
        
    Note:
        Invalid indices are silently ignored to prevent errors.
        
    Example:
        >>> all_fields = ['192.168.1.1', 'user', 'group', '22/Jan/2019', 'GET /test']
        >>> select_fields_by_unified_indices(all_fields, [0, 3, 4])
        ['192.168.1.1', '22/Jan/2019', 'GET /test']
    """
    selected_fields = []
    
    # Select fields by unified indices
    for idx in field_indices:
        if 0 <= idx < len(all_fields):
            selected_fields.append(all_fields[idx])
    
    return selected_fields


# === Internal Implementation Functions ===

def _parse_pattern_tokens(pattern_tokens: List[str]) -> List[Dict]:
    """
    Parse preprocessed pattern tokens to create extraction map.
    
    Returns:
        List of extraction instructions with type and position info
    """
    extraction_map = []
    
    for i, token in enumerate(pattern_tokens):
        if token == 'X':
            # X marker indicates complex content extraction (alphanumeric + symbols)
            extraction_map.append({
                'type': 'complex_content',
                'position': i,
                'token': token
            })
        elif re.match(r'^[^\w\s]+$', token):  # Pure symbols (non-alphanumeric, non-whitespace)
            # Symbol token - indicates boundary for content extraction
            extraction_map.append({
                'type': 'symbol',
                'position': i, 
                'token': token
            })
        elif 'X' in token:  # Mixed symbol with X (like "X" or [X])
            # Enclosed content marker
            extraction_map.append({
                'type': 'enclosed_content',
                'position': i,
                'token': token,
                'symbol_pattern': token.replace('X', '')
            })
    
    return extraction_map


def _extract_content_by_pattern(raw_log: str, extraction_map: List[Dict]) -> List[str]:
    """
    Extract content from raw log using the extraction map.
    """
    extracted = []
    log_pos = 0
    
    # Handle content before first symbol
    if extraction_map:
        first_symbol_pos = _find_first_symbol_in_log(raw_log, extraction_map[0])
        if first_symbol_pos > 0:
            pre_content = raw_log[:first_symbol_pos].strip()
            if pre_content:
                extracted.append(pre_content)
            log_pos = first_symbol_pos
    
    # Process each extraction point
    for i, mapping in enumerate(extraction_map):
        if mapping['type'] == 'symbol':
            # Skip over the symbol in raw log
            symbol_match = _find_symbol_in_log(raw_log, mapping['token'], log_pos)
            if symbol_match:
                log_pos = symbol_match['end']
                
        elif mapping['type'] == 'enclosed_content':
            # Extract content within paired symbols (like [content] or "content")
            content_match = _extract_enclosed_content(raw_log, mapping, log_pos)
            if content_match:
                extracted.append(content_match['content'])
                log_pos = content_match['end']
                
        elif mapping['type'] == 'complex_content':
            # Extract complex content (alphanumeric + symbols)  
            next_symbol_pos = _find_next_symbol_position(raw_log, extraction_map, i + 1, log_pos)
            if next_symbol_pos:
                content = raw_log[log_pos:next_symbol_pos].strip()
                if content:
                    extracted.append(content)
                log_pos = next_symbol_pos
        
        # Extract content between this and next symbol (whitespace gaps)
        if i < len(extraction_map) - 1:
            next_mapping = extraction_map[i + 1]
            content_between = _extract_content_between_symbols(raw_log, log_pos, next_mapping)
            if content_between:
                extracted.append(content_between['content'])
                log_pos = content_between['end']
    
    # Handle content after last symbol
    if log_pos < len(raw_log):
        post_content = raw_log[log_pos:].strip()
        if post_content:
            extracted.append(post_content)
    
    return extracted


def _extract_categorized_content(raw_log: str, extraction_map: List[Dict]) -> Tuple[List[str], List[str]]:
    """
    Extract content with categorization into whitespace vs X-masked fields.
    """
    whitespace_fields = []
    x_fields = []
    log_pos = 0
    
    # Handle content before first symbol (whitespace category)
    if extraction_map:
        first_symbol_pos = _find_first_symbol_in_log(raw_log, extraction_map[0])
        if first_symbol_pos > 0:
            pre_content = raw_log[:first_symbol_pos].strip()
            if pre_content:
                whitespace_fields.append(pre_content)
            log_pos = first_symbol_pos
    
    # Process each extraction point with categorization
    for i, mapping in enumerate(extraction_map):
        if mapping['type'] == 'symbol':
            # Skip over the symbol in raw log
            symbol_match = _find_symbol_in_log(raw_log, mapping['token'], log_pos)
            if symbol_match:
                log_pos = symbol_match['end']
                
        elif mapping['type'] == 'enclosed_content':
            # Extract content within paired symbols -> X-masked category
            content_match = _extract_enclosed_content(raw_log, mapping, log_pos)
            if content_match:
                x_fields.append(content_match['content'])
                log_pos = content_match['end']
                
        elif mapping['type'] == 'complex_content':
            # Extract complex content -> X-masked category
            next_symbol_pos = _find_next_symbol_position(raw_log, extraction_map, i + 1, log_pos)
            if next_symbol_pos:
                content = raw_log[log_pos:next_symbol_pos].strip()
                if content:
                    x_fields.append(content)
                log_pos = next_symbol_pos
        
        # Extract content between symbols -> whitespace category
        if i < len(extraction_map) - 1:
            next_mapping = extraction_map[i + 1]
            content_between = _extract_content_between_symbols(raw_log, log_pos, next_mapping)
            if content_between:
                whitespace_fields.append(content_between['content'])
                log_pos = content_between['end']
    
    # Handle content after last symbol (whitespace category)
    if log_pos < len(raw_log):
        post_content = raw_log[log_pos:].strip()
        if post_content:
            whitespace_fields.append(post_content)
    
    return whitespace_fields, x_fields


def _find_first_symbol_in_log(raw_log: str, first_mapping: Dict) -> int:
    """Find position of first symbol in raw log."""
    if first_mapping['type'] == 'symbol':
        # Look for the symbol pattern
        for char in first_mapping['token']:
            pos = raw_log.find(char)
            if pos >= 0:
                return pos
    elif first_mapping['type'] == 'enclosed_content':
        # Look for opening bracket/quote
        symbol_pattern = first_mapping['symbol_pattern']
        if symbol_pattern:
            pos = raw_log.find(symbol_pattern[0])  # First character of symbol pair
            if pos >= 0:
                return pos
    return 0


def _find_symbol_in_log(raw_log: str, symbol_token: str, start_pos: int) -> Optional[Dict]:
    """Find symbol in raw log starting from position."""
    for char in symbol_token:
        pos = raw_log.find(char, start_pos)
        if pos >= 0:
            return {'start': pos, 'end': pos + 1}
    return None


def _extract_enclosed_content(raw_log: str, mapping: Dict, start_pos: int) -> Optional[Dict]:
    """Extract content within paired symbols like [content] or "content"."""
    symbol_pattern = mapping['symbol_pattern']
    if not symbol_pattern or len(symbol_pattern) < 2:
        return None
        
    open_char = symbol_pattern[0]
    close_char = symbol_pattern[-1]
    
    # Find opening symbol
    open_pos = raw_log.find(open_char, start_pos)
    if open_pos < 0:
        return None
        
    # Find closing symbol  
    close_pos = raw_log.find(close_char, open_pos + 1)
    if close_pos < 0:
        return None
        
    content = raw_log[open_pos + 1:close_pos]
    return {'content': content, 'end': close_pos + 1}


def _find_next_symbol_position(raw_log: str, extraction_map: List[Dict], start_index: int, log_pos: int) -> int:
    """Find position of next symbol in the extraction sequence."""
    for i in range(start_index, len(extraction_map)):
        mapping = extraction_map[i]
        if mapping['type'] == 'symbol':
            symbol_match = _find_symbol_in_log(raw_log, mapping['token'], log_pos)
            if symbol_match:
                return symbol_match['start']
        elif mapping['type'] == 'enclosed_content':
            symbol_pattern = mapping['symbol_pattern'] 
            if symbol_pattern:
                pos = raw_log.find(symbol_pattern[0], log_pos)
                if pos >= 0:
                    return pos
    return len(raw_log)


def _extract_content_between_symbols(raw_log: str, start_pos: int, next_mapping: Dict) -> Optional[Dict]:
    """Extract alphanumeric content between symbols (whitespace gaps)."""
    # Find where the next symbol starts
    end_pos = len(raw_log)
    
    if next_mapping['type'] == 'symbol':
        symbol_match = _find_symbol_in_log(raw_log, next_mapping['token'], start_pos)
        if symbol_match:
            end_pos = symbol_match['start']
    elif next_mapping['type'] == 'enclosed_content':
        symbol_pattern = next_mapping['symbol_pattern']
        if symbol_pattern:
            pos = raw_log.find(symbol_pattern[0], start_pos)
            if pos >= 0:
                end_pos = pos
    
    # Extract only alphanumeric content between symbols
    content = raw_log[start_pos:end_pos].strip()
    if content and re.match(r'^[\w\s.]+$', content):  # Allow alphanumeric + whitespace + periods
        return {'content': content, 'end': end_pos}
    
    return None


# === Test Functions ===

def test_field_extraction():
    """Test field extraction with sample log data."""
    test_cases = [
        {
            'raw': '192.168.1.1 - - [22/Jan/2019:14:18:06] "GET /test" 200',
            'pattern': '. . . - - [X] "X"',
            'expected_all': ['192.168.1.1', '200', '22/Jan/2019:14:18:06', 'GET /test'],
            'expected_whitespace': ['192.168.1.1', '200'],
            'expected_x': ['22/Jan/2019:14:18:06', 'GET /test']
        },
        {
            'raw': '- 1117838570 2005.06.03 R02-M1-N0-C:J12-U11 instruction cache parity error',
            'pattern': '- . . -',  
            'expected_all': ['1117838570', '2005.06.03', 'R02-M1-N0-C:J12-U11', 'instruction cache parity error'],
            'expected_whitespace': ['1117838570', '2005.06.03', 'R02-M1-N0-C:J12-U11', 'instruction cache parity error'],
            'expected_x': []
        }
    ]
    
    print("=== Log Field Extraction Test ===")
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest Case {i}:")
        print(f"Raw:     {test_case['raw']}")
        print(f"Pattern: {test_case['pattern']}")
        
        # Test full extraction
        all_fields = extract_fields(test_case['raw'], test_case['pattern'])
        print(f"All fields:      {all_fields}")
        print(f"Expected all:    {test_case['expected_all']}")
        
        # Test categorized extraction
        whitespace_fields, x_fields = categorize_extracted_fields(test_case['raw'], test_case['pattern'])
        print(f"Whitespace:      {whitespace_fields}")
        print(f"Expected ws:     {test_case['expected_whitespace']}")
        print(f"X-masked:        {x_fields}")
        print(f"Expected X:      {test_case['expected_x']}")
        
        # Test field selection (old dual-index method)
        selected_ws, selected_x = select_fields_by_indices(whitespace_fields, x_fields, [0, 1], [0])
        print(f"Selected ws[0,1]: {selected_ws}")
        print(f"Selected X[0]:   {selected_x}")
        
        # Test unified field selection (new method)
        selected_unified = select_fields_by_unified_indices(all_fields, [0, 1, 4])
        print(f"Unified [0,1,4]: {selected_unified}")
        
        print("-" * 60)


if __name__ == "__main__":
    test_field_extraction()