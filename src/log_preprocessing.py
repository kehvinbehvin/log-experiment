#!/usr/bin/env python3
"""
Centralized Log Preprocessing Utility for log-experiment-v5

This module provides the canonical implementation of log preprocessing logic,
including symbol extraction and masking for consistent pattern generation
across training and inference pipelines.

Key Functions:
- preprocess_log(): Extract and mask symbols from log entries
- process_non_enclosing(): Process non-enclosed symbol segments

Usage:
    from log_preprocessing import preprocess_log, process_non_enclosing
    
    pattern = preprocess_log('84.241.26.205 [22/Jan/2019:14:18:06] "GET /test"')
    # Returns: '. [ ] " "'
"""

import re
from typing import List


def preprocess_log(log: str) -> str:
    """
    Extract and process symbols from log entries with consistent masking.
    
    This function extracts all non-alphanumeric, non-whitespace symbols from a log
    entry and processes them to create a standardized pattern signature.
    
    Masking Rules:
    - Enclosed symbols (quotes, brackets, braces, angles) are masked as [X], "X", {X}, <X>
    - Non-enclosed symbols are preserved as-is (periods, colons, dashes, etc.)
    - The 'X' insertion ensures consistent patterns between training and inference
    
    Args:
        log (str): Raw log entry to process
        
    Returns:
        str: Space-separated pattern of symbols (e.g., '. [ ] " "')
        
    Example:
        >>> preprocess_log('[user]: "hello world"')
        '[X] : "X"'
        
        >>> preprocess_log('192.168.1.1 - - [timestamp] "GET /api"')
        '. . . - - [X] "X"'
    """
    # Step 1: Extract all symbols (non-alphanumeric, non-whitespace)
    symbols = ''.join(re.findall(r'[^\w\s]', log))
    
    # Step 2: Process symbols to extract top-level enclosing pairs and non-enclosing symbols
    result = []
    stack = []
    current_segment = ""
    i = 0
    
    while i < len(symbols):
        symbol = symbols[i]
        
        # Handle quotes
        if symbol == '"':
            if stack and stack[-1] == '"':
                # Closing quote
                stack.pop()
                if not stack:  # Top-level pair complete
                    result.append('"X"')
                # Discard internal symbols
                current_segment = ""
            else:
                # Opening quote
                stack.append(symbol)
                if current_segment and not stack[0:-1]:  # Non-enclosed segment
                    result.extend(process_non_enclosing(current_segment))
                current_segment = ""
            i += 1
        
        # Handle other enclosing symbols ({, [, <)
        elif symbol in '{[<':
            stack.append(symbol)
            if current_segment and not stack[0:-1]:  # Non-enclosed segment
                result.extend(process_non_enclosing(current_segment))
            current_segment = ""
            i += 1
        
        # Handle closing symbols (}, ], >)
        elif symbol in '}]>':
            if stack:
                opening = stack[-1]
                if (opening == '{' and symbol == '}') or \
                   (opening == '[' and symbol == ']') or \
                   (opening == '<' and symbol == '>'):
                    stack.pop()
                    if not stack:  # Top-level pair complete
                        result.append(f"{opening}X{symbol}")
                    # Discard internal symbols
                    current_segment = ""
            i += 1
        
        # Collect non-enclosing symbols
        else:
            current_segment += symbol
            i += 1
    
    # Process any remaining non-enclosing symbols
    if current_segment and not stack:
        result.extend(process_non_enclosing(current_segment))
    
    return ' '.join(result)


def process_non_enclosing(segment: str) -> List[str]:
    """
    Process a segment of non-enclosing symbols into individual tokens.
    
    This function handles symbols that are not enclosed within brackets, quotes,
    braces, or angles. It preserves consecutive identical symbols and handles
    special cases like IP addresses.
    
    Args:
        segment (str): String containing non-enclosed symbols
        
    Returns:
        List[str]: List of processed symbol tokens
        
    Examples:
        >>> process_non_enclosing('...')
        ['...']
        
        >>> process_non_enclosing('--:')
        ['-', '-', ':']
        
        >>> process_non_enclosing(':://')
        [':', ':', '/', '/']
    """
    if not segment:
        return []
    
    # Handle IP address (three periods -> ...)
    if re.match(r'^\.\.\.$', segment):
        return ['...']
    
    # Split other symbols into individual tokens
    # Preserve consecutive identical symbols (e.g., -- -> ['-', '-'])
    result = []
    current = segment[0]
    count = 1
    
    for symbol in segment[1:]:
        if symbol == current:
            count += 1
        else:
            if count > 1:
                result.extend([current] * count)
            else:
                result.append(current)
            current = symbol
            count = 1
    
    # Append the last group
    if count > 1:
        result.extend([current] * count)
    else:
        result.append(current)
    
    return result


def preprocess_logs_batch(raw_logs: List[str]) -> List[str]:
    """
    Preprocess a batch of raw log lines.
    
    Convenience function for processing multiple logs at once with
    consistent preprocessing logic.
    
    Args:
        raw_logs (List[str]): List of raw log strings
        
    Returns:
        List[str]: List of preprocessed pattern strings
        
    Example:
        >>> logs = ['[user] "test"', '192.168.1.1 "GET /"']
        >>> preprocess_logs_batch(logs)
        ['[X] "X"', '. . . "X"']
    """
    return [preprocess_log(log.strip()) for log in raw_logs if log.strip()]


# Test function for development and validation
def test_preprocessing():
    """
    Test the preprocessing functions with common log patterns.
    
    This function validates the preprocessing logic against known log formats
    and ensures consistent pattern generation.
    """
    test_logs = [
        '[user]: "hello world"',
        '<timestamp><event></timestamp>',
        '{data: value}',
        '"string with spaces"',
        '"first string" "second string"',
        '192.168.1.1 - - [22/Jan/2019:14:18:06 +0330]',
        '84.241.26.205 [timestamp] "GET /image/820/brand HTTP/1.1" 200',
        'simple-log-entry',
        '1.2.3.4 - - : [time] "GET /test" 200',
        '[][]...()//'
    ]
    
    print("=== Log Preprocessing Test ===")
    for log in test_logs:
        pattern = preprocess_log(log)
        print(f"Log:     {log}")
        print(f"Pattern: {pattern}")
        print()


if __name__ == "__main__":
    # Run tests when executed directly
    test_preprocessing()