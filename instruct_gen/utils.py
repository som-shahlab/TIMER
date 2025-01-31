
import pandas as pd
from datetime import datetime
from lxml import etree
import tiktoken
import json
from pathlib import Path
from google.cloud import bigquery
from typing import Dict, Any, Tuple, Optional, List, Union

def setup_bigquery_client() -> bigquery.Client:
    return bigquery.Client()

def load_file_content(file_path: str) -> str:
    try:
        with open(file_path, 'r') as file:
            return file.read()
    except IOError as e:
        print(f"Error reading file {file_path}: {e}")
        return ""

def get_last_valid_visit_specialty(client: bigquery.Client, person_id: str, project_id: str) -> Tuple[str, str]:
    query = f"""
    WITH ranked_visits AS (
        SELECT 
            v.visit_occurrence_id,
            v.provider_id,
            p.specialty_source_value,
            ROW_NUMBER() OVER (ORDER BY v.visit_start_date DESC) as visit_rank
        FROM `{project_id}.{project_id}.visit_occurrence` v
        JOIN `{project_id}.{project_id}.provider` p ON v.provider_id = p.provider_id
        WHERE v.person_id = {person_id}
    )
    SELECT specialty_source_value, visit_occurrence_id
    FROM ranked_visits
    WHERE specialty_source_value IS NOT NULL
      AND LOWER(specialty_source_value) != 'Unknown'
      AND TRIM(specialty_source_value) != ''
    ORDER BY visit_rank
    LIMIT 1
    """
    query_job = client.query(query)
    result = query_job.result()
    row = next(iter(result), None)
    return (row.specialty_source_value, row.visit_occurrence_id) if row else ('Unknown', 'Unknown')


def extract_window_dates(window_xml: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract the start dates of the first and last visits in the window.
    
    Args:
        window_xml: XML string containing the window's visits
        
    Returns:
        Tuple of (start_date, end_date) in YYYY-MM-DD format
    """
    try:
        tree = etree.fromstring(window_xml.encode('utf-8'))
        dates = []
        
        # Specifically look for visit elements
        for visit in tree.xpath("//visit"):
            if "start" in visit.attrib:
                date_str = visit.attrib["start"]
                try:
                    # Try multiple date formats
                    date_formats = ['%m/%d/%Y %H:%M %p', '%m/%d/%Y %H:%M', '%m/%d/%Y']
                    parsed_date = None
                    
                    for date_format in date_formats:
                        try:
                            parsed_date = datetime.strptime(date_str, date_format)
                            break
                        except ValueError:
                            continue
                    
                    if parsed_date:
                        dates.append(parsed_date)
                    else:
                        print(f"Warning: Could not parse date {date_str} for visit")
                        
                except ValueError as e:
                    print(f"Warning: Date parsing error {e} for visit date {date_str}")
                    continue
        
        if dates:
            start_date = min(dates).strftime('%Y-%m-%d')
            end_date = max(dates).strftime('%Y-%m-%d')
            return start_date, end_date
        else:
            print(f"Warning: No valid dates found in window")
            return None, None
            
    except etree.XMLSyntaxError as e:
        print(f"Error parsing XML: {e}")
        return None, None

def create_context_windows(
    tree: etree._Element,
    window_size: int,
    tokenizer: tiktoken.Encoding,
    min_window_ratio: float = 0.5
) -> List[Dict[str, Any]]:
    """
    Split the EHR timeline into context windows of specified size, with date tracking.
    Args:
        tree: XML tree containing the patient timeline
        window_size: Maximum number of tokens per window
        tokenizer: Tiktoken tokenizer instance
        min_window_ratio: Minimum ratio of window_size for the last window to be kept
    Returns:
        List of dictionaries containing window information including dates
    """
    visits = tree.xpath("//visit")
    visit_strs = [etree.tostring(visit, pretty_print=True).decode() for visit in reversed(visits)]
    
    windows = []
    current_window = []
    current_length = 0
    
    for visit_str in visit_strs:
        visit_length = len(tokenizer.encode(visit_str))
        
        if current_length + visit_length > window_size:
            if current_window:
                window_xml = f'<patient>\n{"".join(current_window)}</patient>'
                start_date, end_date = extract_window_dates(window_xml)
                windows.append({
                    'xml': window_xml,
                    'token_length': current_length,
                    'start_date': start_date,
                    'end_date': end_date
                })
            current_window = [visit_str]
            current_length = visit_length
        else:
            current_window.append(visit_str)
            current_length += visit_length
    
    # Handle the last window
    if current_window:
        window_xml = f'<patient>\n{"".join(current_window)}</patient>'
        start_date, end_date = extract_window_dates(window_xml)
        windows.append({
            'xml': window_xml,
            'token_length': current_length,
            'start_date': start_date,
            'end_date': end_date
        })
    
    # Process windows based on the rules
    if len(windows) == 1:
        return [windows[0]]
    elif len(windows) > 1:
        if windows[-1]['token_length'] < (window_size * min_window_ratio):
            return windows[:-1]
        return windows
    else:
        return []

def initialize_windowed_jsonl(output_path: Path) -> None:
    """
    Initialize or clear the windowed JSONL file.
    
    Args:
        output_path: Path to the windowed JSONL file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.unlink(missing_ok=True)
    output_path.touch()

def save_window_to_jsonl(window_data: Dict[str, Any], patient_timeline: Dict[str, Any], output_path: Path) -> None:
    """
    Save a single window to the JSONL file.
    Optimized for minimal dictionary operations.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Don't clear the file each time, as we're appending multiple windows
    if not output_path.exists():
        output_path.touch()

    window_entry = {
        "uid": patient_timeline["uid"],
        "person_id": patient_timeline["person_id"],
        "window_index": window_data["window_index"],
        "text": window_data["text"],
    }
    with open(output_path, 'a') as f:
        json.dump(window_entry, f)
        f.write('\n')

def process_ehr_context(
    patient_timeline: Dict[str, Any],
    context_size: Union[str, int],
    output_jsonl: Optional[Path] = None
) -> List[Dict[str, Any]]:
    """
    Process EHR context and split into windows if needed.
    """
    tree = etree.fromstring(patient_timeline["text"].encode('utf-8'))
    tokenizer = tiktoken.get_encoding("cl100k_base")
    
    if context_size == "full":
        xml_str = etree.tostring(tree, pretty_print=True).decode()
        start_date, end_date = extract_window_dates(xml_str)
        tokens = len(tokenizer.encode(xml_str))
        return [{
            **patient_timeline,
            'window_index': 0,
            'window_token_count': tokens,
            'window_percent_full': 100,
            'start_date': start_date,
            'end_date': end_date
        }]
    elif context_size == "last_five":
        visits = tree.xpath("//visit")
        if len(visits) > 5:
            for visit in visits[:-5]:
                visit.getparent().remove(visit)
        filtered_xml = etree.tostring(tree, pretty_print=True).decode()
        start_date, end_date = extract_window_dates(filtered_xml)
        tokens = len(tokenizer.encode(filtered_xml))
        return [{
            **patient_timeline,
            'text': filtered_xml,
            'window_index': 0,
            'window_token_count': tokens,
            'window_percent_full': 100,
            'start_date': start_date,
            'end_date': end_date
        }]
    elif isinstance(context_size, (int, str)) and str(context_size).isdigit():
        window_size = int(context_size)
        if window_size <= 0:
            raise ValueError("Context window size must be positive")
        
        windows = create_context_windows(tree, window_size, tokenizer)
        
        processed_windows = []
        for i, window in enumerate(windows):
            window_tokens = len(tokenizer.encode(window['xml']))
            start_date, end_date = extract_window_dates(window['xml'])
            
            if start_date is None or end_date is None:
                print(f"Warning: Missing dates for window {i} of person {patient_timeline.get('person_id')}")
            
            processed_window = {
                **patient_timeline,
                "text": window['xml'],
                "window_index": i,
                "window_token_count": window_tokens,
                "window_percent_full": (window_tokens / window_size) * 100,
                "start_date": start_date,
                "end_date": end_date
            }
            if output_jsonl:
                save_window_to_jsonl(processed_window, patient_timeline, output_jsonl)
            processed_windows.append(processed_window)
            
        print(f"Created {len(processed_windows)} windows for timeline {patient_timeline.get('person_id')}:")
        for w in processed_windows:
            print(f"Window {w['window_index']}: {w['window_token_count']} tokens "
                  f"({w['window_percent_full']:.1f}% of max size) "
                  f"from {w['start_date']} to {w['end_date']}")
            
        return processed_windows
    else:
        raise ValueError(f"Invalid context_size: {context_size}")

def create_prompt_from_timeline(
    patient_timeline: Dict[str, Any], 
    prompt_file: str, 
) -> str:
    prompt_template = load_file_content(Path(prompt_file))
    ehr_content = patient_timeline["text"]
    prompt_content = prompt_template.format(ehr=ehr_content)
    prompt_format = """
        Using this JSON schema:
            Instruction-Response = {'instruction': str, 'response': str, 'evidence': str}
        Return a `list[Instruction-Response]
    """
    full_prompt = prompt_content + "\n\n" + prompt_format
    
    return full_prompt

def create_persona_prompt_from_timeline(
    patient_timeline: Dict[str, Any], 
    prompt_file: str, 
    person_id: str, 
) -> str:
    client = setup_bigquery_client()
    specialty, visit_occurrence_id = get_last_valid_visit_specialty(client, person_id)
    prompt_template = load_file_content(Path(prompt_file))
    ehr_content = patient_timeline["text"]
    prompt_content = prompt_template.format(speciality=specialty, ehr=ehr_content)
    prompt_format = """
        Using this JSON schema:
            Instruction-Response = {'instruction': str, 'response': str, 'evidence': str}
        Return a `list[Instruction-Response]
    """
    full_prompt = prompt_content + "\n\n" + prompt_format
    
    return full_prompt, specialty, visit_occurrence_id


# for generating temporal evaluation set

def create_context_window_for_eval(
    tree: etree._Element,
    window_size: int,
    tokenizer: tiktoken.Encoding,
) -> Dict[str, Any]:
    """
    Create a single window from the last N tokens of the EHR timeline for evaluation.
    
    Args:
        tree: XML tree containing the patient timeline
        window_size: Maximum number of tokens per window
        tokenizer: Tiktoken tokenizer instance
    Returns:
        Dictionary containing the most recent EHR window that fits within token limit
    """
    visits = tree.xpath("//visit")
    if len(visits) < 3:  # Ensure we have enough visits for temporal evaluation
        return None
        
    visit_strs = [etree.tostring(visit, pretty_print=True).decode() for visit in reversed(visits)]
    
    current_window = []
    current_length = 0
    
    for visit_str in visit_strs:
        visit_length = len(tokenizer.encode(visit_str))
        if current_length + visit_length > window_size:
            break
        current_window.append(visit_str)
        current_length += visit_length
    
    if len(current_window) < 3:  # Ensure window contains enough visits
        return None
        
    window_xml = f'<patient>\n{"".join(current_window)}</patient>'
    start_date, end_date = extract_window_dates(window_xml)
    
    return {
        'xml': window_xml,
        'token_length': current_length,
        'start_date': start_date,
        'end_date': end_date
    }

def process_ehr_for_temporal_eval(
    patient_timeline: Dict[str, Any],
    context_size: int,
) -> Optional[Dict[str, Any]]:
    """
    Process EHR specifically for temporal evaluation, focusing on records
    with sufficient longitudinal information.
    """
    tree = etree.fromstring(patient_timeline["text"].encode('utf-8'))
    tokenizer = tiktoken.get_encoding("cl100k_base")
    
    window = create_context_window_for_eval(tree, context_size, tokenizer)
    if not window:
        return None
        
    return {
        **patient_timeline,
        "text": window['xml'],
        "window_token_count": window['token_length'],
        "window_percent_full": (window['token_length'] / context_size) * 100,
        "start_date": window['start_date'],
        "end_date": window['end_date']
    }
