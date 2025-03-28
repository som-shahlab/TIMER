
import argparse
import ast
from ast import literal_eval
import datetime
import os
import re
from typing import Any, Callable, Dict, Optional, Union

import dateutil
import langchain
import langchain.prompts
from langchain.schema import Document
import langchain_community
from langchain_community.retrievers import BM25Retriever
import lxml.etree
import pandas as pd
import transformers
import tiktoken
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Filter and trim a MedAlign Electronic Health Record (EHR) XML doc"
    )
    parser.add_argument(
        "--model",
        required=True,
        type=str,
        help="Path to HuggingFace model or API name (eg 'ChatGPT', 'MedPalm')",
    )
    parser.add_argument(
        "--tokenizer",
        required=False,
        type=str,
        help="Name of/path to the tokenizer (if different from the model)",
    )
    parser.add_argument(
        "--path_to_ehrs",
        required=True,
        type=str,
        help="Path to directory of .xml EHR files to be truncated",
    )
    parser.add_argument(
        "--path_to_instructions",
        required=True,
        type=str,
        help="File that links each person ID with an instruction/question",
    )
    parser.add_argument(
        "--path_to_save",
        required=True,
        type=str,
        help="Directory where EHRs should be stored",
    )
    parser.add_argument(
        "--target_context_length",
        required=True,
        type=int,
        default=512,
        help="Target truncation length in terms of tokens",
    )
    parser.add_argument(
        "--generation_length",
        type=int,
        default=256,
        help="Target generation length in terms of tokens",
    )
    parser.add_argument(
        "--path_to_prompt_template",
        type=str,
        help="Path to location where prompt template is stored",
    )
    parser.add_argument(
        "--codes_only",
        action="store_true",
        help="Whether timelines should be filtered to only contain structured codes (i.e., no notes). Only one of `codes_only` and `notes_only` should be set.",
    )
    parser.add_argument(
        "--notes_only",
        action="store_true",
        help="Whether timelines should be filtered to only contain clinical notes (i.e., no structured codes). Only one of `codes_only` and `notes_only` should be set.",
    )
    parser.add_argument(
        "--use_RAG",
        action="store_true",
        help='Whether BM25 should be used to retrieve the most relevant visits from each EHR based on its associated instruction (for retrieval augmented generation, or "RAG")',
    )
    parser.add_argument(
        "--include_ehr",
        action="store_true",
        help='Whether ehr should be included in the prompt',
    )
    args = parser.parse_args()
    return args


def get_instructions(path_to_instructions: str) -> Dict[int, Dict[str, Any]]:
    """
    Builds map from Instruction ID to instruction details

    The needed information for creating the map is accomplished by reading
    a CSV file from the user-specified path.

    The CSV file is expected to contain at least the following columns:
    - instruction_id: The ID of the instruction.
    - question: The text of the instruction.
    - person_id: The ID of the associated patient.
    - is_selected_ehr: A flag indicating whether the instruction is selected.


    Parameters:
        path_to_instructions (str): Path to CSV file containing instructions.

    Returns:
        Dict[int, Dict[str, Any]]: A dictionary mapping instruction IDs to a
            dictionary containing instruction text and associated patient ID.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If the CSV file does not contain the expected columns.
    """
    if not os.path.exists(path_to_instructions):
        raise FileNotFoundError(
            f"The specified file {path_to_instructions} does not exist."
        )

    instructions_df = pd.read_csv(path_to_instructions)
    required_columns = {
        "instruction_id",
        "question",
        "person_id",
        "is_selected_ehr",
    }
    if not required_columns.issubset(instructions_df.columns):
        raise ValueError(
            f"The CSV file is missing one or more of the required columns: {required_columns}"
        )

    selected_instructions_df = instructions_df.query("is_selected_ehr == 'yes'")
    instructions_map = {
        row["instruction_id"]: {
            "instruction": row["question"],
            "patient_id": row["person_id"],
        }
        for _, row in selected_instructions_df.iterrows()
    }
    return instructions_map


def extract_patient_id_from_fname(fname: str) -> Optional[int]:
    """
    Extracts and returns the patient ID from a given filename.

    The function expects filenames in the format 'EHR_<patient_id>.xml',
    where <patient_id> is a sequence of digits.

    Parameters:
        fname (str): The filename from which to extract the patient ID.

    Returns:
        Optional[int]: The extracted patient ID as an integer, or None if
                    the filename doesn't match the expected format.
    """
    regex_result = re.search(r"EHR_(\d+)\.xml", fname)
    if regex_result is None:
        return None
    return int(regex_result.group(1))


def get_ehrs(path_to_ehrs: str) -> Dict[int, str]:
    """
    Builds a map from Instruction ID to EHR (Electronic Health Record) timeline.

    EHR timelines are in string format and EHR files are read in from the
    user-specified directory. Each file in the directory should be named
    'EHR_<patient_id>.xml', where <patient_id> is a sequence of digits.


    Parameters:
        path_to_ehrs (str): The path to the directory containing the EHR files.

    Returns:
        Dict[int, str]: A dictionary mapping patient IDs to EHR timelines.

    Raises:
        FileNotFoundError: If the specified directory does not exist.
    """
    if not os.path.isdir(path_to_ehrs):
        raise FileNotFoundError(
            f"The specified directory {path_to_ehrs} does not exist."
        )

    ehr_map = {}
    for fname in os.listdir(path_to_ehrs):
        pt_id = extract_patient_id_from_fname(fname)
        if pt_id is None:
            print(
                f"Warning: File '{fname}' does not match the expected format "
                "and will be skipped."
            )
            continue

        file_path = os.path.join(path_to_ehrs, fname)
        with open(file_path, encoding="utf-8", mode="r") as f:
            ehr = f.read()

        ehr_map[pt_id] = ehr
    return ehr_map


def get_tokenizer(tokenizer_name: str) -> Callable:
    """
    Returns a tokenizer based on the given tokenizer name.

    Parameters:
        tokenizer_name (str): The name of the tokenizer. Acceptable values are:
            - "tiktoken"
            - "chatgpt"
            - "gpt-3.5-turbo"
            - "gpt-4"
            - "gpt-4-turbo"
            - "gpt-4o"
            - "cl100k_base"
            - Any valid tokenizer name recognized by the transformers library.

    Returns:
        Callable: The tokenizer instance.
    """
    if tokenizer_name.lower() in [
        "tiktoken",
        "chatgpt",
        "gpt-3.5-turbo",
        "gpt-4",
        "gpt-4-turbo",
        "gpt-4o",
        "cl100k_base",
    ]:
        return tiktoken.get_encoding("cl100k_base")
    print(tokenizer_name)
    return transformers.AutoTokenizer.from_pretrained(tokenizer_name, legacy=False)


def tag_rgx_expression(include, exclude):
    """Create rgx expression to determine which tags should be included or excluded"""
    if include:
        return re.compile("|".join(include))
    elif exclude:
        return re.compile(f'^((?!({"|".join(exclude)})))')
    return re.compile(".*")


def fetch_nodes_with_tag(start_node, tag_str):
    """Fetch nodes with certain tag value"""
    return start_node.xpath(tag_str)


def cast_dtype(i: str) -> Union[str, datetime.datetime, int, float]:
    """Convert string to its appropriate type"""
    try:
        return ast.literal_eval(i)
    except (ValueError, SyntaxError, TypeError):
        try:

            if isinstance(i, (datetime.datetime, list)):
                return i
            return dateutil.parser.parse(i)
        except ValueError:
            return i


def check_condition(node_value, value, condition):
    """Check a single condition"""

    casted_node_value = cast_dtype(node_value)
    casted_value = cast_dtype(value)

    condition_mapping = {
        "$eq": (lambda x, y: x == y),
        "$ne": (lambda x, y: x != y),
        "$gte": (lambda x, y: x >= y),
        "$gt": (lambda x, y: x > y),
        "$lte": (lambda x, y: x <= y),
        "$lt": (lambda x, y: x < y),
        "$in": (lambda x, y: x in y),
        "$nin": (lambda x, y: x not in y),
    }

    return condition_mapping.get(condition, lambda x, y: False)(
        casted_node_value, casted_value
    )


def check_all_conditions(node, conditions):
    """Check that a node meets all conditions"""
    match = True
    for key, value_conditions in conditions.items():

        if not (key in node.attrib):
            match = False
        elif not value_conditions:
            return True

        for condition, value in value_conditions.items():
            if not check_condition(node.attrib[key], value, condition):
                match = False

    return match


def remove_node(start_node, bad_node, remove_children=True):
    """Remove specified node from its direct parent"""
    parent = bad_node.getparent()
    if parent is not None:
        if not remove_children:
            for child in bad_node:
                parent.append(child)
        parent.remove(bad_node)
    return start_node


def query_xml_str(xml_str, filters):
    """Apply filters to an XML string"""
    is_str_equal = lambda x, y: bool(y.match(x.lower()))

    tree = lxml.etree.ElementTree(lxml.etree.fromstring(xml_str))
    root = tree.getroot()

    parent_tag = filters.get("@parent", None)
    include = filters.get("@include_children", [])
    exclude = filters.get("@exclude_children", [])
    first_n = filters.get("@first", None)
    last_n = filters.get("@last", None)

    rgx_tag_compare = tag_rgx_expression(include, exclude)
    parent_nodes = []
    for parent_node in fetch_nodes_with_tag(root, f".//{parent_tag}"):

        if not check_all_conditions(parent_node, filters.get(parent_tag, {})):
            continue

        for child in parent_node.findall(".//"):

            is_tag_match = is_str_equal(child.tag, rgx_tag_compare)
            if not is_tag_match:
                node = remove_node(parent_node, child, remove_children=False)

            elif not check_all_conditions(child, filters.get(child.tag, {})):
                node = remove_node(parent_node, child, remove_children=False)

        if parent_node.findall(".//"):
            parent_str = lxml.etree.tostring(parent_node, pretty_print=False).decode()
            parent_nodes.append(parent_str)

    if first_n:
        return parent_nodes[:first_n]
    elif last_n:
        return parent_nodes[-last_n:]

    return parent_nodes


def filter_events(ehrs, codes_only=False, notes_only=False):
    print("Filtering events...")

    assert not (
        codes_only and notes_only
    ), "Only one of `notes_only` and `codes_only` should be true"

    pt_ids = ehrs.keys()
    for pt_id_key in pt_ids:
        ehr_as_xml_str = ehrs[pt_id_key]
        if notes_only:
            filters = {"@parent": "visit", "@include_children": ["note"]}
            ehr_visit_strs = query_xml_str(
                xml_str=ehr_as_xml_str,
                filters=filters,
            )

        elif codes_only:
            filters = {"@parent": "visit", "@exclude_children": ["note"]}
            ehr_visit_strs = query_xml_str(
                xml_str=ehr_as_xml_str,
                filters=filters,
            )

        else:
            filters = {"@parent": "visit"}
            ehr_visit_strs = query_xml_str(
                xml_str=ehr_as_xml_str,
                filters=filters,
            )

        ehrs[pt_id_key] = (
            ehr_visit_strs  
        )

    return ehrs


def retrieve_most_relevant_visits(ehr_visit_strs, query, target_length, tokenizer):
    """
    Retrieve and filter relevant EHR visits based on a query and target length.

    This function retrieves electronic health record (EHR) visit strings, sorts them
    by relevance using the BM25Retriever, and constructs a list of final documents
    that fit within a specified character length. The final list ensures that the
    most important visit isn't cut off and is sorted chronologically.

    Parameters:
        ehr_visit_strs (list of str): List of EHR visit strings.
        query (str): Query string to retrieve relevant visits.
        target_length (int): Maximum total token count for the final list of documents.
        tokenizer (Callable): Tokenizer that converts text to tokens (used for tracking context length)

    Returns:
        list[str]: List of EHR visit strings sorted chronologically and constrained by the target length.
    """
    langchain_docs = [
        langchain.schema.Document(page_content=doc) for doc in ehr_visit_strs
    ]

    # `k` is the number of documents to retrieve
    # We retrieve everything and just use the BM25Retriever to sort the documents
    retriever = langchain_community.retrievers.BM25Retriever.from_documents(
        langchain_docs, k=len(langchain_docs)
    )

    # Invoking the retriever means the most relevant documents are sorted first
    sorted_docs = retriever.invoke(query)

    # Define the regex pattern to find the start time
    pattern = r'start="([\d/]+ [\d:]+ ?[APM]{0,2})"'

    docs = []
    dts = []

    # Find the startime of the document
    for doc in sorted_docs:
        doc_content = doc.page_content
        start_dt_match = re.search(pattern, doc_content)
        if start_dt_match:
            start_dt = start_dt_match.group(1)
            parsed = False
            # Try different date formats
            for fmt in (
                "%m/%d/%y %I:%M %p",
                "%m/%d/%Y %I:%M %p",
                "%m/%d/%y %H:%M",
                "%m/%d/%Y %H:%M",
            ):
                try:
                    dts.append(datetime.datetime.strptime(start_dt, fmt))
                    parsed = True
                    break
                except ValueError:
                    continue
            if not parsed:
                print(f"Error parsing date: {start_dt}")
                continue
        else:
            print("Start time not found.")
            dts.append(datetime.datetime.min)
        docs.append(doc_content)

    final_docs = []
    current_length = 0

    # Add documents until we exceed the allocated context length
    for i in range(len(docs)):
        doc_content = docs[i]
        doc_length = len(tokenizer.encode(doc_content))
        final_docs.append((dts[i], doc_content))
        current_length += doc_length
        if current_length > target_length:
            break

    # Sort final_docs chronologically
    final_docs.sort(key=lambda x: x[0])

    # Extract only the document content for the final output
    final_docs_content = [doc_content for _, doc_content in final_docs]

    return final_docs_content


def get_prompt_template(path_to_template="prompt_templates/generic.txt") -> str:
    with open(path_to_template, encoding="utf-8", mode="r") as f:
        prompt_template_str = f.read()
    prompt_template_obj = langchain.prompts.PromptTemplate.from_template(
        prompt_template_str
    )
    return prompt_template_obj


def pack_and_trim_prompts(
    instructions: Dict[int, Dict[str, str]],
    ehrs: Dict[int, str],
    prompt_template: langchain.prompts.PromptTemplate,
    context_length: int,
    generation_length: int,
    tokenizer: Callable,
    use_RAG: bool = False,
    verbose: bool = False,
    include_ehr: bool = True,
) -> Dict[int, str]:
    """
    Returns:
        A map from Instruction ID to prompt
    """
    prompts_map = {}
    for instruction_id in tqdm(instructions.keys()):
        instruction = instructions[instruction_id]["instruction"]
        patient_id = instructions[instruction_id]["patient_id"]
        relevant_ehr = ehrs[patient_id]

        # Calculate how many tokens of EHR we can include in the prompt
        num_tokens_instruction = len(tokenizer.encode(instruction))
        num_tokens_prompt_template = len(tokenizer.encode(prompt_template.template))
        if include_ehr:
            target_ehr_length = (
                context_length
                - generation_length
                - num_tokens_prompt_template
                - num_tokens_instruction
            )
        else:
            target_ehr_length = 0

        if target_ehr_length <= 0:
            prompt_with_truncated_ehr = prompt_template.format(
                question=instruction, ehr=""
            )
        else:
            if use_RAG:
                # Return a list of the most relevant visit strings
                most_relevant_visits = retrieve_most_relevant_visits(
                    ehr_visit_strs=relevant_ehr,
                    query=instruction,
                    target_length=target_ehr_length,
                    tokenizer=tokenizer,
                )
                relevant_ehr = "\n".join(most_relevant_visits)
            else:
                relevant_ehr = "\n".join(relevant_ehr)

            # Do a first pass with a fast tokenizer
            fast_tokenizer = tiktoken.get_encoding("cl100k_base")
            fast_encoded = fast_tokenizer.encode(relevant_ehr)
            fast_encoded_truncated = fast_encoded[-(2 * target_ehr_length) :]
            fast_truncated_ehr = fast_tokenizer.decode(fast_encoded_truncated)

            # Then do a second pass with the actual tokenizer
            encoded_ehr = tokenizer.encode(fast_truncated_ehr)
            truncated_encoded_ehr = encoded_ehr[-target_ehr_length:]
            truncated_ehr = tokenizer.decode(truncated_encoded_ehr)
            prompt_with_truncated_ehr = prompt_template.format(
                question=instruction, ehr=truncated_ehr
            )

        prompts_map[instruction_id] = prompt_with_truncated_ehr

        if verbose:
            print(prompt_with_truncated_ehr)
            print("~" * 20)

    return prompts_map


def main():
    args = parse_args()
    model_name = args.model
    print(
        f"Constructing prompts for model = {model_name} "
        f"\n\twith prompt template = {args.path_to_prompt_template} "
        f"\n\twith target context length = {args.target_context_length} "
        f"\n\twith target generation length = {args.generation_length} "
    )

    ### FETCH INSTRUCTIONS ###
    print("Fetching instructions...")
    instructions = get_instructions(args.path_to_instructions)

    ### FETCH RELEVANT EHRs ###
    print("Fetching patient EHR timelines...")
    ehrs = get_ehrs(args.path_to_ehrs)

    ### LOAD TOKENIZER ###
    print("Loading tokenizer...")
    tokenizer = get_tokenizer(args.tokenizer)

    ### FILTER OUT EVENTS BY TYPE ###
    print("Filtering events...")
    ehrs = filter_events(ehrs, codes_only=args.codes_only, notes_only=args.notes_only)

    ### CONSTRUCT & TRUNCATE PROMPTS ###
    print("Constructing prompts using instructions and EHRs...")
    prompt_template = get_prompt_template(args.path_to_prompt_template)
    filled_prompts = pack_and_trim_prompts(
        instructions=instructions,
        ehrs=ehrs,
        prompt_template=prompt_template,
        context_length=args.target_context_length,
        generation_length=args.generation_length,
        tokenizer=tokenizer,
        use_RAG=args.use_RAG,
        verbose=False,
        include_ehr=args.include_ehr,
    )

    ### SAVE CONSTRUCTED PROMPTS TO DISK ###
    df_rows = []
    for instruction_id in tqdm(instructions.keys()):
        row = {}
        row["instruction_id"] = instruction_id
        patient_id = instructions[instruction_id]["patient_id"]
        row["patient_id"] = patient_id
        row["instruction"] = instructions[instruction_id]["instruction"]
        row["ehr"] = "".join(ehrs[patient_id])
        row["prompt"] = filled_prompts[instruction_id]
        row["prompt_template"] = args.path_to_prompt_template
        row["context_length"] = args.target_context_length
        row["generation_length"] = args.generation_length
        df_rows.append(row)

    prompts_df = pd.DataFrame(df_rows)
    prompts_df.to_csv(args.path_to_save, index=False)
    print("...Prompt construction complete")


if __name__ == "__main__":
    main()
