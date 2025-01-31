
# For dataset format details visit: https://crfm.stanford.edu/2023/03/13/alpaca.html

import copy
import json

import torch
from torch.utils.data import Dataset
import tiktoken
import os

import datetime
import langchain
import langchain.prompts
from langchain.schema import Document
import langchain_community
from langchain_community.retrievers import BM25Retriever
import re
import lxml.etree
import dateutil
import ast
import tiktoken
from typing import Any, Callable, Dict, Optional, Union
import random
import xml.etree.ElementTree as ET
from io import StringIO

def preprocess_ehr(ehr_data):
    if isinstance(ehr_data, str):
        # If it's a single string, assume it's an XML string
        try:
            root = ET.fromstring(ehr_data)
            # Assuming each 'visit' is a direct child of the root
            visits = [ET.tostring(visit, encoding='unicode') for visit in root.findall('visit')]
            return visits
        except ET.ParseError:
            print("Error parsing XML string. Returning original string split by double newlines.")
            return ehr_data.split('\n\n')
    elif isinstance(ehr_data, list):
        # If it's already a list, assume each item is a visit
        return ehr_data
    else:
        raise ValueError("Unexpected EHR data format")

PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}


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
            # After performing the filtering, add the XML-as-string to the list
            # of parent_nodes (essentially comprises a document)
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
            ehr_visit_strs  # Each pt timeline is a list of visits as xml strs
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
    # pattern = r'start="([\d/]+ [\d:]+)"'
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
                    # print(datetime.datetime.strptime(start_dt, fmt))
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
        print(f"i: {i}, dts[i]: {dts[i]}, doc_length: {doc_length}")
        final_docs.append((dts[i], doc_content))
        current_length += doc_length
        if current_length > target_length:
            break
        # We used to also consider the datetime of the added docs so as to not truncate the most
        # relevant document, but that adds complexity to the exposition; better to keep simple
        # else:
        #     # Check if final_docs is not empty and if the current doc is earlier than all other documents in final_docs
        #     if final_docs and dts[i] < min(dt for dt, _ in final_docs):
        #         final_docs.append((dts[i], doc_content))
        #         break
        #     else:
        #         break

    # Sort final_docs chronologically
    final_docs.sort(key=lambda x: x[0])

    # Extract only the document content for the final output
    final_docs_content = [doc_content for _, doc_content in final_docs]

    return final_docs_content


class InstructionDataset(Dataset):
    def __init__(self, dataset_config, tokenizer, partition="train", **kwargs):
        self.ann = json.load(open(dataset_config.data_path))
        datasize=kwargs['kwargs'].get("dataset_size", int(len(self.ann)))
        random.seed(42)
        random.shuffle(self.ann)
        if partition == "train":
            train_size = int(datasize * 0.8)
            self.ann = self.ann[:train_size]
        # elif partition == "test":
        #     train_size = int(datasize * 0.7)
        #     test_size = int(datasize * 0.15)
        #     self.ann = self.ann[train_size:train_size+test_size]
        else:
            train_size = int(datasize * 0.8)
            test_size = int(datasize * 0.2)
            self.ann = self.ann[train_size:train_size+test_size]
            # self.ann = self.ann[train_size + test_size:train_size + 2* test_size]

        self.tokenizer = tokenizer
        self.dataset_config=dataset_config
        self.partition=partition

    def __len__(self):
        return len(self.ann)

    def save_split(self, output_path):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(self.ann, f)

    def __getitem__(self, index):
        context_length=self.dataset_config.context_length
        IGNORE_INDEX = -100  # The default setting in CrossEntropyLoss

        ann = self.ann[index]
        if ann.get("input", "") == "":
            prompt = PROMPT_DICT["prompt_no_input"].format_map(ann)
        else:
            number_tokens_instruction = len(self.tokenizer.encode(ann["instruction"]))
            number_tokens_prompt_template = len(self.tokenizer.encode(PROMPT_DICT["prompt_input"]))
            target_ehr_length = (context_length - number_tokens_instruction - number_tokens_prompt_template)
            if target_ehr_length <= 0 or context_length == 2048:
                print("no prompt triggered")
                prompt = PROMPT_DICT["prompt_no_input"].format_map(ann)
            else:
                if self.dataset_config.use_RAG:
                    print("using RAG")
                    relevant_ehr = ann.get("input")

                    # Preprocess the EHR data
                    preprocessed_ehr = preprocess_ehr(relevant_ehr)

                    # Return a list of the most relevant visit strings
                    most_relevant_visits = retrieve_most_relevant_visits(
                        ehr_visit_strs=preprocessed_ehr,
                        query=ann.get("instruction"),
                        target_length=target_ehr_length,
                        tokenizer=tiktoken.get_encoding("cl100k_base"),
                    )
                    # print(f"len(most_relevant_visits): {len(most_relevant_visits)}")
                    relevant_ehr = "\n".join(most_relevant_visits)
                else:
                    relevant_ehr = "\n".join(ann.get("input"))

                # Do a first pass with a fast tokenizer
                fast_tokenizer = tiktoken.get_encoding("cl100k_base")
                fast_encoded = fast_tokenizer.encode(relevant_ehr)
                fast_encoded_truncated = fast_encoded[-(2 * target_ehr_length) :]
                fast_truncated_ehr = fast_tokenizer.decode(fast_encoded_truncated)

                # Then do a second pass with the actual tokenizer
                encoded_ehr = self.tokenizer.encode(fast_truncated_ehr)
                truncated_encoded_ehr = encoded_ehr[-target_ehr_length:]

                truncated_ehr = self.tokenizer.decode(truncated_encoded_ehr)
                ann['input'] = truncated_ehr
                prompt = PROMPT_DICT["prompt_input"].format_map(ann)
        example = prompt + ann["output"]
        prompt = torch.tensor(
            self.tokenizer.encode(prompt), dtype=torch.int64
        )
        example = self.tokenizer.encode(example)
        example.append(self.tokenizer.eos_token_id)
        example = torch.tensor(
            example, dtype=torch.int64
        )
        labels = copy.deepcopy(example)
        labels[: len(prompt)] = -1
        example_mask = example.ge(0)
        label_mask = labels.ge(0)
        example[~example_mask] = 0
        labels[~label_mask] = IGNORE_INDEX

        return {
            "input_ids": example.tolist(),
            "labels": labels.tolist(),
            "attention_mask":example_mask.tolist(),
        }
