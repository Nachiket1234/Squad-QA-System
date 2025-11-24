"""
Data preprocessing utilities for SQuAD Question Answering
"""

import torch
from transformers import BertTokenizerFast
from typing import Dict, List, Tuple, Optional


def prepare_train_features(examples: Dict, tokenizer, max_length: int = 384, 
                          stride: int = 128, pad_on_right: bool = True) -> Dict:
    """
    Prepare features for training by tokenizing and finding answer positions.
    
    This function handles:
    - Tokenization of question-context pairs
    - Converting character-level answer positions to token positions
    - Handling contexts longer than max_length with stride
    
    Args:
        examples: Dictionary with 'question', 'context', 'answers' keys (batch)
        tokenizer: Hugging Face tokenizer
        max_length: Maximum sequence length
        stride: Stride for sliding window over long contexts
        pad_on_right: Whether to pad on right or left
        
    Returns:
        Dictionary with tokenized inputs and answer positions
    """
    # Tokenize questions and contexts
    # Padding side determines where special tokens are placed
    tokenizer.padding_side = "right" if pad_on_right else "left"
    
    tokenized_examples = tokenizer(
        examples['question'],
        examples['context'],
        truncation='only_second',  # Only truncate context, not question
        max_length=max_length,
        stride=stride,
        return_overflowing_tokens=True,  # Return multiple chunks for long contexts
        return_offsets_mapping=True,  # Need this for answer position mapping
        padding='max_length',
    )
    
    # Map overflow samples back to their original examples
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized_examples.pop("offset_mapping")
    
    # Initialize lists for start and end positions
    start_positions = []
    end_positions = []
    
    # For each tokenized example (there may be multiple chunks per original example)
    for i, offsets in enumerate(offset_mapping):
        # Get the original example index for this chunk
        sample_index = sample_mapping[i]
        
        # Get the answer for this example
        answers = examples['answers'][sample_index]
        
        # If no answer (impossible question), set positions to CLS token
        if len(answers['answer_start']) == 0:
            start_positions.append(0)
            end_positions.append(0)
            continue
        
        # Get answer text and start position
        answer_start_char = answers['answer_start'][0]
        answer_end_char = answer_start_char + len(answers['text'][0])
        
        # Get sequence IDs to identify context tokens
        sequence_ids = tokenized_examples.sequence_ids(i)
        
        # Find the start and end of the context in token positions
        context_start = 0
        context_end = len(sequence_ids) - 1
        
        # Find first context token (sequence_id == 1)
        while context_start < len(sequence_ids) and sequence_ids[context_start] != 1:
            context_start += 1
        
        # Find last context token
        while context_end >= 0 and sequence_ids[context_end] != 1:
            context_end -= 1
        
        # Check if answer is in this chunk
        # If answer is outside this chunk, set to CLS position
        if not (offsets[context_start][0] <= answer_start_char and 
                offsets[context_end][1] >= answer_end_char):
            start_positions.append(0)
            end_positions.append(0)
        else:
            # Find token start position
            token_start_idx = context_start
            while token_start_idx <= context_end and offsets[token_start_idx][0] <= answer_start_char:
                token_start_idx += 1
            start_positions.append(token_start_idx - 1)
            
            # Find token end position
            token_end_idx = context_end
            while token_end_idx >= context_start and offsets[token_end_idx][1] >= answer_end_char:
                token_end_idx -= 1
            end_positions.append(token_end_idx + 1)
    
    tokenized_examples["start_positions"] = start_positions
    tokenized_examples["end_positions"] = end_positions
    
    return tokenized_examples


def prepare_validation_features(examples: Dict, tokenizer, max_length: int = 384,
                                stride: int = 128) -> Dict:
    """
    Prepare features for validation/testing.
    
    Similar to training preparation but keeps additional metadata for evaluation.
    
    Args:
        examples: Dictionary with 'question', 'context', 'id' keys
        tokenizer: Hugging Face tokenizer
        max_length: Maximum sequence length
        stride: Stride for sliding window
        
    Returns:
        Dictionary with tokenized inputs and metadata
    """
    tokenized_examples = tokenizer(
        examples['question'],
        examples['context'],
        truncation='only_second',
        max_length=max_length,
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding='max_length',
    )
    
    # Keep example IDs and offset mappings for post-processing
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    
    # Add example IDs
    tokenized_examples["example_id"] = []
    
    for i in range(len(tokenized_examples["input_ids"])):
        # Get sequence IDs to distinguish question from context
        sequence_ids = tokenized_examples.sequence_ids(i)
        
        # Set offset to None for question tokens and special tokens
        tokenized_examples["offset_mapping"][i] = [
            (o if sequence_ids[k] == 1 else None)
            for k, o in enumerate(tokenized_examples["offset_mapping"][i])
        ]
        
        # Store the example ID
        sample_index = sample_mapping[i]
        tokenized_examples["example_id"].append(examples["id"][sample_index])
    
    return tokenized_examples


def postprocess_qa_predictions(
    examples: Dict,
    features: Dict,
    predictions: Tuple[torch.Tensor, torch.Tensor],
    n_best_size: int = 20,
    max_answer_length: int = 30,
    null_score_threshold: float = 0.0
) -> Dict[str, str]:
    """
    Post-process model predictions to extract final answers.
    
    Args:
        examples: Original examples with context
        features: Tokenized features
        predictions: Tuple of (start_logits, end_logits)
        n_best_size: Number of n-best predictions to consider
        max_answer_length: Maximum length of an answer
        null_score_threshold: Threshold for null answer (SQuAD 2.0)
        
    Returns:
        Dictionary mapping example_id to predicted answer text
    """
    import collections
    import numpy as np
    
    start_logits, end_logits = predictions
    
    # Build a map from example to features
    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = collections.defaultdict(list)
    
    for i, feature in enumerate(features):
        features_per_example[example_id_to_index[feature["example_id"]]].append(i)
    
    # Collect predictions for each example
    all_predictions = {}
    
    for example_index, example in enumerate(examples):
        feature_indices = features_per_example[example_index]
        
        min_null_score = None
        prelim_predictions = []
        
        # Loop through all features associated with this example
        for feature_index in feature_indices:
            start_logit = start_logits[feature_index]
            end_logit = end_logits[feature_index]
            offset_mapping = features[feature_index]["offset_mapping"]
            
            # Get null score (score of CLS token)
            null_score = start_logit[0] + end_logit[0]
            if min_null_score is None or null_score < min_null_score:
                min_null_score = null_score
            
            # Get top n_best start and end positions
            start_indexes = np.argsort(start_logit)[-1 : -n_best_size - 1 : -1].tolist()
            end_indexes = np.argsort(end_logit)[-1 : -n_best_size - 1 : -1].tolist()
            
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Skip invalid predictions
                    if offset_mapping[start_index] is None or offset_mapping[end_index] is None:
                        continue
                    
                    if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                        continue
                    
                    # Get answer text
                    start_char = offset_mapping[start_index][0]
                    end_char = offset_mapping[end_index][1]
                    answer_text = example["context"][start_char:end_char]
                    
                    prelim_predictions.append({
                        "score": start_logit[start_index] + end_logit[end_index],
                        "text": answer_text
                    })
        
        # Get the best prediction
        if len(prelim_predictions) > 0:
            best_prediction = sorted(prelim_predictions, key=lambda x: x["score"], reverse=True)[0]
            all_predictions[example["id"]] = best_prediction["text"]
        else:
            all_predictions[example["id"]] = ""
    
    return all_predictions


def normalize_answer(text: str) -> str:
    """
    Normalize answer text for evaluation.
    
    Args:
        text: Answer text
        
    Returns:
        Normalized text
    """
    import re
    import string
    
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    
    def white_space_fix(text):
        return ' '.join(text.split())
    
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    
    def lower(text):
        return text.lower()
    
    return white_space_fix(remove_articles(remove_punc(lower(text))))


def get_tokens(text: str) -> List[str]:
    """
    Tokenize text for F1 calculation.
    
    Args:
        text: Input text
        
    Returns:
        List of tokens
    """
    if not text:
        return []
    return normalize_answer(text).split()


def compute_exact_match(prediction: str, ground_truth: str) -> int:
    """
    Compute Exact Match score.
    
    Args:
        prediction: Predicted answer
        ground_truth: Ground truth answer
        
    Returns:
        1 if exact match, 0 otherwise
    """
    return int(normalize_answer(prediction) == normalize_answer(ground_truth))


def compute_f1(prediction: str, ground_truth: str) -> float:
    """
    Compute F1 score between prediction and ground truth.
    
    Args:
        prediction: Predicted answer
        ground_truth: Ground truth answer
        
    Returns:
        F1 score
    """
    prediction_tokens = get_tokens(prediction)
    ground_truth_tokens = get_tokens(ground_truth)
    
    # If either is empty
    if len(prediction_tokens) == 0 or len(ground_truth_tokens) == 0:
        return int(prediction_tokens == ground_truth_tokens)
    
    # Calculate overlap
    common = set(prediction_tokens) & set(ground_truth_tokens)
    num_same = len(common)
    
    if num_same == 0:
        return 0
    
    precision = num_same / len(prediction_tokens)
    recall = num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    
    return f1


def evaluate_squad(predictions: Dict[str, str], references: Dict[str, List[str]]) -> Dict[str, float]:
    """
    Evaluate SQuAD predictions using EM and F1 metrics.
    
    Args:
        predictions: Dictionary mapping example_id to predicted answer
        references: Dictionary mapping example_id to list of ground truth answers
        
    Returns:
        Dictionary with 'exact_match' and 'f1' scores
    """
    em_scores = []
    f1_scores = []
    
    for example_id, prediction in predictions.items():
        ground_truths = references.get(example_id, [])
        
        # Compute max scores across all ground truth answers
        em_score = max((compute_exact_match(prediction, gt) for gt in ground_truths), default=0)
        f1_score = max((compute_f1(prediction, gt) for gt in ground_truths), default=0.0)
        
        em_scores.append(em_score)
        f1_scores.append(f1_score)
    
    return {
        'exact_match': sum(em_scores) / len(em_scores) * 100 if em_scores else 0,
        'f1': sum(f1_scores) / len(f1_scores) * 100 if f1_scores else 0
    }
