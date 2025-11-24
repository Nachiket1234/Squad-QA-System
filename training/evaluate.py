"""
Evaluation script for BERT Question Answering on SQuAD
"""

import os
import sys
import torch
import json
from torch.utils.data import DataLoader
from transformers import BertForQuestionAnswering, BertTokenizerFast
from tqdm import tqdm
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from data.dataset import SQuADDataset
from data.preprocessing import compute_exact_match, compute_f1, evaluate_squad


class BertQAEvaluator:
    """Evaluator class for BERT Question Answering."""
    
    def __init__(self, model_path: str, config: Config = None):
        """
        Initialize evaluator.
        
        Args:
            model_path: Path to trained model checkpoint
            config: Configuration object (optional)
        """
        self.model_path = model_path
        self.config = config if config else Config()
        self.device = torch.device(self.config.training.device)
        
        # Load model and tokenizer
        self.tokenizer = BertTokenizerFast.from_pretrained(self.config.model.model_name)
        self.model = BertForQuestionAnswering.from_pretrained(self.config.model.model_name)
        
        # Load checkpoint if provided
        if Path(model_path).exists():
            self.load_checkpoint(model_path)
        
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Evaluator initialized with model from {model_path}")
    
    def load_checkpoint(self, checkpoint_path):
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from {checkpoint_path}")
        
        if 'best_eval_loss' in checkpoint:
            print(f"Best eval loss: {checkpoint['best_eval_loss']:.4f}")
    
    def predict(self, question: str, context: str, n_best: int = 20, max_answer_length: int = 30):
        """
        Predict answer for a single question-context pair.
        
        Args:
            question: Question text
            context: Context text
            n_best: Number of best answers to consider
            max_answer_length: Maximum answer length in tokens
            
        Returns:
            Dictionary with predicted answer and confidence score
        """
        # Tokenize
        encoding = self.tokenizer(
            question,
            context,
            max_length=self.config.model.max_length,
            truncation='only_second',
            padding='max_length',
            return_tensors='pt',
            return_offsets_mapping=True
        )
        
        offset_mapping = encoding.pop('offset_mapping')[0]
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        token_type_ids = encoding['token_type_ids'].to(self.device)
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
        
        start_logits = outputs.start_logits[0].cpu().numpy()
        end_logits = outputs.end_logits[0].cpu().numpy()
        
        # Get sequence IDs
        sequence_ids = encoding.sequence_ids(0)
        
        # Find best answer
        best_answer = self._extract_answer(
            start_logits,
            end_logits,
            offset_mapping,
            sequence_ids,
            context,
            n_best,
            max_answer_length
        )
        
        return best_answer
    
    def _extract_answer(self, start_logits, end_logits, offset_mapping, 
                       sequence_ids, context, n_best, max_answer_length):
        """Extract answer from logits."""
        # Get top n_best start and end positions
        start_indexes = np.argsort(start_logits)[-n_best:].tolist()
        end_indexes = np.argsort(end_logits)[-n_best:].tolist()
        
        valid_answers = []
        
        for start_index in start_indexes:
            for end_index in end_indexes:
                # Skip if not in context
                if sequence_ids[start_index] != 1 or sequence_ids[end_index] != 1:
                    continue
                
                # Skip if end before start or too long
                if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                    continue
                
                # Skip if offset is None (special tokens)
                if offset_mapping[start_index] is None or offset_mapping[end_index] is None:
                    continue
                
                # Extract answer text
                start_char = offset_mapping[start_index][0]
                end_char = offset_mapping[end_index][1]
                answer_text = context[start_char:end_char]
                
                # Calculate score
                score = start_logits[start_index] + end_logits[end_index]
                
                valid_answers.append({
                    'text': answer_text,
                    'score': float(score),
                    'start': int(start_index),
                    'end': int(end_index)
                })
        
        # Sort by score and return best
        if valid_answers:
            valid_answers.sort(key=lambda x: x['score'], reverse=True)
            return valid_answers[0]
        else:
            return {'text': '', 'score': 0.0, 'start': 0, 'end': 0}
    
    def evaluate_dataset(self, dataset_path: str, max_samples: int = None):
        """
        Evaluate on a dataset.
        
        Args:
            dataset_path: Path to SQuAD JSON file
            max_samples: Maximum number of samples to evaluate (for testing)
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Load dataset without tokenization (we'll do it manually)
        dataset = SQuADDataset(dataset_path, tokenizer=None)
        examples = dataset.get_raw_examples()
        
        if max_samples:
            examples = examples[:max_samples]
        
        print(f"Evaluating on {len(examples)} examples...")
        
        predictions = {}
        references = {}
        
        for example in tqdm(examples, desc="Evaluating"):
            question = example['question']
            context = example['context']
            answer_text = example['answer_text']
            example_id = example['id']
            
            # Predict
            prediction = self.predict(question, context)
            
            # Store predictions and references
            predictions[example_id] = prediction['text']
            references[example_id] = [answer_text]  # SQuAD v1.1 has multiple answers, we use first
        
        # Calculate metrics
        metrics = evaluate_squad(predictions, references)
        
        print("\n" + "="*60)
        print("Evaluation Results")
        print("="*60)
        print(f"Exact Match (EM): {metrics['exact_match']:.2f}%")
        print(f"F1 Score:         {metrics['f1']:.2f}%")
        print("="*60)
        
        return {
            'exact_match': metrics['exact_match'],
            'f1': metrics['f1'],
            'predictions': predictions,
            'references': references
        }
    
    def save_predictions(self, predictions: dict, output_path: str):
        """Save predictions to JSON file."""
        with open(output_path, 'w') as f:
            json.dump(predictions, f, indent=2)
        print(f"Predictions saved to {output_path}")
    
    def analyze_errors(self, predictions: dict, references: dict, top_n: int = 10):
        """
        Analyze prediction errors.
        
        Args:
            predictions: Dictionary of predictions
            references: Dictionary of references
            top_n: Number of examples to show
            
        Returns:
            List of error examples
        """
        errors = []
        
        for example_id, pred in predictions.items():
            ref = references[example_id][0]
            
            em = compute_exact_match(pred, ref)
            f1 = compute_f1(pred, ref)
            
            if em == 0:  # Only incorrect predictions
                errors.append({
                    'id': example_id,
                    'prediction': pred,
                    'reference': ref,
                    'f1': f1
                })
        
        # Sort by F1 (worst first)
        errors.sort(key=lambda x: x['f1'])
        
        print("\n" + "="*60)
        print(f"Top {top_n} Error Examples")
        print("="*60)
        
        for i, error in enumerate(errors[:top_n], 1):
            print(f"\n{i}. ID: {error['id']}")
            print(f"   Prediction: '{error['prediction']}'")
            print(f"   Reference:  '{error['reference']}'")
            print(f"   F1 Score:   {error['f1']:.3f}")
        
        return errors


def main():
    """Main evaluation function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate BERT-QA model')
    parser.add_argument('--model_path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to evaluation dataset')
    parser.add_argument('--max_samples', type=int, default=None, help='Maximum samples to evaluate')
    parser.add_argument('--output_path', type=str, default='outputs/predictions.json', help='Output path for predictions')
    
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = BertQAEvaluator(args.model_path)
    
    # Evaluate
    results = evaluator.evaluate_dataset(args.dataset_path, args.max_samples)
    
    # Save predictions
    evaluator.save_predictions(results['predictions'], args.output_path)
    
    # Analyze errors
    evaluator.analyze_errors(results['predictions'], results['references'])


if __name__ == "__main__":
    main()
