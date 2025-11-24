"""
Inference pipeline for BERT Question Answering
"""

import torch
from transformers import BertForQuestionAnswering, BertTokenizerFast
from pathlib import Path
import numpy as np
from typing import Dict, List, Tuple


class QAPredictor:
    """
    Question Answering predictor for deployment.
    
    Wraps model and tokenizer for easy inference.
    """
    
    def __init__(self, model_path: str = None, model_name: str = 'bert-base-uncased'):
        """
        Initialize predictor.
        
        Args:
            model_path: Path to fine-tuned model checkpoint (optional)
            model_name: Pretrained model name (default: bert-base-uncased)
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load tokenizer
        self.tokenizer = BertTokenizerFast.from_pretrained(model_name)
        
        # Load model
        self.model = BertForQuestionAnswering.from_pretrained(model_name)
        
        # Load fine-tuned weights if provided
        if model_path and Path(model_path).exists():
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded fine-tuned model from {model_path}")
        else:
            print(f"Using base model: {model_name}")
        
        self.model.to(self.device)
        self.model.eval()
    
    def predict(
        self,
        question: str,
        context: str,
        max_length: int = 384,
        n_best: int = 20,
        max_answer_length: int = 30,
        return_multiple: bool = False
    ) -> Dict:
        """
        Predict answer for a question given context.
        
        Args:
            question: Question text
            context: Context text
            max_length: Maximum sequence length
            n_best: Number of best answers to consider
            max_answer_length: Maximum answer length in tokens
            return_multiple: If True, return top n_best answers
            
        Returns:
            Dictionary with answer text, confidence score, and position
        """
        # Tokenize
        encoding = self.tokenizer(
            question,
            context,
            max_length=max_length,
            truncation='only_second',
            padding='max_length',
            return_tensors='pt',
            return_offsets_mapping=True
        )
        
        offset_mapping = encoding.pop('offset_mapping')[0]
        
        # Move to device
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
        
        # Extract logits
        start_logits = outputs.start_logits[0].cpu().numpy()
        end_logits = outputs.end_logits[0].cpu().numpy()
        
        # Get sequence IDs
        sequence_ids = encoding.sequence_ids(0)
        
        # Extract answers
        if return_multiple:
            return self._extract_multiple_answers(
                start_logits, end_logits, offset_mapping,
                sequence_ids, context, n_best, max_answer_length
            )
        else:
            return self._extract_best_answer(
                start_logits, end_logits, offset_mapping,
                sequence_ids, context, n_best, max_answer_length
            )
    
    def _extract_best_answer(
        self, start_logits, end_logits, offset_mapping,
        sequence_ids, context, n_best, max_answer_length
    ) -> Dict:
        """Extract single best answer."""
        answers = self._get_valid_answers(
            start_logits, end_logits, offset_mapping,
            sequence_ids, context, n_best, max_answer_length
        )
        
        if answers:
            best = answers[0]
            return {
                'answer': best['text'],
                'score': float(best['score']),
                'start': int(best['start']),
                'end': int(best['end']),
                'confidence': self._calculate_confidence(best['score'])
            }
        else:
            return {
                'answer': '',
                'score': 0.0,
                'start': 0,
                'end': 0,
                'confidence': 0.0
            }
    
    def _extract_multiple_answers(
        self, start_logits, end_logits, offset_mapping,
        sequence_ids, context, n_best, max_answer_length
    ) -> List[Dict]:
        """Extract multiple best answers."""
        answers = self._get_valid_answers(
            start_logits, end_logits, offset_mapping,
            sequence_ids, context, n_best, max_answer_length
        )
        
        results = []
        for ans in answers[:5]:  # Return top 5
            results.append({
                'answer': ans['text'],
                'score': float(ans['score']),
                'start': int(ans['start']),
                'end': int(ans['end']),
                'confidence': self._calculate_confidence(ans['score'])
            })
        
        return results
    
    def _get_valid_answers(
        self, start_logits, end_logits, offset_mapping,
        sequence_ids, context, n_best, max_answer_length
    ) -> List[Dict]:
        """Get all valid answers sorted by score."""
        # Get top indices
        start_indexes = np.argsort(start_logits)[-n_best:].tolist()
        end_indexes = np.argsort(end_logits)[-n_best:].tolist()
        
        valid_answers = []
        
        for start_index in start_indexes:
            for end_index in end_indexes:
                # Skip if not in context
                if sequence_ids[start_index] != 1 or sequence_ids[end_index] != 1:
                    continue
                
                # Skip invalid spans
                if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                    continue
                
                # Skip if offset is None
                if offset_mapping[start_index] is None or offset_mapping[end_index] is None:
                    continue
                
                # Extract answer
                start_char = offset_mapping[start_index][0]
                end_char = offset_mapping[end_index][1]
                answer_text = context[start_char:end_char]
                
                # Calculate score
                score = start_logits[start_index] + end_logits[end_index]
                
                valid_answers.append({
                    'text': answer_text,
                    'score': score,
                    'start': start_index,
                    'end': end_index
                })
        
        # Sort by score
        valid_answers.sort(key=lambda x: x['score'], reverse=True)
        
        return valid_answers
    
    def _calculate_confidence(self, score: float) -> float:
        """
        Convert logit score to confidence percentage.
        
        Uses sigmoid-like transformation.
        """
        # Normalize score (typical range is -10 to 20)
        normalized = (score + 10) / 30
        confidence = max(0.0, min(1.0, normalized)) * 100
        return confidence
    
    def predict_batch(
        self,
        questions: List[str],
        contexts: List[str],
        **kwargs
    ) -> List[Dict]:
        """
        Predict answers for multiple question-context pairs.
        
        Args:
            questions: List of questions
            contexts: List of contexts
            **kwargs: Additional arguments for predict()
            
        Returns:
            List of prediction dictionaries
        """
        results = []
        for question, context in zip(questions, contexts):
            result = self.predict(question, context, **kwargs)
            results.append(result)
        return results
    
    def highlight_answer(self, context: str, answer: str) -> str:
        """
        Return context with answer highlighted.
        
        Args:
            context: Original context
            answer: Answer to highlight
            
        Returns:
            Context with answer wrapped in ** markers
        """
        if not answer or answer not in context:
            return context
        
        # Find answer position
        start = context.find(answer)
        end = start + len(answer)
        
        # Wrap in markers
        highlighted = (
            context[:start] +
            f"**{answer}**" +
            context[end:]
        )
        
        return highlighted


def load_predictor(checkpoint_path: str = None) -> QAPredictor:
    """
    Convenience function to load predictor.
    
    Args:
        checkpoint_path: Path to model checkpoint
        
    Returns:
        QAPredictor instance
    """
    return QAPredictor(checkpoint_path)
