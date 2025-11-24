<div align="center">

# 🤖 BERT Question Answering System

### *Intelligent Question Answering using Transformer Models on SQuAD Dataset*

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/🤗_Transformers-4.30+-yellow.svg)](https://huggingface.co/transformers/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**A production-ready Question Answering system built with BERT and fine-tuned on SQuAD v1.1**

[Demo](#-demo) • [Features](#-features) • [Installation](#-installation) • [Usage](#-usage) • [Documentation](#-documentation)

---

</div>

## 📖 Overview

This project implements a state-of-the-art **extractive question answering system** using BERT (Bidirectional Encoder Representations from Transformers). The system can understand natural language questions and extract precise answers from provided text passages.

### 🎯 Key Capabilities

- ✅ **Answer Questions** from any text passage with high accuracy
- 🎯 **Extract Precise Spans** - finds exact answer locations in context
- 📊 **Confidence Scoring** - provides reliability metrics for predictions
- 📈 **Performance Metrics** - evaluates using Exact Match (EM) and F1 scores
- 🌐 **Interactive Web Interface** - user-friendly Gradio-based demo
- 🔄 **Batch Processing** - handle multiple questions efficiently

### 📊 Technical Specifications

| Component                     | Details                             |
| ----------------------------- | ----------------------------------- |
| **Model**               | BERT-base-uncased (110M parameters) |
| **Dataset**             | SQuAD v1.1                          |
| **Training Examples**   | 87,599 questions                    |
| **Validation Examples** | 10,570 questions                    |
| **Task Type**           | Extractive Question Answering       |

### 🏆 Performance Metrics

After fine-tuning on SQuAD v1.1:

| Metric                               | Score   |
| ------------------------------------ | ------- |
| **Exact Match (EM)**           | ~82-85% |
| **F1 Score**                   | ~88-92% |
| **Answer Extraction Accuracy** | 86%+    |

---

## 🎬 Demo

<div align="center">

### Interactive Web Interface
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/e3b3f4ba-a413-4355-9f32-906d9eb4138a" />


<img width="1919" height="1018" alt="image" src="https://github.com/user-attachments/assets/b34e22f9-6ef5-46ee-96b3-14b654013caf" />



*Live demo showing the BERT QA system answering "What is the boiling point of water at sea level?"

**Try it yourself:** Run `python app.py` or open `notebooks/06_deployment.ipynb`

</div>

---

## ✨ Features

### 🧠 **Smart Question Understanding**

- Handles diverse question types: What, Who, When, Where, How, Why
- Processes questions of varying complexity
- Understands context-dependent queries

### 🎯 **Accurate Answer Extraction**

- Extracts exact answer spans from text
- Handles long documents with sliding window approach
- Maps character positions to token indices with 86%+ accuracy

### 📊 **Comprehensive Evaluation**

- **Exact Match (EM):** Binary metric for exact answer matching
- **F1 Score:** Token-level overlap measurement
- Performance breakdown by question type
- Detailed error analysis and categorization

### 🚀 **Multiple Deployment Options**

1. **Interactive Web UI** - Gradio-based interface
2. **Python API** - Programmatic access
3. **Jupyter Notebooks** - Educational walkthrough
4. **Batch Processing** - Handle multiple questions

### 🔧 **Production-Ready Features**

- Mixed precision training (FP16) for efficiency
- Gradient accumulation support
- Early stopping to prevent overfitting
- Comprehensive logging with TensorBoard
- Model checkpointing and versioning

---

## 📁 Project Structure

```
📦 assignment/
│
├── 📂 data/                          # Data Processing Pipeline
│   ├── dataset.py                    # SQuAD dataset loader
│   ├── preprocessing.py              # Tokenization & metrics
│   └── dataloader.py                 # PyTorch DataLoader utilities
│
├── 📂 training/                      # Training Infrastructure
│   ├── train.py                      # Training loop with checkpointing
│   └── evaluate.py                   # Evaluation with EM/F1 metrics
│
├── 📂 inference/                     # Deployment Modules
│   └── predict.py                    # QAPredictor class for inference
│
├── 📂 notebooks/                     # 📓 Jupyter Notebooks
│   ├── 00_project_overview.ipynb     # Project introduction
│   ├── 01_data_exploration.ipynb     # 📊 Dataset analysis & visualization
│   ├── 02_tokenizer_testing.ipynb    # 🔤 BERT tokenization tutorial
│   ├── 03_data_validation.ipynb      # ✅ Preprocessing validation
│   ├── 04_model_training.ipynb       # 🏋️ Training workflow
│   ├── 05_evaluation_analysis.ipynb  # 📈 Performance evaluation
│   └── 06_deployment.ipynb           # 🚀 Interactive Gradio demo
│
├── 📂 archive/                       # Dataset Files
│   ├── train-v1.1.json              # 87,599 training questions
│   └── dev-v1.1.json                # 10,570 validation questions
│
├── 📂 assets/                        # Images & Resources
│   └── demo_screenshot.png          # Demo interface screenshot
│
├── 📂 checkpoints/                   # Model Checkpoints (created during training)
├── 📂 logs/                          # Training Logs (TensorBoard)
├── 📂 outputs/                       # Evaluation Results
│
├── 📄 config.py                      # Configuration management
├── 📄 requirements.txt               # Python dependencies
├── 📄 app.py                         # Standalone Gradio app
├── 📄 demo_samples.py                # Sample demonstrations
└── 📄 README.md                      # This file
```

---

## 🚀 Installation

### Prerequisites

- Python 3.10 or higher
- pip package manager
- 8GB+ RAM (16GB recommended for training)
- GPU with CUDA support (optional, but recommended for training)

### Step-by-Step Setup

1️⃣ **Clone the Repository**

```bash
git clone <repository-url>
cd assignment
```

2️⃣ **Create Virtual Environment** (Recommended)

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/Mac
source .venv/bin/activate
```

3️⃣ **Install Dependencies**

```bash
pip install -r requirements.txt
```

<details>
<summary><b>📦 Key Dependencies</b></summary>

| Package          | Version  | Purpose                   |
| ---------------- | -------- | ------------------------- |
| `transformers` | ≥4.30.0 | BERT model & tokenizer    |
| `torch`        | ≥2.0.0  | Deep learning framework   |
| `datasets`     | ≥2.14.0 | Dataset loading utilities |
| `gradio`       | ≥3.40.0 | Web interface             |
| `evaluate`     | ≥0.4.0  | Evaluation metrics        |
| `numpy`        | ≥1.24.0 | Numerical computing       |
| `pandas`       | ≥2.0.0  | Data manipulation         |
| `matplotlib`   | ≥3.7.0  | Visualization             |
| `seaborn`      | ≥0.12.0 | Statistical plots         |
| `tensorboard`  | ≥2.13.0 | Training monitoring       |

</details>

4️⃣ **Verify Installation**

```bash
python demo_samples.py
```

You should see sample predictions with confidence scores!

---

## 💻 Usage

### Quick Start - Run Demo

**Option 1: Web Interface (Recommended)**

```bash
python app.py
```

Opens interactive demo at `http://127.0.0.1:7860`

**Option 2: Jupyter Notebook**

```bash
jupyter notebook notebooks/06_deployment.ipynb
```

Run all cells to launch embedded Gradio interface

**Option 3: Python API**

```python
from inference.predict import QAPredictor

# Initialize predictor
predictor = QAPredictor('checkpoints/best_model.pt')

# Ask a question
result = predictor.predict(
    question="What is the capital of France?",
    context="Paris is the capital and most populous city of France."
)

print(f"Answer: {result['answer']}")           # "Paris"
print(f"Confidence: {result['confidence']:.1f}%")  # 95.2%
```

### 📚 Complete Workflow

<details>
<summary><b>1️⃣ Data Exploration</b></summary>

```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```

**What you'll see:**

- 📊 Dataset statistics (87K training, 10K dev questions)
- 📈 Answer length distribution
- 🔍 Question type analysis
- 💡 Sample question-answer pairs

**Key Insights:**

- Average answer length: ~20 characters
- Most common questions: "What" (43%), "Who" (9%), "How" (9%)
- Context length: ~750 characters on average

</details>

<details>
<summary><b>2️⃣ Data Validation</b></summary>

```bash
jupyter notebook notebooks/03_data_validation.ipynb
```

**Validation Steps:**

- ✅ Tokenization correctness
- ✅ Answer span extraction (86% accuracy)
- ✅ DataLoader batching
- ✅ Edge case handling

**Output:** Confirms preprocessing pipeline works correctly

</details>

<details>
<summary><b>3️⃣ Model Training</b></summary>

```bash
jupyter notebook notebooks/04_model_training.ipynb
```

**Training Configuration:**

```python
epochs: 3
batch_size: 16
learning_rate: 3e-5
optimizer: AdamW
warmup_ratio: 0.1
max_length: 384 tokens
stride: 128 tokens
```

**Training Time:** ~2-3 hours on GPU (NVIDIA RTX 3060+)

**Output:** `checkpoints/best_model.pt` (saved automatically)

</details>

<details>
<summary><b>4️⃣ Evaluation & Analysis</b></summary>

```bash
jupyter notebook notebooks/05_evaluation_analysis.ipynb
```

**Evaluation Metrics:**

- Exact Match (EM): 82-85%
- F1 Score: 88-92%
- Performance by question type
- Error analysis with F1 distribution

**Output:** Detailed evaluation report in `outputs/` folder

</details>

<details>
<summary><b>5️⃣ Deployment</b></summary>

```bash
# Web Interface
python app.py

# Or use notebook
jupyter notebook notebooks/06_deployment.ipynb
```

**Features:**

- 🎨 Interactive Gradio UI
- 📝 5 pre-loaded examples
- 💯 Confidence score display
- 🔍 Answer highlighting in context

</details>

---

## 🧠 How It Works

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Question + Context                        │
│          "What is AI?" + "AI is intelligence..."            │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
        ┌────────────────────────┐
        │   BERT Tokenizer       │
        │  [CLS] Q [SEP] C [SEP] │
        └────────┬───────────────┘
                 │
                 ▼
        ┌────────────────────────┐
        │   BERT Encoder         │
        │   (12 layers, 110M     │
        │    parameters)         │
        └────────┬───────────────┘
                 │
                 ▼
        ┌────────────────────────┐
        │   QA Head              │
        │   Start/End Logits     │
        └────────┬───────────────┘
                 │
                 ▼
        ┌────────────────────────┐
        │   Answer Extraction    │
        │   "intelligence        │
        │   demonstrated..."     │
        └────────────────────────┘
```

### Key Components

**1. Input Processing**

- Concatenates question and context: `[CLS] question [SEP] context [SEP]`
- Tokenizes using BERT WordPiece tokenizer
- Maximum length: 384 tokens
- Sliding window with stride 128 for long contexts

**2. BERT Encoding**

- 12 transformer layers
- 768 hidden dimensions
- 12 attention heads
- Contextual embeddings for each token

**3. Answer Prediction**

- Two linear layers predict start/end positions
- Softmax over all token positions
- Highest scoring span selected as answer

**4. Post-Processing**

- Convert token positions to character positions
- Extract answer text from original context
- Calculate confidence score via sigmoid normalization

### Training Strategy

```python
# Optimization
optimizer: AdamW
learning_rate: 3e-5 with linear warmup (10%)
batch_size: 16 (with gradient accumulation)
mixed_precision: FP16 for efficiency

# Regularization  
max_grad_norm: 1.0 (gradient clipping)
early_stopping: patience=2 epochs
dropout: 0.1 (BERT default)

# Data Processing
max_length: 384 tokens
doc_stride: 128 tokens (overlap for long contexts)
```

---

## 📚 Documentation

### Jupyter Notebooks Guide

| Notebook                         | Purpose                    | Runtime | Key Outputs                  |
| -------------------------------- | -------------------------- | ------- | ---------------------------- |
| **00_project_overview**    | Project introduction       | 2 min   | Overview & objectives        |
| **01_data_exploration**    | Dataset analysis           | 5 min   | 4 visualizations + stats     |
| **02_tokenizer_testing**   | BERT tokenization tutorial | 5 min   | Token examples               |
| **03_data_validation**     | Preprocessing validation   | 10 min  | 86% extraction accuracy      |
| **04_model_training**      | Train BERT-QA model        | 2-3 hrs | `best_model.pt` checkpoint |
| **05_evaluation_analysis** | Performance evaluation     | 15 min  | EM/F1 scores + analysis      |
| **06_deployment**          | Interactive demo           | 5 min   | Gradio web interface         |

### Python Modules API

<details>
<summary><b>📦 data/dataset.py</b></summary>

```python
from data.dataset import SQuADDataset

# Load dataset with tokenization
dataset = SQuADDataset(
    json_path='archive/train-v1.1.json',
    tokenizer=tokenizer,
    max_length=384,
    stride=128
)

# Get statistics
stats = dataset.get_statistics()
print(f"Questions: {stats['num_examples']}")
```

**Methods:**

- `__init__(json_path, tokenizer, max_length, stride)`
- `__getitem__(idx)` - Get tokenized example
- `get_statistics()` - Dataset statistics

</details>

<details>
<summary><b>🎯 inference/predict.py</b></summary>

```python
from inference.predict import QAPredictor

predictor = QAPredictor('checkpoints/best_model.pt')

# Single prediction
result = predictor.predict(question, context)
# Returns: {'answer': str, 'confidence': float, 'score': float}

# Batch prediction
results = predictor.predict_batch(questions, contexts)

# Highlight answer
highlighted = predictor.highlight_answer(context, answer)
```

**Methods:**

- `predict(question, context)` - Single question
- `predict_batch(questions, contexts)` - Multiple questions
- `highlight_answer(context, answer)` - Format with highlighting

</details>

<details>
<summary><b>🏋️ training/train.py</b></summary>

```python
from training.train import BertQATrainer
from config import Config

config = Config()
trainer = BertQATrainer(config)

# Train model
trainer.setup()
trainer.train()
```

**Methods:**

- `setup()` - Initialize model, data, optimizer
- `train()` - Main training loop
- `save_checkpoint(path)` - Save model state
- `load_checkpoint(path)` - Load model state

</details>

<details>
<summary><b>📊 training/evaluate.py</b></summary>

```python
from training.evaluate import BertQAEvaluator

evaluator = BertQAEvaluator('checkpoints/best_model.pt')

# Evaluate on dataset
results = evaluator.evaluate_dataset('archive/dev-v1.1.json')

print(f"EM: {results['exact_match']:.2f}%")
print(f"F1: {results['f1']:.2f}%")
```

**Methods:**

- `evaluate_dataset(json_path)` - Full evaluation
- `predict(question, context)` - Single prediction
- `analyze_errors(predictions, references)` - Error analysis

</details>

### Configuration

Edit `config.py` to customize training:

```python
@dataclass
class TrainingConfig:
    epochs: int = 3
    batch_size: int = 16
    learning_rate: float = 3e-5
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    fp16: bool = True
    gradient_accumulation_steps: int = 1
    early_stopping_patience: int = 2

@dataclass
class ModelConfig:
    model_name: str = 'bert-base-uncased'
    max_length: int = 384
    stride: int = 128
```

---

## 🛠️ Advanced Usage

### Hyperparameter Tuning

Recommended hyperparameters to experiment with:

| Parameter     | Options                   | Impact            |
| ------------- | ------------------------- | ----------------- |
| Learning Rate | 2e-5,**3e-5**, 5e-5 | Convergence speed |
| Batch Size    | 8,**16**, 32        | Memory vs speed   |
| Epochs        | 2,**3**, 4          | Training time     |
| Max Length    | 256,**384**, 512    | Context coverage  |
| Stride        | 64,**128**, 192     | Long doc handling |

**Bold** = Default/Recommended

### Custom Dataset

To train on your own data, format it as SQuAD JSON:

```json
{
  "data": [{
    "title": "Your_Topic",
    "paragraphs": [{
      "context": "Your context paragraph...",
      "qas": [{
        "question": "Your question?",
        "id": "unique_id",
        "answers": [{
          "text": "answer span",
          "answer_start": 123
        }]
      }]
    }]
  }]
}
```

Then:

```python
from data.dataset import SQuADDataset
dataset = SQuADDataset('path/to/custom_data.json', tokenizer)
```

### Monitoring Training

**TensorBoard Integration**

```bash
tensorboard --logdir=logs
```

Access at `http://localhost:6006` to view:

- 📉 Training & validation loss curves
- 📊 Learning rate schedule
- 🎯 Gradient norms
- ⏱️ Training speed metrics

### Batch Processing Example

```python
from inference.predict import QAPredictor
import pandas as pd

predictor = QAPredictor('checkpoints/best_model.pt')

# Load questions from CSV
df = pd.read_csv('questions.csv')

# Process in batches
results = predictor.predict_batch(
    questions=df['question'].tolist(),
    contexts=df['context'].tolist()
)

# Save results
df['answer'] = [r['answer'] for r in results]
df['confidence'] = [r['confidence'] for r in results]
df.to_csv('results.csv', index=False)
```

---

## 🐛 Troubleshooting

<details>
<summary><b>❌ Out of Memory (OOM) Errors</b></summary>

**Solutions:**

```python
# 1. Reduce batch size
TrainingConfig(batch_size=8)

# 2. Enable gradient accumulation
TrainingConfig(gradient_accumulation_steps=2)

# 3. Reduce max length
ModelConfig(max_length=256)

# 4. Disable FP16 on incompatible GPUs
TrainingConfig(fp16=False)
```

</details>

<details>
<summary><b>📉 Low Accuracy</b></summary>

**Checklist:**

1. ✅ Verify preprocessing in `03_data_validation.ipynb`
2. ✅ Check answer extraction accuracy >90%
3. ✅ Train for 3-4 epochs (not just 1)
4. ✅ Try different learning rates (2e-5, 5e-5)
5. ✅ Ensure using full training set (87K examples)

</details>

<details>
<summary><b>⏱️ Slow Training</b></summary>

**Optimizations:**

- ✅ Enable mixed precision: `TrainingConfig(fp16=True)`
- ✅ Use GPU if available
- ✅ Increase batch size (if memory allows)
- ✅ Reduce max_length to 256
- ✅ Use fewer warmup steps

**Expected Times:**

- CPU: ~12-15 hours for 3 epochs
- GPU (RTX 3060): ~2-3 hours
- GPU (RTX 4090): ~1 hour

</details>

<details>
<summary><b>🔧 Model Not Found Error</b></summary>

```
FileNotFoundError: checkpoints/best_model.pt not found
```

**Solution:**
Train the model first:

```bash
jupyter notebook notebooks/04_model_training.ipynb
```

Or use base BERT (not fine-tuned):

```python
predictor = QAPredictor()  # No checkpoint path
```

</details>

<details>
<summary><b>📦 Package Installation Issues</b></summary>

**For Windows:**

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

**For Linux/Mac:**

```bash
pip install torch torchvision torchaudio
pip install -r requirements.txt
```

**For Apple Silicon (M1/M2):**

```bash
pip install torch torchvision torchaudio
pip install -r requirements.txt
# Training will use MPS backend automatically
```

</details>

---

## 📖 References & Resources

### 📄 Research Papers

1. **BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding**

   - Devlin et al., 2019
   - [arXiv:1810.04805](https://arxiv.org/abs/1810.04805)
   - Foundation model for this project
2. **SQuAD: 100,000+ Questions for Machine Comprehension of Text**

   - Rajpurkar et al., 2016
   - [arXiv:1606.05250](https://arxiv.org/abs/1606.05250)
   - Dataset source

### 🔗 Useful Links

- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers)
- [SQuAD Leaderboard](https://rajpurkar.github.io/SQuAD-explorer/)
- [BERT Fine-tuning Tutorial](https://huggingface.co/transformers/task_summary.html#question-answering)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)

### 📚 Recommended Reading

- [The Illustrated BERT](http://jalammar.github.io/illustrated-bert/)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (Transformer paper)
- [Question Answering with Transformers](https://huggingface.co/course/chapter7/7)

---

## 🤝 Contributing & Future Improvements

### Potential Enhancements

- [ ] Add support for **SQuAD v2.0** (unanswerable questions)
- [ ] Implement other models (**RoBERTa**, **ALBERT**, **DistilBERT**)
- [ ] Create **FastAPI REST API** for production deployment
- [ ] Add **Docker containerization**
- [ ] Implement **answer re-ranking** strategies
- [ ] Add **multi-language support**
- [ ] Create **Streamlit alternative** interface
- [ ] Add **confidence calibration** techniques
- [ ] Implement **ensemble methods**
- [ ] Add **explainability** features (attention visualization)

### Project Statistics

```
📊 Total Lines of Code: ~3,000+
📁 Files: 25+
📓 Notebooks: 7
🧪 Test Accuracy: 86%
⏱️ Training Time: 2-3 hours
💾 Model Size: ~440MB
🎯 F1 Score: 88-92%
```

---

## 📄 License

This project is created for educational purposes as part of an NLP assignment. The SQuAD dataset is licensed under **CC BY-SA 4.0**. BERT models are licensed under **Apache 2.0**.

---

## 🙏 Acknowledgments

- **Stanford NLP Group** - For the SQuAD dataset
- **Google Research** - For BERT architecture
- **Hugging Face** - For Transformers library
- **PyTorch Team** - For the deep learning framework
- **Gradio Team** - For the web interface framework

---

<div align="center">

### 🎓 Academic Project

**Course:** Natural Language Processing
**Institution:** IIIT Nagpur
**Author:** Nachiket
**Date:** November 2025

---

**⭐ If you find this project helpful, please consider giving it a star!**

[![GitHub stars](https://img.shields.io/github/stars/yourusername/bert-qa-squad?style=social)](https://github.com/yourusername/bert-qa-squad)

</div>

---

## 📞 Contact & Support

For questions, issues, or collaboration:

- 📧 Email: nachiketdoddamani@gmail.com
- 💬 GitHub Issues: [Create an issue](https://github.com/yourusername/bert-qa-squad/issues)
- 📚 Documentation: [Wiki](https://github.com/yourusername/bert-qa-squad/wiki)

---

<div align="center">

**Made with ❤️ and 🤖 using BERT**

*Last Updated: November 24, 2025*

</div>
