# Quick Start Guide

## 🚀 Get Started in 5 Minutes

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Sample Demo
```bash
python demo_samples.py
```

You'll see the system answer 5 sample questions!

### 3. Launch Web Interface
```bash
python app.py
```

Open `http://127.0.0.1:7860` in your browser.

### 4. Explore Notebooks
```bash
jupyter notebook notebooks/
```

Start with `01_data_exploration.ipynb`

---

## 📝 Common Tasks

### Train the Model
```bash
jupyter notebook notebooks/04_model_training.ipynb
```
Run all cells (takes 2-3 hours on GPU)

### Evaluate Performance
```bash
jupyter notebook notebooks/05_evaluation_analysis.ipynb
```

### Use Python API
```python
from inference.predict import QAPredictor

predictor = QAPredictor()
result = predictor.predict(
    "What is AI?",
    "AI is intelligence demonstrated by machines."
)
print(result['answer'])
```

---

## 🆘 Need Help?

- **Data Issues**: Check `03_data_validation.ipynb`
- **Training Errors**: See [Troubleshooting](README.md#troubleshooting)
- **Low Accuracy**: Verify extraction accuracy >90% in validation notebook
- **Slow Performance**: Enable FP16, use GPU, reduce batch size

---

## 📚 Learning Path

1. ✅ Run demo samples → **Understand output format**
2. ✅ Explore dataset → **See SQuAD structure**
3. ✅ Validate preprocessing → **Verify pipeline works**
4. ✅ Train model → **Get fine-tuned checkpoint**
5. ✅ Evaluate → **Analyze performance**
6. ✅ Deploy → **Launch web interface**

---

For complete documentation, see [README.md](README.md)
