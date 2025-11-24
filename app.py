"""Standalone Gradio app for BERT Question Answering."""

import torch
import gradio as gr
from pathlib import Path
from inference.predict import QAPredictor

# Load model
model_path = 'checkpoints/best_model.pt'
predictor = QAPredictor(model_path if Path(model_path).exists() else None)

def answer_question(question, context, show_confidence=True):
    """Answer a question given context."""
    if not question or not context:
        return "Please provide both a question and context.", ""
    
    result = predictor.predict(question, context)
    
    if result['answer']:
        answer_text = f"**Answer:** {result['answer']}"
        if show_confidence:
            answer_text += f"\n\n**Confidence:** {result['confidence']:.1f}%"
        highlighted = predictor.highlight_answer(context, result['answer'])
    else:
        answer_text = "**Answer:** No answer found"
        highlighted = context
    
    return answer_text, highlighted

# Example contexts
examples = [
    [
        "What is the capital of France?",
        "Paris is the capital and most populous city of France. Situated on the Seine River, in the north of the country, it is in the centre of the Île-de-France region."
    ],
    [
        "Who invented the telephone?",
        "Alexander Graham Bell was awarded the first U.S. patent for the telephone in 1876. The invention of the telephone is the culmination of work done by many individuals."
    ],
    [
        "When was the United Nations founded?",
        "The United Nations was established after World War II with the aim of preventing future wars. On 25 April 1945, 50 governments met in San Francisco for a conference and started drafting the UN Charter, which was adopted on 25 June 1945 and took effect on 24 October 1945."
    ],
    [
        "What is photosynthesis?",
        "Photosynthesis is a process used by plants and other organisms to convert light energy into chemical energy that can later be released to fuel the organisms' activities. This chemical energy is stored in carbohydrate molecules, such as sugars."
    ],
    [
        "How many people live in Tokyo?",
        "Tokyo is the capital of Japan and the most populous metropolitan area in the world. The Greater Tokyo Area has a population of approximately 37.4 million people, making it the largest metropolitan area in the world."
    ]
]

# Create interface
demo = gr.Interface(
    fn=answer_question,
    inputs=[
        gr.Textbox(label="Question", placeholder="Enter your question...", lines=2),
        gr.Textbox(label="Context", placeholder="Paste context here...", lines=8),
        gr.Checkbox(label="Show Confidence Score", value=True)
    ],
    outputs=[
        gr.Markdown(label="Answer"),
        gr.Textbox(label="Context with Highlighted Answer", lines=8)
    ],
    examples=examples,
    title="🤖 BERT Question Answering System",
    description="Ask questions about any text! This system uses BERT to extract answers from the provided context. Try the examples below or enter your own question and context.",
    theme=gr.themes.Soft(),
    allow_flagging="never"
)

if __name__ == "__main__":
    print("🚀 Starting BERT Question Answering System...")
    print("📍 Access the interface at: http://127.0.0.1:7860")
    demo.launch(share=False, server_port=7860)
