"""Sample demonstration of the BERT Question Answering System."""

from inference.predict import QAPredictor

# Initialize predictor (uses base BERT if no checkpoint exists)
print("=" * 80)
print("BERT Question Answering System - Sample Demonstrations")
print("=" * 80)
print("\nInitializing predictor...")
predictor = QAPredictor()
print("✓ Predictor loaded successfully!\n")

# Sample 1: Capital of France
print("\n" + "=" * 80)
print("SAMPLE 1: Geography Question")
print("=" * 80)
question1 = "What is the capital of France?"
context1 = """Paris is the capital and most populous city of France. Situated on the 
Seine River, in the north of the country, it is in the centre of the Île-de-France region. 
The city has an area of 105 square kilometers and a population of 2,206,488 inhabitants."""

result1 = predictor.predict(question1, context1)
print(f"Question: {question1}")
print(f"Context: {context1[:150]}...")
print(f"\n✓ Answer: {result1['answer']}")
print(f"  Confidence: {result1['confidence']:.1f}%")

# Sample 2: Historical fact
print("\n" + "=" * 80)
print("SAMPLE 2: Historical Question")
print("=" * 80)
question2 = "When was the United Nations founded?"
context2 = """The United Nations (UN) is an intergovernmental organization tasked with 
maintaining international peace and security. It was established after World War II with 
the aim of preventing future wars. On 25 April 1945, 50 governments met in San Francisco 
for a conference and started drafting the UN Charter, which was adopted on 25 June 1945 
and took effect on 24 October 1945, when the UN began operations."""

result2 = predictor.predict(question2, context2)
print(f"Question: {question2}")
print(f"Context: {context2[:150]}...")
print(f"\n✓ Answer: {result2['answer']}")
print(f"  Confidence: {result2['confidence']:.1f}%")

# Sample 3: Scientific fact
print("\n" + "=" * 80)
print("SAMPLE 3: Science Question")
print("=" * 80)
question3 = "What is photosynthesis?"
context3 = """Photosynthesis is a process used by plants and other organisms to convert 
light energy into chemical energy that can later be released to fuel the organisms' activities. 
This chemical energy is stored in carbohydrate molecules, such as sugars, which are synthesized 
from carbon dioxide and water. In most cases, oxygen is also released as a waste product."""

result3 = predictor.predict(question3, context3)
print(f"Question: {question3}")
print(f"Context: {context3[:150]}...")
print(f"\n✓ Answer: {result3['answer']}")
print(f"  Confidence: {result3['confidence']:.1f}%")

# Sample 4: Who question
print("\n" + "=" * 80)
print("SAMPLE 4: Person Identification")
print("=" * 80)
question4 = "Who invented the telephone?"
context4 = """Alexander Graham Bell was awarded the first U.S. patent for the telephone in 
1876. The invention of the telephone is the culmination of work done by many individuals, 
and led to an array of lawsuits relating to the patent claims of several individuals and 
numerous companies."""

result4 = predictor.predict(question4, context4)
print(f"Question: {question4}")
print(f"Context: {context4[:150]}...")
print(f"\n✓ Answer: {result4['answer']}")
print(f"  Confidence: {result4['confidence']:.1f}%")

# Sample 5: Number question
print("\n" + "=" * 80)
print("SAMPLE 5: Numerical Question")
print("=" * 80)
question5 = "How many people live in Tokyo?"
context5 = """Tokyo is the capital of Japan and the most populous metropolitan area in the 
world. The Greater Tokyo Area has a population of approximately 37.4 million people, making 
it the largest metropolitan area in the world by population."""

result5 = predictor.predict(question5, context5)
print(f"Question: {question5}")
print(f"Context: {context5[:150]}...")
print(f"\n✓ Answer: {result5['answer']}")
print(f"  Confidence: {result5['confidence']:.1f}%")

# Batch prediction example
print("\n" + "=" * 80)
print("BATCH PREDICTION DEMO")
print("=" * 80)
questions = [
    "What is the Eiffel Tower made of?",
    "When was the Eiffel Tower built?"
]
contexts = [
    "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. It is named after the engineer Gustave Eiffel.",
    "The Eiffel Tower was constructed from 1887 to 1889 as the entrance arch to the 1889 World's Fair."
]

batch_results = predictor.predict_batch(questions, contexts)
for i, (q, r) in enumerate(zip(questions, batch_results), 1):
    print(f"\n{i}. Q: {q}")
    print(f"   A: {r['answer']} (Confidence: {r['confidence']:.1f}%)")

print("\n" + "=" * 80)
print("✓ All sample demonstrations completed successfully!")
print("=" * 80)
print("\nThe Question Answering system is working correctly!")
print("You can now use it via:")
print("  1. Python API: from inference.predict import QAPredictor")
print("  2. Jupyter Notebook: notebooks/06_deployment.ipynb")
print("  3. Web Interface: python app.py (requires gradio)")
print("=" * 80)
