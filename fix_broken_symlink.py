# Save as: test_rag_token.py
from transformers import RagTokenizer, RagRetriever, RagTokenForGeneration
import torch

print(" Testing RAG Token Generation")
print("=" * 50)

# Load model components
tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-nq")
retriever = RagRetriever.from_pretrained(
    "facebook/rag-token-nq",
    index_name="exact",
    use_dummy_dataset=False
)
model = RagTokenForGeneration.from_pretrained(
    "facebook/rag-token-nq",
    retriever=retriever
)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

print("✅ Models loaded")

# Test questions with different templates
questions = [
    "What is the capital of France?",
    "Who wrote Romeo and Juliet?",
    "What is the largest planet?"
]

templates = {
    "basic": "{question}",
    "instructional": "Answer the following question using available information: {question}",
    "expert_role": "As an expert, provide a precise answer: {question}"
}

for question in questions:
    print(f"\n Question: {question}")

    for template_name, template in templates.items():
        prompt = template.format(question=question)

        # Use the working method from example
        input_dict = tokenizer.prepare_seq2seq_batch(
            prompt, return_tensors="pt")
        input_ids = input_dict["input_ids"].to(device)

        with torch.no_grad():
            generated = model.generate(input_ids=input_ids)

        answer = tokenizer.batch_decode(generated, skip_special_tokens=True)[0]

        print(f"  {template_name:15}: '{answer}'")

    print("-" * 50)

print("\n If this works, we'll integrate it into your main script!")
