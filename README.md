# 🧠 Philosopher LLM: Fine-Tuning Llama 3 with Stoic Wisdom

[![Python](https://img.shields.io/badge/Python-3.9+-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1-orange?style=for-the-badge&logo=pytorch)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Transformers-yellow?style=for-the-badge)](https://huggingface.co/)
[![Unsloth](https://img.shields.io/badge/⚡%20Unsloth-Optimized-green?style=for-the-badge)](https://github.com/unsloth/unsloth)
[![License: MIT](https://img.shields.io/badge/License-MIT-purple.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

An end-to-end fine-tuning project that transforms a Llama 3-8B model into a Stoic philosopher. This model, trained on Marcus Aurelius's *Meditations*, can analyze modern problems through a framework of Stoic reasoning, providing insightful, in-character guidance.
✨ Project Overview
The core challenge of this project was not simply to make a Large Language Model sound like a philosopher, but to teach it to reason like one. This required moving beyond basic text generation to imbue the model with a specific, abstract framework—Stoicism.
This was achieved through three key stages:
Automated Dataset Creation: Programmatically transforming the raw, unstructured text of Meditations into a high-quality, structured dataset for instruction fine-tuning.
Optimized Fine-Tuning: Using the high-performance Unsloth library to efficiently fine-tune a Llama 3-8B model with QLoRA on a single consumer GPU.
Qualitative Evaluation: Assessing the model's ability to provide coherent, philosophically-sound advice on novel problems.
🛠️ Tech Stack & Architecture
This project leverages a modern, efficient stack for LLM fine-tuning.
Data Generation: Groq API (Llama 3 70B), Python, PyMuPDF (for PDF parsing)
Fine-Tuning: Unsloth, PyTorch, Hugging Face (Transformers, PEFT, TRL, Datasets)
Quantization: bitsandbytes for 4-bit model loading (QLoRA)
Compute: Google Colab (NVIDIA T4 GPU)
Architectural Flow
code
Code
1. PDF Ingestion          2. Automated Dataset Creation       3. Optimized Fine-Tuning
┌───────────────────┐     ┌───────────────────────────────┐     ┌────────────────────────┐
│ Meditations.pdf   ├─────► Python Script (PyMuPDF, Regex)├─────► meditations_dataset.json │
└───────────────────┘     │        ▲                      │     └────────────┬───────────┘
                          │        │ Groq API Call        │                  │
                          │        ▼                      │                  ▼
                          │   (Llama 3 70B)               │     ┌────────────────────────┐
                          └───────────────────────────────┘     │ Unsloth & PEFT (QLoRA) ├─────► Final Model
                                                                └────────────────────────┘
📝 Project Stages
1. Data Ingestion & Automated Dataset Creation
The foundation of any good model is its data. The raw text of Meditations was extracted from a PDF, cleaned using regex to remove artifacts, and split into manageable chunks.
To create the instruction-following dataset, a robust two-step prompting strategy was employed:
Each text chunk was sent to the Groq API (Llama 3 70B) with a prompt instructing it to generate a modern problem and a corresponding Stoic response, separated by a unique delimiter (<--->).
The Python script then reliably parsed this plain text response, handling the final JSON structuring itself. This method proved far more resilient than asking the API to generate perfect JSON directly.
2. Optimized Fine-Tuning with Unsloth
Training was performed using the Unsloth library to maximize efficiency.
Model: unsloth/llama-3-8b-bnb-4bit, a version of Llama 3-8B pre-quantized to 4-bits.
Technique: QLoRA (Quantized Low-Rank Adaptation) was applied. This involves "freezing" the 8 billion parameters of the base model and only training a small set of adapter layers.
Benefits: This approach resulted in a ~2.5x training speedup and ~50% less GPU memory usage, making it possible to fine-tune this powerful model on a free Google Colab instance.
🚀 How to Use the Model
To run inference with the fine-tuned model, you can use the following Python script. Ensure you have Unsloth and its dependencies installed.
code
Python
import torch
from unsloth import FastLanguageModel
from transformers import pipeline

# Load your fine-tuned model
# NOTE: Replace "outputs" with the path where your final model adapters are saved.
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "outputs", # YOUR MODEL HERE
    max_seq_length = 2048,
    dtype = None,
    load_in_4bit = True,
)

# Define the prompt format (must be identical to training)
prompt_format = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

# --- Chat with the Philosopher ---
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

modern_problem = "I got into a petty argument with a friend and I can't let it go."

prompt = prompt_format.format(
    "You are the philosopher Marcus Aurelius. Analyze the user's problem from a Stoic perspective and offer guidance in your authentic voice.",
    modern_problem,
    "" # Empty for generation
)

outputs = pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.7)
print(outputs['generated_text'].split("### Response:").strip())```

## 💡 Challenges & Learnings

*   **Challenge:** The external API (Groq) was unreliable when tasked with generating perfectly formatted JSON, often returning malformed responses that broke the data pipeline.
*   **Learning:** The solution was to simplify the AI's task. By prompting for simple, delimited text and handling the JSON structuring in Python, I created a more robust and fault-tolerant system. This highlights the principle of assigning tasks to the components best suited for them.

*   **Challenge:** The initial PDF parsing treated the entire book as one large, unusable chunk of text.
*   **Learning:** This emphasized the importance of data granularity. Implementing a proper chunking strategy before processing was crucial for providing the API with appropriately sized and meaningful contexts.

## 🔮 Future Work

*   **Build an Interactive UI:** Develop a simple web interface using Gradio or Streamlit to allow users to easily chat with the philosopher.
*   **Expand the Pantheon:** Fine-tune models on the works of other philosophers (e.g., Nietzsche, Plato) to create a "Council of Thinkers."
*   **Quantitative Evaluation:** Implement an evaluation framework using another powerful LLM (like GPT-4) as a "judge" to score the philosophical coherence of the model's responses.

## License

This project is licensed under the [MIT License](LICENSE).
