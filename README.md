# 🧠 Philosopher LLM: Fine-Tuning Llama 3 with Stoic Wisdom

[![Python](https://img.shields.io/badge/Python-3.9+-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1-orange?style=for-the-badge&logo=pytorch)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Transformers-yellow?style=for-the-badge)](https://huggingface.co/)
[![Unsloth](https://img.shields.io/badge/⚡%20Unsloth-Optimized-green?style=for-the-badge)](https://github.com/unsloth/unsloth)
[![License: MIT](https://img.shields.io/badge/License-MIT-purple.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

An end-to-end fine-tuning project that transforms a Llama 3-8B model into a Stoic philosopher. This model, trained on Marcus Aurelius's *Meditations*, can analyze modern problems through a framework of Stoic reasoning, providing insightful, in-character guidance.

## 🚀 Demo Output

Here is an example of the model responding to a modern-day problem after being fine-tuned:

```text
--- Your Modern Problem ---
I feel so much anxiety about my career. I'm constantly worried I'm not doing enough or that I'll fall behind everyone else.

--- Marcus Aurelius's Response ---
Your anxiety stems not from your career itself, but from your judgment of it. The progress of others is external to you and not within your control. Your worry is a creation of your own mind. Discard this judgment and the feeling of injury vanishes. Focus on the task before you, perform it with diligence and virtue, and you will find that the tranquility you seek comes not from outpacing others, but from mastering your own thoughts.
✨ Project Overview
The core challenge of this project was not simply to make a Large Language Model sound like a philosopher, but to teach it to reason like one. This required moving beyond basic text generation to imbue the model with a specific, abstract framework—Stoicism.
This was achieved through two main scripts:
dataset.py: Automates the creation of a high-quality dataset by programmatically transforming the raw text of Meditations into structured instruction-response pairs using an external LLM API.
finetune.ipynb: Leverages the high-performance Unsloth library to efficiently fine-tune a Llama 3-8B model with QLoRA, runnable on a single consumer GPU.
🛠️ Tech Stack
Data Generation: Groq API (Llama 3 70B), Python, PyMuPDF
Fine-Tuning: Unsloth, PyTorch, Hugging Face (Transformers, PEFT, TRL, Datasets)
Quantization: bitsandbytes for 4-bit model loading (QLoRA)
Environment: Jupyter Notebook (.ipynb), Google Colab (or local with GPU)
📂 Project Structure
code
Code
.
├── .venv/                      # Virtual environment (ignored by .gitignore)
├── finetuned_llama.../         # Output directory for the fine-tuned model (ignored)
├── llama3_meditati.../         # Checkpoints from training (ignored)
├── meditations_corpus.pdf      # The raw input book (not committed)
├── dataset.py                  # Script to generate the training dataset
├── finetune.ipynb              # Jupyter Notebook for fine-tuning the model
├── meditations_finetune.json   # The final, structured dataset
├── .gitignore                  # Specifies files for Git to ignore
└── README.md                   # This file
⚙️ Getting Started
Follow these steps to replicate the project.
Prerequisites
Python 3.9+
A GPU (NVIDIA T4 on Google Colab is sufficient)
A Groq API key for data generation
Step 1: Create the Dataset
The first step is to generate the meditations_finetune.json file from the source PDF.
Place the PDF version of Meditations in the root directory and name it meditations_corpus.pdf.
Open dataset.py and paste your Groq API key into the GROQ_API_KEY variable.
Run the script from your terminal:
code
Bash
python dataset.py
This will create the meditations_finetune.json file, containing the ~172 training examples.
Step 2: Fine-Tune the Model
Now, you can run the fine-tuning process using the Jupyter Notebook.
If using Google Colab: Upload finetune.ipynb and the generated meditations_finetune.json to your Colab environment.
If running locally: Ensure you have a compatible GPU and have installed the necessary libraries.
Open finetune.ipynb and execute the cells in order. The notebook will:
Install Unsloth and all dependencies.
Load the base Llama 3-8B model in 4-bit precision.
Apply LoRA adapters.
Load your custom JSON dataset.
Run the training process.
Save the final model adapters to an output directory (e.g., finetuned_llama...).
Step 3: Run Inference
The final cells in finetune.ipynb demonstrate how to load your new, fine-tuned model and chat with it. You can modify the modern_problem variable to ask it for advice on any topic.
💡 Challenges & Learnings
Challenge: The external API (Groq) was unreliable when tasked with generating perfectly formatted JSON.
Learning: The solution was to simplify the AI's task. By prompting for simple, delimited text (<--->) and handling the JSON structuring in Python (dataset.py), the data pipeline became far more robust.
Challenge: Initial attempts at parsing the PDF resulted in one giant, unusable chunk of text.
Learning: This emphasized the importance of data granularity. Implementing a proper chunking strategy was crucial for providing the API with appropriately sized and meaningful contexts.
🔮 Future Work
Build an Interactive UI: Develop a simple web interface using Gradio or Streamlit to allow users to easily chat with the philosopher.
Expand the Pantheon: Fine-tune models on the works of other philosophers (e.g., Nietzsche, Plato) to create a "Council of Thinkers."
Quantitative Evaluation: Implement an evaluation framework using another powerful LLM (like GPT-4) as a "judge" to score the philosophical coherence of the model's responses.
License
This project is licensed under the MIT License.
