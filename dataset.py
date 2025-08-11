import os
import json
import time
import re
import fitz  
from groq import Groq
from tqdm import tqdm

# --- CONFIGURATION ---
INPUT_FILE = "/Users/apple/Desktop/hume/book1.pdf"
OUTPUT_FILE = "meditations_finetuning_dataset.json"
MODEL = "llama3-70b-8192"
CHUNK_SIZE = 2500  
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

PROMPT_TEMPLATE = """
Your task is to act as a data creator. Read the following philosophical passage from Marcus Aurelius. Then, complete two tasks:

1.  **Create a Modern Problem:** Invent a realistic, modern-day problem or frustration that the passage could solve. Label it clearly with `PROBLEM:`.
2.  **Write Aurelius's Response:** Write a response to that problem in the first-person voice of Marcus Aurelius, applying the wisdom from the passage. Label it clearly with `RESPONSE:`.

Keep the response authentic and insightful. Do not mention that you are an AI.

Here is the passage:
---
{passage_text}
---

Structure your entire output like this, and only like this:
PROBLEM: [Your invented modern problem]
<--->
RESPONSE: [Your response in the voice of Marcus Aurelius]
"""

def extract_and_clean_text_from_pdf(pdf_path):
    """Extracts and cleans text from a PDF."""
    print(f"Reading and cleaning text from '{pdf_path}'...")
    try:
        doc = fitz.open(pdf_path)
        full_text = "".join(page.get_text() for page in doc)
        doc.close()
        cleaned_text = re.sub(r'\s+', ' ', full_text).strip()
        print(f"Successfully extracted and cleaned {len(cleaned_text)} characters.")
        return cleaned_text
    except Exception as e:
        print(f"Error reading PDF file: {e}")
        return None

def chunk_text(text, chunk_size):
    """Splits text into smaller chunks."""
    print(f"Splitting text into chunks of {chunk_size} characters...")
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    print(f"Created {len(chunks)} text chunks.")
    return chunks

def process_chunk_simplified(client, chunk):
    """
    Sends a simplified prompt and parses the text response.
    This is more resilient than asking for a JSON object directly.
    """
    retries = 3
    delay = 5
    for attempt in range(retries):
        try:
            # NOTE: We have removed the 'response_format' parameter here.
            chat_completion = client.chat.completions.create(
                messages=[
                    {"role": "user", "content": PROMPT_TEMPLATE.format(passage_text=chunk)}
                ],
                model=MODEL,
                temperature=0.75,
                max_tokens=2048,
            )
            response_text = chat_completion.choices[0].message.content.strip()

            if "<--->" in response_text:
                problem_part, response_part = response_text.split("<--->", 1)
                problem = problem_part.replace("PROBLEM:", "").strip()
                response = response_part.replace("RESPONSE:", "").strip()

                if problem and response:
                    return {
                        "instruction": "You are the philosopher Marcus Aurelius. Analyze the user's problem from a Stoic perspective and offer guidance in your authentic voice.",
                        "input": problem,
                        "output": response
                    }
            tqdm.write(f"\n[Warning] Failed to parse API response: {response_text[:300]}...")

        except Exception as e:
            tqdm.write(f"\n[API Error] An error occurred: {e}. Retrying ({attempt + 1}/{retries})...")
        
        time.sleep(delay)
    
    return None

def main():
    """Main function to orchestrate the entire data generation process."""
    print("--- Starting Final Dataset Generation (V3 - Robust Method) ---")
    
    if GROQ_API_KEY == "YOUR_GROQ_API_KEY_HERE" or not GROQ_API_KEY:
        print("\nFATAL ERROR: Please paste your Groq API key into the script.")
        return

    full_text = extract_and_clean_text_from_pdf(INPUT_FILE)
    if not full_text: return

    text_chunks = chunk_text(full_text, CHUNK_SIZE)
    if not text_chunks: return

    print("\n--- Connecting to Groq and Processing Chunks ---")
    client = Groq(api_key=GROQ_API_KEY)
    all_examples = []
    
    for chunk in tqdm(text_chunks, desc="Generating Examples"):
        generated_example = process_chunk_simplified(client, chunk)
        if generated_example:
            all_examples.append(generated_example)
        else:
            tqdm.write(f"Skipping a chunk after multiple failed attempts.")

    if all_examples:
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(all_examples, f, indent=2, ensure_ascii=False)
        print("\n--- Dataset Generation Complete! ---")
        print(f"Successfully generated {len(all_examples)} examples.")
        print(f"Your final dataset is saved as: {OUTPUT_FILE}")
    else:
        print("\n--- Dataset Generation Failed ---")
        print("No valid examples were generated. Please check the error messages.")

if __name__ == "__main__":
    main()