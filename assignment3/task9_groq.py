import os
import json
from datasets import load_dataset
from groq import Groq

groq_key = os.environ.get("GROQ_KEY")
client = Groq(api_key=groq_key)

# dataset = load_dataset("conll2003")
dataset = load_dataset("lhoestq/conll2003")
test_set = dataset["test"]

num_sentences = 200
subset = test_set.select(range(num_sentences))

def generate_prompt(sentence_text):
    return f"""You are an expert system specialized in Named Entity Recognition (NER).
Your task is to identify all named entities in the provided sentence.
The entity categories are exactly:
- PER (Person)
- LOC (Location)
- ORG (Organization)
- MISC (Miscellaneous)

Analyze the sentence and extract the entities. Return the results strictly as a valid JSON object where keys are the extracted entity texts and values are their corresponding labels (PER, LOC, ORG, or MISC). Do not include any introductory or concluding text.

Sentence: "{sentence_text}"

JSON Output:"""

results = []

model_ver = "llama-3.1-8b-instant"

print(f"Processing 200 sentences with {model_ver}...")

for index, example in enumerate(subset):
    # Reconstruct the sentence string from tokens
    sentence_str = " ".join(example["tokens"])
    # Build prompt
    prompt_content = generate_prompt(sentence_str)

    try:
        response = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt_content
                }
            ],
            model=f"{model_ver}",
            temperature=0.0, # set temp - important from lecture
            response_format={"type": "json_object"}
        )

        # Parse output
        output_text = response.choices[0].message.content
        predicted_entities = json.loads(output_text)

        results.append({
            "index": index,
            "sentence": sentence_str,
            "ground_truth_tokens": example["tokens"],
            "ground_truth_tags": example["ner_tags"],
            "predicted_entities": predicted_entities
        })

    # Capture erros
    except Exception as e:
        print(f"Error encountered at index {index}: {str(e)}")
        results.append({
            "index": index,
            "sentence": sentence_str,
            "error": str(e)
        })

# Export res so I don't have to run again
output_filename = "groq_zero_shot_ner.json"
with open(output_filename, "w", encoding="utf-8") as outfile:
    json.dump(results, outfile, indent=4, ensure_ascii=False)

print(f"Finished processing! Results have been saved to {output_filename}")
