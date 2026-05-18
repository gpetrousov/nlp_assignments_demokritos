res_file = "groq_zero_shot_ner.json"

import json
from seqeval.metrics import classification_report as seqeval_report
from sklearn.metrics import classification_report as sklearn_report

# Standard CoNLL2003 tag mapping
id2label = {
    0: "O",
    1: "B-PER",
    2: "I-PER",
    3: "B-LOC",
    4: "I-LOC",
    5: "B-ORG",
    6: "I-ORG",
    7: "B-MISC",
    8: "I-MISC"
}

# Load your existing JSON file
# Change the filename string below if your file is named differently
input_filename = "task10_results.json"

with open(input_filename, "r", encoding="utf-8") as f:
    results = json.load(f)

y_true = []
y_pred = []

print("Aligning free-form dictionary entities to token-level sequences...")

for item in results:
    # Skip any sentences where the API call failed
    if "error" in item:
        continue

    # 1. Convert ground truth integer IDs (0-8) to string tags ("O", "B-ORG", etc.)
    true_strings = [id2label[idx] for idx in item["ground_truth_tags"]]
    y_true.append(true_strings)

    # 2. Map the extracted dictionary entities back to individual tokens
    tokens = item["ground_truth_tokens"]
    pred_entities = item["predicted_entities"]

    pred_strings = []
    for token in tokens:
        # Strip trailing/leading punctuation to ensure clean lookups
        clean_token = token.strip(",.?!:;\"'")

        # Check if the token (or cleaned token) exists in the model's predictions
        entity_type = pred_entities.get(token) or pred_entities.get(clean_token)

        # Verify it's a valid label type
        if entity_type in ["PER", "LOC", "ORG", "MISC"]:
            pred_strings.append(f"B-{entity_type}")
        else:
            pred_strings.append("O")

    y_pred.append(pred_strings)

# Run the evaluation metrics
print("\n=== Groq Llama-3.1-8B — Entity-level classification report (seqeval) ===")
print(seqeval_report(y_true, y_pred, digits=3, zero_division=0))
