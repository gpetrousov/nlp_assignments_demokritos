res_file = "groq_zero_shot_ner.json"

import json
from seqeval.metrics import classification_report as seqeval_report
from sklearn.metrics import classification_report as sklearn_report

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

    true_strings = [id2label[idx] for idx in item["ground_truth_tags"]]
    y_true.append(true_strings)

    tokens = item["ground_truth_tokens"]
    pred_entities = item["predicted_entities"]

    pred_strings = []
    for token in tokens:
        clean_token = token.strip(",.?!:;\"'")

        entity_type = pred_entities.get(token) or pred_entities.get(clean_token)

        if entity_type in ["PER", "LOC", "ORG", "MISC"]:
            pred_strings.append(f"B-{entity_type}")
        else:
            pred_strings.append("O")

    y_pred.append(pred_strings)

# Fire!
print("\n=== Groq Llama-3.1-8B — Entity-level classification report (seqeval) ===")
print(seqeval_report(y_true, y_pred, digits=3, zero_division=0))
