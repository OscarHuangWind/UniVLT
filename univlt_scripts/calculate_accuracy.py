import json

def normalize_text(text):
    """Normalize text for comparison by removing extra spaces and making lowercase"""
    if text is None:
        return ""
    return " ".join(text.lower().split())

def calculate_accuracy(json_file):
    """Calculate accuracy by comparing answer and answer_gt fields"""
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    total_samples = len(data)
    correct = 0
    
    print(f"\nAnalyzing {total_samples} samples...")
    print("-" * 50)
    
    for idx, entry in enumerate(data):
        answer = normalize_text(entry.get("answer", ""))
        answer_gt = normalize_text(entry.get("answer_gt", "")[0])
        
        is_correct = answer == answer_gt
        if is_correct:
            correct += 1
        
        # Print comparison for mismatches (helpful for debugging)
        # if not is_correct:
        #     print(f"\nMismatch in sample {idx + 1}:")
        #     print(f"Answer:     {entry.get('answer', '')}")
        #     print(f"Ground Truth: {entry.get('answer_gt', '')}")
    
    accuracy = (correct / total_samples) * 100 if total_samples > 0 else 0
    
    print("\n" + "=" * 50)
    print(f"Results Summary:")
    print(f"Total samples: {total_samples}")
    print(f"Correct matches: {correct}")
    print(f"Accuracy: {accuracy:.2f}%")
    print("=" * 50)
    
    return accuracy

if __name__ == "__main__":
    json_file = "example.json"
    calculate_accuracy(json_file)
