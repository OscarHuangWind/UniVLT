import json
import ast
import re
import numpy as np
import pandas as pd
import collections
import json, ast, re

def parse_bboxes(entry, default_label):
    bboxes = []
    if entry is None:
        return bboxes

    raw = entry.strip()

    # 1) Strip ```json ... ``` fences if present
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z]*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw).strip()

    # 2) Try to parse as JSON / python literal (handles dict, list, list-of-dicts, list-of-4)
    obj = None
    for parser in (json.loads, ast.literal_eval):
        try:
            obj = parser(raw)
            break
        except Exception:
            pass

    if obj is not None:
        # Case A: single dict {"bbox_2d":..., "label":...}
        if isinstance(obj, dict):
            if "bbox_2d" in obj:
                lbl = obj.get("label", default_label)
                bboxes.append((lbl, list(map(float, obj["bbox_2d"]))))
            return bboxes

        # Case B: list
        if isinstance(obj, list):
            # B1: [x1,y1,x2,y2]
            if len(obj) == 4 and all(isinstance(x, (int, float)) for x in obj):
                bboxes.append((default_label, list(map(float, obj))))
                return bboxes

            # B2: [{"bbox_2d":...,"label":...}, ...]
            for item in obj:
                if isinstance(item, dict) and "bbox_2d" in item:
                    lbl = item.get("label", default_label)
                    bboxes.append((lbl, list(map(float, item["bbox_2d"]))))
            if bboxes:
                return bboxes

    # 3) Fallback: parse text like "motorcycle [x1, y1, x2, y2]"
    pattern = r"(\w+)\s*\[\s*([\d.]+),\s*([\d.]+),\s*([\d.]+),\s*([\d.]+)\s*\]"
    for cls, x1, y1, x2, y2 in re.findall(pattern, raw):
        bboxes.append((cls, [float(x1), float(y1), float(x2), float(y2)]))

    # 4) Also allow unlabeled "[x1, y1, x2, y2]" inside a sentence (GT often looks like this)
    pattern2 = r"\[\s*([\d.]+),\s*([\d.]+),\s*([\d.]+),\s*([\d.]+)\s*\]"
    if not bboxes:
        m = re.search(pattern2, raw)
        if m:
            bboxes.append((default_label, list(map(float, m.groups()))))

    return bboxes


def compute_iou(box1, box2):
    """
    Computes IoU between two bounding boxes.
    Format: [x1, y1, x2, y2]
    """
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0

def evaluate_bboxes(json_data):
    """
    Evaluates predictions by calculating precision, recall, and F1 score per class.
    """
    tp_counts = collections.defaultdict(int)
    fp_counts = collections.defaultdict(int)
    fn_counts = collections.defaultdict(int)

    for entry in json_data:
        label = "pedestrian" if "pedestrian" in entry["question"].lower() else "motorcycle"

        ans_str = entry["answer"] if isinstance(entry["answer"], str) else " ".join(entry["answer"])
        gt_str  = entry["answer_gt"] if isinstance(entry["answer_gt"], str) else " ".join(entry["answer_gt"])

        preds = parse_bboxes(ans_str, label)
        gts   = parse_bboxes(gt_str,  label)

        pred_boxes = [box for cls, box in preds if cls == label]
        gt_boxes = [box for cls, box in gts if cls == label]

        matched_gt = [False] * len(gt_boxes)

        for pred_box in pred_boxes:
            matched = False
            for i, gt_box in enumerate(gt_boxes):
                if not matched_gt[i] and compute_iou(pred_box, gt_box) >= 0.5:
                    matched_gt[i] = True
                    tp_counts[label] += 1
                    matched = True
                    break
            if not matched:
                fp_counts[label] += 1

        fn_counts[label] += matched_gt.count(False)

    results = {}
    for label in set(tp_counts.keys()) | set(fp_counts.keys()) | set(fn_counts.keys()):
        TP = tp_counts[label]
        FP = fp_counts[label]
        FN = fn_counts[label]
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        results[label] = {"precision": precision, "recall": recall, "f1": f1}

    df = pd.DataFrame(results).T
    print(df)

    # Compute and print average F1 score
    avg_f1 = df["f1"].mean()
    print(f"\nAverage F1 Score: {avg_f1:.3f}")

    return df

if __name__ == "__main__":
    # Replace this with your path to the JSON file
    json_path = "example.json"

    with open(json_path, "r") as f:
        data = json.load(f)

    evaluate_bboxes(data)
