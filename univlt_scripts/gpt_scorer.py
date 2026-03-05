import sys
import torch
import argparse
from PIL import Image
from pathlib import Path
import os
import json
import re
from tqdm import tqdm

# set gpt for scoring
from openai import OpenAI
API_KEY='YOUR_API_KEY'

client = OpenAI(
        api_key=API_KEY,
    )

def eval_gpt_score(args):
    eval_json = json.load(open(args.eval_file, 'r'))
    corr_num = 0
    # for idx, eval_sample in enumerate(eval_json):
    for idx, eval_sample in tqdm(enumerate(eval_json), total=len(eval_json)):
        question = eval_sample['question']
        answer_gt = eval_sample['answer_gt']
        if isinstance(answer_gt, list):
            answer_gt = answer_gt[0]
        answer = eval_sample['answer']
        content = []
        task_prompt = f"Given the question: '{question}', the reference answer: '{answer_gt.strip()}', and the predicted answer {answer}. Please evaluate the correctness of the predicted answer and assign a continuous score between 0 and 1. Please give me the score only."
        content.append({"type": "input_text", "text": task_prompt})

        response = client.responses.create(
                    model='gpt-4.1',
                    input=[
                        {
                            "role": "user",
                            "content": content
                        }
                    ],
                )
        raw = response.output_text.strip()
        match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", raw)
        if match:
            rationale = float(match.group(0))
        else:
            print(f"Failed to parse numeric score from: {raw}")
            rationale = 0.0
        if rationale >= 0.5:
            corr_num += 1
        # print(rationale)
    avg_score = corr_num / len(eval_json)
    eval_json.append({'gpt_score': avg_score})
    out_file = Path(args.output_path) / Path(args.eval_file).name
    with open(out_file, 'w') as f:
        json.dump(eval_json, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_file", type=str, default='/path/to/your/eval_file.json')
    parser.add_argument("--output_path", type=str, default='/path/to/your/output/directory')
    args = parser.parse_args()
    eval_gpt_score(args)