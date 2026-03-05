import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, AutoConfig
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
import json
import argparse
import os
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_id',
        type=str,
        default='/path/to/your/model'
    )
    parser.add_argument(
        '--eval_data',
        type=str,
        nargs='+',
        default=[
        '/path/to/test/json_1',
        '/path/to/test/json_2',
        '/path/to/test/json_n',
        ],
        help='One or more eval JSON file paths.'
    )
    parser.add_argument(
        '--output_path',
        type=str,
        default='/path/to/your/eval_results_folder',
        help='Directory to save result files.'
    )
    parser.add_argument(
        '--output_file',
        type=str,
        default=None,
        help=('Optional fixed output file name. '
              'If multiple eval files are provided and this is set, a per-file name will be derived as '
              '"<stem>__<model_name>.json" and this value will be ignored. '
              'If not set, names default to "results_<model_name>_on_<stem>.json".')
    )
    parser.add_argument(
        '--max_new_tokens',
        type=int,
        default=1024,
        help='Generation max_new_tokens.'
    )
    return parser.parse_args()

def images_exist(image_list):
    return all(os.path.exists(img) for img in image_list)


def generate_message(data_dict):
    message = []
    image_list = data_dict['image']
    image_tmp = '<image>' * len(image_list)
    question = data_dict['conversations'][0]['value'].replace(f'{image_tmp}\n', '')
    content = []
    for image in image_list:
        content.append({"type": "image", "image": image})
    content.append({"type": "text", "text": question})
    message.append({'role': 'user', 'content': content})
    answer = data_dict['conversations'][1]['value']
    return message, answer, question


def build_output_filename(eval_path: str, model_id: str, output_file: str | None) -> str:
    """
    Decide the output filename for a given eval file.
    - If multiple eval files are given, always auto-name per file as "<stem>__<model_name>.json".
    - If a single eval file is given and output_file is provided, use it.
    - Otherwise, default to "results_<model_name>_on_<stem>.json".
    """
    eval_stem = Path(eval_path).stem
    model_name = Path(model_id).name

    if output_file:
        # If the caller supplied a name but is running multiple files, we still do per-file names.
        # The main invocation will override this when len(eval_data) > 1.
        return output_file

    return f"results_{model_name}_on_{eval_stem}.json"


def run_single_eval(eval_path: str, model, processor, max_new_tokens: int):
    data = json.load(open(eval_path, 'r'))
    total_cnt = 0
    correct_cnt = 0
    skipped_cnt = 0
    results = []

    for data_dict in tqdm(data, desc=f"Evaluating {Path(eval_path).name}"):
        image_list = data_dict["image"]
        # normalize in-place FIRST
        img = data_dict.get("image")
        if img is None:
            data_dict["image"] = []
        elif not isinstance(img, list):
            data_dict["image"] = [img]

        # now it's safe
        image_list = data_dict["image"]

        # Skip samples with missing image files
        if not images_exist(image_list):
            skipped_cnt += 1
            continue
        message, answer_gt, question = generate_message(data_dict)
        text = processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(message)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(model.device)

        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()

        if output_text == answer_gt:
            correct_cnt += 1
        total_cnt += 1

        results.append({
            "question_id": data_dict.get("question_id"),
            "question": question,
            "image": data_dict["image"],
            "answer": output_text,
            "answer_gt": [answer_gt]
        })

    return results


if __name__ == '__main__':
    args = parse_args()
    os.makedirs(args.output_path, exist_ok=True)

    # Load model & processor ONCE
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="flash_attention_2",
    )

    min_pixels = 144 * 28 * 28
    max_pixels = 112896*2
    processor = AutoProcessor.from_pretrained(args.model_id, min_pixels=min_pixels, max_pixels=max_pixels)

    multiple = len(args.eval_data) > 1

    for eval_path in args.eval_data:
        # Decide the per-file output name
        if multiple:
            # Ignore --output_file and auto-name
            eval_stem = Path(eval_path).stem
            model_name = Path(args.model_id).name
            output_file = f"{eval_stem}__{model_name}.json"
        else:
            # Single file: honor provided --output_file if set, else default scheme
            output_file = build_output_filename(eval_path, args.model_id, args.output_file)

        output_filepath = os.path.join(args.output_path, output_file)

        # Run evaluation
        results = run_single_eval(eval_path, model, processor, args.max_new_tokens)

        # Save results (match original behavior: only `results`; switch to `output_data` if you prefer)
        with open(output_filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)

        print(f"Results saved to {output_filepath}")