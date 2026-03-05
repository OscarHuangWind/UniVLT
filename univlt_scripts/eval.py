import click
import torch
import ast
import pandas as pd
from datasets import Dataset
from functools import partial 
from pathlib import Path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from benchmark.constants import LINGOQA_TEST, Keys

from evaluate import load
from pycocoevalcap.meteor.meteor import Meteor
from benchmark.judge import LingoJudge
import evaluate

CIDEr = load("Kamichanw/CIDEr")
bleu = evaluate.load("bleu")

@click.command()
@click.option('--predictions_path', type=click.Path(exists=True), help='Path to predictions file.', default='/path/to/your/predictions.csv')
@click.option('--batch_size', type=int, help='Batch size for evaluation.', default=1)
def evaluate(predictions_path: str, batch_size: int):
    """
    Simple script for running evaluation on the LingoQA benchmark.

    Args:
        predictions_path: Path to a .csv file containing the model predictions.
        batch_size: Batch size for evaluation.
    """
    predictions = pd.read_csv(predictions_path)
    predictions = predictions.rename({"answer": Keys.prediction}, axis=1)
    print("predictions path: ", predictions_path)

    references = pd.read_parquet(LINGOQA_TEST)
    references = references[[Keys.question_id, Keys.question, Keys.answer]]
    references = references.groupby([Keys.question_id, Keys.question]).agg({Keys.answer: list}).reset_index()
    predictions[Keys.references] = references[Keys.answer]

    predictions = predictions.fillna('None')
    
    predictions[Keys.references] = predictions[Keys.references].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    # Create dataset from merged data
    dataset = Dataset.from_pandas(predictions)

    judge = LingoJudge().eval().to("cuda:0")
    dataset_evaluated = dataset.map(partial(evaluate_question, judge), batched=True, batch_size=batch_size)
    dataset_filtered = dataset_evaluated.filter(select_correct)
    benchmark_score = dataset_filtered.num_rows/dataset_evaluated.num_rows
    print(f"The overall benchmark score is {benchmark_score*100}%")
    
    predictions = [pred.strip() for pred in dataset['Keys.prediction']]
    references = [[ref.strip() for ref in refs] for refs in dataset['Keys.references']]
    assert len(predictions) == len(references), "Predictions and references must have the same length."
    
    pred_dict = {idx: [pred] for idx, pred in enumerate(predictions)}  
    ref_dict = {idx: refs for idx, refs in enumerate(references)}  
    
    score = CIDEr.compute(predictions=dataset['Keys.prediction'], references=dataset['Keys.references'])
    print('CIDEr score',score['CIDEr'])
    results = bleu.compute(predictions=predictions, references=references)
    print("BLEU score:", results['bleu'])

    meteor_scorer = Meteor()
    meteor_score, _ = meteor_scorer.compute_score(ref_dict, pred_dict)
    print('meteor_score',meteor_score)

def evaluate_question(metric: LingoJudge, data_dict: dict) -> dict:
    """
    Run evaluation for a batch of questions.

    Args:
        metric: the evaluation metric for computing the scores.
        data_dict: the data dictionary containing questions, references, and predictions.

    Out:
        data_dict: updated data dictionary containing information such as
        the maximum score, the probability of correctness, and a boolean
        indicating whether the prediction is correct or not.
    """
    questions = data_dict['question']                   # []
    references = data_dict['Keys.references']           # [[]]
    prediction = data_dict['Keys.prediction']           # [] 

    scores = metric.compute(questions, references, prediction)

    data_dict[Keys.score] = scores
    data_dict[Keys.probability] = torch.sigmoid(scores)
    data_dict[Keys.correct] = scores > 0.0
    return data_dict


def select_correct(data_dict: dict) -> bool:
    """
    Filtering function for selecting the predictions classified as correct.
    """
    return data_dict[Keys.correct]


def attach_debugger():
    import debugpy
    debugpy.listen(5678)
    print("Waiting for debugger!")
    debugpy.wait_for_client()
    print("Attached!")

if __name__ == "__main__":
    # attach_debugger()
    evaluate()
