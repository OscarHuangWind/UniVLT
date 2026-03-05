# run the generate_csv.py script for the json file
python generate_csv.py --input_file example.json

# launch the evaluation
python eval.py --predictions_path example.csv

# launch the gpt evaluation as well
python gpt_scorer.py --eval_file example.json --output_path ./gpt_eval_results
