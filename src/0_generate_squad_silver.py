#%%
import json
import pandas as pd
import numpy as np
import sys


from pathlib import Path
from tqdm.autonotebook import tqdm

from simpletransformers.classification import ClassificationModel
from datasets import load_dataset

#%%
args = sys.argv[1:]
# args = ['0.9', 'train']

cutoff = float(args[0])
split = args[1]

root = (Path(__file__).parent).resolve()

#%%
model = ClassificationModel(
    "roberta", 
    "/home/vonderoh/Github/EG-size/src/q_type_classifier_tuned6", 
    args = {
        "use_multiprocessing": False, # Can cause crash if enabled (but, slower)
        "use_multiprocessing_for_evaluation": False, # Can cause crash if enabled (but, slower)
        "silent": True,
    }
)
#%%
type_map = {
    'Causality': 0,
    'Entity_properties': 1,
    'Temporal_order': 2,
    'Belief_states': 3,
    'Factual': 4,
    'Event_duration': 5,
    'Character_identity': 6,
    'Subsequent_state': 7
}
num_map = {v:k for k,v in type_map.items()}
#%%
dataset_sq = load_dataset('squad')
# %%
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum() # only difference

def predict(question):
    pred_list, logits = model.predict([question])
    pred = pred_list[0]
    prob = [x for x in softmax(logits)[0]][pred]
    return num_map[pred], prob
# %%
context_ids = {}

output_rows = []
for row in tqdm(dataset_sq[split]):
    if row['context'] in context_ids:
        context_id = context_ids[row['context']]
    else:
        context_id = f'{split}-{len(context_ids)}'
        context_ids[row['context']] = context_id
    question = row['question']
    q_type, prob = predict(question)
    if prob < cutoff:
        q_type = 'Unknown'
    output_rows.append({
        **row,
        'question_type': q_type,
        'context_id': context_id,
    })

out_folder = root/'squad_silver'
out_folder.mkdir(parents=True, exist_ok=True)
out_path = out_folder/f'{split}-{cutoff}.jsonl'
with open(out_path, 'w') as f:
    for row in output_rows:
        f.write(json.dumps(row)+'\n')
# %%
