#%% Based on https://towardsdatascience.com/asking-the-right-questions-training-a-t5-transformer-model-on-a-new-task-691ebba2d72c
import numpy as np
import pandas as pd

from pathlib import Path
from lib.t5_model import T5Model as T5Single
from transformers import T5ForConditionalGeneration, T5Tokenizer

import json
import sys

from pathlib import Path

np.random.seed(314)

root = (Path(__file__).parent/'../').resolve()

args = sys.argv[1:]
# args = ['single_model_soft_prompt_patch','5e-5','quail','t5-large'] # ! TODO: Remove this line
############ Config ############

architecture = args[0]
# separate_models
# shared_encoder
# grouped_encoders
# single_model
# single_model_with_token
learning_rate = float(args[1])
token_prefix_length = 10

dataset = args[2]

base_model = args[3]

if len(args) > 4:
    batch_strategy = args[4]
else:
    batch_strategy = 'random'

# ! Add batch? (Required unique seeds)
if base_model == 't5-large':
    model_dir_name = architecture
elif base_model == 't5-3b':
    model_dir_name = f"{architecture}-3b"
else:
    model_dir_name = f"{architecture}-pt-{base_model.split('/')[-1]}"

models_base_name = 'models'
if batch_strategy!='random':
    models_base_name = f'models-{batch_strategy}'

if dataset == 'quail':
    out_folder = Path(models_base_name)/model_dir_name/args[1]
else:
    out_folder = Path(f'{models_base_name}-{dataset}')/model_dir_name/args[1]

################################

arch_map = {
    'separate_models': None,
    'single_model': T5Single,
    'single_model_soft_prompt_patch': T5Single,
    'single_model_pure_soft': T5Single,
    'single_model_soft_from_pure': T5Single,
    'single_model_t5_wta': T5Single,
    'single_model_with_token': T5Single,
    'single_model_with_end_token': T5Single,
    'single_model_with_token_v2': T5Single,
    'single_model_with_token_v3': T5Single,
    'single_model_with_token_question_token_init': T5Single,
    'single_model_with_token_random_token_init': T5Single,
    'single_model_with_token_random_token_init_alt': T5Single,
    'single_model_with_token_random_token_init_with_sep_epochs_20': T5Single,
    'single_model_with_token_random_token_init_with_sep': T5Single,
    'single_model_with_man_prompt_token_init_with_sep': T5Single,
    'single_model_with_man_prompt_token_init': T5Single,
    'single_model_with_token_random_token_init_two_stage': T5Single,
    'single_model_with_token_end_twice': T5Single,
    'single_model_with_sentence_prompt': T5Single,

    'ghanem_et_al_2022': None,
    'ghanem_et_al_2022_true': None,
    'ghanem_et_al_2022_soft': None,
    'ghanem_et_al_2022_soft_shared': None,
    'ghanem_et_al_2022_soft_sane': None,
}
T5Model = arch_map[architecture]

epochs = 100

out_folder.mkdir(parents=True,exist_ok=True)

def load_dataset_df(path):
    rows = []
    with open(path) as dataset_file:
        for line in dataset_file.readlines():
            rows.append(json.loads(line))
    # target_text     input_text      prefix
    df = pd.DataFrame(rows)
    df = df.rename(
        columns={
            'question': 'target_text', 
            'context': 'input_text'
        }
    )
    df = df[df['question_type']!='Unanswerable']
    if '_token_init' in architecture or '_pure_soft' in architecture or '_soft_from_pure' in architecture or '_soft_prompt_patch' in architecture:
        df['input_text'] = df.apply(lambda x: ''.join([f'<{x.question_type}{i}>' for i in range(token_prefix_length)])+ x.input_text, axis=1)

    df['prefix'] = '' # We add the prefix transparantly above
    df['decoder_key'] = df['question_type']
    return df

def count_words(text):
    return len(text.split(' '))

if dataset == 'quail':
    soft_prompt_id_map = {
        'Belief_states': 0,
        'Causality': 1,
        'Character_identity': 2,
        'Entity_properties': 3,
        'Event_duration': 4,
        'Factual': 5,
        'Subsequent_state': 6,
        'Temporal_order': 7,
    }

    df_train = load_dataset_df(root/'data/quail/quail_v1.3/json/train.jsonl')
    df_validation = load_dataset_df(root/'data/quail/quail_v1.3/json/dev.jsonl')
    # df_test = load_dataset_df(root/'data/quail/quail_v1.3/json/challenge.jsonl')
    question_types = [
        "Belief_states",
        "Causality",
        "Character_identity",
        "Entity_properties",
        "Event_duration",
        "Factual",
        "Subsequent_state",
        "Temporal_order",
        # "Unanswerable",
    ]
elif dataset == 'dreamscape':
    df_train = load_dataset_df(root/'data/dreamscape/train_v2.jsonl')
    df_validation = load_dataset_df(root/'data/dreamscape/dev_v2.jsonl')
    # df_test = load_dataset_df(root/'data/dreamscape/test_v2.jsonl')
    question_types = sorted(list(set(df_train['question_type'])))
else:
    raise NotImplementedError('Dataset not currently supported, map to QUAIL format and add another case here...')
#%%
import torch

# Prevents crash, slower than default
torch.multiprocessing.set_sharing_strategy('file_system')

job_name = f"Quail Question Generation with T5 Patch ({architecture}, lr {learning_rate})"

print('job:',job_name)

model_args = {
    "preprocess_inputs": False,
    "reprocess_input_data": True,
    "overwrite_output_dir": True,
    "max_seq_length": 512,
    "train_batch_size": 2,
    "gradient_accumulation_steps": 2,
    "num_train_epochs": epochs,
    "early_stopping_patience": 1,
    "use_early_stopping": True,

    "save_eval_checkpoints": True,
    "save_steps": -1,
    "use_multiprocessing": False, # Can cause crash if enabled (but, slower)
    "use_multiprocessing_for_evaluation": False, # Can cause crash if enabled (but, slower)
    "evaluate_during_training": True,
    "evaluate_during_training_steps": 15000,
    "evaluate_during_training_verbose": True,
    "local_files_only":True,
    "fp16": False,
    "learning_rate": learning_rate,
    "output_dir": str(out_folder.resolve()),
    "best_model_dir": str((out_folder/'best_model').resolve()),
    "batch_strategy": batch_strategy,

    "manual_seed": 314,
    "wandb_project": job_name,
}


if T5Model:
    model = T5Model("t5c", base_model, args=model_args)

    if "_soft_prompt_patch" in architecture:
        if '-3b' in base_model:
            patch_path = root/'src/models/single_model_with_token_random_token_init-3b/1e-4'
        else:
            patch_path = root/'src/models/single_model_with_token_random_token_init/1e-4'
        patch_model = T5ForConditionalGeneration.from_pretrained(patch_path)
        patch_tokenizer = T5Tokenizer.from_pretrained(patch_path)
        patch_embeddings = patch_model.shared.weight

        model.tokenizer = patch_tokenizer

        # Resize embeddings
        with torch.no_grad():
            model.model.resize_token_embeddings(patch_embeddings.shape[0])
            model.model.shared.weight[32000:] = patch_embeddings[32000:]

        del patch_model
        del patch_tokenizer
        del patch_embeddings
    if '_new_soft' in architecture:
        for name, param in model.model.named_parameters():
            if name != 'shared.learned_embedding':
                param.requires_grad = False
        model.train_model(df_train, eval_data=df_validation)
    else:
        model.train_model(df_train, eval_data=df_validation)
else:
    raise Exception('Not implmentated')
# %%
