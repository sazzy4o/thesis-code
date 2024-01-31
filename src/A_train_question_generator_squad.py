#%% Based on https://towardsdatascience.com/asking-the-right-questions-training-a-t5-transformer-model-on-a-new-task-691ebba2d72c
import numpy as np
import pandas as pd

from pathlib import Path
from htawta_true import trainer as true_trainer
from transformers import T5ForConditionalGeneration, T5Tokenizer
from datasets import load_dataset

import sys
import torch

from pathlib import Path

# Speed things up
# torch.backends.cuda.matmul.allow_tf32 = True

root = (Path(__file__).parent/'../').resolve()

args = sys.argv[1:]
# args = ['t5_wta_patch','5e-5','1000','1'] # ! TODO: Remove this line
############ Config ############

architecture = args[0]
learning_rate = float(args[1])
token_prefix_length = 10
take = int(args[2]) if args[2] != 'None' else None

dataset = 'squad'

size = 'large'
# t5-small
# t5-base
# t5-large
# t5-3b
# t5-11b.

seed = int(args[3])

torch.manual_seed(seed) # pytorch random seed
np.random.seed(seed) # numpy random seed
torch.backends.cudnn.deterministic = True

# ! Add batch? (Required unique seeds)
batch_strategy = 'random'

base_model = f't5-{size}'

model_dir_name = f'{architecture}-{take}'

models_base_name = f'models-squad-{seed}'

if dataset == 'squad':
    out_folder = Path(models_base_name)/model_dir_name/args[1]
    ghanem_et_al_2022_prompt_map = {
        'Unset': 'Create a question about the following paragraph',
    }
    question_types = [
        'Unset',
    ]
else:
    raise NotImplementedError('Dataset not currently supported, map to SQUAD format and add another case here...')

epochs = 100

supported_architectures = [
    'ghanem_et_al_2022_true',
    'separate_models',
    'single_model',
    'single_model_with_token_random_token_init',
    'single_model_soft_prompt_patch',
    't5_wta_patch',
    't5_wta_control_length',
    't5_wta_control_length_and_init',
    't5_wta_control_init_striped',
    't5_wta_control_init_start',
]

################################

out_folder.mkdir(parents=True,exist_ok=True)

def load_squad_df(split, take):
    dataset_sq = load_dataset('squad', split=split)
    if take:
        rng = np.random.default_rng(seed=seed)
        select_indexes = list(range(87599))
        rng.shuffle(select_indexes)
        dataset_sq = dataset_sq.select(select_indexes[:take])
    df = pd.DataFrame(dataset_sq)
    df = df.rename(
        columns={
            'question': 'target_text', 
            'context': 'input_text'
        }
    )
    df['question_type'] = 'Unset'
    if architecture == 'single_model_with_token_random_token_init' or architecture == 'single_model_soft_prompt_patch' or architecture == 't5_wta_control_init_striped' or architecture == 't5_wta_control_init_start':
        df['input_text'] = df.apply(lambda x: ''.join([f'<{x.question_type}{i}>' for i in range(token_prefix_length)])+ x.input_text, axis=1)
    elif architecture == 'ghanem_et_al_2022_true':
        df['input_text'] = df.apply(lambda row: ghanem_et_al_2022_prompt_map[row['question_type']]+' </s> '+ row['input_text'], axis=1)
    
    return df

if dataset == 'squad':
    df_train = load_squad_df('train', take)
    df_validation = load_squad_df('validation', None)
else:
    raise NotImplementedError('Dataset not currently supported, map to SQUAD format and add another case here...')
#%%
job_name = f"Quail Question Generation with T5 ({architecture}, lr {learning_rate})"

print('job:',job_name)

model_params = {
    "OUTPUT_PATH": str(out_folder.resolve()),
    "MODEL": base_model,
    # Total batch size = TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_BATCH_SIZE = 8
    "TRAIN_BATCH_SIZE": 4,
    "GRADIENT_ACCUMULATION_BATCH_SIZE": 2, # Equivalent to batch size 8 (slower, but lower memory)
    "VALID_BATCH_SIZE": 16,        
    "TRAIN_EPOCHS": epochs,           
    "LEARNING_RATE": learning_rate,
    "MAX_SOURCE_TEXT_LENGTH": 512,
    "MAX_TARGET_TEXT_LENGTH": 64,
    "early_stopping_patience": 3,  
}

if architecture=='separate_models':
    question_type = args[5]
    print('Question type:',question_type)
    out_folder_resolved = (out_folder/question_type).resolve()
    out_folder_resolved.mkdir(parents=True,exist_ok=True)
    model_params["OUTPUT_PATH"] = str(out_folder_resolved)

    df_train = df_train[df_train['question_type']==question_type]
    df_validation = df_validation[df_validation['question_type']==question_type]

print('model_params:' ,model_params)

model = T5ForConditionalGeneration.from_pretrained(model_params["MODEL"])
tokenizer = T5Tokenizer.from_pretrained(model_params["MODEL"])

model.cuda()

if architecture == 'single_model_with_token_random_token_init':
    new_tokens = [
        tok for i in range(token_prefix_length)
        for tok in
        [
            f"<{x}{i}>" for x in question_types
        ]

    ]

    tokenizer.add_tokens(new_tokens)

    def get_random_regular_token_id(max_id:int):
        while True:
            random_id = np.random.randint(0, max_id)
            if random_id not in tokenizer.all_special_ids:
                return random_id


    # Resize embeddings
    with torch.no_grad():
        old_embeddings = model.get_input_embeddings()
        old_token_count = old_embeddings.weight.shape[0]
        new_embeddings = model._get_resized_embeddings(old_embeddings, old_token_count+len(new_tokens))
        for tok_str in new_tokens:
            new_tok = tokenizer.encode(tok_str)[0]
            new_embeddings.weight[new_tok] = old_embeddings.weight[get_random_regular_token_id(old_token_count)].detach().clone()

        model.set_input_embeddings(new_embeddings)

        old_lm_head = model.get_output_embeddings()
        old_lm_head_count = old_embeddings.weight.shape[0]
        new_lm_head = model._get_resized_lm_head(old_lm_head, old_lm_head_count+len(new_tokens))
        for tok_str in new_tokens:
            new_tok = tokenizer.encode(tok_str)[0]
            new_lm_head.weight[new_tok] = old_lm_head.weight[get_random_regular_token_id(old_token_count)].detach().clone()

        model.set_output_embeddings(new_lm_head)

        model.config.vocab_size = new_embeddings.weight.shape[0]

training_loader, validation_loader, _ = true_trainer.build_data(tokenizer, dataframes=[df_train, df_validation, df_validation], source_text="input_text", target_text="target_text", model_params=model_params)
true_trainer.T5Trainer(model, training_loader, validation_loader, tokenizer, model_params=model_params)
# %%
