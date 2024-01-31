#%% Based on https://towardsdatascience.com/asking-the-right-questions-training-a-t5-transformer-model-on-a-new-task-691ebba2d72c
import numpy as np
import pandas as pd

from pathlib import Path
from htawta_true import trainer as true_trainer
from transformers import T5ForConditionalGeneration, T5Tokenizer
from peft import LoraConfig, get_peft_model, TaskType

import json
import sys
import torch

from pathlib import Path

root = (Path(__file__).parent/'../').resolve()

args = sys.argv[1:]
# args = ['t5_wta_patch','5e-5','quail','large'] # ! TODO: Remove this line
############ Config ############

architecture = args[0]
learning_rate = float(args[1])
token_prefix_length = 10

dataset = args[2]

size = args[3]
# t5-small
# t5-base
# t5-large
# t5-3b
# t5-11b.

seed = int(args[4])

torch.manual_seed(seed) # pytorch random seed
np.random.seed(seed) # numpy random seed
torch.backends.cudnn.deterministic = True

# ! Add batch? (Required unique seeds)
batch_strategy = 'random'

base_model = f't5-{size}'

model_dir_name = architecture

models_base_name = f'models-{size}-{seed}-control'

if dataset == 'quail':
    out_folder = Path(models_base_name)/model_dir_name/args[1]
else:
    out_folder = Path(f'{models_base_name}-{dataset}')/model_dir_name/args[1]

if '_t5_lm' in architecture:
    base_model = 'google/t5-large-lm-adapt'

epochs = 100

supported_architectures = [
    'lora_only_t5_lm',
]

if architecture not in supported_architectures:
    raise NotImplementedError(f'Architecture {architecture} not supported, please add it to the list of supported architectures: {supported_architectures}')

################################

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
    # if architecture == 'single_model_with_token_random_token_init' or architecture == 'single_model_soft_prompt_patch' or architecture == 't5_wta_control_init_striped' or architecture == 't5_wta_control_init_start' or architecture == 'control_t5_lm' or architecture == 't5_wta_control_init_striped_t5_lm' or architecture == 't5_wta_control_init_start_t5_lm':
    #     df['input_text'] = df.apply(lambda x: ''.join([f'<{x.question_type}{i}>' for i in range(token_prefix_length)])+ x.input_text, axis=1)
    # elif architecture == 'ghanem_et_al_2022_true' or architecture == 'ghanem_et_al_2022_t5_lm':
    #     df['input_text'] = df.apply(lambda row: ghanem_et_al_2022_prompt_map[row['question_type']]+' </s> '+ row['input_text'], axis=1)
    # elif architecture == 't5_wta_patch' or architecture == 't5_wta_control_length' or architecture == 't5_wta_control_length_and_init':
    #     df['input_text'] = df.apply(lambda row: t5_wta_patch_prompt_map[row['question_type']]+' </s> '+ row['input_text'], axis=1)
    # elif architecture == 'single_model_with_hard_prompt' or architecture == 'single_model_with_hard_prompt_t5_lm':
    #     df['input_text'] = df.apply(lambda row: 'Create Question'+' </s> '+ row['input_text'], axis=1)
    # elif architecture == 'single_model_with_soft_prompt' or architecture == 'single_model_with_soft_prompt_t5_lm':
    #     df['input_text'] = df.apply(lambda x: ''.join([f'<SP{i}>' for i in range(token_prefix_length)])+ x.input_text, axis=1)

    df['prefix'] = '' # We add the prefix transparantly above
    return df

if dataset == 'quail':
    df_train = load_dataset_df(root/'data/quail/quail_v1.3/json/train.jsonl')
    df_validation = load_dataset_df(root/'data/quail/quail_v1.3/json/dev.jsonl')
    df_test = load_dataset_df(root/'data/quail/quail_v1.3/json/challenge.jsonl')
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
    df_test = load_dataset_df(root/'data/dreamscape/test_v2.jsonl')
    question_types = sorted(list(set(df_train['question_type'])))
else:
    raise NotImplementedError('Dataset not currently supported, map to QUAIL format and add another case here...')
#%%
job_name = f"Quail Question Generation with T5 ({architecture}, lr {learning_rate})"

print('job:',job_name)

model_params = {
    "OUTPUT_PATH": str(out_folder.resolve()),
    "MODEL": base_model,
    # Total batch size = TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_BATCH_SIZE = 8
    "TRAIN_BATCH_SIZE": 8,
    "GRADIENT_ACCUMULATION_BATCH_SIZE": 1, # Equivalent to batch size 8 (slower, but lower memory)
    "VALID_BATCH_SIZE": 8,        
    "TRAIN_EPOCHS": epochs,           
    "LEARNING_RATE": learning_rate,
    "MAX_SOURCE_TEXT_LENGTH": 512,
    "MAX_TARGET_TEXT_LENGTH": 64,
    "early_stopping_patience": 3,  
}

if architecture=='separate_models' or 'lora_only' in architecture:
    question_type = args[5]
    print('Question type:',question_type)
    out_folder_resolved = (out_folder/question_type).resolve()
    out_folder_resolved.mkdir(parents=True,exist_ok=True)
    model_params["OUTPUT_PATH"] = str(out_folder_resolved)

    df_train = df_train[df_train['question_type']==question_type]
    df_validation = df_validation[df_validation['question_type']==question_type]
    df_test = df_test[df_test['question_type']==question_type]

print('model_params:' ,model_params)

model = T5ForConditionalGeneration.from_pretrained(model_params["MODEL"])
tokenizer = T5Tokenizer.from_pretrained(model_params["MODEL"])

if 'lora_only' in architecture:
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q", "v"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM
    )
    model = get_peft_model(model, lora_config, adapter_name="lora_only")
    model.set_adapter('lora_only')
else:
    raise NotImplementedError('Architecture not currently supported, add another case here...')

model.print_trainable_parameters()

device_map = None
if size in ['11b']:
    # device_map = {
    #     0: [0,  1,  2,  3,  4,  5],
    #     1: [6,  7,  8,  9,  10, 11],
    #     2: [12, 13, 14, 15, 16, 17],
    #     3: [18, 19, 20, 21, 22, 23]
    # }
    
    device_map = {
        0: [0,  1,  2],
        1: [3,  4,  5,  6,  7,  8,  9],
        2: [10, 11, 12, 13, 14, 15, 16],
        3: [17, 18, 19, 20, 21, 22, 23]
    }
    model_params["TRAIN_BATCH_SIZE"] = 1
    model_params["GRADIENT_ACCUMULATION_BATCH_SIZE"] = 8
    model.parallelize(device_map)
elif size in ['3b']:
    device_map = {
        0: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        1: [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
    }
    model.parallelize(device_map)
else:
    # model_params["TRAIN_BATCH_SIZE"] = 8
    # model_params["GRADIENT_ACCUMULATION_BATCH_SIZE"] = 1
    # device_map = {
    #     0: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    #     1: [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
    # }
    # model.parallelize(device_map)
    model.cuda()

training_loader, validation_loader, _ = true_trainer.build_data(tokenizer, dataframes=[df_train, df_validation, df_test], source_text="input_text", target_text="target_text", model_params=model_params)
true_trainer.T5Trainer(model, training_loader, validation_loader, tokenizer, model_params=model_params)
# %%
