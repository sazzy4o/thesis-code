#!/bin/bash
for size in "large" # "3b" "base" "small"
do
    for learning_rate in "1e-6" "5e-6" "1e-5" "5e-5" "1e-4" "5e-4"
    do
        for architecture in t5_wta_patch # single_model_soft_prompt_patch # single_model single_model_with_token_random_token_init ghanem_et_al_2022_true
        do
            file=./models-$size/$architecture/$learning_rate/model_files/config.json
            if [ ! -f "$file" ]; then
                sbatch --time=24:00:00 --mem-per-cpu=12000M --mincpus=1 \
                --account def-afyshe-ab --gres=gpu:v100l:1 run_python.sh \
                ./A_train_question_generator.py $architecture $learning_rate quail $size
            fi
        done
    done

    # for learning_rate in "1e-6" "5e-6" "1e-5" "5e-5" "1e-4" "5e-4"
    # do
    #     for architecture in separate_models
    #     do
    #         for question_type in "Belief_states" "Causality" "Character_identity" "Entity_properties" "Event_duration" "Factual" "Subsequent_state" "Temporal_order"
    #         do
    #             file=./models-$size/$architecture/$learning_rate/$question_type/"model_files/config.json"
    #             if [ ! -f "$file" ]; then
    #                 sbatch --time=24:00:00 --mem-per-cpu=20000M --mincpus=2 \
    #                 --account def-afyshe-ab --gres=gpu:a100:2 run_python.sh \
    #                 ./A_train_question_generator.py $architecture $learning_rate quail $size $question_type
    #             fi
    #         done
    #     done
    # done
done