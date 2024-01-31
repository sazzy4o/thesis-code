#!/bin/bash
for seed in 1 2 3
do
    for learning_rate in "5e-6" "1e-5" "5e-5" "1e-4" "5e-4" # "1e-6" 
    do
        echo '' > /dev/null
        for architecture in ghanem_et_al_2022_t5_lm control_t5_lm
        do
            file=./models-large-$seed-control/$architecture-from-silver-$architecture/$learning_rate/model_files/config.json
            if [ ! -f "$file" ]; then
                echo $file
                # if [ "$learning_rate" = "1e-6" ] || [ "$learning_rate" = "5e-6" ]; then
                #     sbatch --time=72:00:00 --mem-per-cpu=10000M --mincpus=1 --exclude=cdr2614,cdr2486,cdr2591 \
                #     --account def-afyshe-ab --gres=gpu:v100l:1 run_python.sh \
                #     ./A_train_question_generator.py $architecture $learning_rate quail large $seed
                # else
                    sbatch --time=24:00:00 --mem-per-cpu=10000M --mincpus=1 --exclude=cdr2614,cdr2486,cdr2591 \
                    --account def-afyshe-ab --gres=gpu:v100l:1 run_python.sh \
                    ./A_train_question_generator_from_another_model.py $architecture $learning_rate quail large $seed $architecture
                # fi
            fi
        done
        # file=./models-large-$seed-control/single_model_with_token_random_token_init-from-silver-soft_attempt_t5_lm/$learning_rate/model_files/config.json
        # if [ ! -f "$file" ]; then
        #     sbatch --time=24:00:00 --mem-per-cpu=10000M --mincpus=1 --exclude=cdr2614,cdr2486,cdr2591 \
        #     --account def-afyshe-ab --gres=gpu:v100l:1 run_python.sh \
        #     ./A_train_question_generator_from_another_model.py single_model_with_token_random_token_init $learning_rate quail large $seed soft_attempt_t5_lm
        # fi
    done
done