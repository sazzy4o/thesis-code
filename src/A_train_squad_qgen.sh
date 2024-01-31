#!/bin/bash
for seed in 2 3 # 1
do
    for learning_rate in "5e-5" "1e-4" "5e-4" # "1e-6" # 1e-6 is too slow, skipping for now
    do
        for take in 10000 5000 2000 1000 
        do
            for architecture in single_model_with_token_random_token_init ghanem_et_al_2022_true
            do
                file=./models-squad-$seed/$architecture-$take/$learning_rate/model_files/config.json
                if [ ! -f "$file" ]; then
                    echo "Running $file"
                    sbatch --time=12:00:00 --mem-per-cpu=10000M --mincpus=1 --exclude=cdr2614,cdr2486,cdr2591 \
                    --account def-afyshe-ab --gres=gpu:v100l:1 run_python.sh \
                    ./A_train_question_generator_squad.py $architecture $learning_rate $take $seed
                fi
            done
        done
    done
done