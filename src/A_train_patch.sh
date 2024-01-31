# for learning_rate in "1e-6" "5e-6" "1e-5" "5e-5" "1e-4" "5e-4"
# do
#     for architecture in single_model_soft_prompt_patch
#     do
#         file=./models/$architecture-3b/$learning_rate/model_args.json
#         if [ ! -f "$file" ]; then
#             sbatch --time=2-00:00:00 --mem-per-cpu=8000M --mincpus=4 \
#             --account def-afyshe-ab --gres=gpu:a100:1 run_python.sh \
#             ./A_train_question_generator_patch.py $architecture $learning_rate quail t5-3b
#         fi
#     done
# done