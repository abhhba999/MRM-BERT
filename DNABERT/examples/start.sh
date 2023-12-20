
export KMER=3
export MODEL_PATH=/ssd2/wls/DNABERT/3-new-12w-0
export DATA_PATH=sample_data/hm6a
export OUTPUT_PATH=./ft/hm6a

python run_finetune.py  --model_type dna  --tokenizer_name=dna3  --model_name_or_path $MODEL_PATH  --task_name dnaprom  --do_train  --do_eval  --data_dir $DATA_PATH  --max_seq_length 55  --per_gpu_eval_batch_size=32    --per_gpu_train_batch_size=32   --learning_rate 3e-5  --num_train_epochs 5.0  --output_dir $OUTPUT_PATH  --evaluate_during_training  --logging_steps 100  --save_steps 4000  --warmup_percent 0.1  --hidden_dropout_prob 0.1  --overwrite_output  --weight_decay 0.01  --n_process 40  --fp16