# GraphCodeBERT -- Code To Comment Generation

## Task Definition

The task is to generate natural language comments for python code, and evaluated by [smoothed bleu-4](https://www.aclweb.org/anthology/C04-1072.pdf) score.

## Training

```shell
 python code/run.py --do_train --do_eval --model_type roberta --model_name_or_path microsoft/graphcodebert-base --train_filename dataset/python/train.jsonl --dev_filename dataset/python/valid.jsonl --output_dir weights/gcb --max_source_length 256 --max_target_length 128 --beam_size 10 --train_batch_size 4 --eval_batch_size 4 --learning_rate 7e-5 --num_train_epochs 15
```

## Testing 

```shell
python code/run.py--do_test --model_type roberta --model_name_or_path microsoft/graphcodebert-base --test_filename dataset/python/test.jsonl --max_source_length 256 --max_target_length 128 --beam_size 10  --eval_batch_size 4 --output_dir test/gcb --load_model_path weights/gcb/checkpoint-best-bleu/pytorch_model.bin
```