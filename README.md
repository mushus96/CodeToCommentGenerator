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

## Retrieval Dataset
We have used {other dataset names} to intermediate fine tune our model. But for the final finetuning we required data which was not readily available anywhere. We decided to scrap the data from official websites of **Scikit Learn, PyTorch, Numpy, Tensorflow**. 

We needed to get the description of each function which can come in python code. By using this extra information model can generate a better comment. 

For scraping we have used *Beautiful soup library*. It saved us a lot of time rather than copying all the functions manually. We retrieved total 5633 functions from these four libraries. 

| Library | No of functions |
|------------|------------------|
| Pandas | 1216
|Scikit Learn | 519 |
|PyTorch| 702|
|Numpy|1869|
|Tensorflow-keras|1327|

To scrap the data we used retrieval_task.ipynb. In this notebook there are two parts, first one is used for Pandas, Scikit, PyTorch, Numpy and second one is used for Tensorflow. We just need to mention links of all the webpages which contains functions. We created a list of links and passed it to loop. It will create that many (no of webpages) excel files and save it to working directory. At the end we have given code to merge all the excel files to one. In the for loop we need to go through all the file names (which are numbered) and in the loop it will call excel_write_all fun to dump in one excel.
