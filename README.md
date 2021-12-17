# Automatic Code Comment Generation using GraphCodeBERT


['GraphCode-Bert'](https://arxiv.org/abs/2009.08366) is a transformer-based pre-trained  model  jointly  trained  on  the  code,  codecomments,  and  the  data-flow  graph  of  the  code. This repository contains the code for automatically generally code documentation in python using 'GraphCodeBERT' model. The jupyter notebook 'Demo.ipynb' contains all the code for for loading the 'CodeSearchNet' dataset for the task, pre-processing the dataset, training the 'GraphCodeBert' model on the dataset and testing it. The following sections contain information about the steps executed as part of the ''Demo.ipynb' notebook.

- The notebook has been tested on Colab. Steps to clone the repo, download the dataset and other dependencies are given in the notebook. It is recommended to run the notebook on Colab and using Drive because of the large size of the models and dataset.
## Dataset

- First, we download 'CodeSearchNet' dataset for python language. The 'curr_lang' variable can be updated to download the dataset for a different language.
- The 'sample' variable specifies the percentage of dataset to be downloaded. For our use-case, we used 10% of the 'CodeSearchNet' dataset.
- Than we removed all the code-comment pairs which have non-ascii charactors. 
- We removed the comments from the code to avoid the model from cheating as it can simply remember the comment from the method itself.
- After removing the comments from the code, we removed all code-comment pairs where the size of the code is smaller than the method, have empty comments and duplicate code-comment pairs.
- We also removed all code-comment pairs where method length is outside the 95th per-centile of the method lengths and converting everything to lowercase.
- Finally, we move all the data in 'training', 'testing' and 'validation' dataframes into the 'python/train.jsonl', 'python/test.jsonl' and 'python/valid.jsonl' which is the expected file format for the models. 

## Task Definition

The task is to generate natural language comments for python code, and evaluated by [smoothed bleu-4](https://www.aclweb.org/anthology/C04-1072.pdf) score.




The following commands can be used to train and test the different combinations of  model architectures, provided the dataset has been loaded correctly based on the Demo.py notebook

## Training

```shell
lang = 'python' # programming language
lr = 5e-5
batch_size = 16 # change depending on the GPU Colab gives you
beam_size = 10
source_length = 256
target_length = max_cmt_len
data_dir = '.'
output_dir = f'model/{lang}'
train_file = f'{data_dir}/{lang}/train.jsonl'
dev_file = f'{data_dir}/{lang}/valid.jsonl'
epochs = 5 
pretrained_model = 'microsoft/graphcodebert-base'
load_model_path = f'model/{lang}/checkpoint-best-bleu/pytorch_model.bin'

! python run.py \
    --do_train \
    --do_eval \
    --do_lower_case \
    --model_type roberta \
    --model_name_or_path {pretrained_model} \
    --train_filename {train_file} \
    --dev_filename {dev_file} \
    --output_dir {output_dir} \
    --max_source_length {source_length} \
    --max_target_length {target_length} \
    --beam_size {beam_size} \
    --train_batch_size {batch_size} \
    --eval_batch_size {batch_size} \
    --learning_rate {lr} \
    --num_train_epochs {epochs} \
    --load_model_path {load_model_path}
```

## Testing 

```shell
batch_size=64
lang = curr_lang 
model_lang='python'
data_dir = '.'
beam_size = 10
source_length = 256
target_length = max_cmt_len
output_dir = f'model/{model_lang}' #f'model/{lang}_new' 
dev_file=f"{data_dir}/{lang}/valid.jsonl"
test_file=f"{data_dir}/{lang}/test.jsonl"
test_model=f"{output_dir}/checkpoint-best-bleu/pytorch_model.bin" #f"{output_dir}/checkpoint-best-bleu/pytorch_model.bin" #checkpoint for test

! python run.py \
    --do_test \
    --model_type roberta \
    --model_name_or_path microsoft/graphcodebert-base \
    --load_model_path {test_model} \
    --dev_filename {dev_file} \
    --test_filename {test_file} \
    --output_dir {output_dir} \
    --max_source_length {source_length} \
    --max_target_length {target_length} \
    --beam_size {beam_size} \
    --eval_batch_size {batch_size} 
```

## Retrieval Dataset
We have used *CodeSearchNet code* dataset to intermediate fine tune our model. But for the final finetuning we required data which was not readily available anywhere. We decided to scrap the data from official websites of **Pandas, Scikit Learn, PyTorch, Numpy, Tensorflow**. 

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
