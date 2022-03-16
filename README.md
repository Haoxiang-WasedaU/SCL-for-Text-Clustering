# self-supervise contrastive learning for long and short texts clustering 
This is code of paper of IEEE SMC 2021:"A Simple and Effective Usage of Self-supervised Contrastive Learning for Text Clustering" [[paper]](https://ieeexplore.ieee.org/abstract/document/9659143)

## Overview

- [`train.py`](./download.sh) : The main script for this this project, including default BERT, loss function, and clustering accuracy metric 
- [`load_data.py`](./load_data.py) : Load the data of sup, unsup
- [`models.py`](./models.py) : Model calsses for a general transformer (from Pytorchic BERT's code) and LSTM model
- [`train.py`](./train.py) : A custom training class(Trainer class) 
- ***utils***
  - [`configuration.py`](./utils/configuration.py) : Set a configuration from json file
  - [`checkpoint.py`](./utils/checkpoint.py) : Functions to load a model from tensorflow's file (from Pytorchic BERT's code)
  - [`optim.py`](./utils.optim.py) : Optimizer (BERTAdam class) (from Pytorchic BERT's code)
  - [`tokenization.py`](./utils/tokenization.py) : Tokenizers adopted from the original Google BERT's code
  - [`utils.py`](./utils/utils.py) : A custom utility functions adopted from Pytorchic BERT's code
- ***Dataprocessing_util***
  - [`backtranslation.py`](./Dataprocessing_util/backtranslation.py) : Backtraslation script 
  - [`fewshotprocessing.py`](./Dataprocessing_util/fewshotprocessing.py) : Fewshot preprocessing script 
  - [`Reuters_processing.py`](./Dataprocessing_util/Reuters_processing.py) : Processing Reuters dataset 
  
## Pre-works

#### - Download pre-trained BERT model 
First, you have to download pre-trained BERT_base from Google's BERT repository. 
After running, you can get the pre-trained BERT_base_Uncased model at **/BERT_Base_Uncased** director and **/data**

preparing datasets: Reuters,20newsgroup, stackoverflow, SearchSnippets

## Example usage
 Please install the required package
 
 And then run main.py script
 
 You can choose different models in main.py script
 
 All the evaluation result will print in your screen, and you can also save them


## Acknowledgement
Thanks to references of [UDA](https://github.com/google-research/uda) and [Pytorchic BERT](https://github.com/dhlee347/pytorchic-bert), I can implement this code.

