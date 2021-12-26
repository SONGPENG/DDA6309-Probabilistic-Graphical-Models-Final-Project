# Task-aware Joint CWS and POS (TCwsPos)

This is the implementation of the final project of the course DDA6309 Probabilistic Graphical Models, The Chinese University of Hong Kong (Shenzhen).

Please contact us at `{pengsong,leqitian}@link.cuhk.edu.cn` if you have any question.


## Requirements

Our code works with the following environment.
* `python=3.6`
* `pytorch=1.1`

## Downloading BERT

In our paper, we use BERT ([paper](https://www.aclweb.org/anthology/N19-1423/)) as the encoder.

For BERT, please download pre-trained BERT-Base Chinese from [Google](https://github.com/google-research/bert) or from [HuggingFace](https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese.tar.gz). If you download it from Google, you need to convert the model from TensorFlow version to PyTorch version.


## Running on Sample Data

Run `run_sample.sh` to train a model on the small sample data under the `sample_data` folder.

## Datasets

We use [Universal Dependencies 2.4](https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-2988) (UD) in our paper.

To obtain and pre-process the data, you can go to `data_preprocessing` directory and run `getdata.sh`. This script will download and process the official data from UD.

All processed data will appear in `data` directory organized by the datasets, where each of them contains the files with the same file names under the `sample_data` directory.

## Training and Testing

You can find the command lines to train and test model on a specific dataset in `run.sh`.

Here are some important parameters:

* `--do_train`: train the model
* `--do_test`: test the model
* `--use_bert`: use BERT as encoder
* `--bert_model`: the directory of pre-trained BERT model
* `--model_name`: the name of model to save 

## Predicting

`run_sample.sh` contains the command line to segment and tag the sentences in an input file ([./sample_data/sentence.txt](./sample_data/sentence.txt)).

Here are some important parameters:

* `--do_predict`: segment and tag the sentences using a pre-trained TCwsPos model.
* `--input_file`: the file contains sentences to be segmented and tagged. Each line contains one sentence; you can refer to [a sample input file](./sample_data/sentence.txt) for the input format.
* `--output_file`: the path of the output file. Words are segmented by a space; POS labels are attached to the resulting words by an underline ("_").
* `--eval_model`: the pre-trained WMSeg model to be used to segment the sentences in the input file.


## To-do List

* Regular maintenance

You can leave comments in the `Issues` section, if you want us to implement any functions.