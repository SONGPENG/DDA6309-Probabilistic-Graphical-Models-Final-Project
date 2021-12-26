mkdir logs

# UD1
mkdir ./models/UD1_tcwspos_bilstm

mkdir ./models/UD1_pcwspos_bilstm

mkdir ./models/UD1_scwspos_bilstm

mkdir ./models/UD1_tcwspos_transformer

mkdir ./models/UD1_pcwspos_transformer

mkdir ./models/UD1_scwspos_transformer

# UD2
mkdir ./models/UD2_tcwspos_bilstm

mkdir ./models/UD2_pcwspos_bilstm

mkdir ./models/UD2_scwspos_bilstm

mkdir ./models/UD2_tcwspos_transformer

mkdir ./models/UD2_pcwspos_transformer

mkdir ./models/UD2_scwspos_transformer

# important parameters
# do_train: train the model
# do_test: test the model
# use_bert: use BERT as encoder
# use_zen: use ZEN as encoder
# bert_model: the directory of BERT/ZEN model
# use_attention: use two-way attention
# source: the toolkit to be use (stanford or berkeley)
# feature_flag: use pos, chunk, or dep knowledge
# model_name: the name of model to save

# training
# Command lines to train our model with POS knowledge from SCT
# use BERT

# UD1
python main_tcwspos.py --do_train --train_data_path=./data/UD1/train.tsv --eval_data_path=./data/UD1/dev.tsv --use_bert --bert_model=./models/BERT/bert-base-chinese/ --use_attention --max_seq_length=300 --max_ngram_size=300  --train_batch_size=16 --eval_batch_size=16 --num_train_epochs=100 --warmup_proportion=0.1 --learning_rate=1e-5  --patient=15 --source=stanford --feature_flag=pos --model_name=UD1_tcwspos_transformer --encoder_type=transformer

python main_pcwspos.py --do_train --train_data_path=./data/UD1/train.tsv --eval_data_path=./data/UD1/dev.tsv --use_bert --bert_model=./models/BERT/bert-base-chinese/ --use_attention --max_seq_length=300 --max_ngram_size=300  --train_batch_size=16 --eval_batch_size=16 --num_train_epochs=100 --warmup_proportion=0.1 --learning_rate=1e-5  --patient=15 --source=stanford --feature_flag=pos --model_name=UD1_pcwspos_transformer --encoder_type=transformer

python main_scwspos.py --do_train --train_data_path=./data/UD1/train.tsv --eval_data_path=./data/UD1/dev.tsv --use_bert --bert_model=./models/BERT/bert-base-chinese/ --use_attention --max_seq_length=300 --max_ngram_size=300  --train_batch_size=16 --eval_batch_size=16 --num_train_epochs=100 --warmup_proportion=0.1 --learning_rate=1e-5  --patient=15 --source=stanford --feature_flag=pos --model_name=UD1_scwspos_transformer --encoder_type=transformer

python main_tcwspos.py --do_train --train_data_path=./data/UD1/train.tsv --eval_data_path=./data/UD1/dev.tsv --use_bert --bert_model=./models/BERT/bert-base-chinese/ --use_attention --max_seq_length=300 --max_ngram_size=300  --train_batch_size=16 --eval_batch_size=16 --num_train_epochs=100 --warmup_proportion=0.1 --learning_rate=1e-5  --patient=15 --source=stanford --feature_flag=pos --model_name=UD1_tcwspos_bilstm --encoder_type=bilstm

python main_pcwspos.py --do_train --train_data_path=./data/UD1/train.tsv --eval_data_path=./data/UD1/dev.tsv --use_bert --bert_model=./models/BERT/bert-base-chinese/ --use_attention --max_seq_length=300 --max_ngram_size=300  --train_batch_size=16 --eval_batch_size=16 --num_train_epochs=100 --warmup_proportion=0.1 --learning_rate=1e-5  --patient=15 --source=stanford --feature_flag=pos --model_name=UD1_pcwspos_bilstm --encoder_type=bilstm

python main_scwspos.py --do_train --train_data_path=./data/UD1/train.tsv --eval_data_path=./data/UD1/dev.tsv --use_bert --bert_model=./models/BERT/bert-base-chinese/ --use_attention --max_seq_length=300 --max_ngram_size=300  --train_batch_size=16 --eval_batch_size=16 --num_train_epochs=100 --warmup_proportion=0.1 --learning_rate=1e-5  --patient=15 --source=stanford --feature_flag=pos --model_name=UD1_scwspos_bilstm --encoder_type=bilstm



# UD2
python main_tcwspos.py --do_train --train_data_path=./data/UD2/train.tsv --eval_data_path=./data/UD2/dev.tsv --use_bert --bert_model=./models/BERT/bert-base-chinese/ --use_attention --max_seq_length=300 --max_ngram_size=300  --train_batch_size=16 --eval_batch_size=16 --num_train_epochs=100 --warmup_proportion=0.1 --learning_rate=1e-5  --patient=15 --source=stanford --feature_flag=pos --model_name=UD2_tcwspos_transformer --encoder_type=transformer

python main_pcwspos.py --do_train --train_data_path=./data/UD2/train.tsv --eval_data_path=./data/UD2/dev.tsv --use_bert --bert_model=./models/BERT/bert-base-chinese/ --use_attention --max_seq_length=300 --max_ngram_size=300  --train_batch_size=16 --eval_batch_size=16 --num_train_epochs=100 --warmup_proportion=0.1 --learning_rate=1e-5  --patient=15 --source=stanford --feature_flag=pos --model_name=UD2_pcwspos_transformer --encoder_type=transformer

python main_scwspos.py --do_train --train_data_path=./data/UD2/train.tsv --eval_data_path=./data/UD2/dev.tsv --use_bert --bert_model=./models/BERT/bert-base-chinese/ --use_attention --max_seq_length=300 --max_ngram_size=300  --train_batch_size=16 --eval_batch_size=16 --num_train_epochs=100 --warmup_proportion=0.1 --learning_rate=1e-5  --patient=15 --source=stanford --feature_flag=pos --model_name=UD2_scwspos_transformer --encoder_type=transformer

python main_tcwspos.py --do_train --train_data_path=./data/UD2/train.tsv --eval_data_path=./data/UD2/dev.tsv --use_bert --bert_model=./models/BERT/bert-base-chinese/ --use_attention --max_seq_length=300 --max_ngram_size=300  --train_batch_size=16 --eval_batch_size=16 --num_train_epochs=100 --warmup_proportion=0.1 --learning_rate=1e-5  --patient=15 --source=stanford --feature_flag=pos --model_name=UD2_tcwspos_bilstm --encoder_type=bilstm

python main_pcwspos.py --do_train --train_data_path=./data/UD2/train.tsv --eval_data_path=./data/UD2/dev.tsv --use_bert --bert_model=./models/BERT/bert-base-chinese/ --use_attention --max_seq_length=300 --max_ngram_size=300  --train_batch_size=16 --eval_batch_size=16 --num_train_epochs=100 --warmup_proportion=0.1 --learning_rate=1e-5  --patient=15 --source=stanford --feature_flag=pos --model_name=UD2_pcwspos_bilstm --encoder_type=bilstm

python main_scwspos.py --do_train --train_data_path=./data/UD2/train.tsv --eval_data_path=./data/UD2/dev.tsv --use_bert --bert_model=./models/BERT/bert-base-chinese/ --use_attention --max_seq_length=300 --max_ngram_size=300  --train_batch_size=16 --eval_batch_size=16 --num_train_epochs=100 --warmup_proportion=0.1 --learning_rate=1e-5  --patient=15 --source=stanford --feature_flag=pos --model_name=UD2_scwspos_bilstm --encoder_type=bilstm



# testing

# UD1
python main_tcwspos.py --do_test --eval_data_path=./data/UD1/test.tsv --eval_model=./models/UD1_tcwspos_bilstm/model.pt

python main_pcwspos.py --do_test --eval_data_path=./data/UD1/test.tsv --eval_model=./models/UD1_pcwspos_bilstm/model.pt

python main_scwspos.py --do_test --eval_data_path=./data/UD1/test.tsv --eval_model=./models/UD1_scwspos_bilstm/model.pt

python main_tcwspos.py --do_test --eval_data_path=./data/UD1/test.tsv --eval_model=./models/UD1_tcwspos_transformer/model.pt

python main_pcwspos.py --do_test --eval_data_path=./data/UD1/test.tsv --eval_model=./models/UD1_pcwspos_transformer/model.pt

python main_scwspos.py --do_test --eval_data_path=./data/UD1/test.tsv --eval_model=./models/UD1_scwspos_transformer/model.pt

# UD2
python main_tcwspos.py --do_test --eval_data_path=./data/UD2/test.tsv --eval_model=./models/UD2_tcwspos_bilstm/model.pt

python main_pcwspos.py --do_test --eval_data_path=./data/UD2/test.tsv --eval_model=./models/UD2_pcwspos_bilstm/model.pt

python main_scwspos.py --do_test --eval_data_path=./data/UD2/test.tsv --eval_model=./models/UD2_scwspos_bilstm/model.pt

python main_tcwspos.py --do_test --eval_data_path=./data/UD2/test.tsv --eval_model=./models/UD2_tcwspos_transformer/model.pt

python main_pcwspos.py --do_test --eval_data_path=./data/UD2/test.tsv --eval_model=./models/UD2_pcwspos_transformer/model.pt

python main_scwspos.py --do_test --eval_data_path=./data/UD2/test.tsv --eval_model=./models/UD2_scwspos_transformer/model.pt