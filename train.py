import pandas as pd
import numpy as np
import sys
import os
import json
import transformers
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, Dataset
from sklearn.preprocessing import LabelEncoder
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
import tensorflow as tf

class my_Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = torch.tensor(labels)

    def __getitem__(self, idx):
        input_ids = self.encodings['input_ids'][idx]
        target_ids = self.labels[idx]
        attention_masks = self.encodings['attention_mask'][idx]
        return {"input_ids": input_ids, "labels": target_ids, "attention_mask": attention_masks}

    def __len__(self):
        return len(self.labels)



f = open ('/u/athbagde/ML_CL/EXIST 2023 Dataset 2/training/EXIST2023_training.json', "r")
  
# Reading from file
data = json.loads(f.read())


training_set  = pd.DataFrame(columns=['id','tweet','lang','sex','age','t1_lb','t2_lb','t3_lb'])

for id in data:
    sample = data[id]
    training_set.loc[len(training_set.index)] = [sample['id_EXIST'],sample['tweet'],
                                                sample['lang'],sample['gender_annotators'],
                                                sample['age_annotators'],sample['labels_task1'],
                                                sample['labels_task2'],sample['labels_task3']
                                                ]

training_set = training_set.set_index(['id','tweet','lang']).apply(pd.Series.explode).reset_index()

f = open ('/u/athbagde/ML_CL/EXIST 2023 Dataset 2/dev/EXIST2023_dev.json', "r")
  
# Reading from file
data = json.loads(f.read())


val_training_set  = pd.DataFrame(columns=['id','tweet','lang','sex','age','t1_lb','t2_lb','t3_lb'])

for id in data:
    sample = data[id]
    val_training_set.loc[len(val_training_set.index)] = [sample['id_EXIST'],sample['tweet'],
                                                sample['lang'],sample['gender_annotators'],
                                                sample['age_annotators'],sample['labels_task1'],
                                                sample['labels_task2'],sample['labels_task3']
                                                ]

val_set =  val_training_set.set_index(['id','tweet','lang']).apply(pd.Series.explode).reset_index()
F_val_set = val_set.loc[val_set['sex']=='F']
F_train_set = training_set.loc[training_set['sex']=='F']

# check if we have cuda installed
if torch.cuda.is_available():
    # to use GPU
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('roberta-large')

features = list(F_train_set['tweet'])
targets = list(F_train_set['t1_lb'])
val_features = list(F_val_set['tweet'])
val_targets = list(F_val_set['t1_lb'])

MAX_LEN = 128
tokenized_feature = tokenizer(
                            # Sentences to encode
                            features, 
                            # Add empty tokens if len(text)<MAX_LEN
                            padding = 'max_length',
                            # Truncate all sentences to max length
                            truncation=True,
                            # Set the maximum length
                            max_length = MAX_LEN, 
                            # Return attention mask
                            return_attention_mask = True,
                            # Return pytorch tensors
                            return_tensors = 'pt'       
                   )
MAX_LEN = 128
val_tokenized_feature = tokenizer(
                            # Sentences to encode
                            val_features, 
                            # Add empty tokens if len(text)<MAX_LEN
                            padding = 'max_length',
                            # Truncate all sentences to max length
                            truncation=True,
                            # Set the maximum length
                            max_length = MAX_LEN, 
                            # Return attention mask
                            return_attention_mask = True,
                            # Return pytorch tensors
                            return_tensors = 'pt'       
                   )
le = LabelEncoder()
le.fit(targets)
target_num = le.transform(targets)

le = LabelEncoder()
le.fit(val_targets)
val_target_num = le.transform(val_targets)

tokenized_feature['input_ids']

batch_size = 64
# Create the DataLoader for our training set
train_data = my_Dataset(tokenized_feature,target_num)
val_data = my_Dataset(val_tokenized_feature,val_target_num)

from transformers import RobertaForSequenceClassification, AdamW, get_linear_schedule_with_warmup
model = RobertaForSequenceClassification.from_pretrained(
    "roberta-large", 
    # Specify number of classes
    num_labels = len(set(targets)), 
    # Whether the model returns attentions weights
    output_attentions = False,
    # Whether the model returns all hidden-states 
    output_hidden_states = False
)

model.cuda()

import evaluate
accuracy = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)


args = TrainingArguments(
    f"RoBERTa-finetuned-task1",
    evaluation_strategy = "epoch",
    save_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=5,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model='accuracy'
)

trainer = Trainer(
    model,
    args,
    train_dataset=train_data,
    eval_dataset=val_data,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()

trainer.save_model(f'F_only_model1')
