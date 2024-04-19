# import streamlit as st
# import transformers
# import tensorflow as tf
# from transformers import AutoTokenizer, TFAutoModelForSeq2SeqLM

# # Load the pre-trained model and tokenizer
# # model_checkpoint = "tf_model/"
# # tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
# # model = TFAutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

# import os

# import sys

# import transformers

# import tensorflow as tf

# from datasets import load_dataset

# from transformers import AutoTokenizer

# from transformers import TFAutoModelForSeq2SeqLM, DataCollatorForSeq2Seq

# from transformers import AdamWeightDecay

# from transformers import AutoTokenizer

# model_checkpoint = "Helsinki-NLP/opus-mt-en-hi"

# raw_datasets = load_dataset("cfilt/iitb-english-hindi")

# tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# max_input_length = 128

# max_target_length = 128

# source_lang = "en"

# target_lang = "hi"

# def preprocess_function(examples):

#     inputs = [ex[source_lang] for ex in examples["translation"]]

#     targets = [ex[target_lang] for ex in examples["translation"]]

#     model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

# # Setup the tokenizer for targets

#     with tokenizer.as_target_tokenizer():

#         labels = tokenizer(targets, max_length=max_target_length, truncation=True)

#         model_inputs["labels"] = labels["input_ids"]

#         return model_inputs

# tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)

# model = TFAutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

# batch_size = 16

# learning_rate = 2e-5

# weight_decay = 0.01

# num_train_epochs = 1

# data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, return_tensors="tf")

# generation_data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, return_tensors="tf", pad_to_multiple_of=128)

# train_dataset = model.prepare_tf_dataset(

# tokenized_datasets["test"],

# batch_size=batch_size,

# shuffle=True,

# collate_fn=data_collator,

# )

# validation_dataset = model.prepare_tf_dataset(

# tokenized_datasets["validation"],

# batch_size=batch_size,

# shuffle=False,

# collate_fn=data_collator,

# )

# generation_dataset = model.prepare_tf_dataset(

# tokenized_datasets["validation"],

# batch_size=8,

# shuffle=False,

# collate_fn=generation_data_collator,

# )

# optimizer = AdamWeightDecay(learning_rate=learning_rate, weight_decay_rate=weight_decay)

# model.compile(optimizer=optimizer)

# model.fit(train_dataset, validation_data=validation_dataset, epochs=1)
# model.save_pretrained("tf_model/")
# tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
# model = TFAutoModelForSeq2SeqLM.from_pretrained("tf_model/")

# # Streamlit app
# st.title("English-Hindi Translation")

# # Get user input
# input_text = st.text_area("Enter English text to translate:", height=200)

# if st.button("Translate"):
#     # Tokenize input text
#     tokenized = tokenizer([input_text], return_tensors='np')
    
#     # Generate translation
#     output = model.generate(**tokenized, max_length=128)
    
#     # Decode the output
#     with tokenizer.as_target_tokenizer():
#         translation = tokenizer.decode(output[0], skip_special_tokens=True)
    
#     st.success(f"Translated text: {translation}")


import streamlit as st
import transformers
import tensorflow as tf
from datasets import load_dataset
from transformers import AutoTokenizer, TFAutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, AdamWeightDecay

# Load the pre-trained model and tokenizer
model_checkpoint = "Helsinki-NLP/opus-mt-en-hi"
raw_datasets = load_dataset("cfilt/iitb-english-hindi")
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
max_input_length = 128
max_target_length = 128
source_lang = "en"
target_lang = "hi"

def preprocess_function(examples):
    inputs = [ex[source_lang] for ex in examples["translation"]]
    targets = [ex[target_lang] for ex in examples["translation"]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=max_target_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_datasets = raw_datasets.map(preprocess_function, batched=True, remove_columns=raw_datasets["train"].column_names)

# Load or create the model
try:
    model = TFAutoModelForSeq2SeqLM.from_pretrained("tf_model/")
except OSError:
    model = TFAutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

batch_size = 16
learning_rate = 2e-5
weight_decay = 0.01
num_train_epochs = 1
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, return_tensors="tf")
generation_data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, return_tensors="tf", pad_to_multiple_of=128)

train_dataset = model.prepare_tf_dataset(
    tokenized_datasets["train"],
    batch_size=batch_size,
    shuffle=True,
    collate_fn=data_collator,
)

validation_dataset = model.prepare_tf_dataset(
    tokenized_datasets["validation"],
    batch_size=batch_size,
    shuffle=False,
    collate_fn=data_collator,
)

generation_dataset = model.prepare_tf_dataset(
    tokenized_datasets["validation"],
    batch_size=8,
    shuffle=False,
    collate_fn=generation_data_collator,
)

optimizer = AdamWeightDecay(learning_rate=learning_rate, weight_decay_rate=weight_decay)
model.compile(optimizer=optimizer)

# Train the model if it's not already trained
try:
    model = TFAutoModelForSeq2SeqLM.from_pretrained("tf_model/")
except OSError:
    model.fit(train_dataset, validation_data=validation_dataset, epochs=num_train_epochs)
    model.save_pretrained("tf_model/")

# Streamlit app
st.title("English-Hindi Translation")

# Get user input
input_text = st.text_area("Enter English text to translate:", height=200)

if st.button("Translate"):
    # Tokenize input text
    tokenized = tokenizer([input_text], return_tensors='np')
    
    # Generate translation
    output = model.generate(**tokenized, max_length=128)
    
    # Decode the output
    with tokenizer.as_target_tokenizer():
        translation = tokenizer.decode(output[0], skip_special_tokens=True)
    
    st.success(f"Translated text: {translation}")