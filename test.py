import streamlit as st
import transformers
import tensorflow as tf
from datasets import load_dataset
from transformers import AutoTokenizer, TFAutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, AdamWeightDecay
import base64
from gtts import gTTS
import time
from io import BytesIO

@st.cache
def load_model(target_lang):
    if target_lang == "hi":
        model_checkpoint = "Helsinki-NLP/opus-mt-en-hi"
    elif target_lang == "ru":
        model_checkpoint = "Helsinki-NLP/opus-mt-en-ru"
    else:
        raise ValueError(f"Invalid target language: {target_lang}")

    raw_datasets = load_dataset("cfilt/iitb-english-hindi")
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    max_input_length = 128
    max_target_length = 128
    source_lang = "en"

    def preprocess_function(examples):
        inputs = [ex[source_lang] for ex in examples["translation"]]
        targets = [ex[target_lang] for ex in examples["translation"]]
        model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)
        # Setup the tokenizer for targets with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=max_target_length, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_datasets = raw_datasets.map(preprocess_function, batched=True, remove_columns=raw_datasets["train"].column_names)

    # Load or create the model
    try:
        model = TFAutoModelForSeq2SeqLM.from_pretrained("tf_model/")
    except OSError:
        model = TFAutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

    return model, tokenizer

def text2Speech(data, lang):
    tts = gTTS(text=data, lang=lang, slow=False)
    mp3_fp = BytesIO()
    tts.write_to_fp(mp3_fp)
    mp3 = mp3_fp.getvalue()
    return mp3

def app():
    st.title("English to Hindi/Russian Translation")

    # Get target language from user
    target_lang = st.selectbox("Select target language", ["hi", "ru"])

    # Get user input
    input_text = st.text_area("Enter English text to translate:", height=200)

    if st.button("Translate"):
        # Load the model and tokenizer
        model, tokenizer = load_model(target_lang)

        # Tokenize input text
        tokenized = tokenizer([input_text], return_tensors='np')

        # Generate translation
        output = model.generate(**tokenized, max_length=128)

        # Decode the output with tokenizer.as_target_tokenizer():
        translation = tokenizer.decode(output[0], skip_special_tokens=True)
        st.success(f"Translated text: {translation}")

        # Convert text to speech after 2 seconds
        def play_audio(audio_data):
            st.audio(audio_data, format='audio/mp3', start_time=0)

        time.sleep(2)  # Wait for 2 seconds
        audio_lang = "hi" if target_lang == "hi" else "ru"
        audio_data = text2Speech(translation, audio_lang)
        play_audio(audio_data)

if __name__ == '__main__':
    app()
