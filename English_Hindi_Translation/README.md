# English-Hindi Translation with Transformers and Streamlit

This repository contains code for a machine translation project that translates English text to Hindi. It leverages the capabilities of the Transformers library and Streamlit to create a web application for user interaction.

## Prerequisites

- Python 3.6 or later
- TensorFlow 2.x
- Transformers library
- Streamlit library
- A GPU is recommended for faster training and inference.

## Installation

1. Clone this repository.
2. Create a new virtual environment (recommended) and activate it.
3. Install the required libraries:

```bash
pip install transformers tensorflow datasets streamlit
```

# Model and Dataset

## This project utilizes the following:

- ### Model:
The pre-trained model used for translation is Helsinki-NLP/opus-mt-en-hi from the Transformers library. This model is a multilingual encoder-decoder architecture trained on a massive dataset of English-Hindi text pairs. It is specifically designed for translating between English and Hindi languages.
Dataset: The training data for the model (if you choose to train it) comes from the IITB English-Hindi dataset (`cfilt/iitb-english-hindi`), available from the Datasets library. This dataset is a collection of English sentences and their corresponding Hindi translations, commonly used for training machine translation models for the English-Hindi language pair.

- ### Dataset: 
The training data for the model (if you choose to train it) comes from the IITB English-Hindi dataset (`cfilt/iitb-english-hindi`), available from the Datasets library. This dataset is a collection of English sentences and their corresponding Hindi translations, commonly used for training machine translation models for the English-Hindi language pair.

# Training the Model (Optional)
The code includes a script for training a translation model using the pre-trained model `Helsinki-NLP/opus-mt-en-hi`. A pre-trained model is already included (tf_model/) for convenience, but you can retrain it if desired.

To train the model:

Run the following command in your terminal:
```bash
python test.py
```

This script will:

- Load the pre-trained model and tokenizer.
- Preprocess the IITB English-Hindi dataset (`cfilt/iitb-english-hindi`).
- Train the model for one epoch (you can adjust this parameter).
- Save the trained model to the `tf_model/ directory`.

# Running the Streamlit App

Make sure you have trained the model (or the `tf_model`/ directory exists).

Run the following command in your terminal:
```bash
streamlit run app.py
```

# Using the Streamlit App

- The app title is "English-Hindi Translation".
- Enter the English text you want to translate in the large text area box.
- Click the "Translate" button.
- The translated Hindi text will be displayed in a success message below the input box.

# Demo Video



https://github.com/adityach007/Gen_AI/assets/108794914/a8e32477-fdc3-4abc-ba3b-a83ffe97d7f8


