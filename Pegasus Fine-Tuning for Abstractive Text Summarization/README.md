# Pegasus Fine-Tuning for Abstractive Text Summarization

This repository contains code for fine-tuning the Pegasus model on the SAMSum dataset for abstractive text summarization.

## Overview

Pegasus is a state-of-the-art model developed by Google Research for abstractive text summarization. This project demonstrates how to fine-tune the Pegasus model on the SAMSum dataset, which contains dialogues along with human-written summaries.

## Technologies Used

- NVIDIA GPU
- Hugging Face Transformers
- Datasets Library
- PyTorch
- NLTK
- Matplotlib
- Py7zr
- Accelerate
- TensorFlow (used indirectly by Hugging Face libraries)

# Pegasus Fine-Tuning for Abstractive Text Summarization

This Python script fine-tunes and evaluates a Pegasus model for abstractive text summarization on the SAMSum dataset. It utilizes several natural language processing (NLP) libraries and tools.

## Key Technologies and Components Used

- **NVIDIA GPU**: Leverages GPU acceleration for faster computation.
- **Hugging Face Transformers**: Used for working with pre-trained transformer models.
- **Datasets Library**: Loads and preprocesses datasets, including SAMSum.
- **PyTorch**: Backend for training and using transformer models.
- **NLTK**: Tokenization using `sent_tokenize`.
- **Matplotlib**: Visualization purposes.
- **Py7zr**: Handling 7z compressed files.
- **Accelerate**: Library for distributed computing in PyTorch.
- **TensorFlow**: Indirectly used as a dependency for Hugging Face libraries.

## Workflow Overview

1. **Installation**: Installs necessary packages and dependencies.
2. **Model and Data Loading**: Loads the Pegasus model and tokenizer from Hugging Face's model hub and loads the SAMSum dataset.
3. **Data Preparation**: Tokenizes text inputs and generates features for training.
4. **Training Setup**: Defines training arguments and sets up the Trainer object for fine-tuning the model.
5. **Model Training**: Trains the model on the SAMSum dataset.
6. **Evaluation**: Defines functions for calculating evaluation metrics, particularly ROUGE scores, on the test dataset. Evaluates the fine-tuned model and computes ROUGE scores.
7. **Model Saving**: Saves the fine-tuned model and tokenizer for later use.
8. **Summarization**: Generates summaries for sample text inputs and compares them with reference summaries.

## Usage

1. Clone the repository:

git clone https://github.com/adityach_007/pegasus-fine-tuning.git

cd pegasus-fine-tuning


4. View the results in the console output and any generated plots.

## Results

The script will fine-tune the Pegasus model on the SAMSum dataset and evaluate it using ROUGE scores. The results will be displayed in the console output and may include metrics such as ROUGE-1, ROUGE-2, ROUGE-L, and ROUGE-Lsum.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to fork the repository and submit pull requests.

## References

- [Pegasus: Pre-training with Extracted Gap-sentences for Abstractive Summarization](https://arxiv.org/abs/1912.08777)
- [SAMSum Dataset](https://arxiv.org/abs/1908.07898)
- [Hugging Face Transformers Documentation](https://huggingface.co/transformers/)
- [NLTK Documentation](https://www.nltk.org/)
- [PyTorch Documentation](https://pytorch.org/)
- [Matplotlib Documentation](https://matplotlib.org/)

