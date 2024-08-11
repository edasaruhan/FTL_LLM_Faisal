
# Sentiment Analysis Using Hugging Face

This project implements a sentiment analysis task using pre-trained models from the Hugging Face library, specifically BERT. The project aligns with the Sustainable Development Goal (SDG) 12: Responsible Consumption and Production.

## Project Structure
- `fine_tuning_script.py`: Python script for loading, fine-tuning, and evaluating the BERT model.
- `dataset/`: Directory for the dataset used in training.
- `results/`: Directory for storing the output of the fine-tuning process.
- `logs/`: Directory for logging the training process.
- `README.md`: This file.

## Steps to Run
1. Install the required libraries: `pip install transformers torch`
2. Run the Python script: `python fine_tuning_script.py`
3. Check the `results/` directory for the fine-tuned model.

## Dataset
The dataset used in this project consists of simple sentences with binary labels indicating positive (1) or negative (0) sentiment.

## Evaluation
The model is evaluated based on accuracy and F1-score before and after fine-tuning.

## License
This project is open-source under the MIT license.
