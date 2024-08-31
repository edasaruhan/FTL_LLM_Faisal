
# Assignment 2: Text Generation and NLP Projects

## Overview

This repository contains code for a text generation application and a natural language processing project focused on text embeddings and classification. An optional semantic search engine is also included.

## Project Structure

- `text_generation_app.py`: Code for the text generation application using Streamlit.
- `text_classification.py`: Code for exploring text embeddings and performing text classification.
- `semantic_search.py`: Code for building a semantic search engine using FAISS.

## Setup Instructions

1. Clone this repository: `git clone <repository_url>`
2. Install the required dependencies: `pip install -r requirements.txt`
3. Run the text generation app: `streamlit run text_generation_app.py`
4. Run the text classification script: `python text_classification.py`
5. Run the semantic search engine: `python semantic_search.py`

## Usage

### Text Generation App

1. Launch the app using Streamlit.
2. Enter a prompt related to SDGs and click "Generate Text."
3. The generated text will be displayed below the input field.

### NLP Project

1. Run `text_classification.py` to explore text embeddings and perform classification.
2. Modify the script to change datasets or embedding types as needed.

### Semantic Search Engine

1. Run `semantic_search.py` and enter a search query.
2. The script will return the top 5 documents related to the query.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
