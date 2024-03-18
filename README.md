# Tweet-Similarity-Analysis-with-Transformer-Embeddings
######This project aims to perform text similarity analysis using Word2Vec embeddings.
## Introduction
###### This project utilizes pre-trained Word2Vec embeddings to calculate text similarity between pairs of texts. It includes preprocessing steps such as tokenization, removing stopwords, and lemmatization before computing the similarity scores.
## Installation
###### To run this project, ensure you have the necessary dependencies installed. You can install the required Python packages using pip:

###### Copy code
#### pip install pandas numpy nltk scikit-learn gensim
###### Additionally, download the NLTK tokenizer resource by running the following command:

#### import nltk
#### nltk.download('punkt')
Usage
Clone the repository:
bash
Copy code
git clone https://github.com/your-username/project-name.git
Navigate to the project directory:
bash
Copy code
cd project-name
Run the Python script:
Copy code
python text_similarity_analysis.py
Dataset
The project uses two Excel files as datasets:

train.xlsx: Contains the training data.
test.xlsx: Contains the test data.
Ensure the datasets are formatted correctly with the required columns (text, user, etc.) before running the code.

Results
The project evaluates the text similarity model using precision, recall, and F1 score metrics on both the training and test datasets. The results are displayed in the console after running the script.

License
This project is licensed under the MIT License.

