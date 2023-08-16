# NLP
What is NLP?  
NLP = natural learning Programmin
# Day 1
## Libraries Download
+ nltk : `!pip install nltk`
+ spacy:  `!pip install spaCy`
+ Gensim:  `!pip install gensim`
+ scikit-learn or sklearn: `!pip install scikit-learn` 
+ Tensorflow: `!pip install tensorflow`
+ Keras: `!pip install keras`
+ Torch and Torchvision: `!pip install torch torchvision`
+ Transformer (Hugging Face): `!pip install transformer`
### Libraries documentation: 
+ https://brew.sh/
+ https://chocolatey.org/
+ NLTK https://www.nltk.org/
+ https://wordnet.princeton.edu/
+ spaCy https://spacy.io/
+ Gensim https://radimrehurek.com/gensim/
+ Scikit-learn https://scikit-learn.org/stable/getting_started.html
+ TensorFlow https://www.tensorflow.org/learn
+ Keras https://keras.io/
+ PyTorch https://pytorch.org
+ Hugging Face transformers https://huggingface.co/docs/transformers/main/en/index
+ JAX https://jax.readthedocs.io/en/latest/notebooks/quickstart.html
# Day 2
## you can access dataset from these website
+ https://archive.ics.uci.edu/ml/datasets.php
+ https://snap.stanford.edu/data/web-Amazon.html

+ https://dumps.wikimedia.org/

+ https://nlp.stanford.edu/sentiment/index.html

+ https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment

+ https://paperswithcode.com/dataset/standardized-project-gutenberg-corpus

+ https://www.cs.cmu.edu/~enron/

+ https://www.kaggle.com/rtatman/blog-authorship-corpus

+ https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection

+ https://cseweb.ucsd.edu/~jmcauley/datasets.html

+ https://wordnet.princeton.edu/download

+ https://github.com/nproellochs/SentimentDictionaries

+ http://help.sentiment140.com/for-students/

+ https://www.cs.jhu.edu/~mdredze/datasets/sentiment/

+ https://www.yelp.com/dataset

+ http://qwone.com/~jason/20Newsgroups/

+ https://www.microsoft.com/en-us/download/details.aspx?id=52419

+ https://www.statmt.org/europarl/

+ http://kavita-ganesan.com/entity-ranking-data/#.Yw1NsuzMKXj

+ https://archive.ics.uci.edu/ml/datasets/Legal+Case+Reports

+ https://rajpurkar.github.io/SQuAD-explorer/

+ https://catalog.ldc.upenn.edu/LDC93s1

+ https://www.imdb.com/interfaces/

+ https://www.reddit.com/r/datasets/comments/1uyd0t/200000_jeopardy_questions_in_a_json_file/
## Reading from a Word Document
+ To read the Word Document you need to install python-docx library
The whole step is shown in file name **RWordDoc.ipynb**
## Data Preprocessing in NLP
It involves
+ Data cleaning and preprocessing
+ Visualization
+ Data Augmentation
+ Distance metrics
+ Modeling
+ Model Evaluation
### Segmentation
While running Segmentation.ipynb file from repo , run this command in bash : python -m spacy download en_core_web_sm
### WOrd tokenization 
YOu will find the repo by name of WordTokenization.ipynb   
Word Tokenization can be done by three different library textblob, Nltk, Spacy .. you will find the example code in the repo
### Part of Speech Tagging (Day 7)
You will find the code in POS.ipynb  
Part Of speech is also known as POS tagging
it might be desired to retain only certain parts of speech, such as nouns. The use cases can be cleaning data before creating a word-counts (bag-of-words) model or further processing that depends on parts of speech, such as named entity recognition (where two nouns occurring together are likely first and last names of a person) and keyphrase extraction.
### N-grams
**You will find N-gram code in N-gram.ipynb file**
N-grams are a contiguous Sequence of N elements. For instance "Natural", "Language" and "processing" are unigrams,"natural language" and "language processing" are bigrams, and "natural language processing" is the trigram of the strirng "natural language processing"  
In many NLP feature generation methods, each word in a sentence is used as an independent unit (token) while encoding data. Instead, getting multi word pairs from a sentence can be beneficial for certain application that contain multi-word keywords or sentiment analysis.
For example: "not happy" bigram versys 'happy' unigram can convet different sentiments for the sentence 'James is not happy'
## Cleaning
### Punctuation removal
or many applications such as category classification and word visualizations, the words used in the text matter and the punctuation does not have relevance to the application. Punctuation can be removed using a regex expression.  
In regex: 
+ \m matches a newline character
+ \w is a word character that matches any single letter number, or underscore (same as [a-zA-Z0-9])
+ \s is for matching white spaces
+ ^ is for matching with everything except the pattren specified  
**You will find Punctuation Removel code in PR.ipynb**
### URL removal (Day 10)
In language document URL removal can be beneficial in reducing overall text length and removing information that does not convet meaning for your application  
In regex, \s matches all whits-spaces character and \S matches with non white spcaed characters
+ | stands for OR and can be used when we want to match multiple pattren with the OR logic.
**You will find the code in url.ipynb file in this repo**
### Emoji Removal (Day 11)
Unicode is an international standard that maintains a mapping of individual character and unique number acrrross devices and programs.  
Each character is represented as a code point.   
These code point are encoded in bytes and can be decoded back to code points.
