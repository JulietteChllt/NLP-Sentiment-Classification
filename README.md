# Tweet Sentiment Classification

This projects explores different ways to perform sentiment analysis on natural text data extracted from Twitter.  The data are tweets that had a positive or a negative emoji, the aim is to recover this information. We are also given some training data. All data can be downloaded at this [address](https://www.aicrowd.com/challenges/epfl-ml-text-classification).

##Libraries

* [Nltk] - Download nltk packages
'''$ python
$ >>> import nltk
$ >>> nltk.download()'''

* [Wordninja] - Install wordninja library
'''$ pip install worldninja'''

* [Gingerit] - Install gingerit library
'''$ pip install gingerit'''

* [Ekphrasis] - Install ekphrasis library
'''$ pip3 install ekphrasis'''

* [Pattern] - Install Pattern library
'''$ pip install pattern'''

* [Gensim] - Install Gensim library
'''$ conda install gensim'''

* [Tensorflow] -Install tensorflow library
'''pip install tensorflow'''

* [GloVe] - Install glove python binary library
'''$ pip install glove-python-binary'''

* [Keras] - Install keras library
'''$ pip install keras'''

* [Seaborn] - Install seaborn library
'''$ pip install seaborn'''

## Processing

A crucial step of the classification is to preprocess the raw tweets given as data. This is done in the [pre_processing](helper/pre_processing.py) file. The two functions *get_pre_process_data* and *get_pre_process_data_test* can perform these different task : 

- Replace the punctuation.
- Remove the letter repetitions.
- Replace the emoji by words, either *positive*, *negative*, or *love*.
- Remove the hashtags of tweets and separate the following words.
- Remove the apostrophe's contraction for a better analysis by the machine learning models.
- Replace the slang, very common in tweets.
- Get rid of meaningless short words and stop words.
- Correct the spelling mistakes.
- Remove the numbers.
- Stemming, reducing words to their root form.
- Lemmatisation, grouping inflected forms of a word.
- Adding the word *positive* (*negative*) in front of words with a positive (negative) meaning.

The processed data will be saved in a file. This will save computational time as some of those task can be really costly.



## Vector representation

One important step in natural language processing is the representation of words. Machine learning algorithm can not perform directly on natural language. There is exist different methods for words embedding, we explored the **GloVe** algorithm in the [glove_algo](helper/glove_algo.ipynb) file. You have the choice to run the algorithm with a pre-trained model from the glove database and fine tuned with our datasets, or just construct a model with only our data. This method gave us a result of 73.2%.

## Classic machine learning

We implemented some classical machine learning algorithm on our preprocessed data using the library **sklearn**. You can run these following algorithm in the [ML_implementations](helper/ML_implementations.ipynb) file :

- Logistic regression.
- Support Vector Machine (SVM).
- SVM with L1_based feature selection.
- Multinomial Naive Bayes.
- Bernouilli Naive Bayes.
- Ridge Classifier.
- AdaBoost.
- Perceptron.
- Passive-Aggresive.
- Nearest Centroid.

Our best results were with the *Logistic Regression* with 85.4% accuracy on the test dataset on AIcrowd.

## Neural networks

Deep and large pre-trained language models are suitable for various natural language processing tasks. In this part of the project we took advantage of existing models and did transfer learning. We tested two models in the []() file and the CNN LSTM in the [cnn_algo](helper/cnn_algo.ipynb)  :

1. DistilBERT is a transformer model, smaller and faster that BERT.
   - Base model (uncased), re-trained distilBert with 4 layers that gave us 82.5%.
   - Base model (cased), re-trained distilBert with 4 layers and keeping the capital letters, with a result of 85%.
2. XtremeDistil is a distilled task-agnostic transformer model that leverages task transfer for learning a small universal model that can be applied to arbitrary tasks and languages.
   - XtremeDistil-l6-h384 re-trained with 2 linear layers and Relu as activation functions, with a result of 88.9%.
   - XtremeDistil-l12-h384 reached 89.5%.
3. CNN Long Short-Term Memory (LSTM)  is an architecture designed for prediction problems on spatial inputs, this is why we use an embedding layer. This algorithm gives us 83%.

## Run script



