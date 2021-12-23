# Tweet Sentiment Classification

This projects explores different ways to perform sentiment analysis on natural text data extracted from Twitter.  The data are tweets that had a positive or a negative emoji, the aim is to recover this information. We are also given some training data. All data can be downloaded at this [address](https://www.aicrowd.com/challenges/epfl-ml-text-classification).

## Libraries

* [Nltk](https://www.nltk.org/ "Nltk") - Download nltk packages
```$ python
   $ import nltk
   $ nltk.download()
   ```
   
* [Wordninja](https://github.com/keredson/wordninja "git wordninja") - Install wordninja library  
```$ pip install worldninja```

* [Gingerit](https://github.com/Azd325/gingerit "git gingerit") - Install gingerit library  
```$ pip install gingerit```

* [Ekphrasis](https://github.com/cbaziotis/ekphrasis "ekphrasis") - Install ekphrasis library  
```$ pip3 install ekphrasis```

* [Pattern](https://github.com/clips/pattern "Pattern") - Install Pattern library  
```$ pip install pattern```

* [Gensim](https://github.com/RaRe-Technologies/gensim "gensim") - Install Gensim library  
```$ conda install gensim```

* [Tensorflow](https://www.tensorflow.org/ "tensor") -Install tensorflow library  
```pip install tensorflow```

* [GloVe](https://github.com/stanfordnlp/GloVe "glove") - Install glove python binary library  
```$ pip install glove-python-binary```

* [Keras](https://keras.io/ "keras") - Install keras library  
```$ pip install keras```

* [Seaborn](https://github.com/mwaskom/seaborn "seaborn") - Install seaborn library  
```$ pip install seaborn```

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
- Adding the word *positive* (*negative*) in front of words with a positive (negative) meaning. (see [dictionnary processing](helper/dictionnaryProcessing.ipynb) file to see how the dictionaries were prepared and [dictionaries unprocessed](Resources/initialDictionnaries) and [dictionaries processed](Resources/Dict_pos_neg) to see the dictionaries before and after processing).
- Using ekphrasis Text preprocessor

The processed data will be saved in a file. This will save computational time as some of those task can be really costly.



## Vector representation

One important step in natural language processing is the representation of words. Machine learning algorithm can not perform directly on natural language. There is exist different methods for words embedding, we explored several including Word2Vector and **GloVe**, all methods can be found in the [word Embedding](helper/WordEmbeddings.ipynb) file alors with our test on glove with a CNN alogrithm. For glove you have the choice to run the algorithm with a pre-trained model from the glove database and fine tuned with our datasets, or construct a model with only our data. The first option yield sligthly better results. Overall glove gave us results around 82%.

## Classic machine learning

We implemented some classical machine learning algorithms on our preprocessed data using the library **sklearn**. You can find and these algorithms in the [ML_implementations](helper/ML_implementations.ipynb) file. The algorithms implemented are :

- Logistic regression.
- Support Vector Machine (SVM) both with l1 and l2 norm.
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
We provide a run.py that re-train our Transformers Neural Networks algorithms with the full data set in Ressources.
The script requires two arguments (transformers_model_name,transformers_tokenizer_name), by default it would execute our best performing model XtremeDistil-l12-h384.
If arguments are provided please consider modifying those parameters in order to achieve best results (i.e best accuracy and speed) :


|                model             | number epochs  | learning_rate |
|----------------------------------|----------------|---------------|
|  DistilBERT base model (uncased) |        3       |     1.e-04    |
|  DistilBERT base model (cased)   |        3       |     1.e-04    |
|      XtremeDistil-l6-h384        |        5       |     5.e-05    |
|      XtremeDistil-l12-h384       |        3       |     5.e-05    |



⚠️ If you want to skip training phase and use already our best already transfer learned model ( modified XtremeDistil-l12-h384), you can directly load our provided model (ModifiedXtremeDistil-l12-h384_full.pth) to compute your predictions.



