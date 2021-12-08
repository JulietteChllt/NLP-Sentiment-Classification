"""
Import the necessary libraries 
"""
import os
import csv
import numpy as np
import pickle
import pattern
from pattern.en import lemma, lexeme
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer
import nltk as nltk
import wordninja as wn
import matplotlib as plt
import seaborn as sns
import re
from ekphrasis.classes.spellcorrect import SpellCorrector
from gingerit.gingerit import GingerIt


"""
Import the necessary datasets : 
positive and negative training sets and 
positive and negative word dictionnaries
"""
PATH_TRAIN_NEG = '../Resources/train_neg.txt'
PATH_TRAIN_POS = '../Resources/train_pos.txt'

PATH_TRAIN_NEG_FULL = '../Resources/train_neg_full.txt'
PATH_TRAIN_POS_FULL = '../Resources/train_pos_full.txt'

PATH_DICT_POS = '../Resources/positive-words.txt'
PATH_DICT_NEG = '../Resources/negative-words.txt'

with open(PATH_TRAIN_POS) as f:
    train_pos = f.read().splitlines()
with open(PATH_TRAIN_NEG) as f:
    train_neg = f.read().splitlines()
with open(PATH_DICT_POS, encoding="ISO-8859-1") as f:
    POSITIVE_WORDS_LIST = set((x.strip() for x in f.readlines()))
with open(PATH_DICT_NEG, encoding="ISO-8859-1") as f:
    NEGATIVE_WORDS_LIST = set((x.strip() for x in f.readlines()))


"""
Download english stopwords library from the nltk package
Initialise stemmer using nltk PorterStemmer function 
Initialise lemmatizer using nltk WordNetLemmatizer function
"""
nltk.download('stopwords')
nltk.download('wordnet')
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()


"""
argument : tweet_list (list of strings)
Replace consecutive punctiation by a specific codeword (consecutive...)
Delete all other punctuation, consecutive spaces or special characters
"""


def replace_ponctuation(tweet_list):

    for i, tweet in enumerate(tweet_list):
        # replace multiple stops by the word 'consecutivestop'
        tweet = re.sub(r"(\.)\1+", ' consecutiveStop ', tweet)
        # replace multiple exclamation by the word 'consecutivequestion'
        tweet = re.sub(r"(\?)\1+", ' consecutiveQuestion ', tweet)
        # replace multiple exclamation by the word 'consecutiveexclamation'
        tweet = re.sub(r"(\!)\1+", ' consecutiveExclamation ', tweet)
        # delete all ponctuaction
        tweet = re.sub(r"[,.;@?!&$\\*\"]+\ *", ' ', tweet)
        # deleting consecutive spaces
        tweet = re.sub(r"\s+", ' ', tweet)
        tweet_list[i] = tweet

    return tweet_list


"""
argument : tweet_list (list of strings)
Remove characters repeted consecutively more than twice to leave only 2 consecutive letters
ex : heeeeeeeey -> heey
"""


def letter_repetition_treatment(tweet_list):

    for i, tweet in enumerate(tweet_list):
        tweet = re.sub(r'(.)\1+', r'\1\1', tweet)
        tweet_list[i] = tweet

    return tweet_list


"""
argument : tweet_list (list of strings)
Replaces specific emojies with the word "positive" or the word "negative"
"""


def emoji_treatment(tweet_list):

    for i, tweet in enumerate(tweet_list):
        # Sad -- :-(, : (, :(, ):, )-:
        tweet = re.sub(r'(:\s?\(|:-\(|\)\s?:|\)-:)', ' negative ', tweet)
        # Cry -- :,(, :'(, :"(
        tweet = re.sub(r'(:,\(|:\'\(|:"\()', ' negative ', tweet)
        # Smile -- :), : ), :-), (:, ( :, (-:, :')
        tweet = re.sub(r'(:\s?\)|:-\)|\(\s?:|\(-:|:\'\))', ' positive ', tweet)
        # Laugh -- :D, : D, :-D, xD, x-D, XD, X-D
        tweet = re.sub(r'(:\s?D|:-D|x-?D|X-?D)', ' positive ', tweet)
        # Love -- <3, :*
        tweet = re.sub(r'(<3|:\*)', ' love ', tweet)
        # Wink -- ;-), ;), ;-D, ;D, (;,  (-;
        tweet = re.sub(r'(;-?\)|;-?D|\(-?;)', ' positive ', tweet)
        tweet_list[i] = tweet

    return tweet_list


"""
argument : tweet_list (list of strings)
Remove the '#' sign for each string and split the words in the hashtag into the most likely combination of words
ex : #Machinelearningisthefuture -> Machine learning is the future
"""


def hashtag_treatment(tweet_list):

    for i, tweet in enumerate(tweet_list):
        tweet = np.array(tweet.split(), dtype='object')
        for word in tweet:
            if '#' in word:
                index = np.where(tweet == word)
                word = " ".join(wn.split(word))
                if (isinstance(tweet, str)):
                    tweet.replace('#', '')
                else:
                    np.put(tweet, index[0][0], word)
                tweet = " ".join(tweet)
                tweet_list[i] = tweet

    return tweet_list


"""
argument : tweet_list (list of strings)
Replace current expression in their contracted form with their full form
ex : I've gotta go -> I have got to go 
"""


def apostrophe_contraction(tweet_list):

    contractions = {
        '\'m': ' am',
        'im': ' I am',
        'ive': 'I have',
        '\'re': ' are',
        '\'ve': ' have',
        '\'s': ' is',
        '\'ll': ' will',
        '\'d': ' would',
        '\'t': ' not',
        'ain\'t': 'not',
        'aint': 'not',
        'can\'t': 'can not',
        'cant': 'can not',
        'don\'t': 'do not',
        'dont': 'do not',
        'isn\'t': 'is not',
        'isnt': 'is not',
        'won\'t': 'will not',
        'wont': 'will not',
        'shouldn\'t': 'should not',
        'shouldnt': 'should not',
        'couldn\'t': 'could not',
        'wouldn\'t': 'would not',
        'aren\'t': 'are not',
        'arent': 'are not',
        'doesn\'t': 'does not',
        'doesnt': 'does not',
        'wasn\'t': 'was not',
        'wasnt': 'was not',
        'weren\'t': 'were not',
        'werent': 'were not',
        'hasn\'t': 'has not',
        'haven\'t': 'have not',
        'havent': 'have not',
        'hadn\'t': 'had not',
        'mustn\'t': 'must not',
        'didn\'t': 'did not',
        'mightn\'t': 'might not',
        'needn\'t': 'need not',
        'imma': 'i am going to',
        'wanna': 'want to',
        'gonna': 'going to',
        'gotta': 'got to',
        'thats': 'that is',
    }
    pat = re.compile(r"\b(%s)\b" % "|".join(contractions))
    tweet_list = [pat.sub(lambda m: contractions.get(
        m.group()), tweet.lower()) for tweet in tweet_list]

    return [re.sub(r"\'", ' ', tweet) for tweet in tweet_list]


"""
argument : tweet_list (list of strings)
Replace current slang words with their meaning using our mapping 
ex : 'cya 2nite gurl' -> 'see you tonight girl' 
"""


def correct_slang(tweet_list):

    slang = {
        '2nite': 'tonight',
        '2night': 'tonight',
        '2': 'to',
        '4': 'for',
        'ab': 'about',
        'ace': 'success',
        'ad': 'awesome person',
        'af': 'very',  # mmmh could do better : word af  -> very word maybe ?
        'aka': 'meaning',
        'asap': 'soon',
        'aww': 'cute',
        'bc': 'because',
        'bf': 'boyfriend',
        'bff': 'best friend',
        'brb': 'I come',
        'btr': 'better',
        'btw': 'by the way',
        'cus': 'because',
        'cuz': 'because',
        'cya': 'see you',
        'da': 'the',
        'dammit': 'damn it',
        'dam': 'damn',
        'der': 'there',
        'dm': 'message me',
        'dunno': 'do not know',
        'dnt': 'do not',
        'dw': 'okay',
        'ew': 'gross',
        'ftw': 'win',
        'fyi': 'for information',
        'gf': 'girlfriend',
        'gotta': 'has',
        'gurl': 'girl',
        'haha': 'laught',
        'hahah': 'laught',
        'hahaha': 'laught',
        'hahahah': 'laught',
        'hahahaha': 'laught',
        'hmu': 'message me',
        'idk': 'do not know',
        'idc': 'do not care',
        'ily': 'love',
        'imo': 'think',
        'irl': 'real life',
        'jk': 'laught',
        'lmao': 'laught',
        'lmk': 'let me know',
        'lil': 'little',
        'lol': 'laught',
        'luv': 'love',
        'ppl': 'people',
        'morn': 'morning',
        'n': 'and',
        'nbd': 'okay',  # no big deal
        'np': 'okay',  # no problem
        'nvm': 'okay',  # never mind
        'omg': 'amazing',  # oh my god
        'omw': "come",
        'r': 'are',
        'rofl': 'laught',
        'roflmao': 'laught',
        'rn': 'now',
        'rt': 'retweet',
        'sch': 'school',
        'tbh': 'honestly',
        'til': 'until',
        'thx': 'thanks',
        'ttyl': 'talk later',
        'u': 'you',
        'ull': 'you will',
        'ur': 'your',
        'w': 'with',
        'wan': 'want',
        'waz': 'what is',
        'wtf': 'seriously',
        'wud': 'would',
        'x': 'kiss',
        'xx': 'kiss',
        'xo': 'kiss',
        'xoxo': 'kiss',
        'xd': 'laught',
        'y': 'why',
        'ya': 'you',
        'yay': 'happy',
        'yolo': 'enjoy',
        'yuck': 'gross',
    }
    pat = re.compile(r"\b(%s)\b" % "|".join(slang))

    return [pat.sub(lambda m: slang.get(m.group()), tweet.lower()) for tweet in tweet_list]


"""
argument : tweet_list (list of strings)
Replace current slang words with their meaning using gingerIt package 
"""


def correct_slang2(tweet_list):

    parser = GingerIt()
    for i, tweet in enumerate(tweet_list):
        if (tweet and len(tweet) < 300):
            t = parser.parse(tweet)
            tweet_list[i] = t.get('result')

    return tweet_list


"""
argument : tweet_list (list of strings)
Remove words that have length one 
"""


def short_word_treatment(tweet_list):

    for i, tweet in enumerate(tweet_list):
        tweet = " ".join([word for word in tweet.split() if len(word) > 1])
        tweet_list[i] = tweet

    return tweet_list


"""
argument : tweet_list (list of strings)
Remove numbers and possible remaining punctuation / special characters
"""


def numbers_treatment(tweet_list):

    for i, tweet in enumerate(tweet_list):
        new_tweet = []
        for word in tweet.split():
            try:
                word = re.sub('[,\.:%_\-\+\*\/\%\_]', '', word)
                float(word)
                new_tweet.append("")
            except:
                new_tweet.append(word)
            tweet_list[i] = " ".join(new_tweet)

    return tweet_list


"""
argument : tweet_list (list of strings)
Replace words that are not uniquely composed of alphabetic characters (ie contain numbers or special characters) 
"""


def non_alphabetic_treatment(tweet_list):

    for i, tweet in enumerate(tweet_list):
        tweet = " ".join([word for word in tweet.split() if word.isalpha()])
        tweet_list[i] = tweet

    return tweet_list


"""
argument : tweet_list (list of strings)
Use spell corrector dictionnary from ekphrasis package to correct spelling mistakes
"""


def correct_spelling(tweet_list):

    sp = SpellCorrector(corpus="english")

    return [sp.correct_text(tweet) for tweet in tweet_list]


"""
argument : tweet_list (list of strings)
Remove english stopwords excluding the one that may help understand the polarity of the tweet
ex : this processing step is an important step -> proccessing step important step
"""


def stopwords_treatment(tweet_list):

    stop_words = set(stopwords.words('english'))
    stop_words.difference_update(
        ['no', 'not', 'but', 'why', 'won', 'won\'t', 'very', 'don\'t', 'against'])

    return [" ".join(w for w in tweet.split() if not w in stop_words) for tweet in tweet_list]


"""
argument : tweet_list (list of strings)
Remove suffix of words 
ex : studies study studying studied -> studi studi studi studi
"""


def stemming_treatment(tweet_list):

    for i, tweet in enumerate(tweet_list):
        tweet = " ".join(stemmer.stem(t) for t in tweet.split())
        tweet_list[i] = tweet

    return tweet_list


# try with wordNet but not that great
"""
argument : tweet_list (list of strings)
Replace words with the root of the word 
ex : studies study studying studied -> study study studying studied
"""


def lemmatizing_treatment(tweet_list):

    for i, tweet in enumerate(tweet_list):
        tweet = " ".join(lemmatizer.lemmatize(t) for t in tweet.split())
        tweet_list[i] = tweet

    return tweet_list


# now with Pattern lemmatizer : seems to work great !
"""
argument : tweet_list (list of strings)
Replace words with the root of the word 
ex : studies study studying studied -> study study study study
"""


def lemmatizing_treatment2(tweet_list):

    for i, tweet in enumerate(tweet_list):
        new_tweet = " ".join([lemma(word) for word in tweet.split()])
        tweet_list[i] = new_tweet

    return tweet_list


"""
argument : tweet_list (list of strings)
Use dictionnaries to identify positive and negative words in string,
append the word "positive" or "negative" to the word in question in the string
ex : great zombie are here -> positive great negative zombie are here
"""


def negative_positive_word_treatment(tweet_list):

    def check_word(word):

        if word in POSITIVE_WORDS_LIST:
            return "positive " + word
        elif word in NEGATIVE_WORDS_LIST:
            return "negative " + word

        return word

    return [" ".join(check_word(w) for w in tweet.split()) for tweet in tweet_list]


"""
argument : booleans to specify which process function to call, and a name to save it
Will load the raw dataset and will process it with the appropriate functions.
"""


def get_pre_process_data(positive=True, full=False, ponctuation=True, letter_repetition=True,
                         emoji=True, hashtag=True, apostroph=True, slang=True, slang2=False,
                         short_word=True, numbers=True, spelling=False, alphabetic=True,
                         stopwords=True, stemming=False, lemmatizing=False, lemmatizing2=True, neg_pos_word=True, save_file_name=""):

    # file already exists
    if save_file_name != "":
        PATH = '../Resources/' + save_file_name
        if os.path.exists(PATH):
            with open(PATH) as f:
                data = csv.reader(f, quoting=csv.QUOTE_ALL)
                return list(data)

    # load the raw data set
    if positive:
        if full:
            with open(PATH_TRAIN_POS_FULL) as f:
                data = f.read().splitlines()
        else:
            with open(PATH_TRAIN_POS) as f:
                data = f.read().splitlines()
    else:
        if full:
            with open(PATH_TRAIN_NEG_FULL) as f:
                data = f.read().splitlines()
        else:
            with open(PATH_TRAIN_NEG) as f:
                data = f.read().splitlines()

    # call the process functions
    if ponctuation:
        data = replace_ponctuation(data)
    if letter_repetition:
        data = letter_repetition_treatment(data)
    if emoji:
        data = emoji_treatment(data)
    if hashtag:
        data = hashtag_treatment(data)
    if apostroph:
        data = apostrophe_contraction(data)
    if slang:
        data = correct_slang(data)
    if slang2:
        data = correct_slang2(data)
    if short_word:
        data = short_word_treatment(data)
    if numbers:
        data = numbers_treatment(data)
    if alphabetic:
        data = non_alphabetic_treatment(data)
    if spelling:
        data = correct_spelling(data)
    if stopwords:
        data = stopwords_treatment(data)
    if stemming:
        data = stemming_treatment(data)
    if lemmatizing:
        data = lemmatizing_treatment(data)
    elif lemmatizing2:
        data = lemmatizing_treatment2(data)
    if neg_pos_word:
        data = negative_positive_word_treatment(data)

    # save data in file
    if save_file_name != "":
        with open(PATH, 'w') as myfile:
            wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
            wr.writerow(data)

    return data


"""
argument : train_pos, train_neg (training datasets)
add second column to training sets to identify tweets from dataset :) (1) and from dataset :( (-1)
"""


def label_data(train_pos, train_neg):

    train_pos = np.array(train_pos).reshape(-1, 1)
    ones = np.ones(shape=(train_pos.shape[0], 1))
    train_pos = np.concatenate((train_pos, ones), axis=1)

    train_neg = np.array(train_neg).reshape(-1, 1)
    neg_ones = np.zeros(shape=(train_neg.shape[0], 1))-1
    train_neg = np.concatenate((train_neg, neg_ones), axis=1)

    return (train_pos, train_neg)
