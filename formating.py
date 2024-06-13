import re, string
from nltk.corpus import stopwords

#Remove punctuations, links, mentions and \r\n new line characters
def strip_all_entities(text): 
    text = text.replace('\r', '').replace('\n', ' ').lower() #remove \n and \r and lowercase
    text = re.sub(r"(@\S+|https?://\S+)", "", text)
    text = re.sub(r"\s+", " ", text).lstrip()
    text = re.sub(r'[^\x00-\x7f]',r'', text) #remove non utf8/ascii characters such as '\x9a\x91\x97\x9a\x97'
    text = re.sub(r'^RT\s+', '', text, flags=re.IGNORECASE) #remove retweet key word "rt" from the beggining of the tweet
    banned_list= string.punctuation + 'Ã'+'±'+'ã'+'¼'+'â'+'»'+'§'
    table = str.maketrans('', '', banned_list)
    text = text.translate(table)
    return text

# remove multiple spaces
def remove_mult_spaces(text): 
    return re.sub("\s\s+" , " ", text)

# remove stop words
def remove_stopwords(text):
    stop_words = stopwords.words('english')
    more_stopwords = ['u', 'im', 'c']
    stop_words = stop_words + more_stopwords
    text = ' '.join(word for word in text.split(' ') if word not in stop_words)
    return text

def clean_text_plus(tweets):
    tweets = tweets.apply(strip_all_entities)
    tweets = tweets.apply(remove_stopwords)
    return tweets 

def clean_text(tweets):
    tweets = tweets.apply(strip_all_entities)
    return tweets 