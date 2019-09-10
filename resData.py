import nltk
from flask import Flask, render_template, request
import numpy as np
import random
from nltk.corpus import stopwords
import string  # to process standard python strings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

f = open('tax_policy.txt', 'r', errors='ignore')
raw = f.read()
raw = raw.lower()  # converts to lowercase
stopWords = stopwords.words('english')

# nltk.download('punkt') # first-time use only
# nltk.download('wordnet') # first-time use only
sent_tokens = nltk.sent_tokenize(raw)  # converts to list of sentences
word_tokens = nltk.word_tokenize(raw)  # converts to list of words

nltk.download('stopwords', quiet=True, raise_on_error=True)
stop_words = set(nltk.corpus.stopwords.words('english'))
tokenized_stop_words = nltk.word_tokenize(' '.join(nltk.corpus.stopwords.words('english')))

lemmer = nltk.stem.WordNetLemmatizer()


# WordNet is a semantically-oriented dictionary of English included in NLTK.
def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]


remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)


def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))


# ------------------greetings match-------------------------------------------------------------
GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up", "hey",)
GREETING_RESPONSES = ["hi", "hey", "*nods*", "hi there", "hello", "I am glad! You are talking to me"]


def greeting(sentence):
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)


# --------------------------- getting the response---------------------------------------------------

def response(user_response):
    robo_response = ''
    # 'stop_words.' % sorted(inconsistent))
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx = vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if (req_tfidf == 0):
        robo_response = robo_response + "I am sorry! I don't understand you"
        return robo_response
    else:
        robo_response = robo_response + sent_tokens[idx]
        return robo_response


# ----------------------    training the bot waht to say when we interact-------------------------------------------------------------

flag = True
print("Bravo: My name is Bravo. I will answer your queries about Chatbots. If you want to exit, type Bye!")
while (flag == True):
    user_response = input()
    user_response = user_response.lower()
    if (user_response != 'bye'):
        if (user_response == 'thanks' or user_response == 'thank you'):
            flag = False
            print("Bravo: You are welcome..")
        else:
            if (greeting(user_response) != None):
                print("Bravo: " + greeting(user_response))
            else:
                sent_tokens.append(user_response)
                word_tokens = word_tokens + nltk.word_tokenize(user_response)
                final_words = list(set(word_tokens))
                print("Bravo: ", end="")
                print(response(user_response))
                sent_tokens.remove(user_response)
    else:
        flag = False
        print("Bravo: Bye! take care..")


@app.route('/home')
def index():
    return render_template('Chatbot.html')


if __name__ == "__main__":
    app.run(debug = True)

'''def process():
     user_input = request.form['user_input']

    bot_response = bot.get_response(user_input)
    bot_response = str(bot_response)
    print("Friend: " + bot_response)
    return render_template('index.html', user_input=user_input,
                               bot_response=bot_response
                               )'''
