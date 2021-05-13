# Imports
import random
import numpy as np
import pickle
import json
from flask import Flask, render_template, request, session
import nltk
from keras.models import load_model
from nltk.stem import WordNetLemmatizer
import train
import regex as re
import warnings

import urllib
import requests
from bs4 import BeautifulSoup

warnings.filterwarnings('ignore')

# initialization of chat
intents = json.loads(open("./Data/intents.json").read())
tokens = pickle.load(open("./Data/tokens.pkl", "rb"))
topics = pickle.load(open("./Data/topics.pkl", "rb"))
count_vectorizer = pickle.load(open("./Data/vectorizer.pkl", "rb"))
model = load_model("./Data/chatbot_IR_prod.h5")
nltk.download('wordnet')
lem = WordNetLemmatizer()

app = Flask(__name__)
app.config['SESSION_TYPE'] = 'memcached'
app.config['SECRET_KEY'] = '12345665412325895548925'

@app.route("/")
def home():
    session["mode"] = 'talk'
    session["learning_tag"] = None
    session["learning_pattern"] = None
    session['prev_topic'] = 'greetings'
    return render_template("index.html")

@app.route("/get", methods=["POST"])
def chatbot_response():
    print('prev_topic', session['prev_topic'])
    message = request.form["msg"]
    res = process_message(message)
    return res


#bot response procedure
def process_message(message):
    mode = session['mode']
    
    #check if command is provided
    is_learning = message.split()[0] == '/learning'
    is_retrain = message.split()[0] =='/retrain'
    is_exit = message.split()[0] == '/exit'   #we can only exit from learning mode to chatting mode

    if is_exit:
        session['mode'] = 'talk'
        return "Learning mode deactivated. We can continue talking"

    elif is_learning:
        session['mode'] = 'learning'
        res = "Learning activated, type one of the topics you want to tech me to respond to:<br> " + str(topics) 
        return res
    
    elif is_retrain:
        train.train()  #send our models to retrain
        return "Model retrained. We will use your input during next update."

    #if learning is activated and message of the user is one of our topics, then store that topic in session and ask for a phrase
    if (mode == 'learning') & (message in topics):
        session['learning_tag'] = message
        return "Topic '" + message + "' recieved. Now type the phrase associated with that topic."

    #if learning mode is activated and learning tag (topic) is provided, then treat a message as a phrase to store
    if (mode == 'learning') & (session['learning_tag'] is not None):
        session['learning_pattern'] = message

        #checking that file is updated succefully
        updated = update_intents(session['learning_tag'], session['learning_pattern']) #check if everything is succefully updated

        #if success, then restore mode to talking, all learning session variables are nullified
        if updated:
            session['mode'] = 'talk'
            session['learning_tag'] = None
            session['learning_pattern'] = None
            return 'We recorded your input. Thank you'
        else:
            return "Something went wrong. Let's try again. Please type /learning to repeat"
            
    elif (mode == 'learning'):
        return "Please type in the correct topic from a list " + str (topics)

    #processing non-learning messages
    message_cleaned = clean_user_input(message) #clean user input. We are sure that all commands are already executed

    #get prediction from the model based on a user input
    ints = predict_topic(message_cleaned, model) #predict the topic based on the user input
    res = get_response(ints, intents) #get the response messsage
    return res


    
#updating intents.json with user input sentences
def update_intents(topic, pattern):    
    for intent in intents['intents']: #loopint thorugh all intents to find our specified one
        if intent["tag"] == topic:
            intent['patterns'].append(pattern) #add phrase to a set of learning phrases
            with open('./Data/intents.json', 'w') as updated_ints:
                json.dump(intents, updated_ints)
            return True
    return False

#function to erase punctuation
def clean_user_input(text):
    text = text.lower()
 
    text = re.sub("\'s", " ", text) # we have cases like "Sam is" or "Sam's" (i.e. his)
    text = re.sub(" whats ", " what is ", text, flags=re.IGNORECASE)
    text = re.sub("\'ve", " have ", text)
    text = re.sub(' the ', ' ', text)
    text = re.sub(' a ', ' ', text)
    text = re.sub("can't", "cannot ", text)
    text = re.sub("n't", " not ", text)
    text = re.sub('"',' ', text)
    text = re.sub("\'m", ' am ', text)
    text = re.sub('\s',' ',text)
    text = re.sub('$','', text)
    #you might need more
    
    text = re.sub("\?",'',text)
    text = re.sub("\,",'', text)
    text = re.sub(";", '', text)
    text = re.sub('!', '', text)
    text = re.sub(":", '', text)
    text = re.sub('-',' ', text)
    text = re.sub('\.','', text)
    text = re.sub("'", ' ', text)
    text = re.sub('$','', text)
    
    text_lemmatized = lemmatize_words(text, lem)
    return text_lemmatized

def lemmatize_words(line, lemmatizer):
    result = ''
    for word in line.split():
        result = result + " " + lemmatizer.lemmatize(word)
     
    return result


#create n_grams from user input
def n_grams_user_input(sentence):
    vectorized = count_vectorizer.transform([clean_user_input(sentence)]) #vectorize cleaned input (get numerical representation)
    vectorized_array = vectorized.toarray() #from sparse matrix to array
    return vectorized_array[0] #reduce dimension


#function to predict the user intent (topic)
def predict_topic(sentence, model):
    n_grams = n_grams_user_input(sentence) #create n-grams s from user input
    
    predicted_topic = model.predict(np.array([n_grams]))[0] #use our stored model to predict the intent
    
    THRESHOLD = 0.25 #arbitrary threshold for output. 
    results = [[i,r] for i,r in enumerate(predicted_topic) if r > THRESHOLD] #store prediction probabilities for topic that exceeded the theshold
    full_result = [[i,r] for i,r in enumerate(predicted_topic)]

    results_list = [] #placeholder for return variable that is list

    if len(results) == 0:
        results_list.append({'intent':'idk', 'message':sentence})
        return results_list

    results.sort(key = lambda x: x[1], reverse = True) #sort by probabilities in descending order


    #print(results) #uncomment if want to see resulting list and assess probabilities (needed if a lot of new data for model behaviour)
    for r in results: #loop through model predictions
        results_list.append({'intent': topics[r[0]], 'message':sentence}) #add name of that intent and its' probability
    return results_list

#function that chooses random response based on the predicted probabilities
def get_response(intents_list, intents_json):
    
    tag = intents_list[0]['intent'] #get the most possible intent

    #if none of topics reached the threshold, then perform google search
    if tag == 'idk':
        google_result_text, description_found = google_search(intents_list[0]['message']) #google the input and boolean to check that text is found
        text_intro = "I am not sure what you mean. Can you try to rephrase?"
        if description_found:
            return text_intro + "<br> Anyways, here what I found: " + "<br>" + google_result_text
        else:
            return text_intro

    #if we detect that definition is needed then check which one exactly is needed
    if tag == 'definitions':
        definition = accurate_definition(intents_list[0]['message'])

        #possibly that we don't know which definition is asked, then respond with one of the basic responses
        if definition is not None:
            session['prev_topic'] = tag #store in session topic that was answered the last
            return definition
    
    if tag == "more":
        tag = session["prev_topic"] #if asked for more, then topic is repeated
        print(tag)

    list_of_intents = intents_json['intents'] #parse all possible intents
    session['prev_topic'] = tag
    for intent in list_of_intents: #loop through all intents
        if intent["tag"] == tag: #when found the predicted intent
            reply = random.choice(intent['responses']) #randomly choose one reply. Modifications can be done. For example, if for some topics 2 replies are needed or even not random selection is performed.
            source = random.choice(intent['sources'])
            full_reply = reply + "<br> " + source 
            return full_reply #return the predicted topic and reply

    return "something went wrong"    
    


def accurate_definition(sentence):
    global_warming = ['global','warming','warmth']
    greenhouse = ['greenhouse','green','house','effect']
    climate_change = ['change', 'changing']

    for word in sentence.split():

        if word in global_warming:
            return "Global warming is the unusually rapid increase in Earth's average surface temperature over the past century primarily due to the greenhouse gases released as people burn fossil fuels."
        elif word in greenhouse:
            return "The greenhouse effect is a natural process that warms the Earth's surface. When the Sun's energy reaches the Earth's atmosphere, some of it is reflected back to space and the rest is absorbed and re-radiated by greenhouse gases."
        elif word in climate_change:
            return 'Climate change is the long-term shift in average weather patterns across the world.<br> Since the mid-1800s, humans have contributed to the release of carbon dioxide and other greenhouse gases into the air. <br> This causes global temperatures to rise, resulting in long-term changes to the climate.'
    return None



def google_search(sentence):
    query = sentence.replace(' ', '+') #make a query separating words and replacing it with '+' (that's how google qury should look like)
    URL = f"https://google.com/search?q={query}&hl=uk&lr=lang_en"

    # desktop user-agent. Standard one.
    USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.97 Safari/537.36"
    headers = {"user-agent" : USER_AGENT}

    #send request to google search
    response = requests.get(URL, headers=headers)
    if response.status_code == 200:  #if response is OK 
        soup = BeautifulSoup(response.text) #initialize the library Beautiful Soup for web parsing
        results = []

        search = soup.find_all('span', {'class':'hgKElc'}) # class for one sort of google descriptions
        for s in search:
            return s.text.split('—')[-1] + '<br> <a href = ' + URL + '> Here is the link</a>', True #we return the first occurence

        search = soup.find_all('span', {'class':"aCOpRe"}) #another class for google descriptions
        for s in search:
            return s.text.split('—')[-1] + '<br> <a href = ' + URL + '> Here is the link</a>', True

    return URL, False #if no description is found, then return the query URL and False to indicate that no desription can be given

if __name__ == "__main__":
    app.run(debug = True)