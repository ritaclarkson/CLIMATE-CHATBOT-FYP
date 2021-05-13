import random #to choose an answer from a set of answers
import json #to read json files
import pickle #to store models and data
import numpy as np #for numerical arrays and operations
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

import nltk #for working with NLP
from nltk.stem import WordNetLemmatizer

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential # Model for stacking layers of Neural Network.
from tensorflow.keras.layers import Dense, Activation, Dropout #Dense if fully-connected layer of nodes, Activation used for activation functions within the models' layers. Dropout is regularization layer.
from tensorflow.keras.optimizers import SGD #Optimizer of the loss. Method that Neural network is using in order to learn

import kerastuner as kt #Hyper parameter optimizer

nltk.download('wordnet')

import warnings
warnings.filterwarnings('ignore')
lem = WordNetLemmatizer() #create lemmatizer instance. Lemmatization is process of determining the lemma of a word based on its intended meaning. So, "Warm", "Warming", "Warmed" will correspond to the same lemma "warm"


def text_preprocessing(text):
    text_cleaned = clean(text)
    text_lemmatized = lemmatize_words(text_cleaned)
    
    return text_lemmatized

def preprocessing(intents):
    tokens, topics, documents, patterns = [], [], [], [] #placeholders for our data.  


    for intent in intents['intents']: #loop thorugh all intents
        for pattern in intent['patterns']: #for each pattern there 
            
            text_preprocessed = text_preprocessing(pattern)
            patterns.append(text_preprocessed)
            
            token_list = nltk.word_tokenize(text_preprocessed) #tokenize that sentence and decompose it to tokens
            tokens.extend(token_list) # extend the token list with new entries
            documents.append((token_list, intent['tag'])) #add tuple of tokens and its' topic

            if intent['tag'] not in topics: #create a new key in the dictionary for new intent
                topics.append(intent['tag'])
                

    tokens = sorted(set(tokens)) #sort tokens 
    topics = sorted(set(topics))

    print('There are {} topics (intents) in our data'.format(len(topics)))
    print('There are {} words in our corpus'.format(len(tokens)))

    
    return tokens, topics, documents, patterns


def clean(text):
    import re    # for regular expressions
    from string import punctuation

    text = text.lower()
 
    text = re.sub("\'s", " is ", text) # we have cases like "Sam is" or "Sam's" (i.e. his)
    text = re.sub(" whats ", " what is ", text, flags=re.IGNORECASE)
    text = re.sub("\'ve", " have ", text)
    text = re.sub(' the ', ' ', text)
    text = re.sub(' a ', ' ', text)
    text = re.sub("can't", "cannot ", text)
    text = re.sub("n't", " not ", text)
    text = re.sub('"',' ', text)
    text = re.sub("\'m", ' am ', text)
    text = re.sub('$','', text)
    #you might need more
    
    text = re.sub("\?",'',text)
    text = re.sub("\,",'', text)
    text = re.sub(";", '', text)
    text = re.sub('!', '', text)
    text = re.sub(":", '', text)
    text = re.sub('-',' ', text)
    text = re.sub('\.','', text)
    text = re.sub("/", ' ', text)
    text = re.sub("'", ' ', text)
    text = re.sub('$','', text)
    
    # Return a list of words
    return text

def lemmatize_words(line):
    result = ''
    
    for word in line.split():
        result = result + " " + lem.lemmatize(word)
     
    return result


def n_grams_from_patterns(patterns, topics, documents, count_vectorizer):
    n_grams = []
    labels = [0] * len(topics)
    for i, pattern in enumerate(patterns):
        pattern_vectorized = count_vectorizer.transform([pattern])
        label = list(labels)
        label[topics.index(documents[i][1])] = 1

        n_grams.append((pattern_vectorized, label))
        
    X = [list(x.toarray()[0]) for x in list(np.array(n_grams)[:,0])]
    y = [y for y in np.array(n_grams)[:,1]]
    
    return X, y

def model_builder(hp):
    model = Sequential() #basic model to connect stack of layers 
  

    # Tune the number of units in the first Dense layer
    # Choose an optimal value between 16-196
    hp_units_1 = hp.Int('units_1', min_value=16, max_value = 256, step = 8) #initiliaze the search parameters
    model.add(Dense(units=hp_units_1, activation='relu')) #add fully-connected layer, activation is rectified linear unit
    
    model.add(Dropout(0.5))
    
    # Tune the number of units in the second Dense layer
    # Choose an optimal value between 16-64
    hp_units_2 = hp.Int('units_2', min_value=16, max_value = 128, step = 4) #same as with the first layer
    model.add(Dense(units=hp_units_2, activation='relu')) #add second layer
    
    model.add(Dropout(0.5)) #drop half of the layers. It is needed to make the model simpler
    
    model.add(Dense(7,activation = 'softmax')) # output layer that consists of nodes. One node per intent that we have. Softmax makes it to look like a probability distribution

    # Tune thyperparameters for optimizer
    # Choose an learning rate from 0.01, 0.02, 0.05, 0.1
    hp_learning_rate = hp.Choice('learning_rate', values=[0.01, 0.02, 0.05, 0.1])
    
    #momentum is defined as additional feaature to learning rate, sort of acceleration in the direction of better outcome
    hp_momentum = hp.Choice('momentum', values = [0.1, 0.2, 0.4, 0.5, 0.7, 0.9])

    #compiling model together
    model.compile(SGD(lr = hp_learning_rate, momentum = hp_momentum, nesterov = True),
                loss='categorical_crossentropy',
                metrics=['accuracy'])

    return model

def train():
    nltk.download('punkt')
    nltk.download('wordnet')
    intents = json.loads(open('./Data/intents.json').read()) #reading our intents files


    tokens, topics, documents, patterns = preprocessing(intents) #here is our variable with data that will be later handed to Neural Network

    pickle.dump(tokens, open("./Data/tokens.pkl", "wb"))
    pickle.dump(topics, open("./Data/topics.pkl", "wb"))

    #change here n-gram length to have later as your input data
    count_vectorizer = CountVectorizer(analyzer='word', ngram_range=(1, 2))
    count_vectorizer.fit(patterns)

    pickle.dump(count_vectorizer,open("./Data/vectorizer.pkl", "wb"))

    X,y = n_grams_from_patterns(patterns, topics,documents, count_vectorizer)

    X_train, X_test, y_train, y_test = train_test_split(np.array(X), np.array(y), test_size=0.1, stratify = np.array(y))

    #initialize the hyper parameter tuner
    tuner = kt.RandomSearch(model_builder,
                        objective='val_accuracy',
                        max_trials=100,
                        directory='trial_hist',
                        project_name='chat_bot',
                        overwrite=True)



    #early call back to stop model when no increase in results 
    stop_early = keras.callbacks.EarlyStopping(monitor='val_loss', patience=8)
    tuner.search(X_train, y_train, epochs=50, validation_split=0.25, callbacks=[stop_early]) #search in hyper parameter space to find the best model

    # Get the optimal hyperparameters
    best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]

    print(f"""The hyperparameter search is complete. The optimal number of units in the first densely-connected layer is {best_hps.get('units_1')},  
        The optimal number of units in the second densely-connected layer is {best_hps.get('units_2')} and 
        the optimal learning rate for the optimizer is {best_hps.get('learning_rate')}.
        The oprimal momentum coefficient is {best_hps.get('momentum')}.
    """)
    #tuner.results_summary()

    # Build the model with the optimal hyperparameters and train it on the data for 75 epochs
    model = tuner.hypermodel.build(best_hps)
    history = model.fit(X_train, y_train, epochs=50, validation_split=0.25)

    val_acc_per_epoch = history.history['val_accuracy']
    best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
    print('Best epoch: %d' % (best_epoch,))
    hypermodel = tuner.hypermodel.build(best_hps) #reproduce the best model

    # Retrain the model with the best amount epochs
    model_pre_prod = hypermodel.fit(X_train, y_train, epochs=best_epoch, validation_split=0.2)
    print(hypermodel.summary())

    hypermodel.save("./Data/chatbot_IR_pre_prod.h5", model_pre_prod) #saving the model optimal model

    eval_result = hypermodel.evaluate(X_test, y_test) #evaluate the result on a completely new data that we left out
    print("[test loss, test accuracy]:", eval_result) #get the accuracy score for the test set

    #retrain the model on the whole data set since optimal model is obtained (and we don't have much data to just leave it)
    hypermodel_prod = tuner.hypermodel.build(best_hps) #reproduce the best model again
    model_prod = hypermodel_prod.fit(np.array(X), np.array(y), epochs=best_epoch)
    hypermodel.save("./Data/chatbot_IR_prod.h5", model_pre_prod) #saving the model for later use

    return "Training complete."

