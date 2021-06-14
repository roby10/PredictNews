import pandas as pd
from io import open
import numpy as np
from numpy import array, asarray, zeros
from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer , one_hot
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

from keras.layers import Dropout, Activation, Embedding, Convolution1D, MaxPooling1D, Input, Dense, BatchNormalization, Flatten, Reshape, Concatenate
from keras.optimizers import Adam

from keras.models import Sequential
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import MaxPooling1D, AveragePooling1D
from keras.layers import GlobalMaxPooling1D, SpatialDropout1D, GRU, SimpleRNN, GlobalAveragePooling1D, TimeDistributed
from keras.layers import Embedding
from keras.layers import Bidirectional
from IPython.display import SVG
from keras.utils.vis_utils import plot_model

import sys
import re

contractions = { 
"ain't": "am not",
"aren't": "are not",
"can't": "cannot",
"can't've": "cannot have",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he would",
"he'd've": "he would have",
"he'll": "he will",
"he's": "he is",
"how'd": "how did",
"how'll": "how will",
"how's": "how is",
"i'd": "i would",
"i'll": "i will",
"i'm": "i am",
"i've": "i have",
"isn't": "is not",
"it'd": "it would",
"it'll": "it will",
"it's": "it is",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"must've": "must have",
"mustn't": "must not",
"needn't": "need not",
"oughtn't": "ought not",
"shan't": "shall not",
"sha'n't": "shall not",
"she'd": "she would",
"she'll": "she will",
"she's": "she is",
"should've": "should have",
"shouldn't": "should not",
"that'd": "that would",
"that's": "that is",
"there'd": "there had",
"there's": "there is",
"they'd": "they would",
"they'll": "they will",
"they're": "they are",
"they've": "they have",
"wasn't": "was not",
"we'd": "we would",
"we'll": "we will",
"we're": "we are",
"we've": "we have",
"weren't": "were not",
"what'll": "what will",
"what're": "what are",
"what's": "what is",
"what've": "what have",
"where'd": "where did",
"where's": "where is",
"who'll": "who will",
"who's": "who is",
"won't": "will not",
"wouldn't": "would not",
"you'd": "you would",
"you'll": "you will",
"you're": "you are"
}

np.random.seed(7)

def clean_text(text, remove_stopwords=True):
    
    #Convert all to lower case
    text=text.encode('ascii',errors='ignore')
    text=text.lower()
    
    #Expand Contractions
    if True:
        text=text.split()
        new_text=[]
        for word in text:
            if word in contractions:
                new_text.append(contractions[word])
            else:
                new_text.append(word)
        text=" ".join(new_text)
        
        
    #Format words and remove unwanted charecters(Noise)
    #Boilerplate
    text = re.sub(r'&amp;', '', text) 
    text = re.sub(r'0,0', '00', text) 
    text = re.sub(r'[-_"\-;%()|.,+&=*%.,!?:#@\[\]]', ' ', text)
    text = re.sub(r'\'', ' ', text)
    text = re.sub(r'\$', '', text)
    text = re.sub(r'u s ', ' united states ', text)
    text = re.sub(r'u n ', ' united nations ', text)
    text = re.sub(r'u k ', ' united kingdom ', text)
    text = re.sub(r' s ', ' ', text)
    text = re.sub(r' yr ', ' year ', text)
    text = re.sub(r' reuters ', '', text)
    
    # Optionally, remove stop words
    if remove_stopwords:
        text = text.split()
        stops = set(stopwords.words("english"))
        text = [w for w in text if not w in stops]
        text = " ".join(text)
    
    
    return  text
                



def main():

	news = pd.read_csv("DJIA/DJIAnew.csv" , encoding = "utf-8",header=None)
	data = pd.read_csv("DJIA/DJIAdata.csv", encoding = "utf-8")


	news = news[[2,3]]
	news = news.groupby([3])[2].apply(' '.join).reset_index()

	i = 0;
	for index, row in news.iterrows():
		if(row.iloc[0] not in list(data.iloc[:,1])):
			news.drop(news.index[i], inplace=True)
			i = i+1

	news = news.reset_index(drop=True)

	df = pd.DataFrame(columns=['Trend'])
	for index, row in news.iterrows():
		var = row.iloc[0]
		close = data.loc[data['Date2'] ==var, 'Close']
		Open  = data.loc[data['Date2'] == var,'Open']
		val = 0
		if (close.values >=  Open.values):
			val =1
		df = df.append({'Trend': val}, ignore_index=True)

	news['Trend'] = df.iloc[:,0].values
	news.columns = ['Date', 'Corpus', 'Trend']


	####################################################
	clean_headlines = []

	for daily_headlines in news['Corpus']:
	    clean_headlines.append(clean_text(daily_headlines))

	news['Corpus'] = clean_headlines
	
	batch_size=32
	epochs= 5
	vocab_size = 85000
	max_length= 1000


	encoded_docs = [one_hot(d, vocab_size) for d in news['Corpus']]
	#print(encoded_docs)
	padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
	print(padded_docs)

	x_train, x_test, y_train, y_test = train_test_split(padded_docs, news['Trend'] , test_size = 0.2, random_state = 2)
	
	tokenizer = Tokenizer(num_words = 1000)
	tokenizer.fit_on_texts(news['Corpus'])
	print(tokenizer.texts_to_sequences([news['Corpus'][15]]))
	
	print(x_train.shape)
	print(x_test.shape)
	print(y_train.shape)
	print(y_test.shape)

	
	#63 cu bidir GRU 50 embedding
	#57 cu simpleRNN

	#65% cu simpleRNN 
	model = Sequential()
	model.add(Embedding(vocab_size, 50,  input_length=max_length))
	model.add(Dropout(0.25))
	model.add(SimpleRNN(100,return_sequences=True))
	model.add(Dense(256))
	model.add(AveragePooling1D())
	model.add(Dropout(0.25))
	model.add(Activation('relu'))
	model.add(Flatten())
	model.add(Dense(1, activation='sigmoid'))


	model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc','binary_accuracy'])

	plot_model(model, to_file='figures/RNN.png', show_shapes=True, show_layer_names=True)

	model.fit(x_train, y_train, batch_size=16, epochs=epochs, validation_split=0.2, verbose=1)
	loss, accuracy,binaryAccuracy = model.evaluate(x_test, y_test, verbose=1)
	print('Accuracy: %f' % (accuracy*100))
	print('binaryAccuracy: %f' % (binaryAccuracy*100))

	sys.exit(0)


if __name__ == "__main__":
    main()