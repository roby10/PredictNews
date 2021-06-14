import pandas as pd
from io import open
from numpy import array
from numpy import asarray
from numpy import zeros
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score 
from sklearn.metrics import confusion_matrix
from collections import Counter
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split


from keras.models import Sequential
from keras.layers import Dropout
from keras.layers import Conv1D
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import MaxPooling1D
from keras.layers import GlobalMaxPooling1D
from keras.layers import Flatten
from keras.layers import Embedding
from keras.layers import Bidirectional

def clean_decription(line):
    words = list(set(line.split()))
    words = list(sorted([x.lower() for x in words]))
    words2 = words[:]
    for word in list(set(words2).intersection(stopwords.words('english'))):
        words.remove(word)
#     out = ' '.join(words)
    out = ' '.join(e for e in words if e.isalnum())
    if out == "":
        print(words)
    return out


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

	print(news.head(10))
	news['Corpus'] = news['Corpus'].map(clean_decription)
	print(news.head(10))

	docs = news['Corpus']
	labels = news['Trend']

	t = Tokenizer()

	t.fit_on_texts(docs)
	vocab_size = len(t.word_index)+1

	encoded_docs = t.texts_to_sequences(docs)
	#modify this
	max_length = 100
	padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')

	print(vocab_size)



	embeddings_index = dict()
	fin = open('glove.6B.300d.txt', encoding='utf8')
	
	for num, line in enumerate(fin):
	    values = line.split()
	    word = values[0]
	    if word in stopwords.words('english'):
	#         print (word)
	        continue
	    coefs = asarray(values[1:], dtype='float32')
	    embeddings_index[word] = coefs

	fin.close()
	max_width = 300
	print('Loaded %s word vectors.' % len(embeddings_index))

	embedding_matrix = zeros((vocab_size, max_width))
	mil=0
	for word, i in t.word_index.items():
	    embedding_vector = embeddings_index.get(word)
	    if embedding_vector is not None:
	        mil+=1
	        embedding_matrix[i] = embedding_vector

	print(embedding_matrix.shape)
	print()

	print('-------------------------------------LSTM here-----------------')
	'''
	# ################ define model#################
	model = Sequential()
	e = Embedding(vocab_size, max_width, weights=[embedding_matrix], input_length=max_length, trainable=True)
	model.add(e)
	model.add(LSTM(32,return_sequences=False))
	# # model.add(GlobalMaxPooling1D(pool_size=2, strides=None, padding='valid'))
	# # model.add(Flatten())
	model.add(Dense(50,  kernel_initializer="normal",activation = 'relu'))
	model.add(Dropout(0.5))
	model.add(Dense(25,  kernel_initializer="normal",activation = 'relu'))
	model.add(Dropout(0.5))
	model.add(Dense(15,  kernel_initializer="normal",activation = 'relu'))
	model.add(Dropout(0.5))
	model.add(Dense(1, activation='sigmoid'))

	# ############## compile the model ##############
	model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc','binary_accuracy'])
	# ############# summarize the model ##############
	print(model.summary())
	
	'''

	################ define model#################
	model = Sequential()
	e = Embedding(vocab_size, max_width, weights=[embedding_matrix], input_length=max_length, trainable=True)
	model.add(e)
	model.add(Bidirectional(LSTM(35, return_sequences=True)))
	model.add(GlobalMaxPooling1D())
	# model.add(Bidirectional(LSTM(10)))
	# model.add(Flatten())
	model.add(Dense(15,  kernel_initializer="normal",activation = 'relu'))
	model.add(Dropout(0.5))
	model.add(Dense(1, activation='sigmoid'))

	############## compile the model ##############
	model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc','binary_accuracy'])
	############# summarize the model ##############
	print(model.summary())

	
	padded_docs_train, padded_docs_test, labels_train, labels_test = train_test_split(padded_docs, labels, test_size=0.2, random_state=69)
	################### fit the model ##########3#############
	model.fit(padded_docs_train, labels_train, epochs=15, validation_split = 0.1,verbose=1)


	# evaluate the model
	loss, accuracy,binaryAccuracy = model.evaluate(padded_docs_test, labels_test, verbose=1)
	print('Accuracy: %f' % (accuracy*100))
	print('binaryAccuracy: %f' % (binaryAccuracy*100))

	'''
	news.iloc[:,1].replace(to_replace="[^a-zA-Z]", value=" ", regex=True, inplace=True)

	news.iloc[:,1] = news.iloc[:,1].str.lower().values

	stop = stopwords.words('english')

	news.columns = ['Date', 'Corpus', 'Trend']
	#stopwords
	#news.iloc[:,1].apply(lambda x: [item for item in x if item not in stop])
	
	#1350 train
	#354 test

	print("------------------LSTM--------------------")

	#print(news)

	train = news[news['Date'] <= 42844]
	test = news[news['Date'] > 42844]

	print(news.shape)
	print(train.shape)
	print(test.shape)
	'''

if __name__ == "__main__":
    main()