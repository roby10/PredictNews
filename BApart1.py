import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from nltk.corpus import stopwords
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score 
from sklearn.metrics import confusion_matrix
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.svm import SVC, LinearSVC

def main():

	news = pd.read_csv("others/BAnew.csv" , encoding = "utf-8",header=None)
	data = pd.read_csv("others/BAdata.csv", encoding = "utf-8")


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
		open  = data.loc[data['Date2'] == var,'Open']
		val = 0
		if (close.values >=  open.values):
			val =1


		df = df.append({'Trend': val}, ignore_index=True)

	news['Trend'] = df.iloc[:,0].values

	news.iloc[:,1].replace(to_replace="[^a-zA-Z]", value=" ", regex=True, inplace=True)

	news.iloc[:,1] = news.iloc[:,1].str.lower().values

	stop = stopwords.words('english')

	news.columns = ['Date', 'Corpus', 'Trend']
	#stopwords
	#news.iloc[:,1].apply(lambda x: [item for item in x if item not in stop])
	
	#1350 train
	#354 test

	print("------------------BA Liniar Regression 1gram--------------------")
	basicvectorizer = CountVectorizer(ngram_range=(1,1))
	#print(news)

	train = news[news['Date'] <= 42800]
	test = news[news['Date'] > 42800]

	print(news.shape)
	print(train.shape)
	print(test.shape)

	# Liniar Regression with 1gram model
	basictrain = basicvectorizer.fit_transform(train.iloc[:,1])
	basicmodel = LogisticRegression()
	basicmodel = basicmodel.fit(basictrain, train["Trend"].astype('int'))
	basictest = basicvectorizer.transform(test.iloc[:,1])
	predictions = basicmodel.predict(basictest)


	print(pd.crosstab(test["Trend"], predictions, rownames=["Actual"], colnames=["Predicted"]))


	print (classification_report(list(test["Trend"]), predictions.tolist()))
	print (accuracy_score(list(test["Trend"]), predictions.tolist()))
	

	#counts = Counter(x for xs in news['Corpus'] for x in xs.split())
	#print(counts)
	print("------------------BA Liniar Regression 2gram--------------------")
	# Liniar Regression with 2gram model
	basicvectorizer2 = CountVectorizer(ngram_range=(2,2))
	basictrain2 = basicvectorizer2.fit_transform(train.iloc[:,1])	
	basicmodel2 = LogisticRegression()
	basicmodel2 = basicmodel2.fit(basictrain2, train["Trend"].astype('int'))
	basictest2 = basicvectorizer2.transform(test.iloc[:,1])
	predictions2 = basicmodel2.predict(basictest2)

	print(pd.crosstab(test["Trend"], predictions2, rownames=["Actual"], colnames=["Predicted"]))
	print (classification_report(list(test["Trend"]), predictions2.tolist()))
	print (accuracy_score(list(test["Trend"]), predictions2.tolist()))
	

	print("------------------BA Random Forest 1gram--------------------")


	basicvectorizer3  = CountVectorizer(ngram_range=(1,1))
	basictrain3 = basicvectorizer3.fit_transform(train.iloc[:,1])
	basicmodel3 = RandomForestClassifier(n_estimators=200, criterion='entropy',max_features='auto')
	basicmodel3 = basicmodel3.fit(basictrain3, train["Trend"].astype('int'))
	basictest3 = basicvectorizer3.transform(test.iloc[:,1])
	predictions3 = basicmodel3.predict(basictest3)

	print(pd.crosstab(test["Trend"], predictions3, rownames=["Actual"], colnames=["Predicted"]))
	print (classification_report(list(test["Trend"]), predictions3.tolist()))
	print (accuracy_score(list(test["Trend"]), predictions3.tolist()))


	print("------------------BA Random Forest 2gram--------------------")


	basicvectorizer4  = CountVectorizer(ngram_range=(2,2))
	basictrain4 = basicvectorizer4.fit_transform(train.iloc[:,1])
	basicmodel4 = RandomForestClassifier(n_estimators=200, criterion='entropy',max_features='auto')
	basicmodel4 = basicmodel4.fit(basictrain4, train["Trend"].astype('int'))
	basictest4 = basicvectorizer4.transform(test.iloc[:,1])
	predictions4 = basicmodel4.predict(basictest4)

	print(pd.crosstab(test["Trend"], predictions4, rownames=["Actual"], colnames=["Predicted"]))
	print (classification_report(list(test["Trend"]), predictions4.tolist()))
	print (accuracy_score(list(test["Trend"]), predictions4.tolist()))



	
	print("------------------BA SVM Lin 1gram--------------------")
	basicvectorizer5 = CountVectorizer(ngram_range=(1,1))
	basictrain5 = basicvectorizer5.fit_transform(train.iloc[:,1])
	basicmodel5 = svm.LinearSVC(C=0.1, class_weight='balanced')
	basicmodel5 = basicmodel5.fit(basictrain5, train["Trend"].astype('int'))
	basictest5 = basicvectorizer5.transform(test.iloc[:,1])
	predictions5 = basicmodel5.predict(basictest5)

	print(pd.crosstab(test["Trend"], predictions5, rownames=["Actual"], colnames=["Predicted"]))
	print (classification_report(list(test["Trend"]), predictions5.tolist()))
	print (accuracy_score(list(test["Trend"]), predictions5.tolist()))


	print("------------------BA SVM Lin 2gram--------------------")
	basicvectorizer6 = CountVectorizer(ngram_range=(2,2))
	basictrain6 = basicvectorizer6.fit_transform(train.iloc[:,1])
	basicmodel6 = svm.LinearSVC(C=0.1, class_weight='balanced')
	basicmodel6 = basicmodel6.fit(basictrain6, train["Trend"].astype('int'))
	basictest6 = basicvectorizer6.transform(test.iloc[:,1])
	predictions6 = basicmodel6.predict(basictest6)

	print(pd.crosstab(test["Trend"], predictions6, rownames=["Actual"], colnames=["Predicted"]))
	print (classification_report(list(test["Trend"]), predictions6.tolist()))
	print (accuracy_score(list(test["Trend"]), predictions6.tolist()))

if __name__ == "__main__":
    main()