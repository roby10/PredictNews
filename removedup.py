import  pandas as pd
from  sklearn.feature_extraction.text import CountVectorizer


def get_jaccard_sum(str1, str2):
	a = set(str1.split())
	b = set(str2.split())
	c = a.intersection(b)
	return float(len(c)) / (len(a) + len(b) - len(c))


def main():
	data = pd.read_csv("others/XOM.csv", encoding = "utf-8",header=None)
	# 0 id
	# 1 news title
	# 2 news corpus
	# 3 date int
	# 4 date format dd/mm/yyyy
	# 5 source
	print(data.size)
	#len(data)
	size = len(data)-1
	l = 0
	while(l < size):
		#print(l)
		date = data.iloc[l,3]
		s = data.iloc[l,2]

		k = l+1

		print(date)

		while(k < size and data.iloc[k,3] == date):

			val = get_jaccard_sum(data.iloc[l,2], data.iloc[k,2])

			if(val > 0.5):
				#print(data.iloc[l,1])
				#print(data.iloc[k,1])
				#print("---------------")
				data.drop(data.index[k], inplace=True)
				size = len(data)-1
			else:
				k = k+1
		l = l+1

	print(data.size)


	data.to_csv("others/XOMnew.csv", encoding = "utf-8", index=False)

if __name__ == "__main__":
    main()