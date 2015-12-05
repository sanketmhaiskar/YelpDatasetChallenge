__author__ = 'Sanket'
import re
from nltk.corpus import stopwords
import json
import sys
from sklearn.metrics import mean_squared_error

reload(sys)
sys.setdefaultencoding('utf8')

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from collections import Counter
from sklearn import cross_validation
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score


def load_file():
    with open('D://stars_reviews.txt',"r") as loadfp:
        stars = []
        review_text = []
        count =0
        for row in loadfp.readlines():
            # skip missing data
            stars_split= row.split("\t")[0]
            review_split = row.split("\t")[1]
            if stars_split and review_split:
                  stars.append(stars_split)
                  # review_text.append(get_clean_review(review_split))
                  review_text.append(review_split)
            count = count+1
            if(count >50000): #To specify the number of Tweets
                break
        return review_text, stars

def load_stopwords():
    return set(stopwords.words("english"))

def load_yelp():
    file_write = open('D://stars_reviews.txt',mode='w+')
    with open('E://Semester 1//SMM//yelp_academic_dataset_review.json') as fp:
        for line in fp.readlines():
            temp=json.loads(line,encoding='ascii')
            file_write.write(str(temp["stars"]))
            file_write.write("\t")
            review = str(temp["text"])
            review = str.replace(review,"\n","")
            file_write.write(review)
            file_write.write("\n")
    print "Done."
    file_write.close()


# preprocess creates the term frequency matrix for the review data set
def preprocess(data,target):
    # data, target = load_file()
    count_vectorizerSVM = CountVectorizer(binary='false',ngram_range=(1,2))
    count_vectorizerNB = CountVectorizer(binary='false',ngram_range=(0,1))
    dataSVM = count_vectorizerSVM.fit_transform(data)
    dataNB = count_vectorizerNB.fit_transform(data)

    tfidf_dataSVM = TfidfTransformer(use_idf=True,smooth_idf=True).fit_transform(dataSVM)
    tfidf_dataNB = TfidfTransformer(use_idf=True,smooth_idf=True).fit_transform(dataNB)
    print "Calculating term frequency."
    return tfidf_dataSVM,tfidf_dataNB


def learn_model(dataSVM,dataNB, target):
    # preparing data for split validation. 70% training, 30% test
    data_train, data_test, target_train, target_test = cross_validation.train_test_split(dataNB, target, test_size=0.2,
                                                                                         random_state=37)
    calculateMajorityClass(target_train)

    print "Using Bernoulli's Naive Bayes"
    classifier = MultinomialNB().fit(data_train, target_train)
    predictedNB = classifier.predict(data_test)
    evaluate_model(target_test, predictedNB)
    # print mean_squared_error(target_test,predictedNB)


    data_train, data_test, target_train, target_test = cross_validation.train_test_split(dataSVM, target, test_size=0.2,
                                                                                         random_state=37)
    calculateMajorityClass(target_train)
    print "Using SVM "
    svm = SVC(probability=False,random_state=33,kernel='linear',shrinking=True)
    classifier = svm.fit(data_train, target_train)
    predicted = classifier.predict(data_test)
    evaluate_model(target_test, predicted)
    # print mean_squared_error(target_test,predicted)


def evaluate_model(target_true, target_predicted):
    print classification_report(target_true, target_predicted)
    print "The accuracy score is {:.2%}".format(accuracy_score(target_true, target_predicted))

def calculateMajorityClass(data):
    counter=Counter(data)
    print counter
    print "Majority Class : " + str((max(counter.values())*100)/sum(counter.values())) + "%"


def main():
    # load_yelp()
    print ("Loading file and getting reviews..")
    data,target = load_file()
    print "Preprocessing ..."
    tf_idfSVM,tf_idfNB = preprocess(data,target)
    learn_model(tf_idfSVM,tf_idfNB,target)

def get_clean_review(raw_review):
    letters_only = re.sub("[^a-zA-Z]", " ", raw_review)
    words = letters_only.lower().split()
    stops = load_stopwords()
    meaningful_words = [w for w in words if not w in stops]
    return( " ".join( meaningful_words ))

main()

