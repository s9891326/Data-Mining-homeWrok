from sklearn import datasets
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
import numpy as np
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from sklearn.model_selection import cross_validate


dataset = datasets.load_files('mini_newsgroups')

stopwords = set(stopwords.words('english'))

tfv = TfidfVectorizer(encoding = 'ISO-8859-1', stop_words = stopwords)

training_tfv = tfv.fit_transform(dataset.data).toarray()

indices = np.random.permutation(training_tfv.shape[0])

feature_attribute = training_tfv[indices]

class_attribute = dataset.target[indices]

alpha_list = range(1, 10, 1)

accuracy_training_list = []

accuracy_test_list = []

for alpha in alpha_list:
    naive_bayes = MultinomialNB(alpha = alpha, fit_prior = True)
    
    cv_results = cross_validate(naive_bayes, feature_attribute, class_attribute, cv = 5, return_train_score = True)
    
    accuracy_training = cv_results['train_score'].mean()
    
    accuracy_training_list.append(accuracy_training)
    
    print('NAIVE BAYES 訓練正確率: ' + str(accuracy_training))
    
    accuracy_test = cv_results['test_score'].mean()
    
    accuracy_test_list.append(accuracy_test)

    print('NAIVE BAYES 測試正確率: ' + str(accuracy_test) + '\n')
    

plt.figure()

#plt.xticks(np.arange(2, 31, step = 2))

#plt.yticks(np.arange(0.7, 1, step = 0.05))

plt.title('NAIVE BAYES')

plt.xlabel('the number of alpha')

plt.ylabel('accuracy')

plt.plot(alpha_list, accuracy_training_list, marker = 'o', label = 'training')

plt.plot(alpha_list, accuracy_test_list, marker = 'o', label = 'test')

plt.legend(loc = 'best')

plt.savefig('NAIVE BAYES.png')

plt.show()



c_list = np.arange(0.1, 1, 0.1)

accuracy_training_list = []

accuracy_test_list = []

for c in c_list:
    svm = LinearSVC(penalty = 'l2', dual = True, C = c, max_iter = 2000)

    cv_results = cross_validate(svm, feature_attribute, class_attribute, cv = 5, return_train_score = True)
    
    accuracy_training = cv_results['train_score'].mean()
    
    accuracy_training_list.append(accuracy_training)
    
    print('SVM 訓練正確率: '+ str(accuracy_training))
    
    accuracy_test = cv_results['test_score'].mean()
    
    accuracy_test_list.append(accuracy_test)

    print('SVM 測試正確率: '+ str(accuracy_test) + '\n')

    
plt.figure()

#plt.xticks(np.arange(2, 31, step = 2))

#plt.yticks(np.arange(0.7, 1, step = 0.05))

plt.title('SVM')

plt.xlabel('the number of C')

plt.ylabel('accuracy')

plt.plot(c_list, accuracy_training_list, marker = 'o', label = 'training')

plt.plot(c_list, accuracy_test_list, marker = 'o', label = 'test')

plt.legend(loc = 'best')

plt.savefig('SVM.png')

plt.show()

