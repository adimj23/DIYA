#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system(' pip install matplotlib')
get_ipython().system(' pip install pandas')
get_ipython().system(' pip install datetime')
get_ipython().system(' pip install seaborn')
get_ipython().system(' pip install sklearn')
get_ipython().system(' pip install nltk')
get_ipython().system(' pip install pydotplus')
get_ipython().system(' pip install ipython')
get_ipython().system(' pip install nbconvert')


# In[10]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn import model_selection
from datetime import datetime
import nltk
from sklearn.metrics import f1_score


# In[11]:


words = set(nltk.corpus.words.words())
app_file_path = '/Users/adisrinivasan/Data Science Internship/donorschoose-application-screening/train.csv'
training_data = pd.read_csv(app_file_path)


# In[12]:


for i in range(len(training_data)):
    training_data['project_submitted_datetime'].values[i] = datetime.strptime(
        training_data['project_submitted_datetime'][i], '%Y-%m-%d %H:%M:%S').month


training_data['project_essay_3'] = training_data['project_essay_3'].fillna('')
training_data['project_essay_4'] = training_data['project_essay_4'].fillna('')


training_data['len_essay_1'] = training_data['project_essay_1'].str.split().str.len()
training_data['len_essay_2'] = training_data['project_essay_2'].str.split().str.len()
training_data['len_essay_3'] = training_data['project_essay_3'].str.split().str.len()
training_data['len_essay_4'] = training_data['project_essay_4'].str.split().str.len()


training_data['essays'] = training_data['project_essay_1'].fillna('') + training_data['project_essay_2'].fillna('') +                           training_data['project_essay_3'].fillna('') + training_data['project_essay_4'].fillna('')

training_data['essays'] = training_data['essays'].apply(lambda x: " ".join(w for w in nltk.wordpunct_tokenize(x)
                                       if w.lower() in words))


# In[13]:


training_data['len_title'] = training_data['project_title'].str.split().str.len()

training_data['len_summary'] = training_data['project_resource_summary'].str.split().str.len()

# training_data['len_subcat'] = training_data['project_subject_subcategories'].str.split().str.len()

# approved = training_data[training_data.project_is_approved == 1]

# cv = CountVectorizer(stop_words='english', min_df=10, binary = True)
# word_count_vector=cv.fit_transform(approved['essays'])

# tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)
# tfidf_transformer.fit(word_count_vector)


# df_idf = pd.DataFrame(tfidf_transformer.idf_, index=cv.get_feature_names(),columns=["idf_weights"])
# df_idf = df_idf.sort_values(by=["idf_weights"])
# # df_idf = df_idf[:50]
# df_idf = df_idf[:25]

# top_words = df_idf.index.tolist()

# for i in top_words:
#     training_data[i] = training_data['essays'].apply(lambda x: x.count(i))
# print(training_data.shape)

vectorizer = TfidfVectorizer(max_features = 75, min_df=5, max_df=0.9, stop_words = 'english')
tf_idf_transformed = vectorizer.fit_transform(training_data['essays'])

df_idf = pd.DataFrame(tf_idf_transformed.toarray(), columns=vectorizer.get_feature_names())

print(df_idf.columns)

training_data = pd.concat([training_data, df_idf], axis=1)

training_data = training_data.drop(['id', 'teacher_id', 'teacher_prefix', 'school_state', 'project_grade_category',
                                    'project_subject_categories', 'project_subject_subcategories',
                                    'project_title', 'project_essay_1', 'project_essay_2',
                                    'project_essay_3', 'project_essay_4', 'essays', 'project_resource_summary',
                                    'teacher_number_of_previously_posted_projects'], axis=1)

print(training_data.columns)


# In[14]:


rejected = training_data[training_data.project_is_approved == 0]
approved = training_data.loc[training_data['project_is_approved'] == 1].sample(n=27734,random_state=1)

normalized_df = pd.concat([rejected, approved])

num_approved = normalized_df.project_is_approved.value_counts()
length = normalized_df.project_is_approved.count()
print(num_approved/length)
print(normalized_df.shape)


# In[15]:


"""
control_features = ['project_submitted_datetime', 'len_essay_1',
                     'len_essay_2', 'len_title', 'len_summary',
                     'school', 'learning', 'classroom', 'learn',
                     'help', 'need', 'work', 'come', 'use', 'able', 'love', 'day', 'class',
                     'make', 'new', 'year', 'time', 'student', 'reading', 'want', 'grade',
                     'allow', 'provide', 'free', 'teach']
"""

y = normalized_df.project_is_approved
X = normalized_df.drop('project_is_approved', axis = 1)


def get_accuracy_score(i):
    train_X, val_X, train_y, val_y = model_selection.train_test_split(X, y, random_state=0, train_size=i/100)
    training_model = DecisionTreeClassifier(max_depth = 3, random_state=0, min_samples_leaf=2)
    training_model.fit(train_X, train_y)
    train_predictions = training_model.predict(train_X)
    return accuracy_score(train_y, train_predictions).astype(float)

def get_val_score(i):
    train_X, val_X, train_y, val_y = model_selection.train_test_split(X, y, random_state=0, train_size=i/100)
    training_model = DecisionTreeClassifier(max_depth = 5, random_state=0, min_samples_leaf=2)
    training_model.fit(train_X, train_y)
    val_predictions = training_model.predict(val_X)
    return accuracy_score(val_y, val_predictions).astype(float)


print(get_val_score(80))


# In[21]:


train_X, val_X, train_y, val_y = model_selection.train_test_split(X, y, random_state=0, 
                                                                  train_size=.8, test_size = .2)
# training_model = DecisionTreeClassifier(max_depth = 5, random_state = 0, min_samples_leaf=2)

# training_model.fit(train_X, train_y)
# val_predictions = training_model.predict(val_X)

# print(val_predictions.sum())
# print(len(val_predictions))

# train_predictions = training_model.predict(train_X)
# print(train_predictions.sum())
# print(len(train_predictions))

def rf_accuracy_score(i):
    train_X, val_X, train_y, val_y = model_selection.train_test_split(X, y, random_state=0, train_size=.8)
    rf_clf = RandomForestClassifier(n_estimators = 100, max_depth = i)
    rf_clf.fit(train_X, train_y)
    train_prediction = rf_clf.predict(train_X)
    return accuracy_score(train_y, train_prediction)

def rf_val_score(i):
    train_X, val_X, train_y, val_y = model_selection.train_test_split(X, y, random_state=0, train_size=.8)
    rf_clf = RandomForestClassifier(n_estimators = 100, max_depth = i)
    rf_clf.fit(train_X,train_y)
    prediction_y =rf_clf.predict(train_X)
    val_prediction = rf_clf.predict(val_X)
    return accuracy_score(val_y, val_prediction)

def f1_accuracy_score(i):
    train_X, val_X, train_y, val_y = model_selection.train_test_split(X, y, random_state=0, train_size=.8)
    rf_clf = RandomForestClassifier(n_estimators = 100, max_depth = i)
    rf_clf.fit(train_X, train_y)
    train_prediction = rf_clf.predict(train_X)
    return f1_score(train_y, train_prediction).astype(float)

def f1_val_score(i):
    train_X, val_X, train_y, val_y = model_selection.train_test_split(X, y, random_state=0, train_size=.8)
    rf_clf = RandomForestClassifier(n_estimators = 100, max_depth = i)
    rf_clf.fit(train_X,train_y)
    prediction_y =rf_clf.predict(train_X)
    val_prediction = rf_clf.predict(val_X)
    return f1_score(val_y, val_prediction)

rf_clf = RandomForestClassifier(n_estimators = 100, max_depth = 15)
rf_clf.fit(train_X,train_y)

prediction_y =rf_clf.predict(train_X)

val_prediction = rf_clf.predict(val_X)

feature_imp = pd.Series(rf_clf.feature_importances_,index=X.columns).sort_values(ascending=False)

print(feature_imp)

print(accuracy_score(train_y, prediction_y))
print(f1_score(val_y, val_prediction))


# In[19]:


acc_df = pd.DataFrame([i, rf_accuracy_score(i)] for i in range(1, 100))

val_df = pd.DataFrame([i, rf_val_score(i)] for i in range(1, 100))

plt.plot(acc_df[0], acc_df[1], label = "Training Accuracy")

plt.plot(val_df[0], val_df[1], label = "Validation Accuracy")

plt.legend(loc='best')

plt.xlabel("Max Depth")
plt.ylabel("Accuracy")
plt.title("Accuracy of Random Forest based on Max Depth")


# In[20]:


train_sizes = [0.1, 0.3, 0.5, 0.7, 0.9, 0.99]
train_sizes, train_scores, validation_scores = model_selection.learning_curve(estimator = rf_clf, X = X, 
                                                                        y = y, train_sizes = train_sizes, cv = 10, 
                                                                        shuffle = True, scoring = 'accuracy', random_state = 1)


train_scores_mean = train_scores.mean(axis = 1)
validation_scores_mean = validation_scores.mean(axis = 1)

# print("ACC AND VAL SCORES:", accuracy_score(val_y, val_predictions), accuracy_score(train_y, train_predictions))
# print(train_scores_mean, validation_scores_mean)

plt.plot(train_sizes, train_scores_mean, label = 'Training accuracy')
plt.plot(train_sizes, validation_scores_mean, label = 'Validation accuracy')
plt.ylabel('Accuracy', fontsize = 14)
plt.xlabel('Training set size', fontsize = 14)
plt.legend()


# In[ ]:


# from sklearn.externals.six import StringIO  
# from IPython.display import Image  
# import pydotplus
# from sklearn.ensemble import RandomForestClassifier
# dot_data = StringIO()

# from sklearn.tree import export_graphviz
# export_graphviz(training_model, out_file=dot_data,  
#                 filled=True, rounded=True,
#                 special_characters=True, feature_names = control_features,class_names=['0','1'])
# graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
# graph.write_png('balanced-decision-tree.png')
# Image(graph.create_png())

