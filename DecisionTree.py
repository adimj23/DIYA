#!/usr/bin/env python
# coding: utf-8

# In[3]:


get_ipython().system(' pip install matplotlib')
get_ipython().system(' pip install pandas')
get_ipython().system(' pip install datetime')
get_ipython().system(' pip install seaborn')
get_ipython().system(' pip install sklearn')
get_ipython().system(' pip install nltk')
get_ipython().system(' pip install pydotplus')
get_ipython().system(' pip install ipython')
get_ipython().system(' pip install nbconvert')


# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn import tree
from sklearn import model_selection
from datetime import datetime
import nltk
from sklearn.metrics import r2_score


# In[2]:


words = set(nltk.corpus.words.words())
app_file_path = '/Users/adisrinivasan/Data Science Internship/donorschoose-application-screening/train.csv'
training_data = pd.read_csv(app_file_path)


for i in range(len(training_data)):
    training_data['project_submitted_datetime'].values[i] = datetime.strptime(
        training_data['project_submitted_datetime'][i], '%Y-%m-%d %H:%M:%S').month
training_data['month'] = training_data['project_submitted_datetime']

print(training_data.shape)


# In[3]:


training_data['project_essay_3'] = training_data['project_essay_3'].fillna('')
training_data['project_essay_4'] = training_data['project_essay_4'].fillna('')


training_data['len_essay_1'] = training_data['project_essay_1'].str.split().str.len()
training_data['len_essay_2'] = training_data['project_essay_2'].str.split().str.len()
training_data['len_essay_3'] = training_data['project_essay_3'].str.split().str.len()
training_data['len_essay_4'] = training_data['project_essay_4'].str.split().str.len()


training_data['essays'] = training_data['project_essay_1'].fillna('') + training_data['project_essay_2'].fillna('') +                           training_data['project_essay_3'].fillna('') + training_data['project_essay_4'].fillna('')

training_data['essays'] = training_data['essays'].apply(lambda x: " ".join(w for w in nltk.wordpunct_tokenize(x)
                                       if w.lower() in words))


training_data['len_title'] = training_data['project_title'].str.split().str.len()

training_data['len_summary'] = training_data['project_resource_summary'].str.split().str.len()


# In[4]:


# approved = training_data[training_data.project_is_approved == 1]

# cv = CountVectorizer(stop_words='english', min_df=10, binary = True)
# word_count_vector=cv.fit_transform(approved['essays'])

# tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)
# tfidf_transformer.fit(word_count_vector)


# df_idf = pd.DataFrame(tfidf_transformer.idf_, index=cv.get_feature_names(),columns=["idf_weights"])
# df_idf = df_idf.sort_values(by=["idf_weights"], ascending = False)
# df_idf = df_idf[:100]
# df_idf = df_idf[:25]

vectorizer = TfidfVectorizer(max_features = 75, min_df=5, max_df=0.9, stop_words = 'english')
tf_idf_transformed = vectorizer.fit_transform(training_data['essays'])

df_idf = pd.DataFrame(tf_idf_transformed.toarray(), columns=vectorizer.get_feature_names())

print(df_idf.columns)

training_data = pd.concat([training_data, df_idf], axis=1)

print(training_data.columns)

# top_words = df_idf.index.tolist()

# print(df_idf)



# for i in top_words:
#     training_data[i] = training_data['essays'].apply(lambda x: x.count(i))
# print(training_data.shape)


# In[5]:


training_data = training_data.drop(['id', 'teacher_id', 'teacher_prefix', 'school_state', 'project_submitted_datetime', 
                                    'project_grade_category', 'project_subject_categories', 
                                    'project_subject_subcategories',
                                    'project_title', 'project_essay_1', 'project_essay_2',
                                    'project_essay_3', 'project_essay_4', 'essays', 'project_resource_summary',
                                    'teacher_number_of_previously_posted_projects'], axis=1)


# In[6]:


num_approved = training_data.project_is_approved.value_counts()
length = training_data.project_is_approved.count()
print(num_approved/length)

training_data.columns.get_loc("project_is_approved")

if 'project_is_approved' in training_data.columns:
    print("hello")


# In[22]:


# control_features = ['project_submitted_datetime', 'len_essay_1',
#                      'len_essay_2', 'len_essay_3', 'len_essay_4', 'len_title',
#                      'school', 'learning', 'classroom', 'learn',
#                      'help', 'need', 'work', 'come', 'use', 'able', 'love', 'day', 'class',
#                      'make', 'new', 'year', 'time', 'student', 'reading', 'want', 'grade',
#                      'allow', 'provide', 'free', 'teach', 'high', 'like', 'project',
#                      'technology', 'way', 'different', 'world', 'best', 'lunch', 'group',
#                      'read', 'needs', 'create', 'home', 'education', 'math', 'teacher',
#                      'hard', 'life', 'working', 'opportunity', 'low', 'eager', 'community', 'just']

# control_features = training_data.drop('project_is_approved')

# print(control_features.columns)

y = training_data.project_is_approved
X = training_data.drop('project_is_approved', axis = 1)


def get_accuracy_score(i):
    train_X, val_X, train_y, val_y = model_selection.train_test_split(X, y, random_state=0, train_size=i/100)
    clf = DecisionTreeClassifier(max_depth = 12, random_state = 0, min_samples_leaf=2)
    clf.fit(train_X,train_y)
    train_pred = clf.predict(train_X)
    return accuracy_score(train_y, train_pred)

def get_val_score(i):
    train_X, val_X, train_y, val_y = model_selection.train_test_split(X, y, random_state=0, train_size=i/100)
    clf = DecisionTreeClassifier(max_depth = 12, random_state = 0, min_samples_leaf=2)
    clf.fit(train_X,train_y)
    val_prediction = clf.predict(val_X)
    return accuracy_score(val_y, val_prediction)

print(get_accuracy_score(99))
print(get_val_score(99))


# In[23]:


acc_df = pd.DataFrame([i, get_accuracy_score(i)] for i in range(1,100))
val_df = pd.DataFrame([i, get_val_score(i)] for i in range(1,100))
plt.plot(acc_df[0], acc_df[1], label = "Training Accuracy")
plt.plot(val_df[0], val_df[1], label = "Validation Accuracy")
plt.legend(loc='best')
plt.xlabel("Training Split %")
plt.ylabel("Accuracy Score")
plt.title("Decision Tree Accuracy based on Training Split %")


# In[9]:


train_X, val_X, train_y, val_y = model_selection.train_test_split(X, y, random_state=0, 
                                                                  train_size=.8, test_size = .2)
clf = DecisionTreeClassifier(max_depth = 5, random_state = 0, min_samples_leaf=2)
training_model.fit(train_X, train_y)
val_predictions = training_model.predict(val_X)

# rf_clf = RandomForestClassifier(n_estimators = 200, max_depth = 100)
# rf_clf.fit(train_X,train_y)

val_predictions = clf.predict(val_X)

print(accuracy_score(val_y, val_predictions))
print(f1_score(val_y, val_predictions))

print(val_predictions.sum())
print(len(val_predictions))

train_predictions = clf.predict(train_X)
print(train_predictions.sum())
print(len(train_predictions))


# In[ ]:


# train_sizes = [1, 1000, 10000, 20000, 50000, 100000]
train_sizes = [.1, .3, .5, .7, .9]
train_sizes, train_scores, validation_scores = model_selection.learning_curve(estimator = rf_clf, X = X, 
                                                                        y = y, train_sizes = train_sizes, cv = 10, 
                                                                        shuffle = True, scoring = "f1")


train_scores_mean = train_scores.mean(axis = 1)
validation_scores_mean = validation_scores.mean(axis = 1)

print("ACC AND VAL SCORES:", accuracy_score(val_y, val_predictions), accuracy_score(train_y, train_predictions))
print(train_scores_mean, validation_scores_mean)

plt.plot(train_sizes, train_scores_mean, label = 'Training accuracy')
plt.plot(train_sizes, validation_scores_mean, label = 'Validation accuracy')
plt.ylabel('Accuracy', fontsize = 14)
plt.xlabel('Training set size', fontsize = 14)
plt.legend(loc='best')


# In[ ]:


# from sklearn.externals.six import StringIO  
# from IPython.display import Image  
# import pydotplus
# dot_data = StringIO()

# from sklearn.tree import export_graphviz
# export_graphviz(training_model, out_file=dot_data,  
#                 filled=True, rounded=True,
#                 special_characters=True, feature_names = control_features,class_names=['0','1'])
# graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
# graph.write_png('decision-tree.png')
# Image(graph.create_png())

