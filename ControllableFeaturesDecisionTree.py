import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn import model_selection
from datetime import datetime
import nltk

words = set(nltk.corpus.words.words())
app_file_path = '/Users/adisrinivasan/Data Science Internship/donorschoose-application-screening/train.csv'
training_data = pd.read_csv(app_file_path)


for i in range(len(training_data)):
    training_data['project_submitted_datetime'].values[i] = datetime.strptime(
        training_data['project_submitted_datetime'][i], '%Y-%m-%d %H:%M:%S').month


training_data['project_essay_3'] = training_data['project_essay_3'].fillna('')
training_data['project_essay_4'] = training_data['project_essay_4'].fillna('')


training_data['len_essay_1'] = training_data['project_essay_1'].str.split().str.len()
training_data['len_essay_2'] = training_data['project_essay_2'].str.split().str.len()
training_data['len_essay_3'] = training_data['project_essay_3'].str.split().str.len()
training_data['len_essay_4'] = training_data['project_essay_4'].str.split().str.len()


training_data['essays'] = training_data['project_essay_1'].fillna('') + training_data['project_essay_2'].fillna('') + \
                          training_data['project_essay_3'].fillna('') + training_data['project_essay_4'].fillna('')

training_data['essays'] = training_data['essays'].apply(lambda x: " ".join(w for w in nltk.wordpunct_tokenize(x)
                                       if w.lower() in words))


training_data['len_title'] = training_data['project_title'].str.split().str.len()

training_data['len_summary'] = training_data['project_resource_summary'].str.split().str.len()

training_data['len_subcat'] = training_data['project_subject_subcategories'].str.split().str.len()

approved = training_data[training_data.project_is_approved == 1]

# Find top 50 keywords of approved projects based on TF-IDF scores
cv = CountVectorizer(stop_words='english', min_df=10, binary = True)
word_count_vector=cv.fit_transform(approved['essays'])

tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)
tfidf_transformer.fit(word_count_vector)


df_idf = pd.DataFrame(tfidf_transformer.idf_, index=cv.get_feature_names(),columns=["idf_weights"])
df_idf = df_idf.sort_values(by=["idf_weights"])
df_idf = df_idf[:50]

top_words = df_idf.index.tolist()

for i in top_words:
    training_data[i] = training_data['essays'].apply(lambda x: x.count(i))


training_data = training_data.drop(['id', 'teacher_id', 'teacher_prefix', 'school_state', 'project_grade_category',
                                    'project_subject_categories', 'project_subject_subcategories',
                                    'project_title', 'project_essay_1', 'project_essay_2',
                                    'project_essay_3', 'project_essay_4', 'essays', 'project_resource_summary',
                                    'teacher_number_of_previously_posted_projects'], axis=1)
                                    
control_features = ['project_submitted_datetime', 'len_essay_1',
                     'len_essay_2', 'len_essay_3', 'len_essay_4', 'len_title',
                     'len_summary', 'school', 'learning', 'classroom', 'learn',
                     'help', 'need', 'work', 'come', 'use', 'able', 'love', 'day', 'class',
                     'make', 'new', 'year', 'time', 'student', 'reading', 'want', 'grade',
                     'allow', 'provide', 'free', 'teach', 'high', 'like', 'project',
                     'technology', 'way', 'different', 'world', 'best', 'lunch', 'group',
                     'read', 'needs', 'create', 'home', 'education', 'math', 'teacher',
                     'hard', 'life', 'working', 'opportunity', 'low', 'eager', 'community', 'just']

y = training_data.project_is_approved
X = training_data[control_features]

def get_accuracy_score(i):
    train_X, val_X, train_y, val_y = model_selection.train_test_split(X, y, random_state=0, train_size=i/100)
    training_model = DecisionTreeClassifier(max_depth = 3, random_state=0, min_samples_leaf=2)
    training_model.fit(train_X, train_y)
    train_predictions = training_model.predict(train_X)
    return accuracy_score(train_y, train_predictions).astype(float)

def get_val_score(i):
    train_X, val_X, train_y, val_y = model_selection.train_test_split(X, y, random_state=0, train_size=i/100)
    training_model = DecisionTreeClassifier(max_depth = 3, random_state=0, min_samples_leaf=2)
    training_model.fit(train_X, train_y)
    val_predictions = training_model.predict(val_X)
    return accuracy_score(val_y, val_predictions).astype(float)
    
from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus
dot_data = StringIO()

# CREATE DECISION TREE MODEL
train_X, val_X, train_y, val_y = model_selection.train_test_split(X, y, random_state=0, train_size=.8, test_size = .1)
training_model = DecisionTreeClassifier(max_depth = 3, random_state=0, min_samples_leaf=2)
training_model.fit(train_X, train_y)
val_predictions = training_model.predict(val_X)
train_predictions = training_model.predict(train_X)

print(accuracy_score(val_y, val_predictions), accuracy_score(train_y, train_predictions))

# VISUALIZE DECISION TREE
export_graphviz(training_model, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True, feature_names = control_features,class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('decision-tree.png')
Image(graph.create_png())
