#!/usr/bin/env python
# coding: utf-8

# In[78]:


import pandas as pd
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import nltk

words = set(nltk.corpus.words.words())
app_file_path = '/Users/adisrinivasan/Data Science Internship/donorschoose-application-screening/train.csv'
training_data = pd.read_csv(app_file_path)


for i in range(len(training_data)):
    training_data['project_submitted_datetime'].values[i] = datetime.strptime(
        training_data['project_submitted_datetime'][i], '%Y-%m-%d %H:%M:%S').month


# In[79]:


sns.barplot(x=training_data['project_submitted_datetime'], y=training_data['project_is_approved'], ci = False)
plt.xlabel("Month of Submission")
plt.ylabel("Acceptance Rate")
plt.title("Acceptance based on Month of Submission")
plt.show()


# In[80]:


training_data['len_essay_1'] = training_data['project_essay_1'].str.split().str.len()
training_data['len_essay_2'] = training_data['project_essay_2'].str.split().str.len()
training_data['len_essay_3'] = training_data['project_essay_3'].str.split().str.len()
training_data['len_essay_4'] = training_data['project_essay_4'].str.split().str.len()


training_data['essays'] = training_data['project_essay_1'].fillna('') + training_data['project_essay_2'].fillna('') +                           training_data['project_essay_3'].fillna('') + training_data['project_essay_4'].fillna('')

training_data['essays'] = training_data['essays'].apply(lambda x: " ".join(w for w in nltk.wordpunct_tokenize(x)
                                       if w.lower() in words))


training_data['len_title'] = training_data['project_title'].str.split().str.len()

training_data['len_summary'] = training_data['project_resource_summary'].str.split().str.len()


# In[86]:


plt.figure(figsize=(6,6))
sns.scatterplot(x = training_data['len_essay_1'], y = training_data['len_essay_2'], 
                hue = training_data['project_is_approved'], s = 5)


plt.xlabel("Length of Essay 1")
plt.ylabel("Length of Essay 2")
plt.title("Acceptance of projects based on Length of Essays 1 & 2")
# plt.legend(labels=['Not Approved', 'Approved'])
plt.xlim(70,220)
plt.ylim(90,350)
plt.show()


# In[82]:


sns.barplot(x = training_data['len_title'], y = training_data['project_is_approved'], ci = False)
plt.xlabel("Title Length")
plt.ylabel("Acceptance Rate")
plt.title("Acceptance Rate based on Length of Project Title")


# In[83]:


plt.figure(figsize=(8,8))
sns.barplot(x = training_data['len_summary'], y = training_data['project_is_approved'], ci = False)
plt.xlabel("Resource Summary Length")
plt.xticks(fontsize=10)
plt.ylabel("Title Length")
plt.title("Acceptance Rate based on Length of Resource Summary")


# In[84]:


plt.figure(figsize=(6,6))
sns.scatterplot(x = training_data['len_essay_3'], y = training_data['len_essay_4'], 
                hue = training_data['project_is_approved'], s = 5)


plt.xlabel("Length of Essay 3")
plt.ylabel("Length of Essay 4")
plt.title("Acceptance of projects based on Length of Essays 3 & 4")
plt.xlim(0, 225)
plt.ylim(0,125)
# plt.legend(labels=['Not Approved', 'Approved'])
plt.show()


# In[85]:


essay_three_split = training_data['project_essay_3'].isna().sum()
filled = training_data['project_essay_3'].count()

print(essay_three_split)
print(filled)

data = [['Filled', filled],  ['Empty', training_data['project_essay_3'].isna().sum()]] 

df = pd.DataFrame(data, columns = ['Filled In?', '# of Projects']) 

plt.title("Third Essay Filled In?")

sns.barplot(x = df['Filled In?'], y = df['# of Projects'], ci = False)

