#!/usr/bin/env python
# coding: utf-8

# In[37]:


import os

# Get the current directory of the notebook
notebook_path = os.path.abspath("__file__")

print("Current notebook path:", notebook_path)


# In[38]:


pip install textblob


# In[1]:


pip install wordcloud


# In[2]:


get_ipython().system('pip install Seaborn')


# In[3]:


get_ipython().system(' pip install cufflinks')


# In[4]:


import numpy as np
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import re
from textblob import TextBlob
from wordcloud import WordCloud
import seaborn as sns
import matplotlib.pyplot as plt
import cufflinks as cf
get_ipython().run_line_magic('matplotlib', 'inline')
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
cf.go_offline();
import plotly.graph_objs as go
from plotly.subplots import make_subplots

import warnings
warnings.filterwarnings("ignore")
warnings.warn("This will not show")
pd.set_option('display.max_columns',None)


# In[5]:


df=pd.read_csv('amazon_sentiment_analysis_df.csv')
df.head()


# In[6]:


df=df.sort_values("wilson_lower_bound",ascending=False)


# In[7]:


df.drop('Unnamed: 0',inplace=True,axis=1)
df.head()


# In[8]:


#  df.isnull().sum()
na_columns_=[col for col in df.columns if df[col].isnull().sum()>0]
n_miss=df[na_columns_].isnull().sum().sort_values(ascending=True)
n_miss
df[na_columns_].isnull().sum()/ df.shape[0]


# In[11]:


#missing values
def missing_values_analysis(df):
    na_columns_=[col for col in df.columns if df[col].isnull().sum()>0]
    n_miss=df[na_columns_].isnull().sum().sort_values(ascending=True)
    ratio_=(df[na_columns_].isnull().sum()/ df.shape[0]*100).sort_values(ascending=True)
    missing_df=pd.concat([n_miss,np.round(ratio_,2)],axis=1,keys=['Missing Values','Ratio'])
    missing_df=pd.DataFrame(missing_df)
    return missing_df

def check_dataframe (df,head=5,tail=5):
    print("Shape".center(82,'~'))
    print('Rows:{}'.format(df.shape[0]))
    print('columns:{}'.format(df.shape[1]))
    print("Types".center(82,'~'))
    print(missing_values_analysis(df))
    print("DUPLICATED VALUES".center(83,'~'))
    print(df.duplicated().sum())
    print("Quantiles".center(82, '~'))
    numeric_df = df.select_dtypes(include=[np.number])  # Select only numeric columns
    print(numeric_df.quantile([0, 0.05, 0.5, 0.95, 1]).T)
    
check_dataframe(df)


# In[12]:


def check_class(dataframe):
    nunique_df=pd.DataFrame({'Variable':dataframe.columns,
                            'Classes':[dataframe[i].nunique()\
                                      for i in dataframe.columns]})
    nunique_df=nunique_df.sort_values('Classes',ascending=False)
    nunique_df=nunique_df.reset_index(drop=True)
    return nunique_df

check_class(df)


# In[13]:


constraints = ['#B34D22', '#EBE00C', '#1FEB0C', '#0C92EB', '#EB0CD5']

def categorical_variable_summary(df, column_name):
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=('Countplot', 'Percentage'),
                        specs=[[{"type": "xy"}, {"type": 'domain'}]])

    fig.add_trace(go.Bar(y=df[column_name].value_counts().values.tolist(),
                         x=[str(i) for i in df[column_name].value_counts().index],
                         text=df[column_name].value_counts().values.tolist(),
                         textfont=dict(size=14),
                         textposition='auto',
                         showlegend=False,
                         name=column_name,
                         marker=dict(color=constraints,
                                     line=dict(color='#DBE6EC', width=1))),
                  row=1, col=1)
    fig.add_trace(go.Pie(labels=df[column_name].value_counts().keys(),
                         values=df[column_name].value_counts().values,
                         textfont=dict(size=18),
                         textposition='auto',
                         showlegend=False,
                         name=column_name,
                         marker=dict(colors=constraints)),
                  row=1, col=2)
    fig.update_layout(title={'text': column_name,
                             'y': 0.9,
                             'x': 0.5,
                             'xanchor': 'center',
                             'yanchor': 'top'},
                      template='plotly_white')
    iplot(fig)


# In[14]:


categorical_variable_summary(df,'overall')


# In[15]:


df.reviewText.head()


# In[16]:


review_example=df.reviewText[2031]
review_example


# In[17]:


review_example=re.sub("[^a-zA-Z]",'',review_example)
review_example


# In[18]:


review_example=review_example.lower().split()


# In[19]:


review_example


# In[20]:


rt=lambda x: re.sub("[^a-zA-Z]",' ',str(x))
df['reviewText']=df['reviewText'].map(rt)
df['reviewText']=df['reviewText'].str.lower()
df.head()


# In[21]:


pip install vaderSentiment


# In[22]:


from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
df[['polarity','subjectivity']]=df['reviewText'].apply(lambda Text:pd.Series(TextBlob(Text).sentiment))

for index, row in df['reviewText'].iteritems():
    score=SentimentIntensityAnalyzer().polarity_scores(row)
    
    neg=score['neg']
    neu=score['neu']
    pos=score['pos']
    if neg>pos:
        df.loc[index, 'sentiment']='Negative'
    elif pos > neg:
        df.loc[index,'sentiment']='Positive'
    else:
        df.loc[index,'sentiment']='Neutral'


# In[23]:


import pandas as pd
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Assuming df is your DataFrame and 'reviewText' is the column with the text reviews

# Calculate polarity and subjectivity using TextBlob
df[['polarity', 'subjectivity']] = df['reviewText'].apply(lambda Text: pd.Series(TextBlob(Text).sentiment))

# Initialize the SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()

# Define a function to classify sentiment
def classify_sentiment(row):
    score = analyzer.polarity_scores(row)
    neg = score['neg']
    neu = score['neu']
    pos = score['pos']
    if neg > pos:
        return 'Negative'
    elif pos > neg:
        return 'Positive'
    else:
        return 'Neutral'

# Apply the function to each row in the 'reviewText' column
df['sentiment'] = df['reviewText'].apply(classify_sentiment)

# Display the DataFrame to verify the results
print(df.head())


# In[24]:


df[df['sentiment']=='Positive'].sort_values('wilson_lower_bound',ascending=False).head(5)


# In[25]:


categorical_variable_summary(df,'sentiment')


# In[ ]:




