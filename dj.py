#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import chart_studio.plotly as py
import plotly.graph_objs as go
from plotly.offline import plot

#for offline plotting
from plotly.offline import download_plotlyjs, init_notebook_mode,plot,iplot
init_notebook_mode(connected=True)


# In[3]:


tata=pd.read_csv(r"C:\Users\91950\Downloads\TCS.csv")
tata.head()


# In[4]:


tata.info()


# In[5]:


tata['Date']=pd.to_datetime(tata['Date'])


# In[6]:


print(f'dataframes contains stock prises between{tata.Date.min()} {tata.Date.max()}')
print(f'total days={(tata.Date.max()-tata.Date.min())}days')


# In[7]:


tata.describe()


# In[8]:


tata[['Open Price','High Price','Low Price','Close Price','WAP']].plot(kind='box')


# In[9]:


#setting layout for the plot
layout=go.Layout(
    title='stock prices of tata',
    xaxis=dict(
        title='Date',
        titlefont=dict(
            family='Courierspace New, monospace',
            size=18,
            color='#7f7f7f'
        )
    ),
    yaxis=dict(
        title='Price',
        titlefont=dict(
            family='Courierspace New, monospace',
            size=18,
            color='#7f7f7f'
        )
    )
)
tata_data=[{'x':tata['Date'],'y':tata['Close Price']}]
plot=go.Figure(data=tata_data,layout=layout)
    
    


# In[29]:


iplot(plot)


# In[11]:


# building the regression model
from sklearn.model_selection import train_test_split
from sklearn.model_selection import *

# for processing 
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

# for model evaluation
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score


# In[12]:


pip install scikit-learn


# In[32]:


#spllit data into train and test set
X = np.array(tata.index).reshape(-1,1)
Y = tata['Close Price']
X_train,X_test,Y_train,Y_test = train_test_split(X,Y, test_size=0.3, random_state=101) 


# In[34]:


#feature scalling
scarler=StandardScaler().fit(X_train)


# In[37]:


from sklearn.linear_model import LinearRegression


# In[38]:


#creating a linear model
lm=LinearRegression()
lm.fit(X_train,Y_train)


# In[47]:


#plot actual and linear model for train dataset
trace0=go.Scatter(
    x=X_train.T[0],
    y=Y_train,
    mode='markers',
    name='Actual'
)
trace1=go.Scatter(
    x=X_train.T[0],
    y=lm.predict(X_train).T,
    mode='lines',
    name='Pridicted'
)
tata_data=[trace0,trace1]
layout.xaxis.title.text='day'
plot2=go.Figure(data=tata_data,layout=layout)
    


# In[51]:


iplot(plot2)


# In[61]:


#calculate score for model evaluation
score=f'''
{'Metric'.ljust(10)}{'Train'.center(20)}{'Test'.center(20)}
{'r2_score'.ljust(10)}{r2_score(Y_train,lm.predict(X_train))}/t{r2_score(Y_test,lm.predict(X_test))}
{'MSE'.ljust(10)}{mse(Y_train,lm.predict(X_train))}/t{mse(Y_test,lm.predict(X_test))}
'''
print(score)


# In[ ]:




