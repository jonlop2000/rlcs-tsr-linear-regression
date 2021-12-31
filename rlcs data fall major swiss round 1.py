#!/usr/bin/env python
# coding: utf-8

# In[24]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# # All Fall Major Stats 

# In[15]:


rlcs_data = pd.read_csv('rlcs-fall-major-data.csv')
print(rlcs_data)


# In[16]:


rlcs_data.head()


# In[17]:


rlcs_data.columns


# ## Total Shot Ratio (TSR)

# In[18]:


print(rlcs_data[["team name","win percentage","shots per game", "shots conceded per game"]])


# In[19]:


win_percentage = rlcs_data["win percentage"]
shots_con_pg = rlcs_data["shots conceded per game"]
shots_pg = rlcs_data["shots per game"]


# In[45]:


tsr = ((shots_pg)/(shots_pg + shots_con_pg))


# ## Linear Regression Model
# 

# In[47]:


plt.xlabel('Total Shot Ratio')
plt.ylabel('Win Percentage')

plt.scatter(tsr,win_percentage)


# In[48]:


#Test Train split for supervised training 
x_train, x_test, y_train, y_test = train_test_split(tsr, win_percentage)


# In[49]:


plt.scatter(x_train, y_train, label = 'Training Data', color = 'r', alpha =.7)
plt.scatter(x_test, y_test, label = 'Testing Data', color = 'g', alpha =.7)
plt.legend()
plt.title("Test train data")
plt.show()


# In[50]:


# Create linear model and train it
LR = LinearRegression()
LR.fit(x_train.values.reshape(-1,1), y_train.values)


# In[51]:


prediction = LR.predict(x_test.values.reshape(-1,1))

plt.plot(x_test, prediction, label = 'Linear Regression', color = 'b')
plt.scatter(x_test, y_test, label = 'Actual Test Data', color = 'g', alpha =.7)
plt.legend()
plt.show()


# In[52]:


LR.predict(np.array([[.5]]))[0]


# In[53]:


LR.score(x_test.values.reshape(-1,1),y_test.values)

