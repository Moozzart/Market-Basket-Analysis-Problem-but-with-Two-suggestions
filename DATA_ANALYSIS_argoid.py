#!/usr/bin/env python
# coding: utf-8

# Importing Important Libraries

# In[62]:


import pandas as pd
import mlxtend
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import pandas as pd #Python data analysis library
import numpy as np #Python scientific computing
import matplotlib.pyplot as plt #For plotting
import matplotlib.mlab as mlab
import seaborn as sns #Python visualization library
from scipy.optimize import curve_fit
from IPython.display import display, HTML
import numpy as np
df=pd.read_csv('transaction_data.csv')


# In[2]:


df['ItemDescription'].nunique()   #total unique items


# In[3]:


df.head()


# In[20]:


data.columns #column names for reference


# Checking for NAN/Null values

# In[4]:


df.isnull().sum()


# There are 2908 null values in item description, they are seen in cell below 

# In[5]:


df[df['ItemDescription'].isnull()==True].head()


# In[98]:


data=df.dropna() #making a copy of dataset without nan values


# In[13]:


data[data['CostPerItem']<0]  #to remove the negative CostPerItem rows


# In[15]:


data[data['NumberOfItemsPurchased']<0]  #to remove the negative NumberOfItemsPurchased rows


# In[24]:


#taking only positive CostPerItem and NumberOfItemsPurchased
data=data[data['CostPerItem']>0]
data=data[data['NumberOfItemsPurchased']>0] 
data.reset_index(inplace =True)


# To check which all products have negative ItemCode

# In[101]:


data[data['ItemCode']==-1]['ItemDescription'].unique()


# Apart from GIRLS PARTY BAG, BOYS PARTY BAG, and PADS TO MATCH ALL CUSHIONS, other products are of no use, most of them are related to basic transaction or fees 

# In[26]:


#Changing the ItemCode of 'BOYS PARTY BAG', 'GIRLS PARTY BAG' and 'PADS TO MATCH ALL CUSHIONS' to 1,2 and 3 respectively from previously set to -1
for i in range(len(data['ItemDescription'])):
    if data['ItemDescription'][i] =='BOYS PARTY BAG':
        data['ItemCode'][i]=1
    if data['ItemDescription'][i] =='GIRLS PARTY BAG':
        data['ItemCode'][i]=2
    if data['ItemDescription'][i] =='PADS TO MATCH ALL CUSHIONS':
        data['ItemCode'][i]=3


# In[27]:


data=data[data['ItemCode']>0] #now removing the rows with negative ItemCode 


# In[28]:


data=data.reset_index()


# # This code cell is not important, I just tried to change the TransactionTime column type to be a DateTime object, You can continue with the cell after but if you want to run,then it'll take 10-15 mins to run this, so better to skip this 

# In[ ]:


months={'Jan':'01',
        'Feb':'02',
        'Mar':'03',
        'Apr':'04',
        'May':'05',
        'Jun':'06',
        'Jul':'07',
        'Aug':'08',
        'Sep':'09',
        'Oct':'10',
        'Nov':'11',
        'Dec':'12'}
for idx,dates in enumerate(data['TransactionTime']):
    dates=dates.replace('IST','')
    date=dates[21:25]+'-'+months[dates[4:7]]+'-'+dates[8:10]+' '+dates[11:19]
    data['TransactionTime'][idx]=date


# In[30]:


data.head()


# In[31]:


data.columns


# In[34]:


#changing the ItemDescription to a string and stripping it. Also doing other important typecastings and dropping the two waste index columns 
data['ItemDescription']=data['ItemDescription'].str.strip()
data.dropna(axis=0, subset=['TransactionId'],inplace=True)
data['TransactionId']=data['TransactionId'].astype('str')
data[~data['TransactionId'].str.contains('C')]
data.drop(['level_0','index'],inplace=True,axis=1)


# In[35]:


data.head()


# In[52]:


#Making a basket dataframe in which all the Transactions are reported with the total of each products separately they bought as the columns 
basket=(data.groupby(['TransactionId','ItemCode'])['NumberOfItemsPurchased'].sum().unstack().reset_index().fillna(0).set_index('TransactionId'))
basket.head()


# In[53]:


#Hot Encoding the transacted products to signify they are bought or not
def encode_units(x):
    if x<=0:
        return 0
    if x>=1:
        return 1
basket_sets=basket.applymap(encode_units)
basket_sets.head()


# # Using Aprori Algorithm Library to get Frequently bought Item sets

# In[54]:


frequent_itemsets=apriori(basket_sets, min_support=0.020, use_colnames=True)
frequent_itemsets.head(10)


# Defining Our Rules, metric used is "Lift"
# 

# ![Capture.JPG](attachment:Capture.JPG)

# In[55]:


rules=association_rules(frequent_itemsets, metric='lift',min_threshold=0.1)


# In[56]:


rules.head()


# In[57]:


#Sorting rules by the lift column that means more lift would be at the top with more chances of goods getting transacted together
rules=rules.sort_values(by='lift',ascending=False) 
rules.head()


# In[72]:


#variations of support vs onfidence
from matplotlib import pyplot as plt
plt.scatter(rules['support'], rules['confidence'], alpha=0.5)
plt.xlabel('support')
plt.ylabel('confidence')
plt.title('Support vs Confidence')
plt.show()


# In[73]:


#variations of support vs lift
plt.scatter(rules['support'], rules['lift'], alpha=0.5)
plt.xlabel('support')
plt.ylabel('lift')
plt.title('Support vs lift')
plt.show()


# In[58]:


#To view all the rows if needed
pd.options.display.max_rows
pd.set_option('display.max_rows', None)


# # Now making a count dataframe in which the number of a particular item  that is purchased per user is given  

# In[59]:


counts=(data.groupby(['UserId','ItemCode'])['NumberOfItemsPurchased'].sum().unstack().reset_index().fillna(0).set_index('UserId'))


# In[60]:


counts.head()


# In[61]:


#to get total items purchased by a user
counts['total_bought']=counts.sum(axis=1)


# In[63]:


#get distribution of number of orders per customer
sns.set_style('whitegrid')
sns.distplot(counts["total_bought"], kde=True, norm_hist=True ,bins=15)


# Here we can clearly see that the data is vastly skewed

# But if we remove the values greater than 25000 in total_bought column, then we see a exponential trend in the graph from the peak

# In[65]:


a=counts[counts['total_bought']<25000] 
a.info()


# In[66]:


b=counts[counts['total_bought']>25000] 
b.info()


# To put into perspective the values above 25000 are just 175 hence as opposed to 4160 below 25000, hence we can remove them to know the pattern of total items bought per customer

# In[67]:


#get distribution of number of orders per customer
sns.set_style('whitegrid')
sns.distplot(a["total_bought"], kde=True, norm_hist=True ,bins=15)


# Exponential trend is clearly visible in this graph, i.e the number of items bought by each customer decreases exponentially 

# Now seeing the trend in the extremely skewed counts['total_bought'] by taking log

# In[68]:


counts['log_total']=np.log(counts['total_bought'])


# In[69]:


#get distribution of number of orders per customer
sns.set_style('whitegrid')
sns.distplot(counts["log_total"], kde=True, norm_hist=True ,bins=15)


# Clearly this seems an approximate gaussian distribution centered at 8

# In[70]:


#To see the counts of every item that has been bought till last date
value_counts=pd.DataFrame(data['ItemCode'].value_counts())
value_counts.reset_index(inplace=True)
value_counts.columns=['ItemCode','times_bought']
value_counts.head(10)


# In[71]:


#to see the number of items purchased by people living in different countries
data.groupby(['Country'])['NumberOfItemsPurchased'].sum()


# # Now doing Recommendation Task

# In[74]:


type(rules['antecedents'][0])


# Since the type of antecedents and consequents are frozen set, I'll change those values to list for easier manipulations

# In[75]:


for i in range(len(rules['antecedents'])):
    rules['antecedents'][i]=list(rules['antecedents'][i])
    rules['consequents'][i]=list(rules['consequents'][i])


# In[76]:


#getting the couplets of itemcode in descending order of their lift values
l=[]
for i in range(len(rules['antecedents'])):
    if len(rules['antecedents'][i])==1 and len(rules['consequents'][i])==1:
        l.append(rules['antecedents'][i]+rules['consequents'][i])
    elif len(rules['antecedents'][i])==2:
        for j in rules['antecedents'][i]:
            l.append([j]+rules['consequents'][i])
    elif len(rules['consequents'][i])==2:
        for j in rules['consequents'][i]:
            l.append(rules['antecedents'][i]+[j])    
        


# In[77]:


l


# In[80]:


#function to determine the next two items that the user could buy after buying 'item_bought'
priority=[]
def find_next_recommendation(item_bought):
    for i in l:
        if item_bought==i[0]:
            priority.append(i[1])
    return priority[:2]


# In[82]:


#Example used is for product 434952
find_next_recommendation(434952) 


# In[87]:


antecedents=[]
for i in (l):
    if i not in antecedents:
        antecedents.append(i[0])
result=[]
for j in (antecedents):
    priority=[]
    result.append([j]+find_next_recommendation(j))
    


# In[90]:


result1=pd.DataFrame(result)


# In[92]:


result1.columns=['Antecedents','Consequent1','Consequent2']


# In[93]:


result1


# So, you can clearly see that very few of the values of consequent2 are NaN, that's because there aren't any cases where after buying that antecedent, there is only one item that is being bought, albeit, I haven't removed them

# # Output file generation

# In[95]:


result1.to_csv(r'C:\Users\Utkarsh Pratap Singh\Desktop\Output.csv', index = False)

