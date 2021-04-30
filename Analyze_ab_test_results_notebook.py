#!/usr/bin/env python
# coding: utf-8

# ## Analyze A/B Test Results
# 
# 
# ## Table of Contents
# - [Introduction](#intro)
# - [Part I - Probability](#probability)
# - [Part II - A/B Test](#ab_test)
# - [Part III - Regression](#regression)
# 
# 
# <a id='intro'></a>
# ### Introduction
# 
# A/B tests are very commonly performed by data analysts and data scientists.  
# For this project, we will be working to understand the results of an A/B test run by an e-commerce website.  My goal is to work through this notebook to help the company understand if they should implement the new page, keep the old page, or perhaps run the experiment longer to make their decision.
# 
# 
# <a id='probability'></a>
# #### Part I - Probability
# 
# To get started, let's import our libraries.

# In[220]:


import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

random.seed(42)


# `1.` Now, read in the `ab_data.csv` data. Store it in `df`.  
# a. Read in the dataset and take a look at the top few rows here:

# In[221]:


df=pd.read_csv("ab_data.csv")
df.head()


# b. Use the cell below to find the number of rows in the dataset.

# In[222]:


df.shape[0]


# c. The number of unique users in the dataset.

# In[223]:


df["user_id"].nunique()


# d. The proportion of users converted.

# In[224]:


df[df["converted"]==1].shape[0]/df["converted"].shape[0]


# e. The number of times the `new_page` and `treatment` don't match.

# In[225]:


treat=df[df["group"]=="treatment"]
t=treat[treat["landing_page"]!="new_page"].shape[0]
new_p=df[df["landing_page"]=="new_page"]
n_p=new_p[new_p["group"]!="treatment"].shape[0]
print(n_p+t)


# f. Do any of the rows have missing values?

# In[226]:


df.describe()


# `2.` For the rows where **treatment** does not match with **new_page** or **control** does not match with **old_page**, we cannot be sure if this row truly received the new or old page.  
# 

# In[227]:


df2=df[((df["group"]=="treatment")&(df["landing_page"]=="new_page") )|((df["group"]=="control")&(df["landing_page"]!="new_page") )]


# In[228]:


# Double Check all of the correct rows were removed - this should be 0
df2[((df2['group'] == 'treatment') == (df2['landing_page'] == 'new_page')) == False].shape[0]


# 3_a. How many unique **user_id**s are in **df2**?

# In[229]:


df2["user_id"].nunique()


# b. There is one **user_id** repeated in **df2**.  What is it?

# In[230]:


df2[df2["user_id"].duplicated()]["user_id"]


# c. What is the row information for the repeat **user_id**? 

# In[231]:


dup_info=df2[df2["user_id"].duplicated()]
dup_info


# d. Remove **one** of the rows with a duplicate **user_id**, but keep the dataframe as **df2**.

# In[232]:


print(df2.shape[0])
df2.drop_duplicates(subset="user_id",keep='first', inplace=True)
print(df2.shape[0])


# 4_a. What is the probability of an individual converting regardless of the page they receive?

# In[233]:


df2[df2["converted"]==1].shape[0]/df2["converted"].shape[0]


# b. Given that an individual was in the `control` group, what is the probability they converted?

# In[234]:


control=df2[df2["group"]=="control"]
control[control["converted"]==1].shape[0]/control["converted"].shape[0]


# c. Given that an individual was in the `treatment` group, what is the probability they converted?

# In[235]:


treatment=df2[df2["group"]=="treatment"]
treatment[treatment["converted"]==1].shape[0]/treatment["converted"].shape[0]


# d. What is the probability that an individual received the new page?

# In[236]:


p_t=df2[df2["group"]=="treatment"].shape[0]/df2.shape[0]
p_c=df2[df2["group"]=="control"].shape[0]/df2.shape[0]
p_npage_c=control[control["landing_page"]=="new_page"].shape[0]/control["landing_page"].shape[0]
p_npage_t=treatment[treatment["landing_page"]=="new_page"].shape[0]/treatment["landing_page"].shape[0]
print(p_t*p_npage_t + p_c*p_npage_c)


# ## I think the provided information may not be sufficient as we have not tested our hypothesis 
# ## testing our hypothesis requires calculating the p value to judge
# 

# <a id='ab_test'></a>
# ### Part II - A/B Test
# 
# Notice that because of the time stamp associated with each event, we could technically run a hypothesis test continuously as each observation was observed.  
# 
# However, then the hard question is do we stop as soon as one page is considered significantly better than another or does it need to happen consistently for a certain amount of time? 
# 
# These questions are the difficult parts associated with A/B tests in general.  
# 
# 
# `1.` For now, consider we need to make the decision just based on all the data provided.  If we want to assume that the old page is better unless the new page proves to be definitely better at a Type I error rate of 5%, what should the null and alternative hypotheses be? 

# ## H0 : Pnew <= Pold   
# ## H1 : Pnew > Pold     

# `2.` Assume under the null hypothesis, $p_{new}$ and $p_{old}$ both have "true" success rates equal to the **converted** success rate regardless of page - that is $p_{new}$ and $p_{old}$ are equal. Furthermore, assume they are equal to the **converted** rate in **ab_data.csv** regardless of the page. <br><br>
# 
# Use a sample size for each page equal to the ones in **ab_data.csv**.  <br><br>
# 
# Perform the sampling distribution for the difference in **converted** between the two pages over 10,000 iterations of calculating an estimate from the null.  <br><br>
# 
# 

# a. What is the **conversion rate** for $p_{new}$ under the null? 

# In[237]:


p_new_null=df2[df2["converted"]==1].shape[0]/df2["converted"].shape[0]
p_new_null


# b. What is the **conversion rate** for $p_{old}$ under the null? <br><br>

# In[238]:


p_old_null=p_new_null  # as given in above cells
p_old_null


# c. What is $n_{new}$, the number of individuals in the treatment group?

# In[239]:


n_new=treatment.shape[0]
n_new


# d. What is $n_{old}$, the number of individuals in the control group?

# In[240]:


n_old=control.shape[0]
n_old


# e. Simulate $n_{new}$ transactions with a conversion rate of $p_{new}$ under the null.  Store these $n_{new}$ 1's and 0's in **new_page_converted**.

# In[256]:


#randomly generated array of 1,0 with prop.=p_new_null

new_page_converted=np.random.choice([1,0],size=n_new,p=[p_new_null,1-p_new_null])
# same as np.random.binomial(1,p_new_null,n_new)
new_page_converted[0:10]


# f. Simulate $n_{old}$ transactions with a conversion rate of $p_{old}$ under the null.  Store these $n_{old}$ 1's and 0's in **old_page_converted**.

# In[257]:


old_page_converted=np.random.choice([1,0],size=n_old,p=[p_old_null,1-p_old_null])
# same as np.random.binomial(1,p_old_null,n_old)
old_page_converted[0:10]


# g. Find $p_{new}$ - $p_{old}$ for the simulated values from part (e) and (f).

# In[258]:


p_new_sim=len(new_page_converted[new_page_converted>0])/len(new_page_converted)
p_old_sim=len(old_page_converted[old_page_converted>0])/len(old_page_converted)
obs=p_new_sim-p_old_sim
print(obs)


# h. Create 10,000 $p_{new}$ - $p_{old}$ values using the same simulation process we used in parts (a) through (g) above. Store all 10,000 values in a NumPy array called **p_diffs**.

# In[259]:


p_diffs=[]
for i in range(10000):
    new_page_converted=np.random.choice([1,0],size=n_new,p=[p_new_null,1-p_new_null])
    
    length1=len(new_page_converted)
    
    ones=len(new_page_converted[new_page_converted>0])
    
    p_new_sim=ones/length1
    
    #------------------
    
    old_page_converted=np.random.choice([1,0],size=n_new,p=[p_old_null,1-p_old_null])
    
    length1=len(old_page_converted)
    
    ones=len(old_page_converted[old_page_converted>0])
    
    p_old_sim=ones/length1
    
    #--------------
    
    p_diffs.append(p_new_sim-p_old_sim)

# ^^  seems complicated ? nevermind just repeating the same cells i used before



p_diffs=np.array(p_diffs)


# i. Plot a histogram of the **p_diffs**.  Does this plot look like what we expected?  

# In[260]:


plt.hist(p_diffs);


# ### yes, it matched the expectations as according to centeral limit theorem the sampling distribution should follow normal distribution   

# j. What proportion of the **p_diffs** are greater than the actual difference observed in **ab_data.csv**?

# In[261]:


(p_diffs>obs).mean()


# ### since P>0.05 then we fail to reject the null hypothesis
# ### which means there might be no difference between the old and the new page 

# # ------------------------------------------------------

# l. We could also use a built-in to achieve similar results.  Though using the built-in might be easier to code, the above portions are a walkthrough of the ideas that are critical to correctly thinking about statistical significance. Fill in the below to calculate the number of conversions for each page, as well as the number of individuals who received each page. Let `n_old` and `n_new` refer the the number of rows associated with the old page and new pages, respectively.

# In[262]:


import statsmodels.api as sm

old_page=df2[df2["landing_page"]=="old_page"]
convert_old =old_page["converted"]

new_page=df2[df2["landing_page"]=="new_page"]
convert_new =new_page["converted"]

n_old =convert_old.shape[0] 
n_new = convert_new.shape[0]

convert_old=convert_old.sum()
convert_new=convert_new.sum()

print(n_old,n_new)


# m. Now lets use `stats.proportions_ztest` to compute the test statistic and p-value.  

# In[263]:


z_statisitc,p_val=sm.stats.proportions_ztest([convert_old,convert_new],[n_old,n_new])
print("z_score = ",z_statisitc," , p_value = ",p_val)


# ### awesome ,the p value nearly matches the simulated value

# n. What do the z-score and p-value computed in the previous question mean for the conversion rates of the old and new pages?  

# ### same meaning as before as the p value matched the previously calculated one (there is no significant difference between the old and the new pages)

# <a id='regression'></a>
# ### Part III - A regression approach
# 
# `1.` In this final part, we will see that the result achieved in the A/B test in Part II above can also be achieved by performing regression.<br><br> 
# 
# a. Since each row is either a conversion or no conversion, what type of regression should we be performing in this case?

# ### Logistic regression as the output is discerete

# b. The goal is to use **statsmodels** to fit the regression model we specified in part **a.** to see if there is a significant difference in conversion based on which page a customer receives. 

# In[264]:


df2["intercept"]=1
df2[["to_be_dropped","ab_page"]]=pd.get_dummies(df2["group"])
df2.drop(labels="to_be_dropped",axis=1,inplace=True)
df2.head()


# c. Use **statsmodels** to instantiate we regression model on the two columns we created in part b., then fit the model using the two columns we created in part **b.** to predict whether or not an individual converts. 

# In[265]:


lg=sm.Logit(df2["converted"],df2[["intercept","ab_page"]])
model=lg.fit()


# d. Provide the summary of our model below, and use it as necessary to answer the following questions.

# In[266]:


model.summary2()


# e. What is the p-value associated with **ab_page**? Why does it differ from the value we found in **Part II**?<br><br>  **Hint**: What are the null and alternative hypotheses associated with the regression model, and how do they compare to the null and alternative hypotheses in **Part II**?

# ### the P value = 0.1899 for the ab_page , the difference is too small to consider the two values different
# 
# ### the null hypothesis is assuming the new conversion rate is less than or equal to the old one
# ### while
# ### the alternative hypothesis is predicting the new conversion rate is going to be higher than the older rate.

# f. Now, considering other things that might influence whether or not an individual converts.  Discuss why it is a good idea to consider other factors to add into the regression model.  Are there any disadvantages to adding additional terms into the regression model?

# ### in my opinion we need to add other factors as we can see that the R squared coeff. is 0 or near 0 which means we need other factors to determine the relation 

# g. Now along with testing if the conversion rate changes for different pages, also add an effect based on which country a user lives in. we will need to read in the **countries.csv** dataset and merge together the datasets on the appropriate rows.  
# 
# Does it appear that country had an impact on conversion?  Don't forget to create dummy variables for these country columns - 

# In[267]:


countries=pd.read_csv("countries.csv")
countries.head()

# we want to assign each user to his or her country
# so we make the index of both dataframes user_id then join them


df2 = df2.set_index('user_id').join(countries.set_index('user_id'))


# In[272]:


df2["country"].unique()


# In[273]:


df2[["US","CA","UK"]]=pd.get_dummies(df2["country"])


# In[274]:


df2.head()


# In[275]:


lg2=sm.Logit(df2["converted"],df2[["intercept","ab_page","US","CA"]])
res=lg2.fit()
res.summary2()


# ### by looking at the p values we may notice that these factors are insignificant

# h. Though we have now looked at the individual factors of country and page on conversion, we would now like to look at an interaction between page and country to see if there significant effects on conversion.  Create the necessary additional columns, and fit the new model.  
# 
# Provide the summary results, and the conclusions based on the results.

# In[277]:


df2["UK_X_page"]=df2["UK"]*df2["ab_page"]
df2["US_X_page"]=df2["US"]*df2["ab_page"]
df2["CA_X_page"]=df2["CA"]*df2["ab_page"]

lg3=sm.Logit(df2["converted"],df2[["intercept","ab_page","US_X_page","CA_X_page"]])
res=lg3.fit()
res.summary2()


# ###  still no significance , let's try UK_X_page

# In[278]:


lg4=sm.Logit(df2["converted"],df2[["intercept","ab_page","US_X_page","UK_X_page"]])
res=lg4.fit()
res.summary2()


# ## Finally , we got some interesting results the P value of US_X_page tells us that our results are now significant (p<alpha)

# ## ------------------------------------------------------------------------------------------------------------------------

# <a id='conclusions'></a>
# #### Conclusions were discussed for each situation individually
# 

# In[279]:


from subprocess import call
call(['python', '-m', 'nbconvert', 'Analyze_ab_test_results_notebook.ipynb'])


# In[ ]:




