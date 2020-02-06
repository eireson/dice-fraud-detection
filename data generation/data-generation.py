#!/usr/bin/env python
# coding: utf-8

# # D&D Dice Fraud Detection Software: Data Generation Script

# ## 0. Global Imports

# Please keep this area tidy. We separate the project into multiple segments, this one concerns generation and assembly of a dice-statistics dataset. This is done so as to avoid environment conflicts specifically around TensorFlow.
#
# Please comment out all testing code that produce printouts.

# In[1]:


import numpy as np
import pandas as pd


# ## 1. Generate Real Data

# Real data is generated by the formula as printed in the book: 4d6dl1 i.e. the sum of four random numbers in (1,6) of which the lowest is dropped

# In[2]:


N_real=5000
N_fake=1000
#Tweak these numbers multiplicatively by the same ratio.


# In[3]:


def FourSixSidedDiceDropLowest():
    randnums= np.random.randint(1,7,4)
    return sum(np.delete(np.sort(randnums),0))

realSample=np.array([[FourSixSidedDiceDropLowest() for i in range(6)] for i in range(N_real)])
print('Real Data generated')


# ## 2. Generate different types of faked data

# We must use a great variety of formulae or algorithms in order to present faked data.
# Main suggestions:
# * More dice, more dropped: 5d6dl2 will make the average go up while keeping the upper bound the same. Up to and including (x+1)d20dlx to make it very obvious
# * Smaller dice: 6d4dl2 has lower upper bound as 4d6dl1 but a higher average
# * Drop fewer dice: more extreme outliers can be generated by removing fewer dice. This pushes the upper bound beyond 18, which is obviously impossible, but the algorithm should see it to know.
# * Roll dice set multiple times and keep the "best": high-average, low-variance stat sets are infrequent but manifestly better. An example ''goodness indicator'' is the following formula: mu / (5 + sigma)
# * For good measure, ''handicap'' sets should also be given, so that the algorithm doesn't automatically think any statistical upper outlier is a cheat

# In[4]:


#This function implements any arbitrary XdYdlZ formula. All dice are assumed identical.
def numbersizedrop(number,size,drop):
    randnums= np.random.randint(1,size+1,number)
    return sum(np.delete(np.sort(randnums),range(drop)))

#This implements the "get lucky" algorithm whereby we keep rolling until
#we find a good statistical outlier with high average and low variance
def keeprolling(number,size,drop,targetcoeff):
    coeff=0
    i=0
    while coeff < targetcoeff:
        tstats= np.array([numbersizedrop(number,size,drop) for i in range(6)])
        i+=1
        if coeff < 2*np.mean(tstats) / (5+ np.std(tstats)):
            coeff =2*np.mean(tstats) / (5+ np.std(tstats))
            stats=tstats
            #print('New candidate at step {2}: coeff. {0} with stats {1}'.format(coeff, stats, i))
    #print('Final candidate at step {2}, coeff {0} with stats {1}'.format(coeff, stats,i))
    return stats





# In[5]:


#We use a concatenation of ndarrays so that the code is somewhat scalable.
#Adding a new generation method is then simply a case of pasting another block.
#This also makes converting the data easier since we can use an enumerator.
fakeSample=np.array([[[numbersizedrop(5,6,2) for i in range(6)] for i in range(N_fake)]])

fakeSample=np.concatenate(
    (
        fakeSample,
         np.array([[[numbersizedrop(6,4,2) for i in range(6)] for i in range(N_fake)]])
    )
    ,axis=0)

fakeSample=np.concatenate(
    (
        fakeSample,
         np.array([[[numbersizedrop(6,20,5) for i in range(6)] for i in range(N_fake)]])
    )
    ,axis=0)

fakeSample=np.concatenate(
    (
        fakeSample,
         np.array([[[numbersizedrop(3,20,2) for i in range(6)] for i in range(N_fake)]])
    )
    ,axis=0)

fakeSample=np.concatenate(
    (
        fakeSample,
         np.array([[[numbersizedrop(4,6,2) for i in range(6)] for i in range(N_fake)]])
    )
    ,axis=0)

fakeSample=np.concatenate(
    (
        fakeSample,
         np.array([[[numbersizedrop(1,20,0) for i in range(6)] for i in range(N_fake)]])
    )
    ,axis=0)

fakeSample=np.concatenate(
    (
        fakeSample,
         np.array([[[numbersizedrop(4,4,0) for i in range(6)] for i in range(N_fake)]])
    )
    ,axis=0)

fakeSample=np.concatenate(
    (
        fakeSample,
         np.array([[[numbersizedrop(2,8,0) for i in range(6)] for i in range(N_fake)]])
    )
    ,axis=0)


# In[6]:


N_fakecats=len(fakeSample)
print('Fake Data Generated with {0} different methods'.format(N_fakecats))


# ## 3. Assemble data in Pandas DataFrames

# A Pandas DataFrame object is most naturally suited for statistical analysis and regression. Note that we should keep the Real and Faked data separate until the end of preprocessing in order to balance the data.

# In[7]:


realDataRaw=pd.DataFrame(realSample, columns=['1','2','3','4','5','6'])
#realDataRaw.describe()
#realDataRaw.head()
#describe() explores the statistics of the DataFrame, very useful to compare the various generation methods


# In[ ]:





# In[8]:


fakeDataLists=[
    pd.DataFrame(fakeSample[i], columns=['1','2','3','4','5','6']) for i in range(np.shape(fakeSample)[0])
]
#DataFrames are necessarily two-dimensional, so to start with we make one frame per method for fake data.
#This allows us to explore the statistics of the fake data before fusing it all together.


# In[9]:


#fakeDataLists[0].describe()


# In[10]:


fakeDataRaw=pd.concat(
    fakeDataLists
    ,keys=[i+1 for i in range(np.shape(fakeSample)[0])]
    #,ignore_index=True
)
#pd.concat takes an optional list of keys to create a hierarchy of column values.
#Alternatively, flatten all the data together with the ignore_index optional argument
#fakeDataRaw.tail()


# We can optionally save the raw data now for later exploration

# In[11]:


realDataRaw.to_pickle('./realDataRaw.zip')
fakeDataRaw.to_pickle('./fakeDataRaw.zip')
print('Saving raw data consisting of {0} real data points, {1} fake data generation methods each with {2} entries'.format(N_real, N_fakecats, N_fake))


# ## 4. Preprocess data

# We must perform a variety of operations on the data before it is analysed by the ML algorithm.
#
# * __Balancing__: Ideally the ML algorithm has about as many data points in each category when trying to sort categorical or binary data, which in this case means aiming for a 50% true/false split. It is easiest to perform this operation here, since we must come up with many different ways of generating fake data, it will be tedious to control how much of the fake data we generate.
# * We **do not perform feature scaling** (removal of mean and standard error from the distribution) since a tell-tale sign of fraud would in fact be a higher than expected average or lower spread. These are vital to the analysis. A second stage of the analysis could work on feature-scaled data to estimate if fraud has been operated on the level of skewness, curtosis or higher moments, but the most obvious and effective instances of fraud happen at the level of average and standard deviation.
# * __Merge__: both sets need to be in one DataFrame by the end of it
# * __Shuffling__: Shuffling the real and fake data inside the DataFrames in order to remove correlations

# In[12]:


realDataRaw['source']=[1 for i in range(N_real)]
fakeDataRaw['source']=[0 for i in range(np.shape(fakeDataRaw)[0])]
fakeDataRaw.tail()
#Tail check to see that the row label structure is fine


# ### Scaling (not implemented as a first pass)

# In[13]:


#from sklearn import preprocessing
#realDataScaled = pd.DataFrame(preprocessing.scale(realDataRaw))
#fakeDataScaled = pd.DataFrame(preprocessing.scale(fakeDataRaw))
#this removes the mean and divides by the average


# ### Balancing

# In[14]:


fakeDataBalanced=pd.concat(
    [
        fakeDataRaw.xs(i+1).sample(int(np.ceil(N_real / N_fakecats)))
     for i in range(N_fakecats)
    ]
)


# ### Merge and Shuffle

# In[15]:


allDataBalanced= pd.concat([realDataRaw, fakeDataBalanced],ignore_index=True)
allDataShuffled=allDataBalanced.sample(frac=1)
allDataShuffled.reset_index(drop=True, inplace=True)


# Now would be a good time to save the raw data in a file. TensorFlow doesn't (natively) handle Pandas Dataframes, but Numpy objects. Since the index structure is now irrelevant, we can convert it to a NumPy array and save it in a compressed form (saves 90% space)

# In[27]:


np.savez_compressed("allDataShuffled.npz",
                    inputs=np.array(allDataShuffled.drop('source',axis=1)),
                    targets=np.array(allDataShuffled['source'])
                   )

print('Job complete, saved {0} balanced and shuffled, fake and real data in equal amounts'.format(2*N_real) )


# This completes the data generation protocol.
