#!/usr/bin/env python
# coding: utf-8

# # Assignment 2: Milestone I Natural Language Processing
# ## Task 1. Basic Text Pre-processing
# #### Student Name: Toh Kah Hie
# #### Student ID: 3936897
# 
# Date: 12 September 2022
# 
# Version: 1.0
# 
# Environment: Python 3 and Jupyter notebook
# 
# Libraries used:
# * sklearn
# * re
# * numpy
# * collections
# * random
# * nltk
# * itertools
# 
# ## Introduction
# In this task, given a directory of job advertisements in the form of text files, the files are loaded and explored. After some initial exploration, those data would be stored into a Dictionary. After that, pre-processing steps would be done on only the **descriptions** of each job advertisements. After pre-processing the descriptions, a Unigram would be generated to get the vocabulary of the descriptions and including the Dictionary, they will be saved into 2 text files named ```vocab.txt``` and ```job_data.txt``` respectively.

# ## Importing libraries 

# In[1]:


# Task 1.1 Loading data and exploration
from sklearn.datasets import load_files
from collections import Counter
import numpy as np
import random
import re

# Task 1.2.1 Tokenization
import nltk
from nltk import RegexpTokenizer
from nltk.tokenize import sent_tokenize
from itertools import chain

# Task 1.2.4 Remove least frequent words
from nltk.probability import *

# Task 1.2.6 Build a unigram
from nltk.util import ngrams

# Task 1.2.8
import csv
import pandas as pd


# ### 1.1 Examining and loading data
# In this task, the data is loaded from the directory into a form of dictionary. The dictionary keys would first be explored to see what fields it has. After that, some initial explorations would be done including
# * Number of data for each field
# * Content of each dictionary field
# * Number of data for each category
# * Total number of data

# In[2]:


# Load the data files
job_data = load_files(r'data',encoding='utf-8')


# #### Task 1.1.1 Number of data under each key of dictionary

# In[3]:


# Get keys of loaded files
job_data.keys()


# According to the [documentation](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_files.html), the 'DESCR' attribute stores the full description of the dataset. Let's check it out

# In[4]:


print(job_data['DESCR'])


# From above, it can be seen that there is no description for the dataset.

# In[5]:


# Skip the last key which is 'DESCR'
for key in list(job_data.keys())[:-1]:
    print("Number of elements in",key,":",len(job_data[key]))
print('\n')
print('Target: ',set(job_data['target']))
print("Target Names:",job_data['target_names'])


# From above it can be seen that there are a total of 776 data and each is assumed to be assigned a number which corresponds to the 4 target names listed. Let's check out some of the data

# #### Task 1.1.2 Map target values to target names

# In[6]:


# Initialize a counter
counter = Counter()
# For each target number
for category_num in range(0,4):
    # Get data with the specific target number
    category_data = np.where(job_data['target'] == category_num)
    # Update the counter
    counter[str(category_num)] += len(category_data[0])
    # Pick a random data under that target
    indx = random.choice(category_data[0])
    print("Category Number",str(category_num),":",job_data['filenames'][indx])


# The code above randomly picks data under each target (0-4). It can be that each target points to target names as follows:
# * 0 - Accounting_Finance
# * 1 - Engineering
# * 2 - Healthcare_Nursing
# * 3 - Sales
# 
# To make sure that each job advertisement points to exactly one category, the total sum of data for all categories should be 776, which is the number of data present in the dataset

# #### Task 1.1.3 Number of data under each target

# In[7]:


# Print number of data under each target
[print("Number of data for target",target,":",str(value)) for target,value in counter.items()]

# Print the sum
print('Sum of data for all targets:',sum(counter.values()))


# The sum of data under each target does come down to 776, which indicates that each data falls under exactly one target. From there, it also shows the number of data for each job category:
# * Accounting_Finance: 191
# * Engineering: 231
# * Healthcare_Nursing: 198
# * Sales: 156

# Now, let's look into some of the contents data

# #### Task 1.1.4 Contents of each data

# In[8]:


indx = random.choice(range(0,len(job_data['data'])))
print("Data at index:",indx)
job_data['data'][indx]


# It seems like within the string there are multiple strings separated by '\n'! Let's split them into individual strings

# In[9]:


job_info = [data.splitlines() for data in job_data['data']]
indx = random.choice(range(0,len(job_data['data'])))
job_info[indx]


# From here it can be seen that for each job, it contains fields like 'Title', 'Webindex','Company' and 'Description'. Let's see if every single data has those fields.

# In[10]:


field_pattern = r'^(\w+): .*'
fieldCounter = Counter()
for data in job_info:
#     for field in data:
    [fieldCounter.update(re.match(field_pattern,field).groups()) for field in data]
fieldCounter


# It can be seen that all data contains the 'Title', 'Webindex' and 'Description' with some missing the 'Company' column. Let's convert all into a dictionary.

# In[11]:


print(list(fieldCounter.keys()))


# #### Task 1.1.5 Save the data into a dictionary

# In[12]:


# Initialize an empty dictionary to store all data
# jobDict = dict.fromkeys(fieldCounter.keys(),[]) * Cannot be used for mutable objects e.g. list
jobDict = {key:[] for key in fieldCounter.keys()}
print(jobDict)

# Pattern to match keys within the data
field_pattern = r'^(\w+): (.*)'

# Function to add data into the dictionary
def addToDict(fields):
    # If the number of fields is less than 4 we could guarantee that the Company column is missing
    # As Company column is the only column with missing values
    for field in fields:
        values = re.match(field_pattern, field).groups()
        jobDict[values[0]].append(values[1])
    if len(fields) < 4:
        jobDict['Company'].append('')


# In[13]:


[addToDict(data) for data in job_info]
for key in jobDict.keys():
    print("Number of data for",key,":",len(jobDict[key]))


# Now that all data is set properly, let's have a look at their data types!

# In[14]:


# Get the first data under each key and check the datatype
for key in jobDict.keys():
    print("Data Type for",key,":",type(jobDict[key][0]),"\nExample value:",jobDict[key][0],"\n")


# It can be seen that they are all strings! It seems appriopriate for 'Title', 'Company' and 'Description' but 'Webindex' looks more appropriate to be in integer form! Let's convert all Webindex into integers!

# In[15]:


# Convert Webindex into integers
jobDict['Webindex'] = [int(webindex) for webindex in jobDict['Webindex']]

# Check the data types again
for key in jobDict.keys():
    print("Data Type for",key,":",type(jobDict[key][0]),"\nExample value:",jobDict[key][0],"\n")


# Now Webindex are all successfully converted into integer format!. Now that all data is in the right format, it is interesting to see if the Webindex is unique across all job advertisements

# In[16]:


print("Number of unique values of Webindex:",len(set(jobDict['Webindex'])))


# Webindex is indeed unique across each job advertisement!

# Lastly, let's add the target names and filenames into the dictionary as well!

# In[17]:


# Add file names into the dictionary
jobDict['Filename'] = job_data['filenames']


# In[18]:


# Map target into target names and save only the target names into the dictionary
jobDict['Target'] = [job_data['target_names'][target] for target in job_data['target']]
for key in jobDict.keys():
    print("Number of data for",key,":",len(jobDict[key]))


# Now that we are done with some initial exploration, let's move on to pre-processing

# ### 1.2 Pre-processing data
# Perform the required text pre-processing steps.

# The pre-processing steps are outlined as follows:
# 1. Extract information from each job advertisement. Perform the following pre-processing steps to the **description** of each job advertisement;
# 
# 2. Tokenize each job advertisement description. The word tokenization must use the following regular expression, ````r"[a-zA-Z]+(?:[-'][a-zA-Z]+)?"````;
# 
# 3. All the words must be converted into the lower case
# 
# 4. Remove words with length less than 2
# 
# 5. Remove stopwords using the provided stop words list (stopwords_en.txt).
# 
# 6. Remove the word that appears only once in the document collection, based on term frequency
# 
# 7. Remove the top 50 most frequent words based on document frequency
# 
# 8. Save all job advertisement text and information in a txt file
# 
# 9. Build a vocabulary of the cleaned job advertisement descriptions, save it in a txt file
# 
# Note that pre-processing would only be done on the description of each job advertisement

# #### Task 1.2.1 Lower case & Tokenization
# Tokenization would be done through the help of the ```nltk``` library with the provided regex pattern ```r"[a-zA-Z]+(?:[-'][a-zA-Z]+)?"```. Before tokenization, all words would be converted into lower cases.

# In[19]:


# Step 1. Take out the descriptions alone for pre-processing
descriptions = jobDict['Description'].copy()


# In[20]:


# Function to tokenize each description
pattern = r"[a-zA-Z]+(?:[-'][a-zA-Z]+)?"
def tokenizeDesc(raw_desc):
    # Step 3. Convert to lower case
    desc = raw_desc.lower()

    # Sentence Segmentation
    # Transform 'Hello World. Bye World.'
    #             to
    # ['Hello World., Bye World.']
    sentences = sent_tokenize(desc)
    
    # Step 2. Tokenize each sentence into tokens
    # Transform ['Hello World., Bye World.']
    #                to
    # [['Hello','World'],['Bye','World']]
    tokenizer = RegexpTokenizer(pattern)
    list_tokens = [tokenizer.tokenize(sentence) for sentence in sentences]
    
    # Flatten the list of lists into a single list
    # Transform [['Hello','World'],['Bye','World']]
    #                to
    # ['Hello','World','Bye','World']
    tokenised_desc = list(chain.from_iterable(list_tokens))
#     print(tokenised_desc)
    return tokenised_desc


# Let's created a function that helps us track the status of the descriptions, including the vocabulary sizes as well as other information!

# In[21]:


# List of statuses to track
stats = ['Vocabulary Size','Number of tokens','Lexical diversity','Average number of words in description','Number of words for longest description','Number of words for shortest description','Standard deviation for number of words']
# Generate a dictionary based on the stats
statusDict = {key:[] for key in stats}


# In[22]:


# Function to get the status of descriptions
def status(descriptions):
    # Get all words from the descriptions
    all_tokens = list(chain.from_iterable(descriptions))
    
    # Get unique words to be the vocabulary of the corpus
    vocab = set(all_tokens)
    
    # Calculate values of different information including the vocabulary size, mean and standard deviation of description length
    desc_length = [len(description) for description in descriptions]
    values = [len(vocab),len(all_tokens),round(len(vocab)/len(all_tokens),3),round(np.mean(desc_length),3),np.max(desc_length),np.min(desc_length),round(np.std(desc_length),3)]
    
    # Update the status into the status dictionary
    for i in range(0,len(values)):
        statusDict[stats[i]].append(values[i])
    
    # Print stats and show changes if changes are made
    for key in statusDict.keys():
        if len(statusDict[key]) > 1:
            print('{0}: {1} -> {2}'.format(key,statusDict[key][-2],statusDict[key][-1]))
        else:
            print('{0}: {1}'.format(key,statusDict[key][0]))


# In[23]:


# Tokenize the descriptions
descriptions = [tokenizeDesc(description) for description in descriptions]


# After tokenization, let's check out some of the processed data as well as the status!

# In[24]:


# Randomly pick a description
indx = random.choice(range(0,len(descriptions)))

print('Before processing:')
# Get original description
print(jobDict['Description'][indx])

print('\nAfter processing:')
# Get tokenized descriptions
print(descriptions[indx])


# In[25]:


status(descriptions)


# It can be seen that around half of the tokens in the corpus are duplicated (Number of tokens vs Vocabulary Size) and the longest description could go up to 815 words!

# #### Task 1.2.2 Remove words with length less than 2
# Assuming that we keep words with exact length of 2

# In[26]:


# Get words in descriptions that has lenght less than 2
short_words = [[token for token in description if len(token)<2] for description in descriptions]

# Flatten them all into a single list
all_short_words = list(chain.from_iterable(short_words))

print("Number of words with length < 2:",len(all_short_words))
print("Number of unique words with length < 2:",len(set(all_short_words)))
print('Unique Words with length less than 2:',set(all_short_words))


# It can be seen that all words with length < 2 are the alphabetical characters. Let's remove them now

# In[27]:


# Step 4. Remove words with length less than 2
descriptions = [[token for token in description if len(token)>=2] for description in descriptions]


# Now that words with length less than 2, it is expected that there would be changes in the status of descriptions

# In[28]:


status(descriptions)


# It can be seen that after removing those words, the vocabulary size is decreased by 26 words and number of tokens is decreased by 6039 tokens

# #### Task 1.2.3 Removing stopwords
# Stopwords listed in the provided file named ````stopwords_en.txt```` would all be removed 

# In[29]:


# Step 5. Remove stopwords from stopwords_en.txt
with open('stopwords_en.txt','r') as f:
    stopwords = f.read().split()
print("Number of stopwords:",len(stopwords))
print("First 10 stopwords:",random.sample(stopwords,10))


# From above it can be seen that there are a total of 571 stopwords with words like "and", 'you're" and "to", which would be removed from the descriptions.

# In[30]:


# Remove words that is present in the stopwords
descriptions = [[token for token in description if token not in stopwords] for description in descriptions]


# In[31]:


status(descriptions)


# From the stats it can be seen that a total of 404 unique words are removed and 73k tokens are removed from the full set of tokens! With stop words removed, the maximum number of words in the description is reduced to 487 words, the average number of words as well as the standard deviation is greatly decreased as well.

# #### Task 1.2.4 Remove words that appears ony once
# Remove words that appeared only once through **term frequency**. This can be done through the help of ```nltk``` library.

# In[32]:


# Get all tokens of all descriptions as a list of tokens
all_tokens = list(chain.from_iterable(descriptions))


# In[33]:


# FreqDist function by nltk provides the frequency distributions of all words in the text
term_freq = FreqDist(all_tokens)


# In[34]:


# items_appear_once = list(filter(lambda x: x[1]==1,term_freq.items()))
# hapaxes() gets words that appears only once
words_appear_once = set(term_freq.hapaxes())
print("Number of words that appears only once:",len(words_appear_once))
print("10 random words within:",random.sample(words_appear_once,10))


# There are a total of 4186 words that appeared only once, including words like "worwickshire", "benefitsmanchester" and others. These words would be removed from the descriptions.

# In[35]:


# Step 6. Remove words that appears only once
descriptions = [[token for token in description if token not in words_appear_once] for description in descriptions]


# In[36]:


status(descriptions)


# After removing such words, the vocabulary size, number of tokens as well as other value counts has all been reduced.

# #### Task 1.2.5 Remove top 50 most frequent words based on document frequency
# This can be through getting only the unique words in each description and with the help of ```ntlk``` as well.

# In[37]:


# We get the unique set of words for each description which cause the number of token in the merged list to be the number of documents that token appeared in
unique_tokens = list(chain.from_iterable([set(description) for description in descriptions]))

# Get FreqDist of unique tokens of descriptions
doc_freq = FreqDist(unique_tokens)

# Get 50 most frequent words
doc_freq.most_common(50)


# In[38]:


# Extract the word from the 50 most frequent words
doc_freq_words = [item[0] for item in doc_freq.most_common(50)]

# Step 7. Remove 50 most frequent words
descriptions = [[token for token in description if token not in doc_freq_words] for description in descriptions]


# In[39]:


status(descriptions)


# That would be the end of pre-processing and the final vocab size is shown above, with 5168 words. Now, it is time to save the neccessary data!

# ## Saving required outputs
# In this task, the vocabulary of descriptions would be generated through the use of unigrams and saved into a text file named ```vocab.txt```. The job advertisement dictionary would also be saved as a txt file named `job_data.txt` for Task 2 use.

# #### Task 1.2.6 Build a Unigram from the vocab generated
# The vocab is built and modified after every pre-processing step and now a Unigram would be generated based on the vocab

# In[40]:


# Step 9. Build vocabulary
def generateVocab(descriptions):
    # Get all tokens of all descriptions as a list of tokens
    all_tokens = list(chain.from_iterable(descriptions))
    # Generate a unigram from the tokens
    unigrams = ngrams(all_tokens,n=1)
    freq_unigram = FreqDist(unigrams)

    # Sort the keys in unigram alphabetically
    sorted_keys = sorted(freq_unigram.keys())
    
    # Get the words from the unigram
    keys = [key[0] for key in sorted_keys]
    
    # Print the first 10 words
    print(keys[:10])
    print("Number of tokens:",len(keys))
    
    return keys


# In[41]:


keys = generateVocab(descriptions)


# #### Task 1.2.7 Saving the unigram into text file
# The generated unigram would be saved into a text file ````vocab.txt```` by the format **word:index**

# In[42]:


# Step 9. Save generated vocabulary to txt file
with open('vocab.txt','w') as f:
    for indx in range(0,len(keys)-1):
        # Write in form of word:index
        f.write('{0}:{1}\n'.format(keys[indx],indx))
    # Write last line without '\n'
    f.write('{0}:{1}'.format(keys[-1],len(keys)-1))


# #### Task 1.2.8 Saving the dataset into a text file
# The pre-processed descriptions would be added into the job advertisement dictionary and all will be saved into text file named `job_data.txt`

# In[43]:


# Step 8. Save all job advertisement text and information in a txt file
# Add processed description into the dictionary as list of strings
jobDict['Processed_description'] = [' '.join(tokens) for tokens in descriptions]

# Generate pandas datafram from the job advertisement dictionary
job_df = pd.DataFrame.from_dict(jobDict)
# Save job_df in form of csv into a text file
job_df.to_csv('job_data.txt',index=False)

# Check out first few data of the dataframe
job_df.head()


# ## Summary
# In this task, given a set of text files categorised under separated folder, I learned how to load all files as a single dictionary of data and perform some basic text pre-processing.
