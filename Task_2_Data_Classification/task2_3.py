#!/usr/bin/env python
# coding: utf-8

# # Assignment 2: Milestone I Natural Language Processing
# ## Task 2&3
# #### Student Name: Toh Kah Hie
# #### Student ID: s3936897
# 
# Date: 22 September 2022
# 
# Version: 1.0
# 
# Environment: Python 3 and Jupyter notebook
# 
# Libraries used:
# * pandas
# * re
# * random
# * sklearn
# * logging
# * gensim
# * matplotlib
# * numpy
# * nltk
# * itertools
# 
# ## Introduction
# 
# **Task 2** lets students generate 3 types of feature representation for Job Advertisement Descriptions namely Count vector representation, TF-IDF weighted vector representation as well as unweighted vector representation. 
# 
# The count vector representation is generated based on the vocabulary produced from Task 1. It is then saved into a text file named ```count_vectors.txt``` with the required formatting. The weighted and unweighted feature representation of the job advertisement descriptions on the other hand is generated based on an in-house trained FastText model.
# 
# **Task 3** asks students to build machine learning models for classifying the category of a job advertisement text and conduct experiments to determine:
# * Which language model performs the best with the chosen machine learning model
# * Whether including the title for each job advertisement for classification boosts the chosen model's accuracy
# 
# The chosen machine learning model in this task is a logistic regression model. The chosen model would be trained based on the 3 generated vector representations in Task 2 and evaluated using a 5-fold cross validation. The one that gave the best performance would then be used and experiment with different types of data including:
# * Only the title of each job advertisement
# * Only the description of each job advertisement (Task 2)
# * A combination of both the title and description of each job advertisement

# ## Importing libraries 

# In[1]:


# Task 2.1
import pandas as pd
import re
import random

# Task 2.2
from sklearn.feature_extraction.text import CountVectorizer

# Task 2.3
import logging
from gensim.models.fasttext import FastText
# 2.3.3
from sklearn.feature_extraction.text import TfidfVectorizer
# 2.3.4
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

# Task 3.1
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from numpy import mean
from numpy import std

# Task 3.2
import nltk
from nltk import RegexpTokenizer
from nltk.tokenize import sent_tokenize
from nltk.probability import *
from nltk.util import ngrams

from itertools import chain


# ## Task 2. Generating Feature Representations for Job Advertisement Descriptions

# In this task, different types of feature representations will be generated, including:
# * Count vector representation
# * Weighted document embeddings
# * Unweighted document embeddings

# ### Task 2.1 Loading the data as well as vocab
# The dataset ```job_data.txt``` as well as the vocabulary ```vocab.txt``` generated previously in Task 1 are loaded for training the language model and generating feature representations.
# 
# The job_data.txt is loaded in the form of csv with columns including
# * Title | Title of job advertisement | String
# * Webindex | Webindex of job advertisement | Integer
# * Company | Company of job advertisement | String
# * Description | Description of job advertisement | String
# * Filename | Which text file the job advertisement come from | String
# * Target | Category of each job advertisement | String
# * Processed_description | Pre-processed description | String

# In[2]:


# Loaded saved dataset
job_df = pd.read_csv('job_data.txt')
job_df.head()


# In[3]:


# Validate the data types
print("Dataypes of each column:")
job_df.dtypes


# In[4]:


# Validate the dataset size
print('Number of rows and columns: ',job_df.shape)


# In[5]:


# Load saved vocabulary
# In the form of word:index
pattern = r'(.+):\d+'
with open('vocab.txt','r') as f:
    vocabulary = f.read().split()

vocab = [re.match(pattern, word).groups()[0] for word in vocabulary]


# Validate the vocabulary size
print("Number of words in vocab:",len(vocab))
print("Random sample of vocab:",random.sample(vocab,10))


# From above, it can be seen that all required data is loaded correctly, with the job advertisement dataset having 776 rows of data and the correct data types as well as the vocabulary having 5168 words.

# ### Task 2.2 Generate Bag-of-words model
# In this task, the **Count** vector representation for each job advertisement description is generated

# #### Task 2.2.1 Generate Count Vectors
# The Count vector representation will be generated based on the loaded vocabulary.

# In[6]:


# Function to generate the count vector representation
def generateCountVec(vocab,data):
    # Use vocab as its vocabulary
    countVectorizer = CountVectorizer(analyzer='word',vocabulary = set(vocab))
    count_features = countVectorizer.fit_transform(data)
    
    # Validate the vector representation by checking if the size is the same as vocab's
    print("Shape of document-by-word matrix:",count_features.shape)
    feature_names = countVectorizer.get_feature_names()
    print("Lenght of vocab is same as the length of feature names?:",vocab == feature_names)
    return count_features


# In[7]:


count_features = generateCountVec(vocab,list(job_df['Processed_description'].values))


# In[8]:


# Function to validate the count vector representation
vocab = sorted(list(vocab))
def validator(data_features, vocab, indx, df):
    # Get the webindex of the job ad
    print("Webindex:", df['Webindex'][indx])
    print("--------------------------------------------\n")
    # Get full description without cleaning
    print("Full Description:",df['Description'][indx],'\n')
    # Get cleaned description
    print("Tokens:",df['Processed_description'][indx])
    print("--------------------------------------------\n")
    # Print vector representation as 'word:count of word'
    print("Vector representation:\n") 
    for word, value in zip(vocab, data_features.toarray()[indx]): 
        if value > 0:
            print(word+":"+str(value), end =' ')


# In[9]:


validator(count_features,vocab,random.randrange(0,job_df.shape[0]),job_df)


# From above, it can be confirmed that count vector representation is successfully and correctly generated.

# ### Task 2.3 Model based on word embeddings
# In this task, a FastText language model will be used and trained on the pre-processed vocabulary to generate the word embeddings. The reason behind this is because according to  [O'Reilly](https://www.oreilly.com/library/view/deep-learning-essentials/9781785880360/12fe4a55-a5d0-4712-bd68-ac043b87a87e.xhtml), FastText takes into account the internal strucutre of words when learning word representation, which would work very better on syntactic tasks with corpus that has words with multiple morphological forms. GloVe model on the other hand focuses on words co-occurences on the whole corpus.
# 
# Considering that this is data about job advertisments, it is reasonable to say that the corpus would be morphologically rich. For example, "accountant", "account", "accounts" for a job that is related to Accounting and Finance.

# In[10]:


# Use logging to see what is happening when training the model
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


# #### Task 2.3.1 Training the FastText model
# The pre-processed description of each job advertisement will be used as corpus to train the FastText language model

# In[11]:


# Function train the FastText language model
def buildFTModel(corpus):
    # Initialize FastText model with vector size of 100 of the word embedding
    modelFT = FastText(vector_size=100)
    
    # Build vocab using the given corpus
    modelFT.build_vocab(corpus_iterable=corpus)
    
    # Train the laugage model based on the vocabulary
    modelFT.train(corpus_iterable=corpus,epochs=modelFT.epochs,total_examples = modelFT.corpus_count,total_words=modelFT.corpus_total_words)
    print(modelFT)
    return modelFT


# In[12]:


# Use the pre-processed description saved in job_df as the corpus to train the FastText model
corpus =[desc.split(' ') for desc in job_df['Processed_description'].values]
modelFT = buildFTModel(corpus)


# From above, it can be seen that the model drops words that appears 5 times or less from the message ```'msg': 'effective_min_count=5 retains 2741 unique words (53.04% of original 5168, drops 2427)'```

# In[13]:


# Extract the KeyedVectors object from the trained model
modelFT_wv = modelFT.wv
print(modelFT_wv)


# From above, it can be seen that the final size of vocabulary generated is 2741. Let's check out the model with some examples

# In[14]:


# Get some words with different morphological forms
word_morphs = {'account':['accounts','accounting','accountant'],'engineer':['engineering','engineers']}


for word in word_morphs.keys():
    # For each word, check if the word exist in the vocabulary
    print(word,"does not exist") if word not in modelFT_wv.key_to_index else ""
    print("#####Checking on word:",word,"\t#####")
    for morph in word_morphs[word]:
        print(word,"does not exist") if word not in modelFT_wv.key_to_index else ""
        print("Similarity between",word,"and",morph,str(modelFT_wv.similarity(word, morph)))
        print()
    print("Most similar words with",word,":",modelFT_wv.most_similar(word, topn=10))
    print('-----------------------------------')


# From above, it can be seen that the model actually captures the similarity of these words! The 10 most similar words listed for each word are also indeed very similar!

# #### Task 2.3.2 Generate unweighted document embeddings
# In this task, the unweighted vector representation of job advertisement descriptions based on the trained model will be generated

# As generating the vector representation requires each description to be a list of tokens rather than a string, a new column with the pre-processed descriptions split into list of tokens will be generated and added into job_df

# In[15]:


# Transform string of tokens for each description into a list of tokens for each description
job_df['Tokens'] = [tokens.split(' ') for tokens in list(job_df['Processed_description'].values)]


# In[16]:


# Check out some of the data
job_df[['Webindex','Processed_description','Target','Tokens']].sample(n=5)


# In[17]:


# Function to generate document vectors based on the pre-trained word embedding model
def gen_docVecs(wordVec,desc_tokens):
    # Create empty dataframe to store all document embeddings of job ad descriptions
    docVecs = pd.DataFrame()
    
    # For each document
    for indx in range(len(desc_tokens)):
        # Load tokens of each document
        tokens = desc_tokens[indx]
        # Create a temporary dataframe
        temp = pd.DataFrame()
        # For each token ib the doc
        for tok_indx in range(len(tokens)):
            try:
                token = tokens[tok_indx]
                # Get the word embedding if it is preset
                word_vec = wordVec[token]
                # Append to temporary dataframe
                temp = temp.append(pd.Series(word_vec),ignore_index=True)
            except:
                pass
        # Take the sum of each column
        doc_vec = temp.sum()
        # Append the value into the dataframe
        docVecs = docVecs.append(doc_vec,ignore_index=True)
    return docVecs


# In[18]:


# Get the unweighted document embeddings
ftext_dv = gen_docVecs(modelFT_wv,job_df['Tokens'])


# Now that the unweighted document embeddings is generated, it has to be checked so that it contains no null values

# In[19]:


ftext_dv.isna().any().sum()


# From above, it can be seen that the unweighted vector representation is successfully generated.

# #### Task 2.3.3 Generate TF-IDF weighted document vectors
# In this task, the weighted document vectors will be generated based on the TF-IDF vectors of the data. As generating this data requires the tf-idf weights of the words, it would be generated as well

# In[20]:


# Dictionary of vocab is generated in the form of index:word for generating the document vectors
vocDict = {indx:vocab[indx] for indx in range(len(vocab))}
vocDict


# In[21]:


# Function that maps between the word_index and the actual word to create a dictionary of word:weight
def doc_word_weights(data_features,voc_dict):
    # List to store word:weight
    tfidf_weights = []
    
    # For each document
    for desc_indx in range(0,data_features.shape[0]):
        # Append each document into the list as a dictionary of word and its weight
        # voc_dict[wrd_indx] gets the actual word from index
        # data_features[desc_indx][0,wrd_indx] gets the tfidf word weights for a specific word in a specific document
        # We only get words that appeared in the document
        tfidf_weights.append({voc_dict[wrd_indx]:data_features[desc_indx][0,wrd_indx] for wrd_indx in data_features[desc_indx].nonzero()[1]})
    return tfidf_weights


# In[22]:


# Function to generate the tfidf weights of word based on the vocab previously generated
def generateTfidfVec(vocab,data):
    tf_vec = TfidfVectorizer(analyzer='word',vocabulary=vocab)
    tfidf_features = tf_vec.fit_transform(data)
    print("Shape of tfidf-features:",tfidf_features.shape)
    return tfidf_features


# In[23]:


tfidf_features = generateTfidfVec(vocab,list(job_df['Processed_description'].values))


# From above, the tf-idf weights of words are successfully generated and is validated that the size of the feature names is the same as the our vocab, which is 5168 words. Let's check out some of them!

# In[24]:


validator(tfidf_features,vocab,random.randint(0,tfidf_features.shape[0]),job_df)


# TF-IDF vectors successfully generated! Let's move on to map the tf-idf weights to the words 

# In[25]:


tfidf_weights = doc_word_weights(tfidf_features,vocDict)


# Now that the tf-idf weights are successfully mapped, let's check out some of the documents

# In[26]:


doc_num = random.randint(0,tfidf_features.shape[0])
print("TF-IDF weights of each word for document with index",doc_num)
tfidf_weights[doc_num]


# Now that is done, we could go on and generate the tf-idf weighted document embeddings now

# In[27]:


# Function to generate the tf-idf weighted document embeddings
def gen_docVecs_weighted(wordVec,desc_tokens,tfidf=[]):
    # Empty final dataframe
    docVecs = pd.DataFrame()
    
    # For each document
    for indx in range(len(desc_tokens)):
        # Get tokens for each document
        tokens = list(set(desc_tokens[indx]))
        # Get a temporary dataframe
        temp = pd.DataFrame()
        
        # For each token within the document
        for tok_indx in range(len(tokens)):
            try:
                token = tokens[tok_indx]
                # Get word embedding if present
                word_vec = wordVec[token]
                
                # Get tf-idf wieght of word if present
                weight = float(tfidf[indx][token]) if tfidf != [] else 1
                # Append to temporary dataframe
                temp = temp.append(pd.Series(word_vec*weight),ignore_index=True)
            except:
                pass
        # Take the sum of each column
        doc_vec = temp.sum()
        
        # Append document value to the dataframe
        docVecs = docVecs.append(doc_vec,ignore_index=True)
    return docVecs


# In[28]:


weighted_ftext_dv = gen_docVecs_weighted(modelFT_wv,job_df['Tokens'],tfidf_weights)


# Now that the weighted document embeddings is generated, it has to be checked so that it contains no null values as well

# In[29]:


weighted_ftext_dv.isna().any().sum()


# TF-IDF weighted document embeddings successfully generated! Let's check out the difference between these document embeddings!

# #### Task 2.3.4 Plot TSNE
# This task plots the embedding vectors in a 2 dimensional space using TSNE to represent the feature space

# In[30]:


def plotTSNE(labels,docVecs,ax,weighted):
    # Get the targets of job advertisements
    targets = sorted(labels.unique())
    # Get 30% of the data for representation
    size = int(len(docVecs)*0.3)
    
    # Randomly get the data from the dataset
    indices = np.random.choice(range(len(features)),size=size,replace=False)
    projected_features = TSNE(n_components=2,random_state=0).fit_transform(docVecs[indices])
    # Plot the points
    for target in targets:
        points = projected_features[(labels[indices]==target)]
        ax.scatter(points[:,0],points[:,1],label=target)
        if weighted:
            ax.set_title("Weighted Feature vector",fontdict=dict(fontsize=15))
        else:
            ax.set_title("Unweighted Feature vector",fontdict=dict(fontsize=15))
    ax.legend()        


# In[31]:


fig, (ax1,ax2) = plt.subplots(1,2,figsize=(8,6))
# Plot unweighted document embeddings
features = ftext_dv.to_numpy()
plotTSNE(job_df['Target'],features,ax1,False)

# Plot tf-idf weighted document embeddings
features = weighted_ftext_dv.to_numpy()
plotTSNE(job_df['Target'],features,ax2,True)


# From there, all feature representations are successfully generated! It is time to save the count vector representation into a text file

# ### Saving outputs
# Save the count vector representation as per spectification.
# - count_vectors.txt

# In[32]:


# Save the count vector representation of job advertisement descriptions
with open('count_vectors.txt','w') as f:
    # For each document
    for desc_indx in range(count_features.shape[0]):
        # Get the webindex of the job
        count = '#' + str(job_df['Webindex'][desc_indx])
        # For each word that appears
        for ftr_indx in count_features[desc_indx].nonzero()[1]:
            count += ','
            value = count_features[desc_indx][0,ftr_indx]
            count += "{}:{}".format(ftr_indx,value)
        f.write(count+'\n')
    f.close()


# ## Task 3. Job Advertisement Classification

# In this task, Logistic Regression model is used to classify the category of a job advertisement from the 3 different vector representations generated in Task 2. A 5-fold cross validation is used to get the performance of each model. The vector representation that provides the best model performance would then be used for different experimentation to determine whether more information equals higher accuracy.

# #### Task 3.1 Q1: Language Model Comparison
# The model used in this project would be the logistic regression model. The labels for each job advertisement used would be the labels stored in job_df.

# In[33]:


# Function for generating classification model evaluate model
# Set the seed to be 0 to have the same train test split across all classification
seed = 0
def classification(features):
    # Perform train and test split for the data given and set the category as job_df['Target']
    X_train, X_test, y_train, y_test,train_indices,test_indices = train_test_split(features, job_df['Target'], list(range(0,len(job_df))),test_size=0.33, random_state=seed)
    
    # Initialize the model
    model = LogisticRegression(max_iter=3000,random_state=seed)
    # Fit the data
    model.fit(X_train, y_train)
    # Get the score
    model.score(X_test, y_test)
    
    # Perform 5-fold Cross validation
    cv = KFold(n_splits=5, random_state=1, shuffle=True)
    scores = cross_val_score(model,X_test,y_test,cv=cv)
    # Get accuracy as well as the standard deviation
    print('Accuracy: Mean: %.3f STD: (%.3f)' % (mean(scores), std(scores)))


# #### Task 3.1.1 Model performance using unweighted document embeddings

# In[34]:


classification(ftext_dv)


# #### Task 3.1.2 Model performance using weighted document embeddings

# In[35]:


classification(weighted_ftext_dv)


# #### Task 3.1.3 Model performance using Count vector representation

# In[36]:


# Classification using count vectors
classification(count_features)


# It seems like among all models, classifying jobs using count vectors returned the best results!

# #### Task 3.2 Q2: Does more information provide higher accuracy?
# Since that in the previous question, using count vectors provided the best results, count vectors would be used for subsequent experiments with titles as well

# #### Task 3.2.1 Classification using only the titles
# Since the titles has not been pre-processed, pre-processing steps done on the job descriptions would be applied to titles as well. Pre-processing function would be copied from Task 1 and used.

# In[37]:


# Step 5. Remove stopwords from stopwords_en.txt
with open('stopwords_en.txt','r') as f:
    stopwords = f.read().split()
print("Number of stopwords:",len(stopwords))
print("First 10 stopwords:",random.sample(stopwords,10))


# In[38]:


# Tokenization based on set pattern
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

# Step 9. Build vocabulary
def generateVocab(descriptions):
    # Get all tokens of all descriptions as a list of tokens
    all_tokens = list(chain.from_iterable(descriptions))
    # Generate a unigram from the tokens
    unigrams = ngrams(all_tokens,n=1)
    freq_unigram = FreqDist(unigrams)

    # print(freq_unigram.items())
    sorted_keys = sorted(freq_unigram.keys())
    keys = [key[0] for key in sorted_keys]
    print(keys[:10])
    print("Number of tokens:",len(keys))
    
    return keys


# In[39]:


# 1.1 Extract titlle
titles = job_df['Title'].copy()

# 1.2 & 1.3 Tokenization & Lower case
titles = [tokenizeDesc(title) for title in titles]

# 1.4 Remove words of length less than 2 
titles = [[token for token in title if len(token)>=2] for title in titles]

# 1.5 Remove words appear in stopwords
titles = [[token for token in title if token not in stopwords] for title in titles]

# 1.6 Remove words appearing only once
all_tokens = list(chain.from_iterable(titles))
term_freq = FreqDist(all_tokens)
words_appear_once = set(term_freq.hapaxes())
titles = [[token for token in title if token not in words_appear_once] for title in titles]

# 1.7 Removing 50 most common words
unique_tokens = list(chain.from_iterable([set(title) for title in titles]))
doc_freq = FreqDist(unique_tokens)
doc_freq_words = [item[0] for item in doc_freq.most_common(50)]
titles = [[token for token in title if token not in doc_freq_words] for title in titles]

# Flatten the list of lists into a single list
titles = [" ".join(token) for token in titles]


# Now that all titles are pre-processed, let's check out some of them

# In[40]:


random.sample(titles,10)


# It seems like some of the titles turn empty after some pre-processing! Thus, some of the pre-processing steps would be removed to keep some of the words. The pre-processing steps removed would be the last few steps, where we skip removing words that only appears once as well as most frequent words.

# In[41]:


# 1.1 Extract titlle
titles = job_df['Title'].copy()

# 1.2 & 1.3 Tokenization & Lower case
titles = [tokenizeDesc(title) for title in titles]

# 1.4 Remove words of length less than 2 
titles = [[token for token in title if len(token)>=2] for title in titles]

# 1.5 Remove words appear in stopwords
titles = [[token for token in title if token not in stopwords] for title in titles]

# Skip 1.6 & 1.7
# titles = [" ".join(token) for token in titles]


# In[42]:


print("Titles with no words:",[(indx,titles[indx]) for indx in range(len(titles)) if len(titles[indx])<1])


# It seems like all titles have at least 1 word now! Let's use these data instead for classification!

# In[43]:


# New vocabulary has to be generated for creating the count vector representation
title_vocab = generateVocab(titles)


# From above it can be seen that the vocabulary size of titles is 954, much less than that of job descriptions (5168)!

# In[44]:


flat_titles = [" ".join(token) for token in titles]
title_countVectorizer = CountVectorizer(analyzer='word',vocabulary = set(title_vocab))
title_count_features = title_countVectorizer.fit_transform(flat_titles)

print("Shape of document-by-word matrix:",title_count_features.shape)
feature_names = title_countVectorizer.get_feature_names()
print("Lenght of vocab is same as the length of feature names?:",title_vocab == feature_names)


# Now that the count vector representation is successfully generated, it is time to evaluate the model's performance using this data

# In[45]:


classification(title_count_features)


# It seems like the model's performance with using only titles is not bad, but it is still not as good as using job descriptions!

# #### Task 3.2.2 Classification using only the description
# Since in the previous question the model performance of using job descriptions is already done, no new classification would be done here.

# In[46]:


classification(count_features)


# #### Task 3.2.3 Classification using a combination of titles and descriptions
# Now, it's time to evaluate the model performance when both the titles and description of each job advertisement is used! The data would be generated through concantenating the pre-processed titles and the pre-processed descriptions for each job advertisement.

# In[47]:


# Concatenate pre-processed titles and descriptions
title_description = [titles[indx] + job_df['Tokens'][indx] for indx in range(job_df.shape[0])]

# Generate the new vocabulary
joined_vocab = generateVocab(title_description)
title_description = [" ".join(token) for token in title_description]


# From above, it can be seen that the number of vocab has increased by 159 words. (5168 to 5327)

# In[48]:


joined_countVectorizer = CountVectorizer(analyzer='word',vocabulary = set(joined_vocab))
joined_count_features = joined_countVectorizer.fit_transform(title_description)

print("Shape of document-by-word matrix:",joined_count_features.shape)
feature_names = joined_countVectorizer.get_feature_names()
print("Lenght of vocab is same as the length of feature names?:",joined_vocab == feature_names)


# In[49]:


classification(joined_count_features)


# It seems like after adding pre-processed titles into the corpus, the accuracy of the model increased! Which answers the question, more information in this dataset does provide a higher accuracy.

# ## Summary
# This assessment allows me to know that there are different ways to represent text documents and complex representations does not necessarily mean you will get a higher classification accuracy.
