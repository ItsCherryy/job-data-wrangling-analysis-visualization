from sklearn.datasets import load_files
import os
import re
import time
from gensim.models.fasttext import FastText
import pickle
import pandas as pd
from nltk import RegexpTokenizer

# Get current working directory
os_path = os.getcwd()
# Get fields of a job
job_fields = ['Title', 'Description','Filename']

# Generate document embeddings
def gen_docVecs(wv,tk_txts):
    # Create empty dataframe to store the document embeddings
    docs_vectors = pd.DataFrame() 

    # For each document
    for i in range(0,len(tk_txts)):
        tokens = tk_txts[i]

        # Initialize a temporary dataframe
        temp = pd.DataFrame()
        # For each token in the document
        for w_ind in range(0, len(tokens)):
            try:
                word = tokens[w_ind]
                # Get word embedding if present
                word_vec = wv[word]
                # Append to temporary dataframe
                temp = temp.append(pd.Series(word_vec), ignore_index = True)
            except:
                pass
        # Take the sum of each column
        doc_vector = temp.sum()
        # Append the value into the dataframe
        docs_vectors = docs_vectors.append(doc_vector, ignore_index = True)
    return docs_vectors

def preprocess(text):
    pattern = r"[a-zA-Z]+(?:[-'][a-zA-Z]+)?"
    text = text.lower().strip()
    tokenizer = RegexpTokenizer(pattern)
    all_tokens = tokenizer.tokenize(text)
    tokens = [token for token in all_tokens if len(token)>=2]
    model_dir = os.path.join(os_path,'models')
    stopwords_dir = os.path.join(model_dir,'stopwords_en.txt')
    with open(stopwords_dir,'r') as f:
        stopwords = f.read().split()
    tokens = [token for token in tokens if token not in stopwords]
    if len(tokens)>0:
        return tokens
    elif len(all_tokens) > 0:
        return all_tokens
    else:
        return text.strip().split(' ')

def predictCategory(text):
    # Get the model and pickle file
    model_dir = os.path.join(os_path,'models')
    modelFT = os.path.join(model_dir,'jobFT.model')
    pklLR = os.path.join(model_dir,'jobFT_LR.pkl')

    # Tokenize the description
    tokenized_data = preprocess(text)

    # Load the FastText model
    jobFT = FastText.load(modelFT)
    jobWV= jobFT.wv

    # Generate document embeddings of tokenized data
    jobFT_dvs = gen_docVecs(jobWV, [tokenized_data])

    # Load the trained Logistic Regression model
    with open(pklLR, 'rb') as file:
        model = pickle.load(file)
    
    # Predict the category based on the description
    y_pred = model.predict(jobFT_dvs)
    y_pred = y_pred[0]

    return y_pred


def writeToFile(title,desc,folder):
    # Get the data directory
    data_dir = os.path.join(os_path,*['static','data',folder])
    
    # Get current time to set as the name of the file
    timestamp = str(round(time.time()*1000))

    # Generate the new file path according to its category
    file_dir = os.path.join(data_dir,timestamp+".txt")

    # Write the content to file
    f = open(file_dir,'w')
    f.write('Title: '+title+'\n')
    f.write('Description: '+desc+'\n')
    f.close()

    # return the filename
    return timestamp


def getJob(folder,filename):
    # Initialize a dictionary for the specified job
    jobDict = {'Title':None,'Description':None}

    # Load the text file of the job
    data_dir = os.path.join(os_path,*['static','data'])
    file_dir = os.path.join(data_dir,*[folder,filename+".txt"])
    f = open(file_dir,'r')
    job = f.read()
    job_lines = job.splitlines()

    # Get the title and description of the job
    field_pattern = r'^(\w+): (.*)'
    for field in job_lines:
        values = re.match(field_pattern, field).groups()
        # Only store the title and description, exlucde company and webindex to the dictionary
        if values[0] not in  ['Company','Webindex']:
            jobDict[values[0]] = values[1]

    # Return the specified job in dictionary format
    return jobDict

def loadLatest():
    # Load full dataset
    jobDict = loadData()
    # Get the latest created job for each category
    latestJob = {name:{key:None for key in job_fields} for name in jobDict.keys()}
    
    # Get the data directory
    data_dir = os.path.join(os_path,*['static','data'])
    # For each job category
    for key in jobDict.keys():
        # Get the full path of each file under a category
        filenames = [os.path.join(data_dir,*[key,filename+".txt"]) for filename in jobDict[key]['Filename']]
        # Get the latest created job
        latest = max(filenames,key=os.path.getctime)
        indx = filenames.index(latest)

        # Loop through the fields of a job
        for field in job_fields:
            # Append data to the dictionary accordingly
            latestJob[key][field] = jobDict[key][field][indx]
    return latestJob

def loadData():
    # Load full data
    data_dir = os.path.join(os_path,*['static','data'])
    job_data = load_files(data_dir,encoding='utf-8')
    
    # Get the job info for each data
    job_info = [data.splitlines() for data in job_data['data']]

    # Generate a nested dictionary to store each job under its respective category
    jobDict = {name:{key:[] for key in job_fields} for name in job_data['target_names']}
    
    field_pattern = r'^(\w+): (.*)'
    dir_pattern = r'^\w:[\\/]+(?:[\w\s]+[\\/]+)+(\w+)\.txt$'

    # For each job
    for indx in range(len(job_info)):
        # Get the target of the job and map it to the target names
        job_target = job_data['target'][indx]
        job_target_name = job_data['target_names'][job_target]

        # Get the filename
        # Only the name of the file, remove all path as well as extension
        job_file_name = job_data['filenames'][indx]
        filename = re.match(dir_pattern,job_file_name).groups()[0]

        # For each field of the job
        for field in job_info[indx]:
            # Get the name and value for the field
            values = re.match(field_pattern, field).groups()

            # Only store the title and description, exlucde company and webindex to the dictionary
            if values[0] not in ['Company','Webindex']:
                jobDict[job_target_name][values[0]].append(values[1])

        # Append the file name for each job
        jobDict[job_target_name]['Filename'].append(filename)

    # Return the loaded job dictionary
    return jobDict