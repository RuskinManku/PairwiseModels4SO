import numpy as np
import pandas as pd
import nltk
import pickle
import random
import json
from sklearn.model_selection import train_test_split
random.seed(420)

def load_dataset():
    """
    Function to load ACL files
    """
    print("Loading files Files")
    #########################
    abstracts_train=[]
    with open('aan/split/train','r') as fil:
        line=fil.readline()
        while line:
            curr_abstract=[]
            with open('aan/txt_tokenized/{}'.format(line.rstrip()),'r') as fil2:
                line_curr=fil2.readline()
                while line_curr:
                    curr_abstract.append(line_curr.rstrip())
                    line_curr=fil2.readline()
            abstracts_train.append(curr_abstract)
            line=fil.readline()
    
    abstracts_train=np.array(abstracts_train)
    ###########################
    abstracts_test=[]
    with open('aan/split/test','r') as fil:
        line=fil.readline()
        while line:
            curr_abstract=[]
            with open('aan/txt_tokenized/{}'.format(line.rstrip()),'r') as fil2:
                line_curr=fil2.readline()
                while line_curr:
                    curr_abstract.append(line_curr.rstrip())
                    line_curr=fil2.readline()
            abstracts_test.append(curr_abstract)
            line=fil.readline()
    
    abstracts_test=np.array(abstracts_test)
    ############################
    abstracts_val=[]
    with open('aan/split/valid','r') as fil:
        line=fil.readline()
        while line:
            curr_abstract=[]
            with open('aan/txt_tokenized/{}'.format(line.rstrip()),'r') as fil2:
                line_curr=fil2.readline()
                while line_curr:
                    curr_abstract.append(line_curr.rstrip())
                    line_curr=fil2.readline()
            abstracts_val.append(curr_abstract)
            line=fil.readline()
    
    abstracts_val=np.array(abstracts_val)

    print("JSON's loaded of shape")
    print(abstracts_train.shape)
    print(abstracts_test.shape)
    print(abstracts_val.shape)

    return abstracts_train,abstracts_test,abstracts_val

def process_dataset(stories):
    print("Processing dataset")
    cnt=0
    new_stories=[]
    for story in stories:
        flag=True
        for sentence in story:
            words=nltk.TreebankWordTokenizer().tokenize(sentence)
            num_words=len(words)
            if(num_words>50):
                cnt+=1
                flag=False
                break
        if flag==True:
            new_stories.append(story)

    print("Removed {} stories".format(cnt))
    return new_stories

"""
For this code, need the AAN dataset, containing the sub-directories sentences, split and txt_tokenized
"""

def main():
    abstracts_train,abstracts_test,abstracts_val=load_dataset()
    with open("acl_train.pkl","wb") as f:
        pickle.dump(abstracts_train,f)
    with open("acl_val.pkl","wb") as f:
        pickle.dump(abstracts_val,f)
    with open("acl_test.pkl","wb") as f:
        pickle.dump(abstracts_test,f)

if __name__ == "__main__":
    main()

