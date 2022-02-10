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
    Function to load SIND json file
    """
    print("Loading JSON Files")
    with open('sis/train.story-in-sequence.json') as f:
        train_json = json.load(f)
    with open('sis/test.story-in-sequence.json') as f:
        test_json = json.load(f)
    with open('sis/val.story-in-sequence.json') as f:
        val_json = json.load(f)

    stories_train=[]
    stories_test=[]
    stories_val=[]

    temp=[]
    for i in range(len(train_json['annotations'])):
        if(i%5==0 and i!=0):
            stories_train.append(temp)
            temp=[]
        if(i==(len(train_json['annotations'])-1)):
            temp.append(train_json['annotations'][i][0]["original_text"])
            stories_train.append(temp)
            break
        temp.append(train_json['annotations'][i][0]["original_text"])
    stories_train=np.array(stories_train)

    temp=[]
    for i in range(len(test_json['annotations'])):
        if(i%5==0 and i!=0):
            stories_test.append(temp)
            temp=[]
        if(i==(len(test_json['annotations'])-1)):
            temp.append(test_json['annotations'][i][0]["original_text"])
            stories_test.append(temp)
            break
        temp.append(test_json['annotations'][i][0]["original_text"])
    stories_test=np.array(stories_test)

    temp=[]
    for i in range(len(val_json['annotations'])):
        if(i%5==0 and i!=0):
            stories_val.append(temp)
            temp=[]
        if(i==(len(val_json['annotations'])-1)):
            temp.append(val_json['annotations'][i][0]["original_text"])
            stories_val.append(temp)
            break
        temp.append(val_json['annotations'][i][0]["original_text"])
    stories_val=np.array(stories_val)

    print("JSON's loaded of shape")
    print(stories_train.shape)
    print(stories_test.shape)
    print(stories_val.shape)

    return stories_train,stories_test,stories_val

"""
For this code, need the "sis" directory containing train/test/val jsons
"""
def main():
    stories_train,stories_test,stories_val=load_dataset()
    with open("sind_train.pkl","wb") as f:
        pickle.dump(stories_train,f)
    with open("sind_val.pkl","wb") as f:
        pickle.dump(stories_val,f)
    with open("sind_test.pkl","wb") as f:
        pickle.dump(stories_test,f)

if __name__ == "__main__":
    main()
