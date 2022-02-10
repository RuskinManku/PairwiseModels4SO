import numpy as np
import pandas as pd
import nltk
import pickle
import random
from sklearn.model_selection import train_test_split
random.seed(420)

def load_dataset():
    print("Loading Dataset")
    rocstories_1=pd.read_csv("rocstories_1.csv")
    rocstories_2=pd.read_csv("rocstories_2.csv")
    
    rocstories_1=np.array(rocstories_1)
    stories=[]
    for story in rocstories_1:
        stories.append(story[2:])
        
    rocstories_2=np.array(rocstories_2)
    for story in rocstories_2:
        stories.append(story[2:])
    stories=np.array(stories)
    print("Dataset Loaded")
    return stories

def split_data(stories):
    stories,stories_test=train_test_split(stories,test_size=(1/10),random_state=420)
    stories_train,stories_val=train_test_split(stories,test_size=(1/9),random_state=420)
    return stories_train,stories_test,stories_val

"""
For this, need rocstories_1.csv and rocstories_2.csv
"""
def main():
    stories=load_dataset()
    #dataset_stats(stories)
    stories_train,stories_test,stories_val=split_data(stories)
    with open("rocstories_train.pkl","wb") as f:
        pickle.dump(stories_train,f)
    with open("rocstories_val.pkl","wb") as f:
        pickle.dump(stories_val,f)
    with open("rocstories_test.pkl","wb") as f:
        pickle.dump(stories_test,f)
    print(stories_train.shape)
    print(stories_test.shape)
    print(stories_val.shape)

if __name__ == "__main__":
    main()