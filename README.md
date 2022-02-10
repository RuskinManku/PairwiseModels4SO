# PairwiseModels4SO
Code accompanying the paper "Local and Global Context-Based Pairwise Models for Sentence Ordering"

- Use transformers==3.0.2 to run the code without errors. 
- All three datasets are openly available.
- During both training time and test time, use the --dataset option to specify which dataset to train on. The dataset should contain the train, val and test files in the current directory. "--dataset=sind", "--dataset=rocstories" and "--dataset=acl" are the options available right now.
- For BERTPair, ALBERTPair and EnsemblePair, use result_local-pairwise.py to get results on the sentence ordering metrics using the 2 decoding mechanisms.
- For Global_BERTPair and Global_ALBERTPair, use result_global-pairwise.py
- For Global-EnsemblePair, use result_global-ensemble.py

