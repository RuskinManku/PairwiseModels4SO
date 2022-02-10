import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from torch.utils.data import TensorDataset,DataLoader
import torch.optim as optim
import pickle
import numpy as np
import sys
import random
random.seed(222)
np.random.seed(111)
import math
import itertools
import shutil
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("GPU/CPU:",torch.cuda.get_device_name(0)) 

class PairEncoder(nn.Module):
    def __init__(self):
        super(PairEncoder, self).__init__()
        self.pair_encoder=transformers.BertModel.from_pretrained('bert-base-uncased')
        self.tokenizer=transformers.BertTokenizer.from_pretrained('bert-base-uncased')
    def forward(self,src_paragraphs):
        batch_tokens_fs,batch_tokens_ss=[],[]
        batch_index_to_pair_dict={}
        for i in range(src_paragraphs.shape[0]):
            curr_sentences=src_paragraphs[i]
            num_sentences=curr_sentences.shape[0]
            for j in range(num_sentences):
                for k in range(num_sentences):
                    if j==k:
                        continue
                    fs_tokens=self.tokenizer.tokenize(curr_sentences[j])
                    ss_tokens=self.tokenizer.tokenize(curr_sentences[k])
                    fs_tokens_id=self.tokenizer.convert_tokens_to_ids(fs_tokens)
                    ss_tokens_id=self.tokenizer.convert_tokens_to_ids(ss_tokens)
                    curr_index=len(batch_tokens_fs)
                    batch_index_to_pair_dict[curr_index]=(i,j,k)
                    batch_tokens_fs.append(fs_tokens_id)
                    batch_tokens_ss.append(ss_tokens_id)
        batch_size=5
        num_examples=len(batch_tokens_fs)
        all_pooled_output=[]
        for k in range(0,num_examples,batch_size):
            start=k
            end=min(k+batch_size,num_examples)
            pad_seq_length = 250
            features = {}
            for i in range(start,end):
                sentence_features = self.tokenizer.prepare_for_model(batch_tokens_fs[i],batch_tokens_ss[i],max_length=pad_seq_length, pad_to_max_length=True, return_tensors='pt',truncation=True)
                for feature_name in sentence_features:
                    if feature_name not in features:
                        features[feature_name] = []
                    features[feature_name].append(sentence_features[feature_name])
            for feature_name in features:
                features[feature_name] = torch.cat(features[feature_name]).to(device)
            pooled_output=self.pair_encoder(**features)[1]
            all_pooled_output.extend(pooled_output)
        all_pooled_output=torch.stack(all_pooled_output).to(device)
        return all_pooled_output,batch_index_to_pair_dict

class pair_wise_model(nn.Module):
    def __init__(self,d_model=768):
        super(pair_wise_model, self).__init__()
        self.encode_pair=PairEncoder()
        self.dropout = nn.Dropout(0.1)
        self.pair_weight=nn.Linear(d_model,2)

    def forward(self, src_paragraphs):
        pooled_output,batch_index_to_pair_dict=self.encode_pair(src_paragraphs)
        pooled_output=self.dropout(pooled_output)
        output_logits=[]
        for i in range(pooled_output.size()[0]):
            idx,j,k=batch_index_to_pair_dict[i]
            classifier_output=self.pair_weight(pooled_output[i])
            output_logits.append(classifier_output)
        output_logits=torch.stack(output_logits).to(device)
        return output_logits,batch_index_to_pair_dict                 


def get_tau_array(permutation_indices):
    num_sentences=len(permutation_indices)
    output_arr=[]
    for i in range(num_sentences):
        for j in range(num_sentences):
            if i==j:
                continue
            if permutation_indices[i]<permutation_indices[j]:
                output_arr.append(0)
            else:
                output_arr.append(1)
    output_arr=np.array(output_arr)
    return torch.from_numpy(output_arr)

def create_permutations(len):
    """
    Function to create a permutation of the paragraph for the dataset. At first we were shuffling,
    but since this is just a pairwise model, no shuffling is required at training time
    """
    indices=np.arange(0,len) 
    return indices

def process_data(paragraphs):
    shuffled_paragraphs=[]
    outputs_tau_lists=[]
    print("\nCreating dataset")
    for paragraph in paragraphs:
        indices_list=create_permutations(len(paragraph))    
        random_permutation=indices_list
        paragraph=np.array(paragraph)
        shuffled_paragraph=paragraph[random_permutation]
        shuffled_paragraphs.append(shuffled_paragraph)
        outputs_tau_lists.append(get_tau_array(random_permutation))
    shuffled_paragraphs=np.array(shuffled_paragraphs)
    y_tensor=torch.stack(outputs_tau_lists).long()
    print("Shuffles paragraphs to final shape {}".format(shuffled_paragraphs.shape))
    print("Y_tensor {}".format(y_tensor.size()))
    return shuffled_paragraphs,y_tensor
    
def create_batches(data_x,data_y,batch_size):
    num_examples=data_x.shape[0]
    indices=np.arange(0,num_examples)
    indices=np.random.permutation(indices)
    data_x=data_x[indices]
    data_y=data_y[indices]
    x_batches=[]
    y_batches=[]
    curr_x_batch=[]
    curr_y_batch=[]
    for i in range(num_examples):
        if (i+1)%batch_size==0 or i==(num_examples-1):
            curr_x_batch.append(data_x[i])
            curr_y_batch.append(data_y[i])
            curr_x_batch=np.array(curr_x_batch)
            curr_y_batch=torch.stack(curr_y_batch)
            x_batches.append(curr_x_batch)
            y_batches.append(curr_y_batch)
            curr_x_batch=[]
            curr_y_batch=[]
        else:
            curr_x_batch.append(data_x[i])
            curr_y_batch.append(data_y[i])
    return x_batches,y_batches
    
def create_batches_different_length(data_x,batch_size):
    len_dict_x={}
    for paragraph in data_x:
        paragraph=np.array(paragraph)
        curr_len=paragraph.shape[0]
        if curr_len in len_dict_x:
            len_dict_x[curr_len].append(paragraph)
        else:
            len_dict_x[curr_len]=[]
            len_dict_x[curr_len].append(paragraph)
    all_x_batches=[]
    all_y_batches=[]
    for len_par,d_x in len_dict_x.items():
        if len_par==1:
            continue
        paragraphs,y_tensors=process_data(d_x)
        curr_x_batch,curr_y_batch=create_batches(paragraphs,y_tensors,batch_size)
        all_x_batches.extend(curr_x_batch)
        all_y_batches.extend(curr_y_batch)
        print("Created {} batches for length {}".format(len(curr_x_batch),len_par))
    return all_x_batches,all_y_batches

def mean_batch_acc(logits,labels):
    """
    Calculates mean accuracy of a batch
    """
    num_examples=logits.size()[0]
    softy=nn.Softmax(dim=1)
    logits=softy(logits)
    correct_values=0
    for i in range(num_examples):
        if(labels[i].item()==0):
            if logits[i][0]>=logits[i][1]:
                correct_values+=1
        if(labels[i].item()==1):
            if logits[i][0]<logits[i][1]:
                correct_values+=1
    return correct_values/num_examples

import argparse
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    args = parser.parse_args()

    if args.dataset=="sind":
        train_paragraphs=np.array(pickle.load(open("sind_train.pkl","rb")))
        val_paragraphs=np.array(pickle.load(open("sind_val.pkl","rb")))
        MODEL_SAVE_NAME="bertpair_sind_model"

    if args.dataset=="rocstories":
        train_paragraphs=np.array(pickle.load(open("rocstories_train.pkl","rb")))
        val_paragraphs=np.array(pickle.load(open("rocstories_val.pkl","rb")))
        MODEL_SAVE_NAME="bertpair_rocstories_model"

    if args.dataset=="acl":
        train_paragraphs=np.array(pickle.load(open("acl_train.pkl","rb")))
        val_paragraphs=np.array(pickle.load(open("acl_val.pkl","rb")))
        MODEL_SAVE_NAME="bertpair_acl_model"

    batch_size=2
    val_x_batches,val_y_batches=create_batches_different_length(val_paragraphs,batch_size)

    pair_model=pair_wise_model()
    pair_model=pair_model.to(device)
    print(pair_model)
    
    criterion = nn.CrossEntropyLoss()
    optimizer_1 = optim.Adam(pair_model.parameters(), lr=5e-6)
    scheduler_1 = torch.optim.lr_scheduler.StepLR(optimizer_1, step_size=1, gamma=0.1)

    CHECKPOINT=None#torch.load("bertpair_sind_model_epoch_5_checkpoint_20078.tar")
    if CHECKPOINT:
        print("----LOADING FROM CHECKPOINT")
        pair_model.load_state_dict(CHECKPOINT["pair_model_state_dict"])
        optimizer_1.load_state_dict(CHECKPOINT["optimizer_1_state_dict"])
        scheduler_1.load_state_dict(CHECKPOINT["scheduler_1_state_dict"])

    RUN_INITIAL_VALIDATION=True
    if RUN_INITIAL_VALIDATION:
        with torch.no_grad():
            print("-----------------------INITIAL VALIDATION STARTED-------------------------------")
            pair_model.eval()
            running_val_loss = 0.0
            mean_total_accuracy=0.0
            for j in range(len(val_x_batches)):
                inputs, labels = val_x_batches[j],val_y_batches[j].to(device)
                labels=labels.view(-1)
                outputs,_ = pair_model(inputs)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item()
                mean_total_accuracy+=mean_batch_acc(outputs,labels)
                sys.stdout.write("\rBatch {} of {}. Loss: {}".format(j+1,len(val_x_batches), loss.item()))
                sys.stdout.flush()
            mean_accuracy=mean_total_accuracy/len(val_x_batches)
            average_val_loss=running_val_loss/len(val_x_batches)
            print("\nInitial validation Loss is {},accuracy:{}".format(average_val_loss,mean_accuracy))

    START_EPOCH=1
    LAST_EPOCH=50 #inclusive
    print("-----------------STARTING TRAINING-----------------------")
    for epoch in range(START_EPOCH,LAST_EPOCH+1):  # loop over the dataset multiple times
        print(f"----------------EPOCH:{epoch}----------------------")
        train_x_batches,train_y_batches=create_batches_different_length(train_paragraphs,batch_size)
        running_train_loss = 0.0 #CHECKPOINT["running_train_loss"]
        pair_model.train()
        print("Curr learning rate _1 :{}".format(scheduler_1.get_last_lr()))
        for i in range(len(train_x_batches)):
            inputs, labels = train_x_batches[i],train_y_batches[i].to(device)
            labels=labels.view(-1)
            optimizer_1.zero_grad()
            
            outputs,_ = pair_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(pair_model.parameters(),1)

            optimizer_1.step()
            running_train_loss += loss.item()
            sys.stdout.write("\rBatch {} of {}. Loss :{}".format(i+1,len(train_x_batches),loss.item()))
            sys.stdout.flush()
            if i==(len(train_x_batches)-1):
                print("\n----------------------RUNNING VALIDATION-----------------------")
                print("Saving model checkpoint")
                torch.save({"pair_model_state_dict":pair_model.state_dict() \
                ,"optimizer_1_state_dict":optimizer_1.state_dict() \
                ,"scheduler_1_state_dict":scheduler_1.state_dict() \
                ,"running_train_loss":running_train_loss},MODEL_SAVE_NAME+f"_epoch_{epoch}_checkpoint_{i+1}.tar")
                with torch.no_grad():
                    pair_model.eval()
                    running_val_loss = 0.0
                    mean_total_accuracy=0.0
                    for j in range(len(val_x_batches)):
                        inputs, labels = val_x_batches[j],val_y_batches[j].to(device)
                        labels=labels.view(-1)
                        outputs,_ = pair_model(inputs)
                        loss = criterion(outputs, labels)
                        running_val_loss += loss.item()
                        mean_total_accuracy+=mean_batch_acc(outputs,labels)
                        sys.stdout.write("\rBatch {} of {}. Loss: {}".format(j+1,len(val_x_batches), loss.item()))
                        sys.stdout.flush()
                    mean_accuracy=mean_total_accuracy/len(val_x_batches)
                    average_val_loss=running_val_loss/len(val_x_batches)
                    print("\nValidation Loss for epoch {} is {},accuracy:{}. Train loss:{}".format(epoch,average_val_loss,mean_accuracy,running_train_loss))
                pair_model.train()
        average_train_loss=running_train_loss/len(train_x_batches)
        print("\nTrain Loss for epoch {} is {}".format(epoch,average_train_loss))
        if epoch<=2:
            print("Taking scheduler step")
            scheduler_1.step()
    print("Training and evaluation Complete")

if __name__ == '__main__':
    main()