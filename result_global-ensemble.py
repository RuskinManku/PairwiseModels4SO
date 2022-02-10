import torch
import pickle
import numpy as np
import itertools
import math
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

MAX_PARAGRAPH_LENGTH=5

def calc_acc(original_abstract,test_permutation):
    sentence_dict={}
    for i,sentence in enumerate(original_abstract):
        sentence_dict[sentence]=i

    tot_sentences=len(original_abstract)
    cor_pos=0
    for j,sentence in enumerate(test_permutation):
        if (j==sentence_dict[sentence]):
            cor_pos+=1

    return cor_pos/tot_sentences

def calc_tau(original_abstract,test_permutation):
    sentence_dict={}
    for i,sentence in enumerate(original_abstract):
        sentence_dict[sentence]=i
    
    tot_sentences=len(original_abstract)
    if tot_sentences==1:
        tau=1.0
        return tau
    inversions=0
    for i in range(tot_sentences):
        for j in range(i+1,tot_sentences):
            sent_1=test_permutation[i]
            sent_2=test_permutation[j]
            if sentence_dict[sent_1]>sentence_dict[sent_2]:
                inversions+=1
    
    tot_pairs=((tot_sentences)*(tot_sentences-1))/2
    tau=1-2*(inversions/tot_pairs)
    return tau

def solve_1(test_paragraph,pair_scores_dict,BEAM_SIZE=64):
    num_sentences=test_paragraph.shape[0]
    beam=[]                 # Contains top BEAM_SIZE tuples (score,solution_list),solution_list is a set
    all_solutions=[]        # Contains all tuples (score,solution_list) at a particular step 
    for step_number in range(num_sentences):
        if step_number==0:
            for i in range(num_sentences):
                all_solutions.append((0,{i},[i]))
            beam=sorted(all_solutions, key=lambda tup: tup[0],reverse=True)
            beam=beam[:BEAM_SIZE]
            all_solutions=[]
        else:
            all_solutions=[]
            for solution in beam:
                curr_list=solution[2]
                curr_solution=solution[1]
                curr_score=solution[0]
                last_ele=curr_list[-1]
                for i in range(num_sentences):
                    if i not in curr_solution:
                        pair_score=pair_scores_dict[(last_ele,i)][0].item()
                        temp_sol=set(curr_solution)
                        temp_sol.add(i)
                        temp_list=curr_list[:]
                        temp_list.append(i)
                        all_solutions.append((curr_score+pair_score,temp_sol,temp_list))
            beam=sorted(all_solutions, key=lambda tup: tup[0],reverse=True)
            beam=beam[:BEAM_SIZE]

    solution_indices=beam[0][2]
    return test_paragraph[solution_indices]

def solve_2(test_paragraph,pair_scores_dict,BEAM_SIZE=64):
    num_sentences=test_paragraph.shape[0]
    beam=[]                 # Contains top BEAM_SIZE tuples (score,solution_list),solution_list is a set
    all_solutions=[]        # Contains all tuples (score,solution_list) at a particular step 
    for step_number in range(num_sentences):
        if step_number==0:
            for i in range(num_sentences):
                all_solutions.append((0,{i},[i]))
            beam=sorted(all_solutions, key=lambda tup: tup[0],reverse=True)
            beam=beam[:BEAM_SIZE]
            all_solutions=[]
        else:
            all_solutions=[]
            for solution in beam:
                curr_list=solution[2]
                curr_solution=solution[1]
                curr_score=solution[0]
                for i in range(num_sentences):
                    if i not in curr_solution:
                        all_pair_score=0
                        for ele in curr_list:
                            pair_score=pair_scores_dict[(ele,i)][0].item()
                            all_pair_score+=pair_score
                        temp_sol=set(curr_solution)
                        temp_sol.add(i)
                        temp_list=curr_list[:]
                        temp_list.append(i)
                        all_solutions.append((curr_score+all_pair_score,temp_sol,temp_list))
            beam=sorted(all_solutions, key=lambda tup: tup[0],reverse=True)
            beam=beam[:BEAM_SIZE]

    solution_indices=beam[0][2]
    return test_paragraph[solution_indices]


#------------------------------------Import appropiate class-----------------------------------
from Global_Ensemblepair import GlobalInformationModule,pair_wise_model

import argparse
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    args = parser.parse_args()

    global_model_bert=GlobalInformationModule(model_name="BERT")
    global_model_bert=global_model_bert.to(device)

    global_model_albert=GlobalInformationModule(model_name="ALBERT")
    global_model_albert=global_model_albert.to(device)

    pair_model=pair_wise_model()
    pair_model=pair_model.to(device)

    #-------------------------Upload corresponding checkpoint
    checkpoint=torch.load("weights/global-ensemblepair_sind_model_epoch_1_checkpoint_40155.tar")
    global_model_bert.load_state_dict(checkpoint["global_model_bert_state_dict"])
    global_model_albert.load_state_dict(checkpoint["global_model_albert_state_dict"])
    pair_model.load_state_dict(checkpoint["pair_model_state_dict"])

    global_model_bert.eval()
    global_model_albert.eval()
    pair_model.eval()

    print(global_model_bert)
    print(global_model_albert)
    print(pair_model)
    
    if args.dataset=="sind":
        test_paragraphs=pickle.load(open("sind_test.pkl","rb"))

    if args.dataset=="rocstories":
        test_paragraphs=pickle.load(open("rocstories_test.pkl","rb"))

    if args.dataset=="acl":
        test_paragraphs=pickle.load(open("acl_test.pkl","rb"))


    tot_acc_1=0
    perfect_matches_1=0
    tot_tau_1=0
    tot_acc_2=0
    perfect_matches_2=0
    tot_tau_2=0
    pair_accs=[]
    tot_pair_acc=0
    for i in range(test_paragraphs.shape[0]):
        paragraph=np.array(test_paragraphs[i])
        print("Solving for {} out of {}".format(i+1,test_paragraphs.shape[0]))
        num_sentences=paragraph.shape[0]
        print(num_sentences)
        if num_sentences==1:
            tot_acc_1+=1.0
            tot_tau_1+=1.0
            perfect_matches_1+=1.0
            tot_acc_2+=1.0
            tot_tau_2+=1.0
            perfect_matches_2+=1.0
            pair_accs.append(100*1.0)
            tot_pair_acc+=100*1.0
            continue
        original_paragraph=paragraph
        indices=np.arange(0,original_paragraph.shape[0])
        shuffled_indices=np.random.permutation(indices)
        test_paragraph=original_paragraph[shuffled_indices]
        test_paragraph_model=np.expand_dims(test_paragraph,axis=0)
        with torch.no_grad():
            pair_embeddings_bert,data_after_level_two_bert,batch_index_to_pair_dict = global_model_bert(test_paragraph_model)
            pair_embeddings_albert,data_after_level_two_albert,_ = global_model_albert(test_paragraph_model)
            output_logits = pair_model(pair_embeddings_bert,data_after_level_two_bert,batch_index_to_pair_dict, \
                                             pair_embeddings_albert,data_after_level_two_albert)
        #Calculating pairwise results
        correct_values=0
        softmax=torch.nn.Softmax(dim=0)
        pair_scores_dict={}
        for idx,tup in batch_index_to_pair_dict.items():
            score=output_logits[idx]
            _,j,k=tup
            pair_scores_dict[j,k]=softmax(score)
        for pair,score in pair_scores_dict.items():
            j,k=pair
            if(shuffled_indices[j]<shuffled_indices[k] and score[0]>=score[1]):
                correct_values+=1
            if(shuffled_indices[j]>shuffled_indices[k] and score[0]<score[1]):
                correct_values+=1
        tot_pairs=(num_sentences)*(num_sentences-1)
        pair_acc=correct_values/tot_pairs
        pair_accs.append(100*pair_acc)
        tot_pair_acc+=100*pair_acc
        ####
        solution_paragraph=solve_1(test_paragraph,pair_scores_dict)
        curr_acc=calc_acc(original_paragraph,solution_paragraph)
        tot_acc_1+=curr_acc
        if int(curr_acc)==1:
            perfect_matches_1+=1
        curr_tau=calc_tau(original_paragraph,solution_paragraph)
        tot_tau_1+=curr_tau
        ####
        solution_paragraph=solve_2(test_paragraph,pair_scores_dict)
        curr_acc=calc_acc(original_paragraph,solution_paragraph)
        tot_acc_2+=curr_acc
        if int(curr_acc)==1:
            perfect_matches_2+=1
        curr_tau=calc_tau(original_paragraph,solution_paragraph)
        tot_tau_2+=curr_tau
        ####
        print("Tau:{}, Acc:{}, Perfect matches till now:{}".format(curr_tau,curr_acc,perfect_matches_2))
        print("Curr PMR:{}, Curr tau:{}".format(perfect_matches_2/(i+1),tot_tau_2/(i+1)))
        print("Total accuracy:{},Total tau:{}".format(tot_acc_2,tot_tau_2))
        print("Curr pair accuracy :{},Avg pair accuracy :{}".format(pair_acc,tot_pair_acc/(i+1)))
        print("\n")
    avg_acc_1=tot_acc_1/test_paragraphs.shape[0]
    avg_tau_1=tot_tau_1/test_paragraphs.shape[0]
    pmr_1=perfect_matches_1/test_paragraphs.shape[0]
    avg_acc_2=tot_acc_2/test_paragraphs.shape[0]
    avg_tau_2=tot_tau_2/test_paragraphs.shape[0]
    pmr_2=perfect_matches_2/test_paragraphs.shape[0]
    print("Final results")
    print("Average tau_1         : {}".format(avg_tau_1))
    print("Average accuracy_1    : {}".format(avg_acc_1))
    print("Perfect match ratio_1 : {}".format(pmr_1))
    print("Average tau_2         : {}".format(avg_tau_2))
    print("Average accuracy_2    : {}".format(avg_acc_2))
    print("Perfect match ratio_2 : {}".format(pmr_2))
    pair_accs=np.array(pair_accs)
    print("Mean pair accuracy    : {}".format(np.mean(pair_accs)))
    print("STD pair accuracy    : {}".format(np.std(pair_accs,ddof=1)))

if __name__ == '__main__':
    main()