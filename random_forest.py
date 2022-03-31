import json
from collections import Counter
import tqdm
import time
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneOut
loo = LeaveOneOut()


with open("modeldb-metadata.json") as peach:
    data_lib = json.load(peach)

with open("new_tags.json") as cucumber:
    new_tags_lib = json.load(cucumber) #new_tags_lib is a dict with key as model iD, and its values are the list of papers it cites

# Filtering out the high and low frequency of the new tags (cited paper)
freq_new_tags_lib=[]
for val in new_tags_lib.values():
    freq_new_tags_lib.extend(val)

count_paper=Counter(freq_new_tags_lib)

prepro_new_tags=[]
for item,val in count_paper.items():
    if 75>val>2:
        prepro_new_tags.append(item)

filtered_new_tags_lib={}
for key,val in new_tags_lib.items():
    filt_papers=[]
    for p in val:
        if p in prepro_new_tags:
            filt_papers.append(p)
    filtered_new_tags_lib[key]=filt_papers


freq_filtered_new_tags_lib=[]
for val in filtered_new_tags_lib.values():
    freq_filtered_new_tags_lib.extend(val)

count_filtered_paper=Counter(freq_filtered_new_tags_lib)


paper_dct=[]
for key in count_filtered_paper.keys():
    paper_dct.append(key)

# print(paper_dct)


print(len(paper_dct))

def find_within_item(item):
        return item['object_id']


def find_within_list(list_of_items):
    try:
        s=[]
        for item in list_of_items:
            s.append(find_within_item(item))
        return s
    except:
        return []
    

def find_within_model(model):
    s=[]
    for v in model.values():
        s.extend(find_within_list(v))
    return s

def find_within_data(dat):
    s={}
    for key,val in dat.items():
        s[key]=find_within_model(val)
    return s


models=find_within_data(data_lib)


row1=[]
row2=[]
for (key, val),(key1, val1) in tqdm.tqdm(zip(new_tags_lib.items(),models.items())):
    column=[]
    for p in paper_dct:
        if p in val:
            column.append(1)
        else:
            column.append(0)
    row1.append(column)
    assert(key == key1)
    if 65417 in val1:
        row2.append(1)
    else:
        row2.append(0)

assert(len(row2) == len(row1) == 1706)

X_all=np.array(row1)
y_all=np.array(row2)

# print(row1)
# print(row2)

# Leave one out CV
loo = LeaveOneOut()
confusion_mtx=np.zeros((2,2))
for train_index, test_index in tqdm.tqdm(loo.split(X_all)):
    # print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X_all[train_index], X_all[test_index]
    y_train, y_test = y_all[train_index], y_all[test_index]
    # print(X_train, X_test, y_train, y_test)
    fit_start = time.perf_counter()
    gnb =  RandomForestClassifier()
    y_pred = gnb.fit(X_train, y_train).predict(X_test)
    print(f"fit time: {time.perf_counter() - fit_start} seconds")
    print("Number of mislabeled points out of a total %d points : %d" % (X_test.shape[0], (y_test != y_pred).sum()))
    if y_test==1 and y_pred==1:
        confusion_mtx[0][0]+=1
    elif y_test==1 and y_pred==0:
        confusion_mtx[1][0]+=1
    elif y_test==0 and y_pred==1:
        confusion_mtx[0][1]+=1
    else:
        confusion_mtx[1][1]+=1

print(confusion_mtx)


# # Regular spliting w. train/test sets
# X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.3, random_state=0)

# fit_start = time.perf_counter()
# gnb = RandomForestClassifier()

# y_pred = gnb.fit(X_train, y_train).predict(X_test)
# print(f"fit time: {time.perf_counter() - fit_start} seconds")
# print("Number of mislabeled points out of a total %d points : %d" % (X_test.shape[0], (y_test != y_pred).sum()))

# confusion_mtx=np.zeros((2,2))
# for m,n in zip(y_test, y_pred):
#     if m==1 and n==1:
#         confusion_mtx[0][0]+=1
#     elif m==1 and n==0:
#         confusion_mtx[1][0]+=1
#     elif m==0 and n==1:
#         confusion_mtx[0][1]+=1
#     else:
#         confusion_mtx[1][1]+=1

# print(confusion_mtx)