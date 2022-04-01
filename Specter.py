import json
from pprint import pprint
import numpy as np
import tqdm
import pandas as pd
from collections import Counter
import requests
from pprint import pprint
import time
import xml.etree.ElementTree as ET



with open("modeldb-metadata.json") as peach:
    data_lib = json.load(peach)

#find the list of all "object_id" and store it in "unique_list"
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


def find_meta_within_data(dat):
    s=[]
    for v in dat.values():
        s.extend(find_within_model(v))
    return s

#link each model ID with its multiple model paper ID (m-n relationship)
model_paper={}
for item,val in data_lib.items():
    try:    
        model_paper[item]=find_within_list(val["model_paper"])
    except:
        model_paper[item]=[]

#print(model_paper['232875'])
# pprint(model_paper)
#print(find_within_list(data_lib["232876"]["model_paper"]))

with open("Cited_paper.json") as cucumber:
    cite_lib = json.load(cucumber) #cite_lib is a dict with key as model paper iD, and its values are the bundle of things including the paper it cites

#pprint(cite_lib['232876'])


# for each model paper id, gets its corresponding pubmed id
def get_pubmed_id(item):
    for key,val in item.items():
        if key == "pubmed_id":
            return val["value"]
        

pmid_dct={} #pmid_dct is a dict with key as model iD, and its values are the list of its model papers' corresponding pubmed id. 
# model_tofix={}
for model_id, paper_id in tqdm.tqdm(cite_lib.items()):
    try:
        pmid_dct[model_id]=[]
        # model_tofix[model_id]=[]
        for s in paper_id.values():
            pmid_dct[model_id].append(get_pubmed_id(s))
            # if get_pubmed_id(s)==None:
            #     model_tofix[model_id].append(s['id'])
    except:
        pmid_dct[model_id]=[]

# for k,v in model_tofix.items():
#     if v!=[]:
#         pprint(str(k)+str(model_tofix[k]))



# for each model paper ID, taking its corresponding pmids and getting abstracts and titles, then you'd call process_pmids and it would return a data structure
def lookup_pmids(pmids, delay_interval=1):
    time.sleep(delay_interval)
    return ET.fromstring(
        requests.post(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
            data={
                "db": "pubmed",
                "retmode": "xml",
                "id": ",".join(str(pmid) for pmid in pmids),
            },
        ).text
    )


def parse_paper(paper):
    abstract_block = paper.find(".//Abstract")
    mesh_heading = paper.find(".//MeshHeadingList")
    try:
        pmid = int(paper.find(".//PMID").text)
    except AttributeError:
        raise Exception("Bad paper? " + ET.tostring(paper, method="text").decode())
    title = paper.find(".//ArticleTitle")
    if title is None:
        title = paper.find(".//BookTitle")

    if abstract_block is not None:
        abstract = [
            {
                "section": item.get("Label"),
                "text": ET.tostring(item, method="text").decode(),
            }
            for item in abstract_block
            if item.tag == "AbstractText"
        ]
    else:
        abstract = ""
    assert title is not None
    title = ET.tostring(title, method="text").decode()
    mesh = []
    if mesh_heading is not None:
        for item in mesh_heading:
            item = item.find("DescriptorName")
            if item is not None:
                mesh.append(item.text)
    return pmid, {"mesh": mesh, "AbstractText": abstract, "ArticleTitle": title}


def process_pmids(pmids, delay_interval=1):
    results = {}
    papers = lookup_pmids(pmids, delay_interval=delay_interval)
    for paper in papers:
        pmid, parsed_paper = parse_paper(paper)
        results[pmid] = parsed_paper
    return results




# # pubmed_papers is a dictionary where the key is the model ID, and its value is a list of model paper ID and its assocaited data structure from PubMed
# pubmed_papers={}
# for key, val in tqdm.tqdm(pmid_dct.items()):
#     try:
#         pubmed_papers[key]=process_pmids(val, delay_interval=1)
#     except:
#         print(f"bad model: {key}")


# with open("pubmed_papers.json", "w") as apple:
#     apple.write(json.dumps(pubmed_papers))



# build a persistent dictionary (i.e. it only does the calculation once) for Specter embeddings
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("allenai/specter")
model = AutoModel.from_pretrained("allenai/specter")

#loading "pubmed_papers" which is a dictionary where the key is the model ID, and its value is a list of model paper ID and its assocaited data structure from PubMed
with open("pubmed_papers.json") as peach:
    pubmed_papers_0 = json.load(peach)

# print(len(pubmed_papers_0))

pubmed_papers={}
for val in pubmed_papers_0.values():
    for key1,val1 in val.items():
        pubmed_papers[key1]=val1

# print(len(pubmed_papers))

# pprint(pubmed_papers)



# for each pubmed id, gets its corresponding abstract
def get_abstract(item):
    for key,val in item.items():
        if key == "AbstractText":
            try:
                return val[0]["text"]
            except:
                return ''
        

import shelve
# print("computing SPECTER embeddings...")
# we can use a persistent dictionary so we don't have to redo this
with shelve.open("yujie-specter") as embeddings:
    for pmid, paper in tqdm.tqdm(pubmed_papers.items()):
        if pmid not in embeddings:
            data = [paper["ArticleTitle"] + tokenizer.sep_token + get_abstract(paper)]
            inputs = tokenizer(
                data, padding=True, truncation=True, return_tensors="pt", max_length=512
            )
            result = model(**inputs)
            # take the first token in the batch as the embedding
            embeddings[pmid] = result.last_hidden_state[:, 0, :].detach().numpy()[0]

# with shelve.open("yujie-specter") as embeddings:
#     pprint(embeddings)
#     print(len(embeddings))


# with open("embeddings_pubmed.json", "w") as apple:
#     apple.write(json.dumps(embeddings))



# create y vector

models=find_within_data(data_lib)

metatags=find_meta_within_data(data_lib)

# create a list of all unique metadata
unique_list = []
for x in metatags:
    if x not in unique_list:
        unique_list.append(x)

#print(unique_list)

# check_meta is a dictionary where the key is the pmid, the value is whether 
check_meta={}
for key, val in tqdm.tqdm(pubmed_papers_0.items()):
    for pmid in val.keys():
        for meta in unique_list:
            if meta in models[key]:
                check_meta[(pmid,meta)]=1
            else:
                check_meta[(pmid,meta)]=0

print("Finished creating check_meta")
#pprint(check_meta)



# Gaussian Naive Bayes
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut
loo = LeaveOneOut()


with shelve.open("yujie-matrices") as meta_matrices:
   
    for meta in tqdm.tqdm(unique_list):
        if str(meta) not in meta_matrices:
            row1=[]
            row2=[]
            with shelve.open("yujie-specter") as embeddings:
                for key, value in tqdm.tqdm(embeddings.items()):
                    row1.append(value)  
                    row2.append(check_meta[(key,meta)])


            assert(len(row2) == len(row1) == 1568)

            #print(row1[57])

            X_all=np.array(row1)
            y_all=np.array(row2)


            # # Regular spliting w. train/test sets
            # X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.3, random_state=0)


            # fit_start = time.perf_counter()
            # #gnb = GaussianNB()
            # adaboo = AdaBoostClassifier(n_estimators=100, random_state=0)

            # y_pred = adaboo.fit(X_train, y_train).predict(X_test)
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

            # print("adaboo")
            # print(confusion_mtx)


            # Leave one out CV
            loo = LeaveOneOut()
            confusion_mtx=np.zeros((2,2))
            for train_index, test_index in tqdm.tqdm(loo.split(X_all)):
                # print("TRAIN:", train_index, "TEST:", test_index)
                X_train, X_test = X_all[train_index], X_all[test_index]
                y_train, y_test = y_all[train_index], y_all[test_index]
                # print(X_train, X_test, y_train, y_test)
                fit_start = time.perf_counter()
                gnb = GaussianNB()
                #neigh = KNeighborsClassifier(n_neighbors=3)
                # adaboo = AdaBoostClassifier(n_estimators=100, random_state=0)
                # logireg = LogisticRegression(random_state=0)

                y_pred = gnb.fit(X_train, y_train).predict(X_test)
                # print(f"fit time: {time.perf_counter() - fit_start} seconds")
                # print("Number of mislabeled points out of a total %d points : %d" % (X_test.shape[0], (y_test != y_pred).sum()))
                if y_test==1 and y_pred==1:
                    confusion_mtx[0][0]+=1
                elif y_test==1 and y_pred==0:
                    confusion_mtx[1][0]+=1
                elif y_test==0 and y_pred==1:
                    confusion_mtx[0][1]+=1
                    # with shelve.open("yujie-specter") as embeddings:
                    #     for key, value in embeddings.items():
                    #         if (row1[int(test_index)]==value).all()==True:
                    #             print(key)

                else:
                    confusion_mtx[1][1]+=1

            #print("gaussian")
            meta_matrices[str(meta)]=confusion_mtx


        

    with shelve.open("yujie-matrices") as meta_matrices:    
        print(meta_matrices)