import json
from pprint import pprint
import numpy as np
import tqdm
import pandas as pd
from collections import Counter
import requests
import json as json



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

#find the list of all "object_id" for each model and store it in "corpus"
#def find_within_item(item):
        return str(item['object_id']) + ":" + item['object_name']


#def find_within_list(list_of_items):
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
    s=[]
    for v in dat.values():
        s.extend(find_within_model(v))
    return s





corpus=find_within_data(data_lib)  

#print(len(corpus))


#link each "object_id" to its assocaited "object_name"
def link_within_item(item):
    dict1 = {}
    dict1[item['object_name']] = item['object_id']
    return dict1


def link_within_list(list_of_items):
    try:
        s={}
        for item in list_of_items:
            s.update(link_within_item(item))
        return s
    except:
        return {}
    

def link_within_model(model):
    s={}
    for v in model.values():
        s.update(link_within_list(v))
    return s

def link_within_data(dat):
    s={}
    for v in dat.values():
        s.update(link_within_model(v))
    return s


linkage=link_within_data(data_lib)

#print(linkage['GENESIS'])

#link each model ID with its multiple model paper ID (m-n relationship)
model_paper={}
for item,val in data_lib.items():
    try:    
        model_paper[item]=find_within_list(val["model_paper"])
    except:
        model_paper[item]=[]

#print(model_paper['232875'])
#print(find_within_list(data_lib["232876"]["model_paper"]))


cite_dic={}
for model_id, paper_id in tqdm.tqdm(model_paper.items()):
    cite_dic[str(model_id)]={}
    cites={}
    for s in paper_id:
        try: 
            cites[s] = requests.get(f"http://modeldb.science/api/v1/papers/{s}").json() 
        except:
            cites[s] =[]
        
    cite_dic[str(model_id)].update(cites)
       

#cite_dic={}
#for model_id, paper_id in tqdm.tqdm(model_paper.items()):
    #cite_dic[str(model_id)] = {s: requests.get(f"http://modeldb.science/api/v1/papers/{s}").json() for s in paper_id}

#cite_dic={
    #str(model_id): {s: requests.get(f"http://modeldb.science/api/v1/papers/{s}").json() for s in paper_id}
    #for model_id, paper_id in tqdm.tqdm(model_paper.items())
#}



with open("Cited_paper_34.json", "w") as apple:
    apple.write(json.dumps(cite_dic))