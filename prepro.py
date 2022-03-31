import json
import gensim
from gensim.test.utils import datapath
import pickle 
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
from pprint import pp, pprint
from gensim.corpora import Dictionary
from gensim.models.ldamodel import LdaModel
from gensim.models import CoherenceModel
from multiprocessing import freeze_support
import numpy as np
import tqdm
import pandas as pd
from collections import Counter
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB

freeze_support()


with open("modeldb-metadata.json") as peach:
    data_lib = json.load(peach)

with open("new_tags.json") as cucumber:
    new_tags_lib = json.load(cucumber) #new_tags_lib is a dict with key as model iD, and its values are the list of papers it cites


#count=0
#for val in new_tags_lib.values():
    #if val !=[]:
        #count+=1
#print(count)



#link each "object_id" to its assocaited "object_name". Key is 'object_name', value is 'object_id'
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




##Step 1: Filter out the tags with extreme (high/low) frequency. We keep abs. freq. (3,115) here

def find_within_item(item):
    if '(web link to model)' in item['object_name'] or '(web link to method)' in item['object_name']:
        index=item['object_name'].rfind('(')
        m=item['object_name'][:index-1]
        try:
            return linkage[m] #return the tag's ID w/o "(web link to model)", only if there is a corresponding one. For example, there is only "Neuronvisio (web link to model)", but no "Neuronvisio"
        except:
            return item['object_id']
    if item['object_name'] == "Brian 2": #consolidate the tags "Brian 2" to "Brian"
        return linkage['Brian']
    else:
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
    s=[]
    for v in dat.values():
        s.extend(find_within_model(v))
    return s


# Filtering out the high and low frequency of the original tags (metadata fields)
freq=find_within_data(data_lib)  
count=Counter(freq)
#print(count)
prepro=[]
for item,val in count.items():
    if val>3 and val < 115:
        prepro.append(item)

#print(len(prepro))
#print(len(count))


# Filtering out the high and low frequency of the new tags (cited paper)
freq_new_tags_lib=[]
for val in new_tags_lib.values():
    freq_new_tags_lib.extend(val)

count_paper=Counter(freq_new_tags_lib)

prepro_new_tags=[]
for item,val in count_paper.items():
    if val>2 and val < 75:
        prepro_new_tags.append(item)

filtered_new_tags_lib={}
for key,val in new_tags_lib.items():
    filt_papers=[]
    for p in val:
        if p in prepro_new_tags:
            filt_papers.append(p)
    filtered_new_tags_lib[key]=filt_papers
    

## Step 2: Building corpus. If meta-tag falls into desired frequency, we include them as tags for each model; 
# also incorporate the papers cited by all of the paper IDs, and assign them as new tags for each model.

def find_within_item_prepro(item):
    if '(web link to model)' in item['object_name'] or '(web link to method)' in item['object_name']:
        index=item['object_name'].rfind('(')
        m=item['object_name'][:index-1]
        if linkage[m] in prepro:
            return str(linkage[m])+ ":" + m
    if item['object_id'] in prepro and item['object_id'] != 182684: #182684 tag is the "Allen Institute (2015)"". We remove them for their bias
        return str(item['object_id']) + ":" + item['object_name']
    
    if item['object_name'] == "Brian 2":
        return str(linkage['Brian'])+ ":" + "Brian"

    else:
        return None

def find_within_list_prepro(list_of_items):
    try:
        s=[]
        for item in list_of_items:
            item_name = find_within_item_prepro(item)
            if item_name is not None:
                s.append(item_name)
        return s
    except:
        return []
    


# for each model, return its meta-tags, along with all paper tags
def find_within_model_prepro(model):
    s=[]
    for key,val in model.items():
        s.extend(find_within_list_prepro(val))
        if key == "id":
            s.extend(filtered_new_tags_lib[str(val)]) #filtered_new_tags_lib is a dict with key as model iD, and its values are the list of papers it cites (with paper frequency =(2,75))
    return s


def find_within_data_prepro(dat):
    s=[]
    for v in dat.values():
        s.append(find_within_model_prepro(v))
    return s

def find_within_data_prepro_withid(dat):
    s={}
    for key,val in dat.items():
        s[key]=find_within_model_prepro(val)
    return s

corpus=find_within_data_prepro(data_lib)  

corpus_withid=find_within_data_prepro_withid(data_lib)

#pprint(corpus_withid)

#print(corpus_withid['87284'])


if __name__ == "__main__":

    
    

    #print(len(corpus))

    dct = gensim.corpora.Dictionary(corpus)

    #print(dct)


    #pprint(dct)

    corpus2 = [dct.doc2bow(text) for text in corpus]

    #pprint(corpus2)

    #pprint([[(dct[id], freq) for id, freq in word] for word in corpus2])

    # # Build LDA model

    # lda_built = LdaModel(corpus2, id2word=dct, num_topics=15)

    #pprint(lda_built.print_topics())

    # # Save model to disk.
    temp_file = datapath("model")
    # lda_built.save(temp_file)

    # # Load a potentially pretrained model from disk.
    lda_sample = LdaModel.load(temp_file)

    # words = lda_sample.show_topic(1, topn = 200)

    # print(words)

    # print(corpus[1])

    # ques_vec = dct.doc2bow(corpus[1])
    
    # topic_vec = lda_built[ques_vec]

    # print(topic_vec)

    #print(dct[corpus2[567][0][0]])
    #print(corpus[567])

    print(dct.doc2bow(corpus_withid['87284']))
    
    for index, score in sorted(lda_sample[dct.doc2bow(corpus_withid['87284'])], key=lambda tup: -1*tup[1]):
        s=[]
        total=0
        grand_total = 0
        for (a,b) in lda_sample.show_topic(index, topn = 50000):
            grand_total += b
            if a in corpus_withid['87284']:
                v=str(b)+"*"+a
                s.append(v) 
                total+=b        
        print("\nScore: {}\t \nTopic: {}\t \n{}".format(score, index, s))
        print("total: ", total)
        print("grandtotal", grand_total)

    # pprint(lda.print_topics())

    # for idx, topic in lda.print_topics(-1):
    #     print('Topic: {} \nWords: {}'.format(idx, topic))



    # vis = gensimvis.prepare(lda, corpus2, dct)

    # pyLDAvis.save_html(vis, "LDA_15.html")

    # import os
    # os.system("open test.html")
