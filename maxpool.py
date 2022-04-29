import scispacy
import spacy
from gensim.models import Word2Vec

# Data preprocessing
rawtext={} #rawtext: key is pmid, value is article title and the abstract
for pmid, paper in tqdm.tqdm(pubmed_papers.items()):
    rawtext[pmid] = paper["ArticleTitle"] + get_abstract(paper)

nlp = spacy.load("en_core_sci_sm")

import shelve
with shelve.open("yujie-scispacy") as scients:
    for pmid, content in tqdm.tqdm(rawtext.items()):
        if pmid not in scients:
            text=content
            doc = nlp(text)
            scients[pmid]=str(doc.ents)
            


# prepare the corpus for w2c model
with shelve.open("yujie-scispacy") as scients:
    w2c_dict=[]
    scients_dict={}
    for pmid,val in tqdm.tqdm(scients.items()):
        sentences=list(val.split(','))
        w2c_dict.append(sentences)   #w2c_dict is a list containing all the unique words from 1568 pmid papers
        scients_dict[pmid]=sentences   #scients_dict: key is pmid, value is list of words extracted from article title and the abstract


# builiding the w2c model
# a=[]
# a.append(w2c_dict)

vec_size=400
        
w2c_model = Word2Vec(a, min_count=1, vector_size=vec_size)

words = w2c_model.wv.index_to_key

we_dict = {word:w2c_model.wv[word] for word in tqdm.tqdm(words)}



# creating the max pooling matrix
aggregate_max_pool={}
for pmid,val in tqdm.tqdm(scients_dict.items()):
    model_vec=[] #model_vec is a matrix combining the 400-d vectors for all the words within a pmid model
    for word in val:
        model_vec.append(we_dict[word])
    
    
    max_pool=[]
    
    for i in range(vec_size):    
        #pooling_list=[]       #pooling_list is a vector that combined the value of each word's vector value at a certain position (from position 1 to 400)
        #for val in model_vec:
        #    pooling_list.append(val[i])
        #max_pool.append(max(pooling_list))  #max_pool is a 400x1 vector: each dimension correspond to the max value out of the pooling list at each position 
        max_pool.append(max(val[i] for val in model_vec))
    aggregate_max_pool[pmid]=max_pool   




maxpool_matrix = list(aggregate_max_pool.values())
maxpool_matrix=np.array(maxpool_matrix)  # maxpool_matrix is a 1568x400 matrix: for each pmid, it has a 400-d vector that contains the max value (most salient feature) out of all the words
