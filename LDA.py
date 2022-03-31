import json
import gensim
import gensim.corpora as corpora
import pickle 
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
from pprint import pprint
from gensim.corpora import Dictionary
from gensim.models.ldamodel import LdaModel
from gensim.models import CoherenceModel
from multiprocessing import freeze_support
import numpy as np
import tqdm
import pandas as pd
from collections import Counter


freeze_support()

with open("modeldb-metadata.json") as peach:
    data_lib = json.load(peach)



#find the list of all "object_id" for each model and store it in "corpus"
def find_within_item(item):
        return str(item['object_id']) + ":" + item['object_name']


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
        s.append(find_within_model(v))
    return s




if __name__ == "__main__":

    


    corpus=find_within_data(data_lib)  



    #pprint(corpus)

    #print(len(corpus))

    dct = corpora.Dictionary(corpus)



    #pprint(dct)

    corpus2 = [dct.doc2bow(text) for text in corpus]

    #pprint(corpus2)

    #pprint([[(dct[id], freq) for id, freq in word] for word in corpus2])

    # Build LDA model
    #lda = gensim.models.LdaMulticore(corpus=corpus2, id2word=dct, num_topics=5, random_state=100, chunksize=100,passes=10, per_word_topics=True)


    lda = LdaModel(corpus2, num_topics=20)

    #pprint(lda.print_topics())





    vis = gensimvis.prepare(lda, corpus2, dct)

    pyLDAvis.save_html(vis, "test.html")

    import os
    os.system("open test.html")

    


    # Compute Coherence Score
    #coherence_model_lda = CoherenceModel(model=lda, texts=corpus, dictionary=dct, coherence='c_v')
    #coherence_lda = coherence_model_lda.get_coherence()
    #print('\nCoherence Score: ', coherence_lda)



    # supporting function
    def compute_coherence_values(corpus_, dictionary, k, a, b):
    
        lda = gensim.models.LdaMulticore(corpus=corpus_, id2word=dictionary,num_topics=k, random_state=100,chunksize=100, passes=10, alpha=a,eta=b)
    
        coherence_model_lda = CoherenceModel(model=lda, texts=corpus, dictionary=dictionary, coherence='c_v')
    
        return coherence_model_lda.get_coherence()

    grid = {}
   
    # Topics range
    min_topics = 35
    max_topics = 60
    step_size = 1
    topics_range = range(min_topics, max_topics, step_size)
    # Alpha parameter
    alpha = list(np.arange(0.01, 1, 0.3))
    alpha.append('symmetric')
    alpha.append('asymmetric')
    # Beta parameter
    beta = list(np.arange(0.01, 1, 0.3))
    beta.append('symmetric')

  

    model_results = {'Topics': [], 'Coherence': []}
    
    # Can take a long time to run
    
       
    for k in tqdm.tqdm(topics_range):
        cv = compute_coherence_values(corpus_=corpus2, dictionary=dct, k=k, a=0.8, b=0.2)
        # Save the model results
        model_results['Topics'].append(k)
        model_results['Coherence'].append(cv)
                    
            
    pd.DataFrame(model_results).to_csv('lda_tuning_results_alpha4.csv', index=False)
 