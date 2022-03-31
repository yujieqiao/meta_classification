import json

from pprint import pprint

import numpy as np
import tqdm
import pandas as pd
from collections import Counter
import requests




with open("modeldb-metadata.json") as peach:
    data_lib = json.load(peach)

with open("Cited_paper.json") as cucumber:
    cite_lib = json.load(cucumber) #cite_lib is a dict with key as model paper iD, and its values are the bundle of things including the paper it cites

#pprint(cite_lib['232876'])



def get_paper_id_name(item):
    new_tags=[]
    #print(f"item: {item}")
    for key,val in item.items():
        if key == "references":
            for paper in val["value"]:
                paper_id = paper["object_id"]
                try:
                    a=str(paper_id) + ":" + requests.get(f"http://modeldb.science/api/v1/papers/{paper_id}").json()["title"]["value"]
                    new_tags.append(a)
                except:
                    ...
    return new_tags

#pprint(get_paper_id_name(cite_lib['232876']['239073']))


paper={} #paper is a dict with key as model iD, and its values are the list of papers it cites
for model_id, paper_id in tqdm.tqdm(cite_lib.items()):
    try:
        paper[model_id]=[]
        for s in paper_id.values():
            paper[model_id].extend(get_paper_id_name(s))
    except:
        print(f"bad model: {model_id}")

#pprint(paper)

with open("new_tags.json", "w") as apple:
    apple.write(json.dumps(paper))


