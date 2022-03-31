import json
from pprint import pprint
from collections import Counter


with open("new_tags.json") as cucumber:
    new_tags_lib = json.load(cucumber) #new_tags_lib is a dict with key as model iD, and its values are the list of papers it cites

freq_new_tags_lib=[]
for val in new_tags_lib.values():
    freq_new_tags_lib.extend(val)



#print(freq_new_tags_lib)

count=Counter(freq_new_tags_lib)

#pprint(count)

prepro_new_tags=[]
for item,val in count.items():
    if val>2 and val < 75:
        prepro_new_tags.append(item)

print(len(prepro_new_tags))
print(len(count))

filtered_new_tags_lib={}
for key,val in new_tags_lib.items():
    filt_papers=[]
    for p in val:
        if p in prepro_new_tags:
            filt_papers.append(p)
    filtered_new_tags_lib[key]=filt_papers

#pprint(filtered_new_tags_lib)

print(len(filtered_new_tags_lib))