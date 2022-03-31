a={'mesh': ['Animals', 'Cell Division', 'Circadian Rhythm', 'DNA Replication', 'Feeding Behavior', 'Female', 'Hexachlorocyclohexane', 'Liver', 'Rats'], 'AbstractText': [{'section': None, 'text': 'Stimulation of hepatic DNA synthesis can be achieved in the intact rat by alpha-hexachlorocyclohexane (alpha-HCH = alpha-benzene hexachloride). The extent of stimulation is high in the morning and low in the evening. These rhythmic variations in the rate of DNA synthesis are synchronized indirectly by the light-dark rhythm, but directly by the animal\'s feeding habits: Rats eat preferentially during the night. If the diurnal rhythm of food intake is abolished, the rhythmic fluctuations in the rate of DNA synthesis are no longer detectable; if rats are adapted to daily feeding periods of only 5 h, these fluctuations are pronounced and almost synchronized. Further experiments show that the time of feeding determines the time of DNA replication. It is concluded that food intake provides a "2nd stimulus" or permissive factor, which is required for the induction of DNA synthesis in a certain critical stage of the prereplicative phase. Labelling experiments with orotic acid suggest that foot intake initially induces an increase of RNA synthesis. The results indicate that controlled feeding schedules provide the possibility to synchronize, in the living animal, a proliferating population of hepatocytes. A hypothesis is derived which offers an explanation for the generation of the diurnal rhythm of cell proliferation in the liver.'}], 'ArticleTitle': "[Feeding rhythms and the diurnal rhythm of cell proliferation in pharmacologically induced liver growth (author's transl)]."}

def get_abstract(item):
    for key,val in item.items():
        if key == "AbstractText":
            return val[0]["text"]

print(get_abstract(a))

# for each model paper id, gets its corresponding pubmed id
def get_pubmed_id(item):
    for key,val in item.items():
        if key == "pubmed_id":
            return val["value"]

pmid_dct={} #pmid_dct is a dict with key as model iD, and its values are the list of its model papers' corresponding pubmed id. 
model_tofix={}
for model_id, paper_id in tqdm.tqdm(cite_lib.items()):
    try:
        pmid_dct[model_id]=[]
        for s in paper_id.values():
            pmid_dct[model_id].append(get_pubmed_id(s))
            if get_pubmed_id(s)==None:
                model_tofix[model_id]=s['id']
    except:
        pmid_dct[model_id]=[]

pprint(model_tofix)