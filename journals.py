import re
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


journal_pmid_list=[]
with open('JComputNeurosci.txt') as f:
    lines1 = f.readlines()
    pmid_list1=re.findall("PMID: ([0-9]+)", str(lines1))

journal_pmid_list.extend(pmid_list1)

with open('FrontComputNeurosci.txt') as o:
    lines2 = o.readlines()
    pmid_list2=re.findall("PMID: ([0-9]+)", str(lines2))
    
journal_pmid_list.extend(pmid_list2)

with open('JMathNeurosci.txt') as p:
    lines3 = p.readlines()
    pmid_list3=re.findall("PMID: ([0-9]+)", str(lines3))

journal_pmid_list.extend(pmid_list3) 

with open('NeuralComput.txt') as q:
    lines4 = q.readlines()
    pmid_list4=re.findall("PMID: ([0-9]+)", str(lines4))

journal_pmid_list.extend(pmid_list4)

journal_pmid_dict={}
for i, pmid in enumerate(journal_pmid_list):
    journal_pmid_dict[i]=list(pmid.split())


# print(journal_pmid_dict)


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
        results[str(pmid)] = parsed_paper
    return results




# pubmed_papers is a dictionary where the key is the model ID, and its value is a list of model paper ID and its assocaited data structure from PubMed
journal_papers={}
for val in tqdm.tqdm(journal_pmid_dict.values()):
    try:
        journal_papers.update(process_pmids(val, delay_interval=1))
        
    except:
        print(f"bad model: {pmid}")


with open("journal_papers.json", "w") as pear:
    pear.write(json.dumps(journal_papers))

