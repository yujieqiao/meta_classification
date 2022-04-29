from transformers import *

tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')

tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_cased')
model = AutoModel.from_pretrained('allenai/scibert_scivocab_cased')