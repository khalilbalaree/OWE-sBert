import requests
import spacy
import collections

nlp = spacy.load("en_core_web_md")

filename = input("Please input the filename: ")

with open(filename, 'r') as f:
    query = f.read().replace('\n', ' ')
    sentences = query.split('.')

doc = nlp(query)

base_url = "https://api.dbpedia-spotlight.org/en/annotate"
params = {"text": query, "confidence":0.1}
headers = {"accept": "application/json"}
r = requests.get(base_url, params=params, headers=headers)
if r.status_code != 200:
    print("Connection error")
l = r.json().get('Resources')

entities = []
for ll in l:
    entity = ll.get('@surfaceForm')
    if entity not in entities:
        entities.append(entity)

ner_entities = []
filter = ["DATE","TIME","MONEY", "PERCENT", "QUANTITY", "ORDINAL", "CARDINAL"]
for ent in doc.ents:
    if ent.label_ not in filter and ent.text not in entities:
        if ent.text not in ner_entities:
            ner_entities.append(ent.text)

for e in entities:
    for ne in ner_entities:
        if e in ne:
            ner_entities.remove(ne)
        if ne in e:
            ner_entities.remove(ne)

# remove duplicate entities
_filter = []
for e in ner_entities:
    sp = e.split(' ')
    _filter.extend(sp)
duplicate = [item for item, count in collections.Counter(_filter).items() if count > 1]
for du in duplicate:
    ner_entities.remove(du)

# description assignment
result = {}
for sent in sentences:
    for e in ner_entities:
        if e in sent:
            try:
                # _str = result[e]
                result[e] += sent.split(e)[1]
            except:
                result[e] = e + sent.split(e)[1]

print(result)
