# Out-of-knowledge-base entities linking

## 1. Introduction:
This repository is the code part of the CMPUT 692 class project: A study of out-of-KB entities.

## 2. Dependencies:

### For finding OOKB entities from text

```
pip install spacy
python -m spacy download en_core_web_md
```

### For training and testing model

```
pip install -U sentence-transformers
pip install torch
pip install numpy
pip install tqdm
```

## 3. Dataset & model download:
After downloading, unzip and put in the corresponding directory.
* [openKE_scripts/dbpedia50_openKE](openKE_scripts/dbpedia50_openKE)
* [open_world/dbpedia50](open_world/dbpedia50)
* [open_world/openke_models](open_world/openke_models)

## 4. Execution:

### Recognize out-of-KB entities in a document
```
cd open_world
python OOKB_entities_recognition.py
```
The program will ask for a file name. A sample document is in [open_world/example_text/sample.txt](open_world/example_text/sample.txt)

### Training a closed-world model:
Thanks to OpenKE KGC framework
```
cd openKE_scripts/openke
bash make.sh
cd ..
python train_transe300D_DBpedia50k.py
```

### Traning a open-world model:
```
cd open-world
python run_open_world.py --mode train
```

### Benchmarking:
```
cd open-world
python run_open_world.py --mode benchmark
```

### Predict a single OOKB entity with the description:
Prepare a file with the following format:  
First line: ID of the relation  
Second line: description of the entity

```
cd open-world
python run_open_world.py --mode predict --file [filename]
```
A sample file is in [open_world/example_text/test_predict.txt](open_world/example_text/test_predict.txt)

## 5. Acknowledge:
* Training the closed-world model using OpenKE (https://github.com/thunlp/OpenKE)
* Inspired by the OWE model (https://github.com/haseebs/OWE)
