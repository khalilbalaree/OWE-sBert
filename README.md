# Out-of-knowledge-base entities linking

## 1. Dependencies:

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

## 2. Execution:

### Training a closed-world model:
1. Download the OpenKE KGC framework from https://github.com/thunlp/OpenKE.
2. Use the python scripts in [/openke-scripts](/openke-scripts).
3. Copy the checkpoint files to [/open-world/checkpoint](/open-world/checkpoint) 

### Traning a open-world model:
```
cd open-world
python3 run_open_world.py --mode train
```

### Benchmarking:
```
cd open-world
python3 run_open_world.py --mode benchmark
```

### Predict a single OOKB entity with the description:
Prepare a file with the following format:  
First line: ID of the relation  
Second line: description of the entity

```
cd open-world
python3 run_open_world.py --mode predict --file [filename]
```
