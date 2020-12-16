import openke
from openke.config import Trainer, Tester
from openke.module.model import TransE
from openke.module.loss import MarginLoss
from openke.module.strategy import NegativeSampling
from openke.data import TrainDataLoader, TestDataLoader
import pickle
import pathlib

# # dataloader for training
train_dataloader = TrainDataLoader(
	in_path = "./dbpedia50_openKE/kb2E/", 
	nbatches = 100,
	threads = 8,  
	bern_flag = 1
)

# dataloader for test
test_dataloader = TestDataLoader("./dbpedia50_openKE/kb2E/", "link")

# define the model
transe = TransE(
	ent_tot = train_dataloader.get_ent_tot(),
	rel_tot = train_dataloader.get_rel_tot(),
	dim = 300)


# define the loss function
model = NegativeSampling(
	model = transe, 
	loss = MarginLoss(),
	batch_size = train_dataloader.get_batch_size()
)

# train the model
trainer = Trainer(model = model, data_loader = train_dataloader, train_times = 1000, alpha = 0.01, use_gpu = True, opt_method='adagrad')
trainer.run()
transe.save_checkpoint('./checkpoint/transe.ckpt')

embeddings = transe.get_parameters()
directory = './models/dbpedia50/transe300/'
pathlib.Path(directory).mkdir(exist_ok=True, parents=True)

other_name_map = {
    'ent_embeddings.weight': 'entities.p',
    'rel_embeddings.weight': 'relations.p'
}

def save_torch_embedding_as_numpy(embedding, filename):
    with open(filename, "wb") as f:
        pickle.dump(embedding, f)

for emb_name, filename in other_name_map.items():
    print("Saving to %s" % (directory + filename))
    save_torch_embedding_as_numpy(embeddings[emb_name], directory + filename)

# test the model
transe.load_checkpoint('./checkpoint/transe.ckpt')
tester = Tester(model = transe, data_loader = test_dataloader, use_gpu = True)
tester.run_link_prediction(type_constrain = False)