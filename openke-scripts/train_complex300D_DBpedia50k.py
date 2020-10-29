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
	in_path = "./dbpedia50/kb2E/", 
	nbatches = 100,
	threads = 8, 
	sampling_mode = "normal", 
	bern_flag = 1
)

# dataloader for test
test_dataloader = TestDataLoader("./dbpedia50/kb2E/", "link")

# define the model
transe = TransE(
	ent_tot = train_dataloader.get_ent_tot(),
	rel_tot = train_dataloader.get_rel_tot(),
	dim = 200, 
	p_norm = 1, 
	norm_flag = True)


# define the loss function
model = NegativeSampling(
	model = transe, 
	loss = MarginLoss(margin = 1.0),
	batch_size = train_dataloader.get_batch_size()
)

# train the model
trainer = Trainer(model = model, data_loader = train_dataloader, train_times = 300, alpha = 1, use_gpu = True)
trainer.run()
transe.save_checkpoint('./checkpoint/transe.ckpt')

embeddings = transe.get_parameters()
directory = './models/dbpedia50/transe300/'
pathlib.Path(directory).mkdir(exist_ok=True, parents=True)

complex_name_map = {
    'ent_re_embeddings.weight': 'entities_r.p',
    'ent_im_embeddings.weight': 'entities_i.p',
    'rel_re_embeddings.weight': 'relations_r.p',
    'rel_im_embeddings.weight': 'relations_i.p'
}

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