import openke
from openke.config import Trainer, Tester
from openke.module.model import ComplEx
from openke.module.loss import SoftplusLoss
from openke.module.strategy import NegativeSampling
from openke.data import TrainDataLoader, TestDataLoader
import pickle
import pathlib

# # dataloader for training
train_dataloader = TrainDataLoader(
	in_path = "./dbpedia50/kb2E/", 
	nbatches = 100,
	threads = 12, 
	sampling_mode = "normal", 
	bern_flag = 1,
    neg_ent = 25,
	neg_rel = 0
)

# dataloader for test
test_dataloader = TestDataLoader("./dbpedia50/kb2E/", "link")

# define the model
complEx = ComplEx(
	ent_tot = train_dataloader.get_ent_tot(),
	rel_tot = train_dataloader.get_rel_tot(),
	dim = 300
)

# define the loss function
model = NegativeSampling(
	model = complEx, 
	loss = SoftplusLoss(),
	batch_size = train_dataloader.get_batch_size(), 
	regul_rate = 1.0
)

# train the model
trainer = Trainer(model = model, data_loader = train_dataloader, train_times = 2000, alpha = 0.5, use_gpu = True, opt_method='adagrad')
trainer.run()
complEx.save_checkpoint('./checkpoint/complEx.ckpt')

embeddings = complEx.get_parameters()
directory = './models/dbpedia50/complex300/'
pathlib.Path(directory).mkdir(exist_ok=True, parents=True)

complex_name_map = {
    'ent_re_embeddings.weight': 'entities_r.p',
    'ent_im_embeddings.weight': 'entities_i.p',
    'rel_re_embeddings.weight': 'relations_r.p',
    'rel_im_embeddings.weight': 'relations_i.p'
}


def save_torch_embedding_as_numpy(embedding, filename):
    with open(filename, "wb") as f:
        pickle.dump(embedding, f)

for emb_name, filename in complex_name_map.items():
    print("Saving to %s" % (directory + filename))
    save_torch_embedding_as_numpy(embeddings[emb_name], directory + filename)

# test the model
complEx.load_checkpoint('./checkpoint/complEx.ckpt')
tester = Tester(model = complEx, data_loader = test_dataloader, use_gpu = True)
tester.run_link_prediction(type_constrain = False)