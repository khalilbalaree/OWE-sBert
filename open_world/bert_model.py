from transformers import DistilBertTokenizer, DistilBertModel
import torch.nn as nn
import torch

class mapper(nn.Module):
    def __init__(self):
        super(mapper,self).__init__()
        self.model = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.linear = nn.Linear(768, 300, bias=True)
        self.act = nn.Tanh()
    
    def forward(self, encoded):
        bert_outputs = self.model(**encoded)[0]
        first_token = bert_outputs[:, 0]
        a = self.linear(first_token)
        b = self.act(a)
        return b

class text_tokens():
    def __init__(self):
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    
    def get_tokens(self, text):
        return self.tokenizer(text, return_tensors='pt',padding=True, truncation=True)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)
# model = mapper().to(device)
# tokenizer = text_tokens()
# text = 'blow ya mind song american hip hop recording artist styles p featuring vocals production swizz beatz song taken third studio album super gangster extraordinary gentleman released october 9 2007 album lead single blow ya mind considered sequel 2002 debut single good times also produced swizz beatz official remix features swizz beatz along styles p d-block cohorts jadakiss sheek louch remix appears video game grand theft auto iv music video directed todd angkasuwan styles p cameo appearances video made idris elba fellow d-block member sheek louch video styles p sitting bench starts hallucinations song peaked number 19 billboard hot rap tracks chart number 51 hot r b/hip-hop songs chart'
# o = model(tokenizer.get_tokens(text).to(device))
# print(o)