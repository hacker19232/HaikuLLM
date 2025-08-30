print("hello")
print("saved")
import pickle
import torch
from llm import *

device = 'cpu'
context = torch.zeros((1,1,), dtype=torch.long, device = device)

model = pickle.load(open("llmhaiku.sav",'rb'))
print(decode(model.generate(idx = torch.zeros((1,1), dtype=torch.long), max_var_size=500)[0].tolist()))
