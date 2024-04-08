#importing the libraries
import random
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm.notebook import tqdm
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW,AutoModelForQuestionAnswering, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from transformers import DebertaTokenizer, DebertaModel, BartTokenizer
import math

# Use a GPU if you have one available (Runtime -> Change runtime type -> GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set seeds for reproducibility
random.seed(26)
np.random.seed(26)
torch.manual_seed(26)

#Loading tokenizer
tokenizer = AutoTokenizer.from_pretrained("michiyasunaga/BioLinkBERT-base", do_lower_case=True)

#Loading Model 
#change model file path accordingly here
model = AutoModelForSequenceClassification.from_pretrained("./trained-model")
model.to(device) # Send the model to the GPU if we have one

#Reading of dataset using Pandas 
df = pd.read_csv('./data/data.csv')
train_data_df, dev_data_df = train_test_split(df, test_size = 0.2, random_state = 42)

#Helper function for predicting
def predict(passage,question):
  sequence = tokenizer.encode_plus(passage,question, return_tensors="pt")['input_ids'].to(device)
  
  logits = model(sequence)[0]
  probabilities = torch.softmax(logits, dim=1).detach().cpu().tolist()[0]
  proba_yes = round(probabilities[1], 2)
  proba_no = round(probabilities[0], 2)

  print(f"Question: {question}, Yes: {proba_yes}, No: {proba_no}")

  if (proba_yes >= proba_no):
    return True
  else:
    return False 
  
passage = "" #Provide necessary scientific abstract here
questions = "" #Provide the right question here. Refer to the data file on how the question annotation was done

predict(passage,questions)





