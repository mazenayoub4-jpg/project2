import torch
import torch.nn as nn
import torch.optim as optim
import torch numpy as np
import pandas as pd
import re
from sk.learn.metrics import f1_score recall_score, accuracy_score
from sk.learn model_selection import train_test_split
from matplotlib import pyplot as plt
"text": = 
"i love this movie"
"this movie is amazing"
"i enjoyed this flim" 
"not my favorite"
"it was okay"
"absoulutey fantastic"
"could be better"
] 
label = [1, 1, 0, 0]
word_to_idx = {
"i": 0
"love": 1,
"this": 2,
"movie" 3,
"is" 4,
"amazing" 5,
"hate" 6,
"terrible": 7
}
def code_sentence(sentence, word_to_idx):
  return [word_to_idx[word] for word in sentence.split
  encoded_texts = [encode_sentence(t, word_to_idx) for t in texts]
  return encoded_texts
  max_len = max([len(t) for t in encoded_texts])
  x_train, x_test, y_train, y_test = train_test
  import torch
import torch.nn as nn
class SentimentModel(nn.Module):
  def __init__(self, vocab_size, embed_dim):
    super().__init__()
   self.Embedding = nn.Embedding(vocab_size, embed_dim)
   self.fc = nn.Linear(embed_dim, 1)
   self.sigmoid = nn.Sigmoid()
   def forward(self, x):
    embedded = self.embedding(x)
    avg_embed = embedded.mean(embedded, dim=1)
    output = self.fc(avg_embed)
    return self.sigmoid(output)
    vocab_size = len(word_to_idx)
embed_dim = 8
learning_rate = 0.001
num_epochs = 200
model = SentimentModel(vocab_size, embedding_dim)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
if (epoch+1) % 20 == 0:
  print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
  X_tensor = torch.tensor(X_train).float()
y_tensor = torch.tensor(y_train).float().unsqueeze(1)
f1 = f1 score(y = y_test, y_pred = y_pred)
recall = recall_score(y_test, y_pred)
conf_matrix = confusion matrix (y_test, y_pred)
print(f"F1 Score: {f1:.4f}')
print(f"Recall: {recall:,4f}')
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred)
