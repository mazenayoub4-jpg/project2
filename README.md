# Imports
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,
f1_score, recall_score, confusion_matrix, classification_report,
confusion_matrix
# Dataset (Example - can be replaced later)
"text": =
("i love this movie", 1)
("this is amazing" 1),
"i enjoyed this film" 0),
"not my favorite" 0),
"it was okay" 1),
"absoulutey fantastic" 0),
"could be better" 0)
]
label = [1, 1, 0, 0, 1, 0, 0]
# Preprocessing
def tokenize(sentence):
  return sentence.lower().split()
  # Encode sentences
def tokenize_text(text):
  return text.split()
  word_to_idx = {"<PAD>":0, "UNK">:1}
  for sentence in texts:
    for word in tokenize(sentence):
      if word not in word_to_idx:
        word_to_idx[word] = len(word_to_idx)
  return word_to_idx
  # Padding
def encode_sentence(sentence, word_to_idx):
  return [word_to_idx.get(word, word_to_idx["<UNK>"]) for word in tokenize(sentence)]
  encoded_texts = [encode_sentence(t, word_to_idx) for t in texts]
  return encoded_texts
  max_len = max(len(s) for (s) in encoded_texts)
  def pad_sequence(seq, max_len):
    return seq + [word_to_idx["<PAD>"]] * (max_len - len(seq))
    padded_texts = [pad_sequence(s, max_len) for s in encoded_texts]
  return padded_texts
  # Train / Test Split
X_train, X_test, y_train, y_test = train_test_split(padded_texts, labels, test_size=0.25, random_state=42)
X_train = torch.tensor(X_train)
X_test = torch.tensor(X_test)
y_train = torch.tensor(y_train).float().unsqueeze(1)
y_test =  torch.tensor(y_test).float().unsqueeze(1)
# Model
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
    #Training Setup
vocab_size = len(word_to_idx)
embed_dim = 16
learning_rate = 0.001
num_epochs = 200
model = SentimentModel(vocab_size, embed_dim)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
#Training Loop
for epoch in range(num_epochs):
  model.train()
  optimizer.zero_grad()
  outputs = model(X_train)
  loss = criterion(outputs, y_train)
  loss.backward()
  optimizer.step()
if (epoch + 1) % 20 == 0:
  print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")
  #Evaluation
model.eval()
with torch.no_grad():
  y_pred = model(X_test)
  y_pred_label = (y_pred > 0.5).int()
  print("\Accuracy:", accuracy_score(y_test, y_pred_label))
  print("F1 Score:", f1_score(y_test, y_pred_label))
  print("\nClassification Report:\n", classification_report(y_test, y_pred_label))
  print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_label))
1 # Dataset & Preprocessing 
data = [
    ("i love this movie", 1), ("this is amazing", 1),
    ("not my favorite", 0), ("i hate this", 0),
    ("this is terrible", 0)
]
texts, label = zip(*data)
word_to_idx = {"<PAD>":0, "<UNK>": 1}
for sentence in texts:
  for word in sentence.lower().split():
    if word not in word_to_idx:
      word_to_idx[word] = len(word_to_idx)  

encoded_texts = 
[torch.tensor([word_to_idx.get(w, 1) for w in t.lower().split()) for t in texts]
X = pad_sequence(encoded_texts, batch_first=True)
y = torch.tensor(labels).float().unsqueeze(1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# Step 1: Using DataLoader
class SentimentDataset(Dataset):
def__init__(self, X, y):
  self.X = X
  self.y = y
  def__len__(self):
    return len(self.X)
  def__getitem__(self, idx):
    return self.X[idx], self.y[idx]
  train_loader =
  DataLoader(SentimentDataset(X_train, y_train), batch_size=2, shuffle=True)

  # Step 2: The LSTM Architecture
  class advancedLSTM(nn.Module):
    def__init__(self, vocab_size, 
    embed_dim, hidden_dim):
    super(AdvancedLSTM, self).__init__()
    self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
    self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
    self.fc = nn.Linear(hidden_dim, 1)
    self.sigmoid = nn.Sigmoid()
    def forward(self, x):
      embedded = self.embedding(x)
      # LSTM returns output and (hidden state, cell state)
      _,(hidden, _) = self.lstm(embedded)
      # Use the last hidden state for classifcation 
      return self.sigmoid(self.fc(hidden[-1]))

     # Step 3 integrated Training
     model =
     AdvancedLSTM(len(word_to_idx),
    embed_dim=16, hidden_dim-32)
     optimizer = optimAdam(model.parameters(), ir=0.01)
     criterion = nn.BCELoss()
     
     for epoch in range(50):
      for batch_X, batch_y in train_loader:
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

      if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1} completed")
