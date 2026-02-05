# Imports
import torch
import torch.nn as nn
import torch.optim as optim
from sk.learn.model_selection import train_test_split
from sk.learn.metrics import accuracy_score,
f1_score, recall_score, confusion_matrix, classification_report,
confusion_matrix
# Dataset (Example - can be replaced later)
"text": =
("i love this movie", 1)
("this is amazing" 1),
"i enjoyed this flim"
"not my favorite"
"it was okay"
"absoulutey fantastic"
"could be better"
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
