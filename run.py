
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from transformers import AutoModel , AutoTokenizer


# Data Organization
pos_path = "/content/drive/MyDrive/Colab_Resources/train_pos_full.txt"
neg_path = "/content/drive/MyDrive/Colab_Resources/train_neg_full.txt"
test_path = "/content/drive/MyDrive/Colab_Resources/test_data.txt"

with open(pos_path) as f:
  pos_data = f.read().splitlines()
with open(neg_path) as f:
  neg_data = f.read().splitlines()
with open(test_path) as f:
  test_data = f.read().splitlines()


# Util function to remove duplicates and blank tweets
def unique_non_empty(tweet_list):
    tweet_list = list(set(tweet_list))
    return [t for t in tweet_list if t]

pos_tweets = []
neg_tweets = []
sentiments = []

for tweet in pos_data :
  pos_tweets.append(tweet)

pos_tweets = unique_non_empty(pos_tweets)
for i in range(len(pos_tweets)):
  sentiments.append(1)

for tweet in neg_data :
  neg_tweets.append(tweet)

neg_tweets= unique_non_empty(neg_tweets)
for i in range(len(neg_tweets)):
  sentiments.append(0)

tweets = pos_tweets + neg_tweets


test_tweet = []
for tweet in test_data:
  test_tweet.append(tweet)

# Generating whole merged data set along with labels
data_set = {
    'tweet': tweets,
    'sentiment': sentiments  
}

df = pd.DataFrame(data_set)


# If no arguments then executer ces deux lignes. else argument 0 model_name et argument 1 tokenizer_name
model_name = 'microsoft/xtremedistil-l12-h384-uncased'
tokenizer_name = 'microsoft/xtremedistil-l12-h384-uncased'


# Creating Pytorch Dataset 
class TweeterDataset(torch.utils.data.Dataset):
  def __init__(self, df,tokenizer_name):
    self.df = df
    self.maxlen = 280
    self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

  def __len__(self):
    return len(self.df)

  def __getitem__(self, index):
    tweet = self.df['tweet'].iloc[index].split()
    tweet = ' '.join(tweet)
    sentiment = int(self.df['sentiment'].iloc[index])

    encodings = self.tokenizer.encode_plus(
        tweet,
        add_special_tokens=True,
        max_length=self.maxlen,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )

    return {
        'input_ids': encodings.input_ids.flatten(),
        'attention_mask': encodings.attention_mask.flatten(),
        'labels': torch.tensor(sentiment, dtype=torch.long)
    }

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

train_dataset = TweeterDataset(train_df)
valid_dataset = TweeterDataset(test_df)


# Creating training and validation data loaders
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

valid_loader = torch.utils.data.DataLoader(
    valid_dataset,
    batch_size=32
)

for batch in train_loader:
  print(batch['input_ids'].shape)
  print(batch['attention_mask'].shape)
  print(batch['labels'].shape)
  break


# Model Architecture for classification based on model name
class SentimentClassifier(nn.Module):
  
  def __init__(self,model_name):
    super(SentimentClassifier, self).__init__()
    self.pretrained = AutoModel.from_pretrained(model_name)
    self.dropout = nn.Dropout(0.3)
    if (model_name == 'XtremeDistil-l12-h384' | model_name == 'XtremeDistil-l6-h384'):
      output_size = 384
    elif (model_name == 'XtremeDistil-l6-h256'):
      output_size = 256
      output_size = 384
    else :
      output_size = 768
    self.linear1 = nn.Linear(output_size, 120)
    self.relu1 = nn.ReLU()
    self.drop1 = nn.Dropout(0.3)
    self.linear2 = nn.Linear(120, 2)
    self.relu2 = nn.ReLU()

  def forward(self, input_ids, attention_mask):
    outputs = self.pretrained(input_ids, attention_mask, return_dict=False)[0]
    x = self.relu1(self.linear1(outputs))
    x = self.dropout(x)
    x = self.linear2(x)
    return x[:, 0, :]


model = SentimentClassifier()
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# Model parameters
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.AdamW(model.parameters(), lr=5e-5)
epochs = 5

mean_training_losses = []
mean_eval_losses = []
best_model = None
best_val = 99

# Training Loop
for epoch in range(epochs):
  train_losses = []
  eval_losses = []
  # TRAIN
  model.train()
  train_loop = tqdm(train_loader)
  for batch in train_loop:
    optimizer.zero_grad()
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['labels'].to(device)
    output = model(input_ids, attention_mask)
    loss = criterion(output, labels)
    train_losses.append(loss.item())
    loss.backward()
    nn.utils.clip_grad_norm(model.parameters(), max_norm=1.0)
    optimizer.step()

    train_loop.set_description(f"Training Epoch: {epoch}")
    train_loop.set_postfix(loss=loss.item())

  # VALIDATION
  model.eval()
  valid_loop = tqdm(valid_loader)
  for batch in valid_loop:
    
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['labels'].to(device)
    output = model(input_ids, attention_mask)
    loss = criterion(output, labels)
    eval_losses.append(loss.item())
    valid_loop.set_description(f"Validation Epoch: {epoch}")
    valid_loop.set_postfix(loss=loss.item())
  
  mean_val = np.mean(eval_losses)
  if(best_val > mean_val):
    best_val= mean_val
    best_model = model
  mean_eval_losses.append(mean_val)
  mean_training_losses.append(np.mean(train_losses))

tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)


# Test Data organization
test_tweet = []
for tweet in test_data:
  test_tweet.append(tweet)

new_data_set = {
    'tweet': test_tweet,
}

evaluation_df = pd.DataFrame(new_data_set)


# Generating Predictions from trained model
evaluation_res = []
for i in range(len(evaluation_df)):
  print(i)
  test_sample = evaluation_df['tweet'].iloc[i]

  encodings = tokenizer.encode_plus(
      test_sample,
      add_special_tokens=True,
      max_length=256,
      padding='max_length',
      truncation=True,
      return_attention_mask=True,
      return_tensors='pt'
  )

  with torch.no_grad():
    model.to(device)
    preds = model(encodings['input_ids'].to(device), encodings['attention_mask'].to(device)).to('cpu')
    preds = np.argmax(preds)
    output = preds.item()
    evaluation_res.append(output)


# Replacing 0 predicted labels with -1 
new_result = [x if x ==1 else -1 for x in evaluation_res]

df_to_submit = pd.DataFrame(new_result)
df_to_submit.rename(columns={0: 'Prediction'}, inplace=True)
df_to_submit.insert(0, 'Id', range(1, 1 + len(df_to_submit)))

# Exporting results to csv
df_to_submit.to_csv('/content/drive/MyDrive/Colab_Resources/'+model_name, index=False)

# Saving trained model
torch.save(model, '/content/drive/MyDrive/Colab_Resources/'+model_name+'.pth')

