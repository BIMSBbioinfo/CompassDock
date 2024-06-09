import pandas as pd
from sklearn.model_selection import train_test_split

# Load your dataset from the text file
data = pd.read_csv('common_ids.txt', header=None)

# 80-20 Split for fine-tuning
train, val = train_test_split(data, test_size=0.2, random_state=42)

# Saving the datasets
train.to_csv('finetune_train.txt', index=False, header=False)
val.to_csv('finetune_val.txt', index=False, header=False)
