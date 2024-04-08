import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

folder = "./fig"
data = "./data"
file = "abstracts_data.csv"

df = pd.read_csv(f'{data}/{file}')

df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)

# 以80%的資料作為訓練集，20%的資料作為驗證集
train_data_df, dev_data_df = train_test_split(
    df_shuffled, test_size=0.2, random_state=42)

# 資料打亂
train_data_df_shuffled = train_data_df.sample(
    frac=1, random_state=42).reset_index(drop=True)

# 分割資料極為True和False
df_shuffled_true = train_data_df_shuffled[train_data_df_shuffled['AMES'] == True]
df_shuffled_false = train_data_df_shuffled[train_data_df_shuffled['AMES'] == False]

n_samples = min(len(df_shuffled_true), len(df_shuffled_false))


df_balanced_true = df_shuffled_true.sample(n=n_samples, random_state=42)
df_balanced_false = df_shuffled_false.sample(n=n_samples, random_state=42)

# print(df_balanced_true.shape[0])
# print(df_balanced_true.shape[0])

df_shuffled_balanced = pd.concat([df_balanced_true, df_balanced_false])

df_shuffled_balanced = df_shuffled_balanced.sample(
    frac=1, random_state=42).reset_index(drop=True)
# print(df_shuffled_balanced.shape[0])


# # Count the number of samples in each category
# true_count = len(df_shuffled_balanced[df_shuffled_balanced['AMES'] == True])
# false_count = len(df_shuffled_balanced[df_shuffled_balanced['AMES'] == False])

# # Create a bar plot
# plt.bar(['True', 'False'], [true_count, false_count])

# # Set the labels and title
# plt.xlabel('Category')
# plt.ylabel('Count')
# plt.title('Counts of df_shuffled_true and df_shuffled_false')

# # Display the counts on the bars
# for i, count in enumerate([true_count, false_count]):
#     plt.text(i, count, str(count), ha='center', va='bottom')

# # Display the plot
# plt.savefig(f'{folder}/ames_chart_new.png')

# train_data_df.to_csv(f'{data}/train_data.csv', index=False)
# dev_data_df.to_csv(f'{data}/dev_data.csv', index=False)
# df_shuffled_balanced.to_csv(f'{data}/train_data_balanced.csv', index=False)
