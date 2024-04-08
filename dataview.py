import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

folder = './fig'
# 讀取 CSV 檔案
df = pd.read_csv('./data/abstracts_data.csv')

# 計算 'AMES' 欄位中 True 的數量
count_True = df['AMES'].value_counts().loc[True]
print(f'Number of True values in AMES column: {count_True}')
count_False = df['AMES'].value_counts().loc[False]
print(f'Number of False values in AMES column: {count_False}')

# 製作長條圖
ax = sns.countplot(x='AMES', data=df)
plt.xlabel('AMES')
plt.ylabel('Count')
plt.title('Count of True and False values in AMES column')

# 將數字轉換為整數類型，然後格式化輸出
for p in ax.patches:
    height = int(p.get_height())  # 將高度轉換為整數類型
    ax.annotate(f'{height}', (p.get_x() + p.get_width() /
                2, height), ha='center', va='bottom')


plt.savefig(f'{folder}/ames_chart.png')
