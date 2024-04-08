import csv
import time
import os
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
from bs4 import BeautifulSoup as soup

folder = './data'
# 讀取 CSV 檔案
df = pd.read_csv(f'{folder}/HGMD_pubmed.csv')


Path = "C:/Program Files/Google/Chrome/Application/chrome.exe"  # 使用原始字符串或者雙引號表示路徑
# 初始化 Chrome 選項
options = webdriver.ChromeOptions()
# 設置瀏覽器路徑
options.binary_location = Path
# 初始化 WebDriver，並傳遞 Chrome 選項
driver = webdriver.Chrome(options=options)

# 檢查是否已經存在標頭
output_file_path = f'{folder}/abstracts_data.csv'
headers_exist = os.path.exists(output_file_path)

# 初始化索引計數器
index_counter = 0

# 初始化已處理過的 Pubmed_id 集合
processed_pubmed_ids = set()

# 如果標頭已存在，則找到最大的索引值
if headers_exist:
    with open(output_file_path, 'r', newline='', encoding='utf-8') as abstract_file:
        reader = csv.reader(abstract_file)
        next(reader)  # 跳過標頭行
        for row in reader:
            index_counter = int(row[0])
            processed_pubmed_ids.add(row[1])  # 將 Pubmed_id 加入集合中

# 打開 CSV 文件來寫入摘要
with open(output_file_path, 'a', newline='', encoding='utf-8') as abstract_file:
    writer = csv.writer(abstract_file)
    # 如果標頭不存在，則寫入標頭
    if not headers_exist:
        writer.writerow(['Index', 'Pubmed_id', 'Abstract',
                        'questions', 'AMES'])  # 寫入CSV文件標頭

    # 迭代所有 PubMed ID
    for _, row in df.iterrows():
        Pubmed_id = str(int(row['Pubmed_id']))

        # 檢查是否已處理過這個 Pubmed_id
        if Pubmed_id in processed_pubmed_ids:
            print(f"Skipping PubMed ID {Pubmed_id} as it's already processed")
            continue

        # 打開 PubMed 網站
        driver.get(f'https://pubmed.ncbi.nlm.nih.gov/{Pubmed_id}/')

        try:
            # 使用 WebDriverWait 等待摘要元素加載完成
            abstract_elem = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CLASS_NAME, 'abstract')))
            page_source = driver.page_source
            # 使用 BeautifulSoup 解析網頁內容
            page_soup = soup(page_source, 'html.parser')
            # 找到所有 class 為 "abstract-content" 的 div 元素
            abstracts = page_soup.find_all("div", class_="abstract-content")
        except Exception as e:
            print(f"Error retrieving abstract for PubMed ID {Pubmed_id}: {e}")
            continue

        if not abstracts:
            print(f"No abstract found for PubMed ID {Pubmed_id}")
            continue

        abstracts = abstracts[0].text.strip()
        if not abstracts:
            print(f"No abstract text found for PubMed ID {Pubmed_id}")
            continue

        # 將索引計數器增加 1
        index_counter += 1

        # 寫入CSV文件
        writer.writerow([index_counter, Pubmed_id, abstracts,
                         'Is this variant pathogenic in this article?', row['HGMD_class']])
        print(f'Saved abstract for PubMed ID {Pubmed_id}')

# 關閉瀏覽器
driver.quit()
