from library import *
from data_loader import *

# Set seeds for reproducibility
random.seed(26)
np.random.seed(26)
torch.manual_seed(26)

tokenizer = AutoTokenizer.from_pretrained(
    "michiyasunaga/BioLinkBERT-base", do_lower_case=True)

model = AutoModelForSequenceClassification.from_pretrained(
    "michiyasunaga/BioLinkBERT-base")
model.to(device)  # Send the model to the GPU if we have one

# Helper functions for implementing layerwise learning rate decay
learning_rate = 1.3818e-05
layerwise_learning_rate_decay = 0.9
weight_decay = 0.01
adam_epsilon = 1e-6
use_bertadam = False

# scheduler params
num_epochs = 100
num_warmup_steps = 0
_model_type = 'bert'


def get_optimizer_grouped_parameters(
    model, model_type,
    learning_rate, weight_decay,
    layerwise_learning_rate_decay
):
    no_decay = ["bias", "LayerNorm.weight"]
    # initialize lr for task specific layer
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if "classifier" in n or "pooler" in n],
            "weight_decay": 0.0,
            "lr": learning_rate,
        },
    ]
    # initialize lrs for every layer
    num_layers = model.config.num_hidden_layers
    layers = [getattr(model, model_type).embeddings] + \
        list(getattr(model, model_type).encoder.layer)
    layers.reverse()
    lr = learning_rate
    for layer in layers:
        lr *= layerwise_learning_rate_decay
        optimizer_grouped_parameters += [
            {
                "params": [p for n, p in layer.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,
                "lr": lr,
            },
            {
                "params": [p for n, p in layer.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
                "lr": lr,
            },
        ]
    return optimizer_grouped_parameters


grouped_optimizer_params = get_optimizer_grouped_parameters(
    model, _model_type,
    learning_rate, weight_decay,
    layerwise_learning_rate_decay
)
optimizer = AdamW(
    grouped_optimizer_params,
    lr=learning_rate,
    eps=adam_epsilon,
    correct_bias=not use_bertadam
)
scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=num_epochs
)

(learning_rates1, learning_rates2, learning_rates3, learning_rates4,
 learning_rates5, learning_rates6, learning_rates7, learning_rates8,
 learning_rates9, learning_rates10, learning_rates11, learning_rates12,
 learning_rates13, learning_rates14) = [[] for i in range(14)]


def collect_lr(optimizer):
    learning_rates1.append(optimizer.param_groups[0]["lr"])
    learning_rates2.append(optimizer.param_groups[2]["lr"])
    learning_rates3.append(optimizer.param_groups[4]["lr"])
    learning_rates4.append(optimizer.param_groups[6]["lr"])
    learning_rates5.append(optimizer.param_groups[8]["lr"])
    learning_rates6.append(optimizer.param_groups[10]["lr"])
    learning_rates7.append(optimizer.param_groups[12]["lr"])
    learning_rates8.append(optimizer.param_groups[14]["lr"])
    learning_rates9.append(optimizer.param_groups[16]["lr"])
    learning_rates10.append(optimizer.param_groups[18]["lr"])
    learning_rates11.append(optimizer.param_groups[20]["lr"])
    learning_rates12.append(optimizer.param_groups[22]["lr"])
    learning_rates13.append(optimizer.param_groups[24]["lr"])
    learning_rates14.append(optimizer.param_groups[26]["lr"])


collect_lr(optimizer)
for epoch in range(num_epochs):
    optimizer.step()
    scheduler.step()
    collect_lr(optimizer)


def encode_data(tokenizer, passages, questions, max_length):
    """Encode the question/passage pairs into features than can be fed to the model."""
    input_ids = []
    attention_masks = []

    for passage, question in zip(passages, questions):
        encoded_data = tokenizer.encode_plus(
            passage, question, max_length=max_length, pad_to_max_length=True, truncation='longest_first')
        encoded_pair = encoded_data["input_ids"]
        attention_mask = encoded_data["attention_mask"]

        input_ids.append(encoded_pair)
        attention_masks.append(attention_mask)

    return np.array(input_ids), np.array(attention_masks)


# Loading data
passages_train = df_shuffled_balanced.Abstract.values
questions_train = df_shuffled_balanced.questions.values
answers_train = df_shuffled_balanced.AMES.values.astype(int)

pubmed_id = df_shuffled_balanced.Pubmed_id.values
passages_dev = dev_data_df.Abstract.values
questions_dev = dev_data_df.questions.values
answers_dev = dev_data_df.AMES.values.astype(int)

# Encoding data
max_seq_length = 512
input_ids_train, attention_masks_train = encode_data(
    tokenizer, questions_train, passages_train, max_seq_length)
input_ids_dev, attention_masks_dev = encode_data(
    tokenizer, questions_dev, passages_dev, max_seq_length)

train_features = (input_ids_train, attention_masks_train, answers_train)
dev_features = (input_ids_dev, attention_masks_dev, answers_dev)

# Preparing pytorch dataloader for model training
batch_size = 2

train_features_tensors = [torch.tensor(
    feature, dtype=torch.long) for feature in train_features]
dev_features_tensors = [torch.tensor(
    feature, dtype=torch.long) for feature in dev_features]

train_dataset = TensorDataset(*train_features_tensors)
dev_dataset = TensorDataset(*dev_features_tensors)

train_sampler = RandomSampler(train_dataset)
dev_sampler = SequentialSampler(dev_dataset)

train_dataloader = DataLoader(
    train_dataset, sampler=train_sampler, batch_size=batch_size)
dev_dataloader = DataLoader(
    dev_dataset, sampler=dev_sampler, batch_size=batch_size)

train_loss_values = []
dev_acc_values = []
state = []


# train the model
for epoch in tqdm(range(num_epochs), desc="Epoch"):
    # Training
    epoch_train_loss = 0  # Cumulative loss
    model.train()
    print("Epoch：", epoch+1)
    for step, batch in enumerate(train_dataloader):

        input_ids = batch[0].to(device)
        attention_masks = batch[1].to(device)
        labels = batch[2].to(device)
        model.zero_grad()
        outputs = model(input_ids, token_type_ids=None,
                        attention_mask=attention_masks, labels=labels)

        loss = outputs[0]
        epoch_train_loss += loss.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
    epoch_train_loss = epoch_train_loss / len(train_dataloader)
    train_loss_values.append(epoch_train_loss)
    print("Epoch loss is", epoch_train_loss)

    # Evaluation
    epoch_dev_accuracy = 0  # Cumulative accuracy
    model.eval()

    for batch in dev_dataloader:

        input_ids = batch[0].to(device)
        attention_masks = batch[1].to(device)
        labels = batch[2]

        with torch.no_grad():
            outputs = model(input_ids, token_type_ids=None,
                            attention_mask=attention_masks)

        logits = outputs[0]
        logits = logits.detach().cpu().numpy()

        predictions = np.argmax(logits, axis=1).flatten()
        labels = labels.numpy().flatten()

        epoch_dev_accuracy += np.sum(predictions == labels) / len(labels)

    epoch_dev_accuracy = epoch_dev_accuracy / len(dev_dataloader)
    print("Epoch accuracy is", epoch_dev_accuracy)
    temp = model
    state.append(temp)
    dev_acc_values.append(epoch_dev_accuracy)

    best_accuracy = max(dev_acc_values)


# Best accuracy
if epoch_dev_accuracy == best_accuracy:
    print("Saving the best model with accuracy:", best_accuracy)
    best_model = state[-1]  # Save the model from the last epoch
    best_model.save_pretrained('./trained-model')
else:
    print("Best accuracy:", best_accuracy)

print('-'*100)


# ---------------------------------------------------------------------------------------------------------- #

# 初始化混淆矩陣元素
true_positive, false_positive, true_negative, false_negative = 0, 0, 0, 0
# 假設你已經計算了混淆矩陣元素
conf_matrix = [[true_negative, false_positive],
               [false_negative, true_positive]]

# 定義混淆矩陣的標籤
labels = ['True', 'False']

# 使用 seaborn 的 heatmap 來顯示混淆矩陣
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=labels, yticklabels=labels)

# 加入標題和軸標籤
plt.title('Confusion Matrix')
plt.xlabel('Actual')
plt.ylabel('Predicted')

# 顯示圖形
plt.savefig(f'{folder}/confusion_matrix.png')

# 初始化空列表來存儲 false negative 和 false positive 的資料
false_negative_data = []
false_positive_data = []

# 透過 DataLoader 進行預測
for batch in dev_dataloader:
    input_ids = batch[0].to(device)
    attention_masks = batch[1].to(device)
    labels = batch[2].to(device)

    with torch.no_grad():
        outputs = model(input_ids, token_type_ids=None,
                        attention_mask=attention_masks)

    logits = outputs[0]
    predictions = torch.argmax(logits, dim=1)

    # 計算混淆矩陣元素 前面是實際值 後面是預測值
    true_positive += torch.sum((predictions == 1) & (labels == 1)).item()
    false_positive += torch.sum((predictions == 1) & (labels == 0)).item()
    true_negative += torch.sum((predictions == 0) & (labels == 0)).item()
    false_negative += torch.sum((predictions == 0) & (labels == 1)).item()

    # 找出 false negative 和 false positive 的索引
    fn_indices = ((predictions == 0) & (labels == 1)).nonzero()
    fp_indices = ((predictions == 1) & (labels == 0)).nonzero()

    # 提取 false negative 和 false positive 的資料並儲存
    for idx in fn_indices:
        idx = idx.item()
        passage = passages_dev[idx]
        question = questions_dev[idx]
        correct_label = answers_dev[idx]
        pubmed_id = dev_data_df.iloc[idx].Pubmed_id
        false_negative_data.append(
            (passage, question, correct_label, pubmed_id))

    for idx in fp_indices:
        idx = idx.item()
        passage = passages_dev[idx]
        question = questions_dev[idx]
        correct_label = answers_dev[idx]
        pubmed_id = dev_data_df.iloc[idx].Pubmed_id
        false_positive_data.append(
            (passage, question, correct_label, pubmed_id))


# 初始化集合來存儲已經寫入的資料
written_data = set()

# 將 false negative 和 false positive 資料寫入檔案
with open(f'{folder}/False.txt', 'w') as f:
    dic = {'0': 'False', '1': 'True'}
    for idx, (passage, question, correct_label, pubmed_id) in enumerate(false_negative_data):
        data_tuple = (passage, question, correct_label, pubmed_id)
        if data_tuple not in written_data:
            f.write(f"false_negative {idx + 1}:\n")
            f.write(f"PubMed ID: {pubmed_id}\n")  # 將PubMed文章ID添加到文本中
            f.write(f"Abstract: {passage}\n")
            f.write(f"Question: {question}\n")
            f.write(f"Correct Label: {dic[str(correct_label)]}\n\n")
            # 將已寫入的資料加入集合
            written_data.add(data_tuple)

    for idx, (passage, question, correct_label, pubmed_id) in enumerate(false_positive_data):
        data_tuple = (passage, question, correct_label, pubmed_id)
        if data_tuple not in written_data:
            f.write(f"false_positive {idx + 1}:\n")
            f.write(f"PubMed ID: {pubmed_id}\n")  # 將PubMed文章ID添加到文本中
            f.write(f"Abstract: {passage}\n")
            f.write(f"Question: {question}\n")
            f.write(f"Predict: {dic[str(correct_label)]}\n\n")
            f.write(f"Correct Label: {dic[str(correct_label)]}\n\n")
            # 將已寫入的資料加入集合
            written_data.add(data_tuple)


# 打印混淆矩陣元素
print("True Positive:", true_positive)
print("False Positive:", false_positive)
print("True Negative:", true_negative)
print("False Negative:", false_negative)
