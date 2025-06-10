import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding
from datasets import load_dataset, Dataset, DatasetDict
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import pandas as pd
import numpy as np
from deep_translator import GoogleTranslator

# Carregar o dataset GoEmotions (em inglês)
dataset = load_dataset("go_emotions")

MAP_6_LABELS = {
    "Satisfação": {0, 1, 4, 5, 8, 13, 15, 17, 18, 20, 21, 23},  # alegria, aprovação, gratidão, etc.
    "Frustração": {2, 3, 6, 9, 10, 11, 12, 14, 16, 19, 24, 25}, # desapontamento, tristeza, etc.
    "Confusão": {7},                                            # confusão
    "Urgência/Pressão": {22},                                   # urgência
    "Raiva/Irritação": {26},                                    # raiva
    "Neutralidade": {27}                                        # neutro
}
LABEL2ID = {
    "Satisfação": 0,
    "Frustração": 1,
    "Confusão": 2,
    "Urgência/Pressão": 3,
    "Raiva/Irritação": 4,
    "Neutralidade": 5
}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

def map_emotion_6(emotions):
    if not emotions:
        return LABEL2ID["Neutralidade"]
    main_emotion = emotions[0]
    for label, indices in MAP_6_LABELS.items():
        if main_emotion in indices:
            return LABEL2ID[label]
    return LABEL2ID["Neutralidade"]

def translate_batch(texts, batch_size=30):
    translated_texts = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        batch_translated = []
        for text in batch:
            try:
                translated = GoogleTranslator(source='en', target='pt').translate(text)
                batch_translated.append(translated)
            except Exception:
                batch_translated.append(text)
        translated_texts.extend(batch_translated)
    return translated_texts

def prepare_data(split):
    texts = dataset[split]['text']
    emotions = dataset[split]['labels']
    labels = [map_emotion_6(emo) for emo in emotions]
    df = pd.DataFrame({'text': texts, 'label': labels})
    max_samples = 500 if split == 'train' else 100
    df = df.sample(min(len(df), max_samples), random_state=42)
    print(f"Traduzindo {len(df)} textos do conjunto {split}...")
    translated_texts = translate_batch(df['text'].tolist())
    df['text_pt'] = translated_texts
    return df

print("Preparando dados de treinamento...")
train_df = prepare_data('train')
print("Preparando dados de teste...")
test_df = prepare_data('test')

train_dataset = Dataset.from_pandas(
    train_df[['text_pt', 'label']].rename(columns={'text_pt': 'text', 'label': 'labels'})
)
test_dataset = Dataset.from_pandas(
    test_df[['text_pt', 'label']].rename(columns={'text_pt': 'text', 'label': 'labels'})
)

datasets = DatasetDict({
    'train': train_dataset,
    'test': test_dataset
})

MODEL_NAME = 'neuralmind/bert-base-portuguese-cased'
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=6,
    problem_type="single_label_classification",
    id2label=ID2LABEL,
    label2id=LABEL2ID
)

def preprocess_function(examples):
    tokens = tokenizer(
        examples['text'],
        truncation=True,
        padding='max_length',
        max_length=256
    )
    tokens["labels"] = examples["labels"]
    return tokens

tokenized_datasets = datasets.map(preprocess_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    acc = accuracy_score(labels, predictions)
    return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}

training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=4,
    learning_rate=4e-5,
    weight_decay=0.01,
    logging_steps=50,
    save_steps=100,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['test'],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

trainer.train()

eval_result = trainer.evaluate()
print("Resultados na validação:", eval_result)

exemplos = [
    "Fiquei muito satisfeito com o atendimento.",
    "Estou frustrado por não ter conseguido resolver.",
    "Não entendi o que aconteceu, estou confuso.",
    "Preciso de uma resposta imediatamente!",
    "Esse erro me deixou com muita raiva.",
    "Ok, obrigado pela informação."
]
inputs = tokenizer(exemplos, padding=True, truncation=True, return_tensors="pt").to(model.device)
outputs = model(**inputs)
preds = torch.argmax(outputs.logits, dim=1)
rotulos = [ID2LABEL[i] for i in range(6)]
for texto, pred in zip(exemplos, preds):
    print(f"Texto: {texto}\nSentimento previsto: {rotulos[pred]}\n")

model.save_pretrained("sentiment_model_ptbr_finetuned")
tokenizer.save_pretrained("sentiment_model_ptbr_finetuned")
