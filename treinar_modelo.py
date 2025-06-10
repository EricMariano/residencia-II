import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding
from datasets import load_dataset, Dataset, DatasetDict
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import pandas as pd
import numpy as np
from deep_translator import GoogleTranslator

# Carregar o dataset GoEmotions (em inglês)
print("Carregando o dataset GoEmotions...")
dataset = load_dataset("go_emotions")
print("Dataset GoEmotions carregado com sucesso.")

# --- Mapeamento expandido e REFINADO das emoções para 6 categorias ---
# Mapeamento para: 0: Satisfação, 1: Frustração, 2: Confusão, 3: Urgência/Pressão, 4: Raiva / Irritação, 5: Neutralidade

emotion_mapping_to_6_categories = {
    # Satisfação (0) - Inclui alegria, admiração, aprovação, gratidão, etc.
    0: 0,   # admiration
    1: 0,   # amusement
    4: 0,   # approval
    5: 0,   # caring
    8: 0,   # desire (movido para Satisfação, como um desejo positivo)
    13: 0,  # excitement (prazer do excitamento)
    15: 0,  # gratitude
    17: 0,  # joy
    18: 0,  # love
    20: 0,  # optimism
    21: 0,  # pride
    23: 0,  # relief

    # Frustração (1) - Inclui desapontamento, tristeza, arrependimento, e embaraço (como desapontamento consigo).
    9: 1,   # disappointment
    10: 1,  # disapproval
    12: 1,  # embarrassment (movido para Frustração, como desapontamento consigo)
    16: 1,  # grief
    24: 1,  # remorse
    25: 1,  # sadness

    # Confusão (2) - Inclui curiosidade e realização (quando leva a uma nova compreensão)
    6: 2,   # confusion
    7: 2,   # curiosity
    22: 2,  # realization (se o "realizar" for no sentido de entender algo que antes era confuso)

    # Urgência/Pressão (3) - Foco em medo, nervosismo.
    14: 3,  # fear (muitas vezes associado a urgência)
    19: 3,  # nervousness (associado a pressão)

    # Raiva / Irritação (4) - Inclui raiva, aborrecimento e nojo (como aversão forte).
    2: 4,   # anger
    3: 4,   # annoyance
    11: 4,  # disgust (movido para Raiva/Irritação, como aversão forte)

    # Neutralidade (5) - Explicitamente neutro ou onde o mapeamento não é claro.
    27: 5,  # neutral
    # Se alguma emoção original não estiver mapeada acima, ela cairá para Neutralidade por padrão.
}


def map_emotion_to_6_categories(emotions):
    """Mapeia as emoções originais do GoEmotions para as 6 categorias definidas."""
    if not emotions:
        return 5  # Retorna 'Neutralidade' se a lista de emoções estiver vazia

    main_emotion_id = emotions[0]
    # Retorna o mapeamento, ou 'Neutralidade' (5) se a emoção não estiver no dicionário
    return emotion_mapping_to_6_categories.get(main_emotion_id, 5)

def translate_batch(texts, batch_size=30):
    """Traduz uma lista de textos para o português em lotes."""
    translated_texts = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        batch_translated = []
        for text in batch:
            try:
                translated = GoogleTranslator(source='en', target='pt').translate(text)
                batch_translated.append(translated)
            except Exception as e:
                # Em caso de erro na tradução, retorna o texto original
                print(f"Erro ao traduzir texto: '{text[:50]}...' Error: {e}. Usando texto original.")
                batch_translated.append(text)
        translated_texts.extend(batch_translated)
    return translated_texts

def prepare_data(split):
    """Prepara e traduz os dados de um split específico do dataset."""
    texts = dataset[split]['text']
    emotions = dataset[split]['labels']
    labels = [map_emotion_to_6_categories(emo) for emo in emotions]
    df = pd.DataFrame({'text': texts, 'label': labels})

    # --- AUMENTO DA QUANTIDADE DE AMOSTRAS PARA MELHOR ACCURACY ---
    max_samples = 3000 if split == 'train' else 600
    df = df.sample(min(len(df), max_samples), random_state=42) # Amostragem para limitar o tamanho
    print(f"Traduzindo {len(df)} textos do conjunto '{split}'...")
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

# ---

# CONFIGURAÇÃO DO MODELO E TREINAMENTO

MODEL_NAME = 'neuralmind/bert-base-portuguese-cased'
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

num_labels = 6
id2label_map = {
    0: 'Satisfação', 1: 'Frustração', 2: 'Confusão',
    3: 'Urgência/Pressão', 4: 'Raiva / Irritação', 5: 'Neutralidade'
}
label2id_map = {v: k for k, v in id2label_map.items()}

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=num_labels,
    problem_type="single_label_classification",
    id2label=id2label_map,
    label2id=label2id_map
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
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted', zero_division=0)
    acc = accuracy_score(labels, predictions)
    return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}

# --- Configurações de treinamento: Removidos argumentos de avaliação para evitar o TypeError ---
training_args = TrainingArguments(
    output_dir="./results_6_sentiments",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=4, # AUMENTADO PARA 4 ÉPOCAS
    learning_rate=4e-5,
    weight_decay=0.01,
    logging_steps=50,
    save_steps=100,
    report_to="none",
)

# Inicializando o Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['test'],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

print("Iniciando treinamento do modelo...")
trainer.train()

# Avaliando o modelo no conjunto de teste
eval_result = trainer.evaluate()
print("\nResultados na validação (6 sentimentos):", eval_result)

# ---

# TESTE COM EXEMPLOS NOVOS E SALVAR O MODELO

exemplos_6_sentimentos = [
    "O serviço foi excelente, fiquei muito satisfeito com tudo.", # Satisfação (0)
    "Este problema é muito frustrante, não consigo avançar de jeito nenhum.", # Frustração (1)
    "Não entendi a explicação, estou completamente confuso sobre isso.", # Confusão (2)
    "Preciso que isso seja resolvido com a máxima urgência! O prazo é amanhã.", # Urgência/Pressão (3)
    "Estou com muita raiva de como isso foi tratado, é inaceitável.", # Raiva / Irritação (4)
    "A reunião foi bastante neutra, sem grandes novidades ou discussões.", # Neutralidade (5)
    "Estou animado com as novas oportunidades que surgiram.", # Satisfação (0)
    "Isso é realmente irritante e me deixa com a paciência esgotada.", # Raiva / Irritação (4)
    "Não sei o que pensar sobre essa situação, é muito incerto.", # Confusão (2)
    "O prazo está apertado, sinto uma pressão enorme para entregar o trabalho.", # Urgência/Pressão (3)
    "Recebi um presente maravilhoso, estou muito feliz!", # Satisfação (0)
    "Atrasaram minha entrega de novo, estou bastante frustrado.", # Frustração (1)
    "Não consigo conectar, isso é tão confuso.", # Confusão (2)
    "Responda-me o mais rápido possível, por favor!", # Urgência/Pressão (3)
    "Essa demora me deixa furioso.", # Raiva / Irritação (4)
    "Ok, entendi. Nada mais a dizer." # Neutralidade (5)
]

# Tokenizando os exemplos para inferência
inputs_6_sentimentos = tokenizer(
    exemplos_6_sentimentos,
    padding=True,
    truncation=True,
    return_tensors="pt"
).to(model.device)

# Realizando a inferência
outputs_6_sentimentos = model(**inputs_6_sentimentos)
preds_6_sentimentos = torch.argmax(outputs_6_sentimentos.logits, dim=1)

# Mapeando os IDs de volta para os rótulos de texto
rotulos_6_sentimentos = list(id2label_map.values())

print("\n--- Previsões para 6 sentimentos ---")
for i, (texto, pred) in enumerate(zip(exemplos_6_sentimentos, preds_6_sentimentos)):
    print(f"Exemplo {i+1}: {texto}\nSentimento previsto: {rotulos_6_sentimentos[pred]}\n")

# Salvando o modelo e o tokenizer ajustados
model.save_pretrained("sentiment_model_ptbr_finetuned_6_sentiments")
tokenizer.save_pretrained("sentiment_model_ptbr_finetuned_6_sentiments")
print("Modelo e tokenizer salvos em 'sentiment_model_ptbr_finetuned_6_sentiments'.")