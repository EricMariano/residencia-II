import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def verificar_modelo():
    """Verifica se o modelo foi treinado e está funcionando"""
    
    modelo_path = "./sentiment_model_ptbr_finetuned"
    
    # 1. Verificar se os arquivos do modelo existem
    print("=== VERIFICAÇÃO DO MODELO ===\n")
    
    arquivos_necessarios = [
        "config.json",
        "model.safetensors",  # ou pytorch_model.bin
        "tokenizer_config.json",
        "vocab.txt"
    ]
    
    print(f"1. Verificando se o diretório '{modelo_path}' existe...")
    if not os.path.exists(modelo_path):
        print(f"❌ ERRO: Diretório '{modelo_path}' não encontrado!")
        print("   O modelo ainda não foi treinado ou salvo.")
        return False
    else:
        print(f"✅ Diretório encontrado!")
    
    print(f"\n2. Verificando arquivos do modelo...")
    arquivos_encontrados = os.listdir(modelo_path)
    print(f"   Arquivos encontrados: {arquivos_encontrados}")
    
    # 3. Tentar carregar o modelo
    print(f"\n3. Tentando carregar o modelo...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(modelo_path)
        model = AutoModelForSequenceClassification.from_pretrained(modelo_path)
        print("✅ Modelo carregado com sucesso!")
        
        # 4. Informações do modelo
        print(f"\n4. Informações do modelo:")
        print(f"   - Número de labels: {model.config.num_labels}")
        print(f"   - Labels: {model.config.id2label}")
        print(f"   - Tipo: {model.config.model_type}")
        
        # 5. Fazer um teste rápido
        print(f"\n5. Teste de predição...")
        textos_teste = [
            "Estou muito feliz!",
            "Que dia terrível!",
            "Normal, nada demais."
        ]
        
        for texto in textos_teste:
            inputs = tokenizer(texto, return_tensors="pt", truncation=True, padding=True)
            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)
                pred = torch.argmax(outputs.logits, dim=-1)
                label = model.config.id2label[pred.item()]
                confianca = probs[0][pred.item()].item()
            
            print(f"   Texto: '{texto}'")
            print(f"   → Sentimento: {label} (confiança: {confianca:.2%})\n")
        
        print("✅ MODELO FUNCIONANDO PERFEITAMENTE!")
        return True
        
    except Exception as e:
        print(f"❌ ERRO ao carregar modelo: {str(e)}")
        return False

if __name__ == "__main__":
    verificar_modelo()
