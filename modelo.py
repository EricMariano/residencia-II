import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class AnalisadorSentimentos:
    def __init__(self, caminho_modelo="./sentiment_model_ptbr_finetuned"):
        """
        Carrega o modelo treinado de análise de sentimentos em português
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Carregar tokenizer e modelo
        self.tokenizer = AutoTokenizer.from_pretrained(caminho_modelo)
        self.model = AutoModelForSequenceClassification.from_pretrained(caminho_modelo)
        self.model.to(self.device)
        self.model.eval()
        
        # Mapeamento de labels
        self.id2label = {0: 'positivo', 1: 'negativo', 2: 'neutro'}
        
    def analisar(self, texto):
        """
        Analisa o sentimento de um texto

        
        """
        try:
            # Tokenizar
            inputs = self.tokenizer(
                texto,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt"
            ).to(self.device)
            
            # Fazer predição
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)
                pred = torch.argmax(logits, dim=-1)
            
            # Obter label e confiança
            label_id = pred.item()
            confianca = probs[0][label_id].item()
            
            return {
                "texto": texto,
                "sentimento": self.id2label[label_id],
                "confianca": round(confianca, 4),
                "probabilidades": {
                    "positivo": round(probs[0][0].item(), 4),
                    "negativo": round(probs[0][1].item(), 4),
                    "neutro": round(probs[0][2].item(), 4)
                },
                "sucesso": True
            }
        except Exception as e:
            return {
                "erro": str(e),
                "sucesso": False
            }

    def analisar_multiplos(self, textos):
        """
        Analisa múltiplos textos de uma vez (batch processing)
        """
        try:
            # Tokenizar todos os textos
            inputs = self.tokenizer(
                textos,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt"
            ).to(self.device)
            
            # Fazer predições
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)
                preds = torch.argmax(logits, dim=-1)
            
            # Processar resultados
            resultados = []
            for i, (texto, pred, prob) in enumerate(zip(textos, preds, probs)):
                label_id = pred.item()
                resultados.append({
                    "texto": texto,
                    "sentimento": self.id2label[label_id],
                    "confianca": round(prob[label_id].item(), 4),
                    "probabilidades": {
                        "positivo": round(prob[0].item(), 4),
                        "negativo": round(prob[1].item(), 4),
                        "neutro": round(prob[2].item(), 4)
                    }
                })
            
            return {
                "resultados": resultados,
                "total_analisado": len(resultados),
                "sucesso": True
            }
        except Exception as e:
            return {
                "erro": str(e),
                "sucesso": False
            }