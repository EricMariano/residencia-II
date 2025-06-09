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
        self.id2label = {
            0: 'Satisfação', 1: 'Frustração', 2: 'Confusão',
            3: 'Urgência/Pressão', 4: 'Raiva / Irritação', 5: 'Neutralidade'
        }        
    def analisar(self, texto):
        """
        Analisa o sentimento de um texto com base em 6 rótulos definidos:
        'Satisfação', 'Frustração', 'Confusão', 'Urgência/Pressão', 
        'Raiva / Irritação' e 'Neutralidade'
        """
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

        # Gerar todas as probabilidades com base nos rótulos
        probabilidades = {
            self.id2label[i]: round(probs[0][i].item(), 4)
            for i in range(len(self.id2label))
        }

        return {
            "texto": texto,
            "sentimento": self.id2label[label_id],
            "confianca": round(confianca, 4),
            "probabilidades": probabilidades,
            "sucesso": True
        }




    def analisar_multiplos(self, textos):
        """
        Analisa múltiplos textos de uma vez (batch processing)
        com base nos rótulos definidos em id2label.
        """
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
            probabilidades = {
                self.id2label[j]: round(prob[j].item(), 4)
                for j in range(len(self.id2label))
            }
            resultados.append({
                "texto": texto,
                "sentimento": self.id2label[label_id],
                "confianca": round(prob[label_id].item(), 4),
                "probabilidades": probabilidades
            })

        return {
            "resultados": resultados,
            "total_analisado": len(resultados),
            "sucesso": True
        }

