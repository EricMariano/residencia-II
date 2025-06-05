from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import uvicorn
from datetime import datetime
import os

from modelo import AnalisadorSentimentos

# Criar instância da API
app = FastAPI(
    title="API de Análise de Sentimentos em Português",
    description="API para análise de sentimentos",
    version="1.0.0"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Verificar se o modelo existe
MODELO_PATH = "./sentiment_model_ptbr_finetuned"
if not os.path.exists(MODELO_PATH):
    raise Exception(f"Modelo não encontrado em {MODELO_PATH}. Execute o treinamento primeiro!")

# Inicializar o modelo
print("Carregando modelo...")
analisador = AnalisadorSentimentos(MODELO_PATH)
print("Modelo carregado com sucesso!")

# Modelos Pydantic
class TextoSimples(BaseModel):
    texto: str = Field(..., min_length=1, max_length=1000, description="Texto para análise")

class TextosMultiplos(BaseModel):
    textos: List[str] = Field(..., min_items=1, max_items=100, description="Lista de textos")

class RespostaSentimento(BaseModel):
    texto: str
    sentimento: str
    confianca: float
    probabilidades: dict
    timestamp: datetime = Field(default_factory=datetime.now)

# Estatísticas simples (em produção, use um banco de dados)
stats = {
    "total_requisicoes": 0,
    "sentimentos_analisados": {"positivo": 0, "negativo": 0, "neutro": 0}
}

# Rotas
@app.get("/")
def home():
    return {
        "mensagem": "API de Análise de Sentimentos em Português",
        "modelo": "BERT Portuguese fine-tuned",
        "endpoints": {
            "/analisar": "POST - Analisa um texto",
            "/analisar-multiplos": "POST - Analisa múltiplos textos",
            "/estatisticas": "GET - Estatísticas de uso",
            "/docs": "Documentação interativa",
            "/health": "Status da API"
        }
    }

@app.post("/analisar", response_model=RespostaSentimento)
async def analisar_sentimento(dados: TextoSimples, background_tasks: BackgroundTasks):
    """
    
    Retorna:
    - sentimento: positivo, negativo ou neutro
    - confianca: nível de confiança da predição (0-1)
    - probabilidades: probabilidade de cada classe
    """
    
    resultado = analisador.analisar(dados.texto)
    
    if not resultado["sucesso"]:
        raise HTTPException(status_code=500, detail=resultado["erro"])
    
    # Atualizar estatísticas em background
    background_tasks.add_task(
        atualizar_stats, 
        resultado["sentimento"]
    )
    
    return RespostaSentimento(
        texto=resultado["texto"],
        sentimento=resultado["sentimento"],
        confianca=resultado["confianca"],
        probabilidades=resultado["probabilidades"]
    )

@app.post("/analisar-multiplos")
async def analisar_multiplos_sentimentos(dados: TextosMultiplos, background_tasks: BackgroundTasks):
    """
    Analisa o sentimento de múltiplos textos em português.
    Máximo de 100 textos por requisição.
    """
    # Filtrar textos vazios
    textos_validos = [t.strip() for t in dados.textos if t.strip()]
    
    if not textos_validos:
        raise HTTPException(status_code=400, detail="Todos os textos estão vazios")
    
    resultado = analisador.analisar_multiplos(textos_validos)
    
    if not resultado["sucesso"]:
        raise HTTPException(status_code=500, detail=resultado["erro"])
    
    # Atualizar estatísticas
    for res in resultado["resultados"]:
        background_tasks.add_task(atualizar_stats, res["sentimento"])
    
    return resultado

@app.get("/estatisticas")
def obter_estatisticas():
    """
    Retorna estatísticas de uso da API
    """
    return {
        "total_requisicoes": stats["total_requisicoes"],
        "sentimentos_analisados": stats["sentimentos_analisados"],
        "modelo_info": {
            "nome": "BERT Portuguese fine-tuned",
            "classes": ["positivo", "negativo", "neutro"]
        }
    }

@app.get("/health")
def health_check():
    """
    Verifica o status da API e do modelo
    """
    # Testar o modelo
    teste = analisador.analisar("teste")
    modelo_ok = teste["sucesso"]
    
    return {
        "status": "ok" if modelo_ok else "erro",
        "modelo_carregado": modelo_ok,
        "timestamp": datetime.now()
    }

def atualizar_stats(sentimento: str):
    """Função auxiliar para atualizar estatísticas"""
    stats["total_requisicoes"] += 1
    if sentimento in stats["sentimentos_analisados"]:
        stats["sentimentos_analisados"][sentimento] += 1

if __name__ == "__main__":
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        reload=False  # Em produção, sempre False
    )