
# 📊 API de Análise de Sentimentos em Português

Esta é uma API construída com **FastAPI** para analisar sentimentos de textos em português. O modelo utilizado é um BERT fine-tuned para classificação de sentimentos (`positivo`, `negativo`, `neutro`).

---

## ⚙️ Requisitos

- Python 3.8+
- `virtualenv` (opcional, mas recomendado)

---

## 📦 Instalação

1. **Clone o repositório**

```bash
git clone https://github.com/seu-usuario/residencia-II.git
cd residencia-II
```

2. **Crie e ative o ambiente virtual**

```bash
python -m venv venv
# Ative no Windows
venv\Scripts\activate
# Ou no Linux/Mac
source venv/bin/activate
```

3. **Instale as dependências**

```bash
pip install -r requirements.txt
```

4. **Verifique se o diretório `sentiment_model_ptbr_finetuned` existe**

Este diretório deve conter o modelo treinado. Caso não exista, treine ou baixe o modelo primeiro.

---

## ▶️ Executando a API

Execute o seguinte comando para iniciar o servidor:

```bash
uvicorn main:app
```

A API estará disponível em:

```
http://127.0.0.1:8000
```

---

## 📚 Endpoints

| Método | Rota                  | Descrição                                      |
|--------|-----------------------|-----------------------------------------------|
| GET    | `/`                   | Informações sobre a API                        |
| GET    | `/docs`               | Interface interativa da API (Swagger UI)      |
| GET    | `/health`             | Verifica o status da API                      |
| GET    | `/estatisticas`       | Retorna estatísticas de uso da API            |
| POST   | `/analisar`           | Analisa o sentimento de um único texto        |
| POST   | `/analisar-multiplos` | Analisa o sentimento de vários textos         |

---

## 📤 Exemplo de uso

### Requisição (POST `/analisar`)
```json
{
  "texto": "Esse produto é maravilhoso!"
}
```

### Resposta
```json
{
  "texto": "Esse produto é maravilhoso!",
  "sentimento": "positivo",
  "confianca": 0.94,
  "probabilidades": {
    "positivo": 0.94,
    "negativo": 0.03,
    "neutro": 0.03
  },
  "timestamp": "2025-06-08T12:34:56.789"
}
```

---

## 🧠 Modelo

- Arquitetura: BERT
- Dataset: Adaptado para português (ex: `tweets`, `reviews`)
- Classes: `positivo`, `negativo`, `neutro`
