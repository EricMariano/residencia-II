# 🚀 Residência de Software II - Squad 7

Repositório da API de Análise de Sentimentos em Português com integração ao PostgreSQL.

> ⚠️ **Importante**: Lembrem de adicionar a `venv` no `.gitignore`!

---

## 📋 Visão Geral

Esta API utiliza:
- **FastAPI** - Framework web moderno e rápido
- **PostgreSQL 16** - Banco de dados com suporte a notificações assíncronas
- **BERT** - Modelo de análise de sentimentos fine-tuned para português

---

## ⚙️ Requisitos

- Python 3.8+
- PostgreSQL 16
- `virtualenv` (recomendado)

---

## 📦 Instalação

### 1. Clone o repositório

```bash
git clone https://github.com/seu-usuario/residencia-II.git
cd residencia-II

# Criar ambiente virtual
python -m venv venv

# Ativar no Windows
venv\Scripts\activate

# Ativar no Linux/Mac
source venv/bin/activate
```
### 2. Instalando depedências
```bash
pip install -r requirements.txt
```
### 3. Treinando modelo
```bash
python treinarnodelo.py
# (Verifique se o diretório sentiment_model_ptbr_finetuned foi criado corretamente.)
```

🗄️ Configuração do PostgreSQL
Comunicação Assíncrona (LISTEN/NOTIFY)
A API utiliza o mecanismo de notificações do PostgreSQL para reagir a eventos em tempo real.

1. Criar a função de trigger
Execute no Query Tool do PostgreSQL:
```sql
Copy SQL
CREATE OR REPLACE FUNCTION notificar_nova_acao_insert()Add commentMore actions
RETURNS TRIGGER AS $$
BEGIN
    -- Envia o acao_id como payload para o canal 'nova_acao_channel'
    PERFORM pg_notify('nova_acao_channel', NEW.acao_id::text);
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;
2. Criar o trigger na tabela
sql
Copy SQL
CREATE TRIGGER trigger_cs_acoes_insert
AFTER INSERT ON cs_acoes
FOR EACH ROW
EXECUTE FUNCTION notificar_nova_acao_insert();
```
▶️ Executando a API
bash
Copy Code
uvicorn main:app --reload
A API estará disponível em: http://127.0.0.1:8000

d commentMore actions
## 📚 Endpoints

| Método | Rota                  | Descrição                                      |
|--------|-----------------------|-----------------------------------------------|
| GET    | `/`                   | Informações sobre a API                        |
| GET    | `/docs`               | Interface interativa da API (Swagger UI)      |
| GET    | `/health`             | Verifica o status da API                      |
| GET    | `/estatisticas`       | Retorna estatísticas de uso da API            |
| POST   | `/analisar`           | Analisa o sentimento de um único texto        |
| POST   | `/analisar-multiplos` | Analisa o sentimento de vários textos         |

``` bash
POST /analisar
Content-Type: application/json

{
  "texto": "Esse produto é maravilhoso!"
}
```

# 🧠 Sobre o Modelo
Arquitetura: BERT (Bidirectional Encoder Representations from Transformers)
Fine-tuning: Adaptado para português brasileiro
Classes de sentimento:
- ✅ Positivos: Satisfação;
- ❌ Negativos: Raiva/Irritação, Frustração, Urgência/Pressão, Confusão;
- ➖ Neutro: Neutralidade.

📝 Notas Importantes
- Certifique-se de que o PostgreSQL está rodando antes de iniciar a API
O modelo deve estar treinado e salvo em ./sentiment_model_ptbr_finetuned
Configure as variáveis de ambiente para conexão com o banco de dados
Use .gitignore para excluir arquivos sensíveis e a pasta venv
