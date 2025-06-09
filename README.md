
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
# Repositório da API – Squad 7 (Residência de Software II)

Lembrem do `.gitignore` na venv de vocês, por favor.  
Estamos usando **FastAPI** e **PostgreSQL 16**.

---

## Comunicação Assíncrona com PostgreSQL (LISTEN / NOTIFY)

Para que a API funcione corretamente, será necessário usar o mecanismo integrado do PostgreSQL: `LISTEN` e `NOTIFY`.  
Isso permite a comunicação assíncrona entre processos conectados ao mesmo banco de dados.

### O que fazer?

Precisamos adicionar ao banco de dados um **trigger** que envie notificações para o canal `'nova_acao_channel'`.  
A API poderá então escutar esse canal e reagir em tempo real a novas ações inseridas.

### 1 Criar a Função de Trigger

```sql
CREATE OR REPLACE FUNCTION notificar_nova_acao_insert()
RETURNS TRIGGER AS $$
BEGIN

PERFORM pg_notify('nova_acao_channel', NEW.acao_id::text);

RETURN NEW;
END;
$$
LANGUAGE plpgsql;
```

-- Envia o acao_id da nova linha como payload da notificação para o canal 'nova_acao_channel'.
-- O payload precisa ser uma string, então convertemos o ID para texto.
-- O valor de retorno é ignorado para triggers AFTER, mas é boa prática retornar NEW.

### 2 Criar o Trigger na Tabela cs_acoes

-- Este trigger acionará a função notificar_nova_acao_insert() após cada operação de INSERT.
-- Crie o trigger na tabela cs_acoes do banco de dados usando o Query Tool

```sql
CREATE TRIGGER trigger_cs_acoes_insert
AFTER INSERT ON cs_acoes
FOR EACH ROW
EXECUTE FUNCTION notificar_nova_acao_insert();

$$
```
 ### Treinamento e Verificação do Modelo
 
 Para garantir que a API vai funcionar, o modelo treinado deve estar disponível no diretório ./sentiment_model_ptbr_finetuned.

Treine o modelo, rodando o script treinarnodelo.py e após isso verifique se o mesmo foi gerado corretamente


