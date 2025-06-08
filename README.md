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


 ### 3 Atualização da tabela cs_sentimentos

Para que os dados da análise de sentimentos seja armazenados corretamente, a tabela cs_sentimentos precisa de uma pequena aleração. Execute o script para garantir o funcionamento correto

 ```sql 
ALTER TABLE cs_sentimentos
ADD COLUMN confianca numeric(5,4),
ADD COLUMN prob_positivo numeric(5,4),
ADD COLUMN prob_negativo numeric(5,4),
ADD COLUMN prob_neutro numeric(5,4);
```