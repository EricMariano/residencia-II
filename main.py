import asyncio  # progamação assíncrona
import asyncpg
from fastapi import FastAPI, HTTPException, status
from modelo import AnalisadorSentimentos


# --- Configuração do Banco
DB_CONFIG = {
    "database": "sq07_db",
    "user": "postgres",
    "password": "45987956aA@",
    "host": "localhost",
    "port": "5433"
}

# canal postegres que o listener irá escutar
LISTEN_CHANNEL = "nova_acao_channel"

app = FastAPI(
    title="API de Análise de Sentimentos em Português",
    description="API para análise de sentimentos",
    version="1.0.0"
)

# Pool de conexões para reutilização
db_pool = None  # para evitar novas conexões a cada requisição


async def init_db_pool():
    """Inicializa o pool de conexões asyncpg."""
    global db_pool
    try:
        # criação do pool, await pausada até que o pool seja criado, mínimo de 1 conexão e máximo de 10
        db_pool = await asyncpg.create_pool(**DB_CONFIG, min_size=1, max_size=10)
        print("Pool de conexões com o banco de dados inicializado.")
    except Exception as e:
        print(f"Erro crítico ao inicializar pool de conexões: {e}")
        # Em um cenário real, a aplicação pode precisar parar aqui
        raise


async def close_db_pool():
    """Fecha o pool de conexões."""
    global db_pool
    if db_pool:
        await db_pool.close()  # fecha todas as conexões
        print("Pool de conexões fechado.")


async def fetch_action_details(pool, acao_id: int):
    """Busca descricao e agent_id da ação com base no ID."""
    async with pool.acquire() as conn:  # adquire uma conexão do pool, ela é libreada de volta ao fim do bloco with
        try:
            query = "SELECT acao_id, descricao, agent_id FROM cs_acoes WHERE acao_id = $1"
            # consulta sql de forma assíncrona. fetchrow busca no máximo uma linha
            result = await conn.fetchrow(query, acao_id)
            if result:
                print(
                    f"Detalhes da Ação Recuperados: ID={result['acao_id']}, AgentID={result['agent_id']}")
                return dict(result)
            else:
                print(f"Ação com ID {acao_id} não encontrada.")
                return None
        except Exception as e:
            print(f"Erro ao buscar detalhes da ação {acao_id}: {e}")
            return None


async def salvar_sentimento_async(pool, acao_id: int, sentimento: str, confianca: float, prob_positivo: float, prob_negativo: float, prob_neutro: float):
    """Salva o sentimento no banco de dados de forma assíncrona."""
    async with pool.acquire() as conn:
        async with conn.transaction():
            try:
                exists = await conn.fetchval("SELECT 1 FROM cs_sentimentos WHERE acao_id = $1", acao_id)
                if exists:
                    print(f"Sentimento já existe para acao_id: {acao_id}. Ignorando.")
                    return False

                await conn.execute(
                    "INSERT INTO cs_sentimentos (acao_id, sentimento, confianca, prob_positivo, prob_negativo, prob_neutro) VALUES ($1, $2, $3, $4, $5, $6);",
                    acao_id, sentimento, confianca, prob_positivo, prob_negativo, prob_neutro
                )
                print(f"Sentimento salvo para acao_id: {acao_id}")
                return True
            except asyncpg.UniqueViolationError:
                print(f"Sentimento já existe para acao_id: {acao_id} (detectado por constraint). Ignorando.")
                return False
            except Exception as e:
                print(f"Erro de banco ao salvar sentimento detalhado para acao_id {acao_id}: {e}")
                return False

# --- Lógica de Processamento da Notificação ---


async def processar_nova_acao(pool, acao_id: int):
    """Orquestra a busca de detalhes, análise e salvamento do sentimento."""
    print(f"Processando acao_id: {acao_id}")
    # busca os detalhes da acao no banco
    action_details = await fetch_action_details(pool, acao_id)

    if action_details and action_details.get('descricao'):
        descricao = action_details['descricao']
        # agent_id = action_details.get('agent_id') # agent_id recuperado, usar se necessário

        sentimento_result = analisador_sentimentos.analisar(descricao)

        if sentimento_result and sentimento_result["sucesso"]:
            sentimento = sentimento_result["sentimento"]
            confianca = sentimento_result["confianca"]
            prob_positivo = sentimento_result["probabilidades"]["positivo"]
            prob_negativo = sentimento_result["probabilidades"]["negativo"]
            prob_neutro = sentimento_result["probabilidades"]["neutro"]
            await salvar_sentimento_async(pool, acao_id, sentimento, confianca, prob_positivo, prob_negativo, prob_neutro)
        else:
            print(f"Análise de sentimento falhou ou vazia para acao_id: {acao_id}. Erro: {sentimento_result.get('erro', 'Desconhecido')}")
    else:
        print(
            f"Não foi possível obter detalhes ou descrição para acao_id: {acao_id}")

# --- Listener de Notificações do PostgreSQL ---


async def notification_handler(connection, pid, channel, payload):
    """Callback chamado quando uma notificação é recebida."""
    print(f"Notificação recebida no canal '{channel}': Payload='{payload}'")
    try:
        acao_id = int(payload)
        # Chama o processamento em background (não bloqueia o listener)
        # Usa o pool global que já deve estar inicializado
        if db_pool:
            asyncio.create_task(processar_nova_acao(db_pool, acao_id))
        else:
            print("Erro: Pool de conexões não inicializado para processar a notificação.")
    except ValueError:
        print(f"Erro: Payload recebido não é um ID de ação válido: {payload}")
    except Exception as e:
        print(f"Erro inesperado no handler da notificação: {e}")


async def listen_for_actions():
    """Conecta ao DB, escuta o canal e processa notificações indefinidamente."""
    conn = None
    while True:  # Loop para tentar reconectar em caso de falha
        try:
            # Usa uma conexão dedicada para o LISTEN, não do pool principal
            conn = await asyncpg.connect(**DB_CONFIG)
            print(
                f"Conexão de escuta estabelecida. Escutando no canal '{LISTEN_CHANNEL}'...")
            await conn.add_listener(LISTEN_CHANNEL, notification_handler)

            # Mantém a conexão de escuta ativa
            while True:
                # Espera indefinidamente por notificações. O add_listener/handler cuidam do resto.
                # O sleep evita um loop apertado caso algo dê errado na espera interna.
                await asyncio.sleep(3600)

        except (asyncpg.exceptions.PostgresConnectionError, ConnectionRefusedError, OSError) as e:
            print(
                f"Erro na conexão de escuta: {e}. Tentando reconectar em 10 segundos...")
            if conn and not conn.is_closed():
                try:
                    # Tenta remover o listener antes de fechar
                    await conn.remove_listener(LISTEN_CHANNEL, notification_handler)
                except Exception as rem_e:
                    print(f"Erro ao remover listener na reconexão: {rem_e}")
                await conn.close()
            conn = None  # Garante que tentaremos uma nova conexão
            await asyncio.sleep(10)
        except Exception as e:
            print(
                f"Erro inesperado no loop de escuta: {e}. Tentando reconectar em 10 segundos...")
            if conn and not conn.is_closed():
                try:
                    await conn.remove_listener(LISTEN_CHANNEL, notification_handler)
                except Exception as rem_e:
                    print(
                        f"Erro ao remover listener no erro inesperado: {rem_e}")
                await conn.close()
            conn = None
            await asyncio.sleep(10)

# --- Eventos Startup/Shutdown da Aplicação FastAPI ---

analisador_sentimentos = None

@app.on_event("startup")
async def startup_event():
    global analisador_sentimentos
    analisador_sentimentos = AnalisadorSentimentos()
    await init_db_pool()
    asyncio.create_task(listen_for_actions())
    print("Listener de ações iniciado em background.")


@app.on_event("shutdown")
async def shutdown_event():
    await close_db_pool()
    print("Aplicação encerrada.")


# Indica o funcionamento da API

@app.get("/")
def read_root():
    return {"Status": "API está rodando"}


# Lógica iniciada pelo LISTENER

# Para rodar: uvicorn main:app --reload
