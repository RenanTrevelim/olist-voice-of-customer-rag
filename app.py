# ==================================================
# 1. IMPORTAÇÕES
# ==================================================

import streamlit as st

from langchain_core.messages import HumanMessage, AIMessage
from src.main import criar_grafo


# ==================================================
# CONFIGURAÇÃO DA PÁGINA
# ==================================================

st.set_page_config(
    page_title="Olist | Voz do Cliente",
    page_icon="💬",
    layout="centered"
)

# ==================================================
# APRESENTAÇÃO DO PROJETO
# ==================================================

st.markdown(
    """
    <style>
        .voc-cabecalho {
            padding: 30px 28px;
            margin-bottom: 22px;
            border: 1px solid #2c4264;
            border-radius: 18px;
            background: linear-gradient(135deg, #14243d, #101923);
        }

        .voc-etiqueta {
            color: #93c5fd;
            font-size: 12px;
            font-weight: 700;
            letter-spacing: 2px;
            margin-bottom: 12px;
        }

        .voc-titulo {
            color: #f8fafc;
            font-size: clamp(28px, 5vw, 40px);
            font-weight: 750;
            line-height: 1.2;
            margin: 0 0 14px;
        }

        .voc-descricao {
            color: #cbd5e1;
            font-size: 16px;
            line-height: 1.6;
            margin: 0;
        }

        .voc-temas {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-top: 22px;
        }

        .voc-tema {
            color: #bfdbfe;
            background: #1c304c;
            border: 1px solid #304c70;
            border-radius: 20px;
            padding: 5px 12px;
            font-size: 12px;
        }
    </style>

    <section class="voc-cabecalho">
        <div class="voc-etiqueta">OLIST · VOICE OF CUSTOMER</div>
        <h1 class="voc-titulo">Entenda a voz dos seus clientes.</h1>
        <p class="voc-descricao">
            Explore elogios, reclamações e experiências de compra.
            Faça perguntas e consulte os relatos que fundamentam
            cada resposta.
        </p>
        <div class="voc-temas">
            <span class="voc-tema">Entrega</span>
            <span class="voc-tema">Qualidade dos produtos</span>
            <span class="voc-tema">Experiência de compra</span>
        </div>
    </section>
    """,
    unsafe_allow_html=True
)

# Mostrar as sugestões apenas antes de começar a conversa
if not st.session_state.get("mensagens"):
    st.markdown("**Por onde começar?**")

    st.markdown(
        """
        - O que os clientes relatam sobre a entrega?
        - Quais comentários mencionam produtos com defeito?
        - O que os clientes elogiam na experiência de compra?
        """
    )

    st.caption(
        "Você também pode continuar o assunto com perguntas como "
        "“E sobre pedidos incompletos?”."
    )


# ==================================================
# 3. CARREGAR O GRAFO
# ==================================================

@st.cache_resource
def carregar_chatbot():
    """Prepara e reutiliza o grafo entre as interações."""
    return criar_grafo()


with st.spinner("Preparando o assistente..."):
    chatbot = carregar_chatbot()


# ==================================================
# 4. HISTÓRICO DA SESSÃO
# ==================================================

if "mensagens" not in st.session_state:
    st.session_state.mensagens = []

with st.sidebar:
    st.header("Conversa")

    if st.button("Nova conversa"):
        st.session_state.mensagens = []
        st.rerun()

    st.caption(
        "As respostas usam os comentários recuperados. "
        "Elas não representam uma análise estatística de toda a base."
    )


# ==================================================
# 5. EXIBIR UMA MENSAGEM
# ==================================================

def exibir_mensagem(mensagem):
    """Mostra a mensagem e os detalhes disponíveis."""

    with st.chat_message(mensagem["papel"]):
        if mensagem.get("assunto"):
            st.caption(mensagem["assunto"])

        st.markdown(mensagem["conteudo"])

        if mensagem.get("contexto"):
            with st.expander("Avaliações recuperadas"):
                st.text(mensagem["contexto"])

        if mensagem.get("trilha"):
            with st.expander("Caminho percorrido"):
                for etapa in mensagem["trilha"]:
                    st.write(f"- {etapa}")


for mensagem in st.session_state.mensagens:
    exibir_mensagem(mensagem)


# ==================================================
# 6. RECEBER A PERGUNTA
# ==================================================

pergunta = st.chat_input("O que você quer saber sobre as avaliações?")

if pergunta and pergunta.strip():
    pergunta = pergunta.strip()

    mensagem_usuario = {
        "papel": "user",
        "conteudo": pergunta
    }

    exibir_mensagem(mensagem_usuario)

    # Converter as mensagens anteriores para o formato do LangChain
    historico = []

    for mensagem in st.session_state.mensagens:
        if mensagem["papel"] == "user":
            historico.append(
                HumanMessage(content=mensagem["conteudo"])
            )
        else:
            historico.append(
                AIMessage(content=mensagem["conteudo"])
            )

    entrada = {
        "pergunta": pergunta,
        "historico": historico,
        "pergunta_interpretada": "",
        "consulta_busca": "",
        "contexto": "",
        "evidencias_suficientes": False,
        "motivo_avaliacao": "",
        "tentativas": 0,
        "resposta": None,
        "trilha": []
    }

    # Executar o fluxo do LangGraph
    try:
        with st.spinner("Consultando as avaliações..."):
            resultado = chatbot.invoke(entrada)

        resposta = resultado["resposta"]

        if resposta is None:
            st.error("Não foi possível gerar uma resposta. Tente novamente.")

        else:
            mensagem_assistente = {
                "papel": "assistant",
                "assunto": resposta.assunto,
                "conteudo": resposta.resposta,
                "contexto": resultado["contexto"],
                "trilha": resultado["trilha"]
            }

            # Salvar somente a interação concluída
            st.session_state.mensagens.append(mensagem_usuario)
            st.session_state.mensagens.append(mensagem_assistente)

            exibir_mensagem(mensagem_assistente)

    except Exception as erro:
        st.error("Não foi possível concluir a consulta.")
        st.exception(erro)