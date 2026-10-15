# ==================================================
# CONFIGURAÇÃO E TIPOS
# ==================================================
import os
import operator
from pathlib import Path
from typing import Annotated, Literal, TypedDict
from dotenv import load_dotenv

# ==================================================
# LEITURA DOS DADOS
# ==================================================
import pandas as pd

# ==================================================
# VALIDAÇÃO COM PYDANTIC
# ==================================================
from pydantic import BaseModel, Field, ValidationError

# ==================================================
# MODELO E PROMPTS
# ==================================================
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ==================================================
# DOCUMENTOS, EMBEDDINGS E BANCO VETORIAL
# ==================================================
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ==================================================
# LANGGRAPH
# ==================================================
from langgraph.graph import StateGraph, START, END

# Subir de src para a raiz do projeto
BASE_DIR = Path(__file__).resolve().parent.parent

CAMINHO_CSV = BASE_DIR / "dados" / "processed" / "dados_rag.csv"


# ==================================================
# 1. CARREGANDO DOCUMENTOS
# ==================================================

def cria_documentos():
    """Lendo o arquivo CSV e criando documentos dele."""

    dados = pd.read_csv(CAMINHO_CSV, keep_default_na=False)

    documentos = []

    for _, linha in dados.iterrows():
        texto = str(linha["review_comment_message"]).strip()

        if not texto:
            continue

        documento = Document(
            page_content=texto,
            metadata = {
                "fonte": CAMINHO_CSV.name,
                "titulo": str(linha["review_comment_title"]).strip(),
                "nota": int(linha["review_score"])
            }
        )

        documentos.append(documento)

    return documentos

# ==================================================
# 2. DIVIDIR DOCUMENTOS
# ==================================================

def dividir_documentos(documentos):
    """Recebe os documentos e divide eles em chunks."""

    divisor = RecursiveCharacterTextSplitter(
        chunk_size=650,
        chunk_overlap=150
    )

    trechos = divisor.split_documents(documentos)

    print(f"\nTotal de Documentos -> {len(documentos)}")
    print(f"\nTotal de Trechos -> {len(trechos)}")

    return trechos

# ==================================================
# 3. CRIAR EMBEDDINGS + BANCO VETORIAL
# ==================================================

def criar_banco(trechos):
    """Criando os embeddings e o banco vetorial."""

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    banco = FAISS.from_documents(
        documents=trechos,
        embedding = embeddings
    )

    return banco

# ==================================================
# 4. RECUPERAR DOCUMENTOS
# ==================================================

def recuperar_documentos(pergunta, banco):
    """Busca os trechos e organiza o contexto com suas fontes."""

    documentos = banco.similarity_search(
        pergunta,
        k=10
    )

    partes = []

    for numero, documento in enumerate(documentos, start=1):
        parte = (
            f"[F{numero}]\n"
            f"FONTE: {documento.metadata['fonte']}\n"
            f"TITULO: {documento.metadata['titulo']}\n"
            f"NOTA: {documento.metadata['nota']}\n"
            f"CONTEÚDO: {documento.page_content}"
        )

        partes.append(parte)

    contexto = "\n\n".join(partes)

    return contexto

# ==================================================
# 5. CRIANDO MODELO
# ==================================================

def criar_modelo():
    """Carregar a configuração e criar o modelo."""

    load_dotenv()

    if not os.getenv("GROQ_API_KEY"):
        raise ValueError("Configure GROQ_API_KEY no arquivo .env")

    modelo = ChatGroq(model="openai/gpt-oss-20b", temperature=0)

    return modelo

# ==================================================
# 6. CLASSE PYDANTIC
# ==================================================

class RespostaConsulta(BaseModel):
    """Organiza a análise das avaliações recuperadas."""

    assunto: str = Field(
        description="Assunto principal identificado no contexto. Retorne vazio se não identificado."
    )

    resposta: str = Field(
        description=(
            "Resposta objetiva à pergunta, baseada exclusivamente no contexto recuperado."
        )
    )

# ==================================================
# 7. CRIANDO CHAIN DA RESPOSTA
# ==================================================

def chain_resposta(modelo):
    """Conecta o prompt ao modelo com saída estruturada."""

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
            Você auxilia uma equipe de e-commerce a compreender
            os relatos dos clientes nas avaliações de compras.

            Responda à pergunta usando somente as avaliações
            presentes no contexto recuperado.

            USO DO CONTEXTO

            - Utilize o contexto para responder a pergunta.
            - Não invente nada, pegue somente informações que foram coletadas pelo contexto.
            - Utilize sempre os metadados se estiverem disponíveis, nada de inventar.

            ESTRUTURA DA SAÍDA

            - Assunto: informe o assunto que a pergunta se refere, utilize do contexto para 
            extrair essa informação.
            - Resposta: apresente uma síntese clara, objetiva e diretamente relacionada à pergunta. 
            Dê sempre uma resposta direta ao assunto, procure não ficar repetindo a mesma palavra várias vezes. 
            Otimize ao máximo a sua resposta, para não deixar dúvidas a pessoa que perguntou. 
            Caso não ter contexto suficiente, fale que não foi capaz de encontrar aquele assunto nos dados.
            """
        ),
        (
            "human",
            """
            Pergunta:
            {pergunta}

            Contexto recuperado:
            {contexto}
            """
        ),
    ])

    modelo_estruturado = modelo.with_structured_output(
        RespostaConsulta
    )

    chain = prompt | modelo_estruturado

    return chain

pergunta = input("[PERGUNTA]: ").strip()

documentos = cria_documentos()
trechos = dividir_documentos(documentos)
banco = criar_banco(trechos)

contexto = recuperar_documentos(pergunta, banco)

modelo = criar_modelo()
gerar_resposta = chain_resposta(modelo)

resposta = gerar_resposta.invoke({
    "pergunta": pergunta,
    "contexto": contexto
})

print("\n=== ASSUNTO ===")
print(resposta.assunto)

print("\n=== RESPOSTA ===")
print(resposta.resposta)