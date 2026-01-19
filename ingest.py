import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
# Troca aqui: Sai OpenAI, entra Google
from langchain_google_genai import GoogleGenerativeAIEmbeddings 
# Nota: Removemos o SupabaseVectorStore pois ele força UUIDs. Vamos inserir manualmente.
from supabase.client import create_client

load_dotenv()

# Configuração
supabase = create_client(os.environ.get("SUPABASE_URL"), os.environ.get("SUPABASE_KEY"))
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

def ingest_data(file_path, doc_type):
    if not os.path.exists(file_path):
        print(f"Erro: Arquivo {file_path} não encontrado.")
        return

    print(f"🔄 Processando {doc_type}...")

    # 1. Carrega o arquivo
    loader = TextLoader(file_path, encoding='utf-8')
    documents = loader.load()

    # 2. Picota em pedaços menores (Chunks)
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)

    # 3. Gera Embeddings e Salva Manualmente (Para funcionar com BigInt)
    for i, doc in enumerate(docs):
        try:
            # Gera o vetor (768 dimensões)
            vector = embeddings.embed_query(doc.page_content)
            
            # Monta o objeto para o banco
            # NÃO enviamos o 'id', assim o banco gera o BigInt sozinho
            data = {
                "content": doc.page_content,
                "metadata": {"category": doc_type},
                "embedding": vector
            }
            
            # Insere direto na tabela
            supabase.table("documents").insert(data).execute()
            print(f"   ✅ Salvo trecho {i+1}/{len(docs)}")
            
        except Exception as e:
            print(f"   ❌ Erro ao salvar trecho {i+1}: {e}")

    print(f"Sucesso! Todos os pedaços de '{doc_type}' foram processados.")

# Rodar a ingestão
if __name__ == "__main__":
    # Certifique-se que a pasta 'data' existe e contém os arquivos
    ingest_data("data/planos_comerciais.txt", "general")
    ingest_data("data/manual_tecnico.txt", "technical")