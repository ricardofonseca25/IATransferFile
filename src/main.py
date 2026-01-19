import os
from datetime import datetime
from dotenv import load_dotenv

# UI e Terminal
from rich.console import Console
from rich.prompt import Prompt
from rich.panel import Panel

# LangChain & Gemini
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from supabase.client import create_client

# LangGraph
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, MessagesState, START, END

load_dotenv()

class GraphApp:
    # CORREÇÃO 1: Usando o modelo correto (1.5)
    def __init__(self, model: str = "gemini-2.5-flash"):
        self.llm = ChatGoogleGenerativeAI(model=model, temperature=0)
        
        self.supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        
        self.thread = {"configurable": {"thread_id": "transferfile_chat"}}
        self.graph = self.build_graph()

    def get_context(self, query: str, category: str):
        """Busca no Supabase via RPC."""
        try:
            query_vector = self.embeddings.embed_query(query)
            params = {
                "query_embedding": query_vector,
                "match_threshold": 0.5,
                "match_count": 3,
                "filter": {"category": category}
            }
            response = self.supabase.rpc("match_documents", params).execute()
            docs = response.data 
            
            if not docs:
                return ""
            return "\n\n".join([d['content'] for d in docs])
        except Exception:
            return "" # Falha silenciosa para não quebrar o fluxo

    def make_specialist_node(self, persona: str, category: str, node_name: str):
        """Cria nó especialista com RAG."""
        def node(state: MessagesState) -> MessagesState:
            print(f"   [DEBUG] 🤖 Agente Ativado: {node_name}") # Debug visual
            last_message = state["messages"][-1].content
            
            context_text = self.get_context(last_message, category)
            
            if not context_text:
                context_instruction = "Não encontrei informações no banco de dados sobre isso. Peça desculpas."
            else:
                context_instruction = f"Use este contexto técnico/comercial:\n{context_text}"

            prompt = ChatPromptTemplate.from_messages([
                ("system", "{system_persona}\n\n{instruction}"),
                ("human", "{messages}")
            ])
            chain = prompt | self.llm
            response = chain.invoke({
                "messages": state["messages"],
                "system_persona": persona,
                "instruction": context_instruction
            })
            return {"messages": response}
        return node

    def build_graph(self):
        # NÓ 1: Geral
        general_agent = self.make_specialist_node(
            "Você é o Comercial do TransferFile. Fale de planos e preços.", 
            "general", "Comercial"
        )
        
        # NÓ 2: Técnico
        tech_agent = self.make_specialist_node(
            "Você é o Engenheiro do TransferFile. Fale de Stack, React, API e Código.", 
            "technical", "Técnico"
        )
        
        # NÓ 3: Revisor
        def reviewer_node(state: MessagesState) -> MessagesState:
            prompt = ChatPromptTemplate.from_messages([
                ("system", "Revisor: Melhore a formatação da resposta anterior. Seja conciso."),
                ("human", "{messages}")
            ])
            chain = prompt | self.llm
            response = chain.invoke({"messages": state["messages"]})
            return {"messages": response}

        # --- ROTEADOR INTELIGENTE (USA IA) ---
        def router_logic(state: MessagesState):
            last_msg = state["messages"][-1].content
            
            # Pergunta para o Gemini qual é a intenção
            router_prompt = ChatPromptTemplate.from_messages([
                ("system", """
                 Você é um roteador de classificação.
                 Se a pergunta for sobre PREÇO, PLANOS, VALORES ou GERAL do produto -> Responda 'GENERAL'.
                 Se a pergunta for sobre CÓDIGO, API, REACT, FLUXO TÉCNICO, INSTALAÇÃO, ERROS -> Responda 'TECHNICAL'.
                 Responda APENAS uma palavra.
                 """),
                ("human", "{question}")
            ])
            
            # Cadeia de decisão rápida
            chain = router_prompt | self.llm | StrOutputParser()
            decision = chain.invoke({"question": last_msg}).strip().upper()
            
            print(f"   [DEBUG] 🔀 Roteador Decidiu: {decision}") # Debug visual
            
            if "TECHNICAL" in decision:
                return "tech_node"
            return "general_node"

        # Montagem
        builder = StateGraph(MessagesState)
        builder.add_node("general_node", general_agent)
        builder.add_node("tech_node", tech_agent)
        builder.add_node("reviewer_node", reviewer_node)

        builder.add_conditional_edges(START, router_logic, {
            "general_node": "general_node",
            "tech_node": "tech_node"
        })

        builder.add_edge("general_node", "reviewer_node")
        builder.add_edge("tech_node", "reviewer_node")
        builder.add_edge("reviewer_node", END)

        memory = MemorySaver()
        return builder.compile(checkpointer=memory)

    def ask(self, prompt: str):
        initial_state = {"messages": [prompt]}
        response = self.graph.invoke(input=initial_state, config=self.thread)
        return response["messages"][-1].content

# --- Execução ---
def main():
    try:
        app = GraphApp()
        console = Console()
        console.print(Panel("🚀 [bold cyan]TransferFile AI[/bold cyan] Online!\n[green]Digite 'sair' para encerrar.", style="bold magenta"))
        
        while True:
            user_input = Prompt.ask("[bold blue]Você[/bold blue]")
            if user_input.strip().lower() == 'sair':
                break
            
            # Removemos o status spinner para você ver os prints de DEBUG
            resposta = app.ask(user_input)
            console.print(Panel(f"[white]{resposta}[/white]", title="[bold green]Assistente[/bold green]", border_style="green"))
            
    except Exception as e:
        print(f"Erro Crítico: {e}")

if __name__ == "__main__":
    main()