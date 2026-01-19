Descrição Geral:

Este projeto é um Chatbot de Terminal (Modelo TESTE), projetado para atuar como um "especialista" no produto "TransferFile".

Ele não é apenas um bot de perguntas e respostas; utilizamos uma arquitetura multi-agente orquestrada pelo LangGraph e Memória de Curto Prazo. Para garantir que ele não invente informações (as famosas "alucinações"), implementamos o padrão RAG (Retrieval-Augmented Generation), buscando dados reais direto da nossa base de conhecimento.

Detalhes da Arquitetura:

Neste projeto, o "cérebro" é dividido em 3 Agentes Especialistas, cada um com uma responsabilidade clara:

Agente Técnico: Cuida de tudo que envolve código, API, JSON e integração.

Agente Comercial: Domina os planos, preços e funcionalidades de negócio.

Agente Fiscalizador: O "guardião dos bons costumes". Ele analisa as respostas dos outros agentes antes de chegarem ao usuário, garantindo que sejam educadas, úteis e bem formatadas.

Fonte de Dados: Todo o conhecimento vem de um Supabase Vector DB, onde armazenamos os manuais técnicos e comerciais vetorizados.

Imagem em anexo para demonstrar o fluxo dos agentes.
