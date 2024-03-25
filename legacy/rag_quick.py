import os

import streamlit as st
from streamlit.logger import get_logger
# from langchain.callbacks.base import BaseCallbackHandler
# from langchain_community.graphs import Neo4jGraph
# from dotenv import load_dotenv
# from utils import (
#     create_vector_index,
# )
from chains import (
    load_embedding_model,
    load_llm,
    configure_llm_only_chain,
    configure_qa_rag_chain,
    generate_ticket,
)

# load_dotenv(".env")

# # url = os.getenv("NEO4J_URI")
# # username = os.getenv("NEO4J_USERNAME")
# # password = os.getenv("NEO4J_PASSWORD")
ollama_base_url = os.getenv("OLLAMA_BASE_URL")
embedding_model_name = os.getenv("EMBEDDING_MODEL")
llm_name = os.getenv("LLM")
# # Remapping for Langchain Neo4j integration
# # os.environ["NEO4J_URL"] = url

# logger = get_logger(__name__)



# # # if Neo4j is local, you can go to http://localhost:7474/ to browse the database
# # neo4j_graph = Neo4jGraph(url=url, username=username, password=password)
# # embeddings, dimension = load_embedding_model(
# #     embedding_model_name, config={"ollama_base_url": ollama_base_url}, logger=logger
# # )
# # create_vector_index(neo4j_graph, dimension)


# # class StreamHandler(BaseCallbackHandler):
# #     def __init__(self, container, initial_text=""):
# #         self.container = container
# #         self.text = initial_text

# #     def on_llm_new_token(self, token: str, **kwargs) -> None:
# #         self.text += token
# #         self.container.markdown(self.text)


# # llm = load_llm(llm_name, logger=logger, config={"ollama_base_url": ollama_base_url})

# # llm_chain = configure_llm_only_chain(llm)
# # rag_chain = configure_qa_rag_chain(
# #     llm, embeddings, embeddings_store_url=url, username=username, password=password
# # )

# # # Streamlit UI

# # def chat_input():
# #     user_input = st.chat_input("What coding issue can I help you resolve today?")

# #     if user_input:
# #         with st.chat_message("user"):
# #             st.write(user_input)
# #         with st.chat_message("assistant"):
# #             st.caption(f"RAG: {name}")
# #             stream_handler = StreamHandler(st.empty())
# #             result = output_function(
# #                 {"question": user_input, "chat_history": []}, callbacks=[stream_handler]
# #             )["answer"]
# #             output = result
# #             st.session_state[f"user_input"].append(user_input)
# #             st.session_state[f"generated"].append(output)
# #             st.session_state[f"rag_mode"].append(name)


# # def display_chat():
# #     # Session state
# #     if "generated" not in st.session_state:
# #         st.session_state[f"generated"] = []

# #     if "user_input" not in st.session_state:
# #         st.session_state[f"user_input"] = []

# #     if "rag_mode" not in st.session_state:
# #         st.session_state[f"rag_mode"] = []

# #     if st.session_state[f"generated"]:
# #         size = len(st.session_state[f"generated"])
# #         # Display only the last three exchanges
# #         for i in range(max(size - 3, 0), size):
# #             with st.chat_message("user"):
# #                 st.write(st.session_state[f"user_input"][i])

# #             with st.chat_message("assistant"):
# #                 st.caption(f"RAG: {st.session_state[f'rag_mode'][i]}")
# #                 st.write(st.session_state[f"generated"][i])

# #         with st.expander("Not finding what you're looking for?"):
# #             st.write(
# #                 "Automatically generate a draft for an internal ticket to our support team."
# #             )
# #             st.button(
# #                 "Generate ticket",
# #                 type="primary",
# #                 key="show_ticket",
# #                 on_click=open_sidebar,
# #             )
# #         with st.container():
# #             st.write("&nbsp;")


# # def mode_select() -> str:
# #     options = ["Disabled", "Enabled"]
# #     return st.radio("Select RAG mode", options, horizontal=True)


# # name = mode_select()
# # if name == "LLM only" or name == "Disabled":
# #     output_function = llm_chain
# # elif name == "Vector + Graph" or name == "Enabled":
# #     output_function = rag_chain


# # def open_sidebar():
# #     st.session_state.open_sidebar = True


# # def close_sidebar():
# #     st.session_state.open_sidebar = False


# # if not "open_sidebar" in st.session_state:
# #     st.session_state.open_sidebar = False
# # if st.session_state.open_sidebar:
# #     new_title, new_question = generate_ticket(
# #         neo4j_graph=neo4j_graph,
# #         llm_chain=llm_chain,
# #         input_question=st.session_state[f"user_input"][-1],
# #     )
# #     with st.sidebar:
# #         st.title("Ticket draft")
# #         st.write("Auto generated draft ticket")
# #         st.text_input("Title", new_title)
# #         st.text_area("Description", new_question)
# #         st.button(
# #             "Submit to support team",
# #             type="primary",
# #             key="submit_ticket",
# #             on_click=close_sidebar,
# #         )


# # display_chat()
# # chat_input()



import bs4
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
# from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = get_logger(__name__)


llm = load_llm(llm_name, logger=logger, config={"ollama_base_url": ollama_base_url})
embeddings, dimension = load_embedding_model(
    embedding_model_name, config={"ollama_base_url": ollama_base_url}, logger=logger
)

# Load, chunk and index the contents of the blog.
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)

retriever = vectorstore.as_retriever()
prompt = hub.pull("rlm/rag-prompt")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

with st.form('my_form'):
    question = st.text_area('Enter Question:', example_question)
    submitted = st.form_submit_button('Submit')         
    if submitted:
        rag_chain.invoke(question)
        
