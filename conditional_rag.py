import os
from urllib import response
from click import prompt
from dotenv import load_dotenv
load_dotenv()
from langchain_groq import ChatGroq
from typing import TypedDict,Annotated
from langgraph.graph import StateGraph,START,END
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.graph.message import add_messages
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from rich import print

#build rag

embedding=HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")

def build_retriver(pdf_path:str):
    loader=PyPDFLoader(pdf_path)
    documents=loader.load()

    splitter=RecursiveCharacterTextSplitter(chunk_size=800,
                                            chunk_overlap=200)
    chunks=splitter.split_documents(documents)

    vectorstore=FAISS.from_documents(chunks,embedding)

    return vectorstore.as_retriever(search_kwargs={"k":4},
                                    search_type="mmr")

academic_retriever=build_retriver("academics_handbook.pdf")
fees_retriever=build_retriver("fee_structure.pdf")

#llm
llm=ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct",
             temperature=0.4)


#statae

class state(TypedDict):
    programme:str
    messages:Annotated[list,add_messages]
    query:str
    retrieved_context:str


def category_type(state:state)->dict:
    """
    select one category academic,fee and general
    """

    last_message=state["messages"][-1].content
    prompt = f"""

    You are an intelligent message classifier.

    Your task is to classify the user's message into exactly one of these categories:

    1. academic
    - Questions about admissions, courses, syllabus, exams, assignments, classes, faculty, attendance, results, placements, internships, timetables, or any academic-related topic.

    2. fees
    - Questions about tuition fees, fee payment, scholarships, refunds, hostel fees, due dates, fines, or any payment-related topic.

    3. general
    - Any greeting, casual conversation, or questions that do not belong to the above categories.

    Rules:
    - Return ONLY one word.
    - The output must be exactly one of:
    - academic
    - fees
    - general
    - Do not provide any explanation or extra text.

    User Message:
   {last_message}

    """
    response=llm.invoke(prompt)
    category=response.content.strip().lower()

    return {
        "query":category
    }


def academic_context(state:state)->dict:
    """retrieve relevant context from academic handbook based on user query"""
    last_message=state["messages"][-1].content
         
    retrived=academic_retriever.invoke(last_message)
    context="\n\n".join([doc.page_content for doc in retrived])
    return {
        "retrieved_context":context
    }

def fee_context(state:state)->dict:
    """retrieve relevant context from fee structure pdf"""
    last_message=state["messages"][-1].content
         
    retrived=fees_retriever.invoke(last_message)
    context="\n\n".join([doc.page_content for doc in retrived])
    return {
        "retrieved_context":context
    }

def general_context(state:state)->dict:
    """generate context for general queries"""
    return {
        "retrieved_context":"NO_RETRIEVAL_NEEDED"
    }

def response_node(state: state) -> dict:
    """
    Generates the final answer, personalized using the student's programme.
    """

    query = state["messages"][-1].content
    programme = state.get("programme", "Unknown")
    context = state["retrieved_context"]

    if context == "NO_RETRIEVAL_NEEDED":
        prompt = (
            f"You are a friendly college assistant talking to a {programme} student. "
            f"Answer this question using your own general knowledge.\n\n"
            f"Question: {query}"
        )

    else:
        prompt = (
            f"You are a college assistant helping a {programme} student. "
            f"Use the following context from the official college documents to answer "
            f"the question accurately. If the context mentions specific figures for "
            f"different programmes, highlight the one relevant to {programme} if possible.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            f"Give a clear, friendly, and precise answer."
        )

    response = llm.invoke(prompt)

    return {
        "messages": [
            ("ai", response.content.strip())
        ]
    }
    

  # router ppoint toward one specific retriever based on the category of the query


def router(state: state):
    """
    Routes the query to the appropriate context retrieval function based on the category.
    """
    category = state["query"]

    if category == "academic":
        return "academic_rag"
    elif category == "fees":
        return "fees_rag"
    else:
        return "general"
    
#graph
graph=StateGraph(state)
#node

graph.add_node("category",category_type)
graph.add_node("general",general_context)
graph.add_node("academic_rag",academic_context)
graph.add_node("fees_rag",fee_context)
graph.add_node("response",response_node)
#edge

graph.add_edge(START,"category")
graph.add_conditional_edges("category",router)
graph.add_edge("general","response")
graph.add_edge("academic_rag","response")
graph.add_edge("fees_rag","response")
graph.add_edge("response",END)



app=graph.compile()

print("chose onr peogram among given three:\n\n")

print("1. BCA\n2. BBA\n3. B.Com(H)\n")

print("enter 1,2 or 3 to select your programme:\n")

programme_choice = input()

if programme_choice == "1":
    programme = "BCA"
elif programme_choice == "2":
    programme = "BBA"
elif programme_choice == "3":
    programme = "B.Com(H)"
else:
    programme = "Unknown"
print(f"You have selected: {programme}\n")


while True:
    user_input = input("Enter your query (or type 'exit' to quit): ")

    if user_input.lower() == "exit":
        break

    

    initial_state = {
        "programme": programme,
        "messages":[("user", user_input)],
        "query": "",
        "retrieved_context": ""
    }

    final_response = app.invoke(initial_state)

    ai_response = final_response["messages"][-1].content

    print(f"Assistant: {ai_response}\n")
