import os
from typing import TypedDict
from rich import print

# 1st create state

class pipelineState(TypedDict):
    raw_input:str
    edit_data:str
    script_data:str
    final_output:str

from dotenv import load_dotenv
load_dotenv()
from langchain_groq import ChatGroq

llm=ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct",temperature=0.7)

def edited_data(state:pipelineState)->dict:
    """
    cleam up grammer,remove typos and refine tone
    """
    prompt = (
    "You are an expert copyeditor. "
    "Clean up the following raw text. "
    "Fix grammar and spelling. "
    f"Text:\n{state['raw_input']}"
    )
    
    response=llm.invoke(prompt)

    return {
        "edit_data": response.content.strip()
    }

def generate_script(state: pipelineState) -> dict:
    """
    Generate an engaging content script from the edited text.
    """

    prompt = (
        "You are an expert content writer.\n\n"
        "Using the edited text below, create an engaging script suitable "
        "for a YouTube video. Keep it clear, natural, and interesting.\n\n"
        f"Edited Text:\n{state['edit_data']}"
    )

    response = llm.invoke(prompt)

    return {
        "script_data": response.content.strip()
    }

def convert_to_hinglish(state: pipelineState) -> dict:
    """
    Convert the generated script into natural Hinglish.
    """

    prompt = (
        "You are an expert content writer.\n\n"
        "Convert the following English script into natural, conversational Hinglish. "
        "Keep the meaning, tone, and flow the same. "
        "Use Hindi words written in English letters (Roman Hindi). "
        "Return only the converted Hinglish script.\n\n"
        f"Script:\n{state['script_data']}"
    )

    response = llm.invoke(prompt)

    return {
        "final_output": response.content.strip()
    }

# now state and node are ready 
# now we have to connect node with the help of edge

from langgraph.graph import StateGraph,END,START
#graph created

graph=StateGraph(pipelineState)

#add node in graph

graph.add_node("editor",edited_data)
graph.add_node("script_writer",generate_script)
graph.add_node("translator",convert_to_hinglish)

#add edge (sequential workflow - one after another)


graph.add_edge(START,"editor")
graph.add_edge("editor","script_writer")
graph.add_edge("script_writer","translator")
graph.add_edge("translator",END)

app=graph.compile()

result=app.invoke({
    "raw_input":"Artifical inteligence is changing the way peoples work and learn. It can automate many task, but human creativity is still very importent."
})

print(result["final_output"])