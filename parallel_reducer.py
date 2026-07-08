import os
from dotenv import load_dotenv
from typing import TypedDict, Annotated
from langchain_groq import  ChatGroq
from langgraph.graph import StateGraph,START,END

load_dotenv()
llm = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct",
               temperature=0.1
               )


def merge_score(existing:dict,newly:dict)->dict:
    if existing is None:
        return newly
    return {
        **existing,**newly
    }


#create state

class analyzeState(TypedDict):
    raw_text:str
    score:Annotated[dict[str,int],merge_score]#reducer

#cnode

def toxicity_score(state:analyzeState)->dict:
    """
    analyzing toxicity and hate sppech!
    """
    prompt = (
    "You are a content moderation expert.\n\n"
    "Analyze the following text and assign integer scores from 0 to 10 for:\n"
    "- toxicity\n"
    "- hate_speech\n\n"
    "Scoring Guidelines:\n"
    "- 0 = None\n"
    "- 1-3 = Low\n"
    "- 4-6 = Moderate\n"
    "- 7-8 = High\n"
    "- 9-10 = Extreme\n\n"
    "Return ONLY a valid Python dictionary in this exact format:\n"
    "{'toxicity': <int>, 'hate_speech': <int>}\n\n"
    "Do not include any explanation, markdown, or extra text.\n\n"
    f"Text:\n{state['raw_text']}"
    )
    response=llm.invoke(prompt)
    try:
        score=int(response.content.strip())
    except ValueError:
        score=0
    return {
        "score":{"toxicity_score":score}
    }

def copyright_score(state: analyzeState) -> dict:
    """
    Analyzing copyright infringement risk in the provided text.
    """
    prompt = (
        "You are a copyright risk analysis expert.\n\n"
        "Analyze the following text and assign an integer score from 0 to 10 for copyright risk.\n\n"
        "Scoring Guidelines:\n"
        "- 0 = No copyright risk\n"
        "- 1-3 = Low risk\n"
        "- 4-6 = Moderate risk\n"
        "- 7-8 = High risk\n"
        "- 9-10 = Extreme risk\n\n"
        "Return ONLY a single integer from 0 to 10.\n\n"
        "Do not include any explanation, markdown, or extra text.\n\n"
        f"Text:\n{state['raw_text']}"
    )

    response = llm.invoke(prompt)

    try:
        score = int(response.content.strip())
    except ValueError:
        score = 0

    return {
        "score": {
            "copyright_score": score
        }
    }


def cultural_regional_sensitivity_score(state: analyzeState) -> dict:
    """
    Analyzing cultural and regional sensitivity risk in the provided text.
    """
    prompt = (
        "You are a cultural and regional sensitivity analysis expert.\n\n"
        "Analyze the following text and assign an integer score from 0 to 10 "
        "for cultural and regional sensitivity risk.\n\n"
        "Scoring Guidelines:\n"
        "- 0 = No cultural or regional sensitivity issue\n"
        "- 1-3 = Low risk\n"
        "- 4-6 = Moderate risk\n"
        "- 7-8 = High risk\n"
        "- 9-10 = Extreme risk\n\n"
        "Consider whether the text may be offensive, disrespectful, stereotypical, "
        "or insensitive toward any culture, region, language, tradition, nationality, "
        "community, or local belief system.\n\n"
        "Return ONLY a single integer from 0 to 10.\n\n"
        "Do not include any explanation, markdown, or extra text.\n\n"
        f"Text:\n{state['raw_text']}"
    )

    response = llm.invoke(prompt)

    try:
        score = int(response.content.strip())
    except ValueError:
        score = 0

    return {
        "score": {
            "cultural_regional_sensitivity_score": score
        }
    }


graph=StateGraph(analyzeState)


graph.add_node("toxicity_node",toxicity_score)
graph.add_node("copyright_node",copyright_score)
graph.add_node("cultural_node",cultural_regional_sensitivity_score)

#fan out 

graph.add_edge(START,"toxicity_node")
graph.add_edge(START,"copyright_node")
graph.add_edge(START,"cultural_node")


# fan in 

graph.add_edge("toxicity_node",END)
graph.add_edge("copyright_node",END)
graph.add_edge("cultural_node",END)


app=graph.compile()


script= ("People from different regions have unique traditions, languages, food habits, "
       " and ways of living. While discussing any culture or community, it is important "
       " to avoid stereotypes and respect local beliefs, customs, and identities.")
    
initial={
    "raw_text":script,
    "score":{}

}

response=app.invoke(initial)


print(response["score"])






