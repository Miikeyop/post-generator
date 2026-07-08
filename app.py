import os
import time
from typing import TypedDict, Annotated

import streamlit as st
from dotenv import load_dotenv

from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain_groq import ChatGroq
from langchain_mistralai import ChatMistralAI
from langchain_tavily import TavilySearch

load_dotenv()

# ──────────────────────────────────────────────────────────────────────────
#  PAGE CONFIG
# ──────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LinkedIn Post Generator",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp { background: #f5f6f8; color: #1a1a1a; }

    /* Force dark, readable text on markdown / labels / expander / status —
       but NOT inside buttons or inputs, which have their own backgrounds. */
    h1, h2, h3, h4, h5, h6,
    .stMarkdown, .stMarkdown p,
    [data-testid="stMarkdownContainer"],
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stCaptionContainer"],
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"],
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
    [data-testid="stExpander"] summary,
    [data-testid="stExpander"] summary span,
    [data-testid="stExpander"] summary p,
    [data-testid="stStatusWidget"] [data-testid="stMarkdownContainer"],
    [data-testid="stStatusWidget"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stWidgetLabel"] p {
        color: #1a1a1a !important;
    }

    [data-testid="stSidebar"] { background: #ffffff; }

    /* Top toolbar (hamburger menu, Deploy button, sidebar-collapse arrow) */
    [data-testid="stHeader"] {
        background: #f5f6f8 !important;
    }
    [data-testid="stToolbar"],
    [data-testid="stToolbarActions"],
    [data-testid="stDecoration"] {
        background: transparent !important;
    }
    [data-testid="stHeader"] button,
    [data-testid="stHeader"] svg,
    [data-testid="stSidebarCollapseButton"] svg,
    [data-testid="stBaseButton-headerNoPadding"] svg {
        fill: #1a1a1a !important;
        color: #1a1a1a !important;
    }
    [data-testid="stHeader"] button:hover,
    [data-testid="stSidebarCollapseButton"] button:hover {
        background: #e8eaed !important;
    }
    [data-testid="stSidebarCollapseButton"] button {
        background: #eceef1 !important;
        border-radius: 6px !important;
    }
    [data-testid="stDeployButton"] button {
        background: #ffffff !important;
        color: #1a1a1a !important;
        border: 1px solid #d5d8dc !important;
    }
    [data-testid="stDeployButton"] button p,
    [data-testid="stDeployButton"] button span {
        color: #1a1a1a !important;
    }

    [data-testid="stExpander"] {
        background: #ffffff;
        border: 1px solid #e2e5e9;
        border-radius: 10px;
    }
    [data-testid="stExpander"] summary {
        background: #ffffff !important;
    }

    [data-testid="stStatusWidget"] {
        background: #ffffff;
        border-radius: 10px;
    }

    /* Text areas / text inputs: white background, dark readable text */
    .stTextArea textarea,
    .stTextInput input {
        background-color: #ffffff !important;
        color: #1a1a1a !important;
        border: 1px solid #d5d8dc !important;
    }
    .stTextArea textarea::placeholder,
    .stTextInput input::placeholder {
        color: #8a8f98 !important;
    }

    /* Buttons: keep their label readable regardless of theme */
    .stButton button, .stDownloadButton button {
        color: #ffffff !important;
        border: none !important;
    }
    .stButton button[kind="primary"] {
        background-color: #ff4b4b !important;
    }
    .stDownloadButton button {
        background-color: #31333f !important;
    }
    .stButton button p, .stDownloadButton button p,
    .stButton button span, .stDownloadButton button span {
        color: #ffffff !important;
    }

    .post-card {
        background: #ffffff;
        border: 1px solid #e2e5e9;
        border-radius: 14px;
        padding: 28px 32px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
        font-size: 1.02rem;
        line-height: 1.6;
        white-space: pre-wrap;
        color: #1a1a1a;
    }
    .draft-card {
        background: #fafbfc;
        border: 1px solid #e8eaed;
        border-radius: 10px;
        padding: 18px 22px;
        white-space: pre-wrap;
        line-height: 1.55;
        color: #333;
    }
    .badge-approved {
        display:inline-block; padding: 3px 12px; border-radius: 20px;
        background:#e6f4ea; color:#137333; font-weight:600; font-size:0.85rem;
    }
    .badge-rejected {
        display:inline-block; padding: 3px 12px; border-radius: 20px;
        background:#fce8e6; color:#c5221f; font-weight:600; font-size:0.85rem;
    }
    .step-label {
        font-weight: 600; color: #444; font-size: 0.95rem;
        text-transform: uppercase; letter-spacing: 0.04em;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("✍️ AI LinkedIn Post Generator")
st.caption("A writer agent drafts your post (searching the web when needed), a reviewer agent critiques it, and they iterate together until it's publish-ready.")

# ──────────────────────────────────────────────────────────────────────────
#  GRAPH DEFINITION  (same logic as the original script, adapted for reuse)
# ──────────────────────────────────────────────────────────────────────────

class State(TypedDict):
    topic: str
    messages: Annotated[list, add_messages]
    draft: str
    review_feedback: str
    is_approved: bool
    attempt: int
    max_attempts: int


WRITER_SYSTEM_PROMPT = (
    "You are an expert LinkedIn content writer. Your job is to write "
    "engaging, professional LinkedIn posts about the given topic. "
    "If the topic requires up-to-date information, statistics, or "
    "current trends, use the web search tool to gather fresh context "
    "before writing. If you have already received feedback on a "
    "previous draft, carefully address every point in the new draft. "
    "Rules for good LinkedIn posts: strong hook in the first line, "
    "1 clear takeaway, easy to skim (short paragraphs), around "
    "150–200 words, ends with a question or call-to-action to invite "
    "engagement. Do not use hashtags."
)

REVIEWER_SYSTEM_PROMPT = (
    "You are a strict LinkedIn content reviewer. You judge whether a "
    "post is publish-ready. Evaluate against these criteria:\n"
    "1. Strong hook in the first line\n"
    "2. One clear, valuable takeaway\n"
    "3. Easy to skim — uses short paragraphs\n"
    "4. Roughly 150-200 words\n"
    "5. Ends with an engaging question or CTA\n"
    "6. Professional but human tone (not corporate-robotic)\n"
    "7. No hashtags\n\n"
    "Respond in exactly this format:\n"
    "VERDICT: APPROVED or REJECTED\n"
    "FEEDBACK: <one short paragraph explaining why>\n\n"
    "Be strict but fair. Approve only if the post genuinely meets all "
    "criteria. Reject if even one criterion is clearly missing."
)


@st.cache_resource(show_spinner=False)
def build_app():
    search_tool = TavilySearch(max_results=3)
    tools = [search_tool]

    writer_llm = ChatMistralAI(model="mistral-small-2603", temperature=0.7)
    writer_llm_with_tools = writer_llm.bind_tools(tools)
    reviewer_llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.2)

    def writer_node(state: State) -> dict:
        attempt = state.get("attempt", 0) + 1
        topic = state["topic"]
        previous_feedback = state["review_feedback"]

        if attempt == 1:
            user_message = (
                f"Write a LinkedIn post on this topic {topic}"
                f"if you need current info search the web first "
            )
        else:
            user_message = (
                f"your previous draft on '{topic}' was rejected"
                f"Here is the reviewer's feedback \n\n {previous_feedback}\n\n"
                f"Write a new, improved draft that fixes every issue mentiond"
                f"do not repeat the same mistake"
            )
        messages = [("system", WRITER_SYSTEM_PROMPT), ("human", user_message)]
        response = writer_llm_with_tools.invoke(messages)

        return {
            "messages": [("human", user_message), response],
            "attempt": attempt,
        }

    tool_node = ToolNode(tools)

    def generate_node(state: State) -> dict:
        """Runs after a tool call — lets the writer read the search
        results and actually produce the post (or call the tool again
        if it needs more info)."""
        response = writer_llm_with_tools.invoke(state["messages"])
        return {"messages": [response]}

    def extract_draft_node(state: State) -> dict:
        last_message = state["messages"][-1]
        draft = last_message.content
        return {"draft": draft}

    def reviewer_node(state: State) -> dict:
        draft = state["draft"]
        prompt = f"review this LinkedIn post draft : \n{draft}\ngive your reviews"
        response = reviewer_llm.invoke(
            [("system", REVIEWER_SYSTEM_PROMPT), ("human", prompt)]
        )
        review_text = response.content.strip()

        is_approved = "APPROVED" in review_text.upper().split("FEEDBACK")[0]

        if "FEEDBACK:" in review_text:
            feedback = review_text.split("FEEDBACK:", 1)[1].strip()
        else:
            feedback = review_text

        return {"review_feedback": feedback, "is_approved": is_approved}

    def should_use_tool(state: State):
        last_message = state["messages"][-1]
        if getattr(last_message, "tool_calls", None):
            return "tools"
        return "extract_draft"

    def should_stop_looping(state: State):
        if state["is_approved"]:
            return END
        if state["attempt"] >= state.get("max_attempts", 3):
            return END
        return "writer"

    graph = StateGraph(State)
    graph.add_node("writer", writer_node)
    graph.add_node("tools", tool_node)
    graph.add_node("generate", generate_node)
    graph.add_node("extract_draft", extract_draft_node)
    graph.add_node("reviewer", reviewer_node)

    graph.add_edge(START, "writer")
    graph.add_conditional_edges("writer", should_use_tool)
    graph.add_edge("tools", "generate")
    graph.add_conditional_edges("generate", should_use_tool)
    graph.add_edge("extract_draft", "reviewer")
    graph.add_conditional_edges("reviewer", should_stop_looping)

    return graph.compile()


# ──────────────────────────────────────────────────────────────────────────
#  SIDEBAR
# ──────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    topic = st.text_area(
        "Topic",
        placeholder="e.g. Why small teams ship AI products faster than big ones",
        height=110,
    )
    max_attempts = st.slider("Max attempts", 1, 5, 3)
    run_btn = st.button("🚀 Generate Post", type="primary", use_container_width=True)

    st.divider()
    st.caption("**Agents**")
    st.caption("📝 Writer — Mistral Small (+ Tavily web search)")
    st.caption("🔎 Reviewer — Llama 3.3 70B via Groq")

    missing = [k for k in ["MISTRAL_API_KEY", "GROQ_API_KEY", "TAVILY_API_KEY"] if not os.getenv(k)]
    if missing:
        st.divider()
        st.warning("Missing API keys in environment:\n\n" + "\n".join(f"- `{k}`" for k in missing))

# ──────────────────────────────────────────────────────────────────────────
#  MAIN AREA
# ──────────────────────────────────────────────────────────────────────────
if "result" not in st.session_state:
    st.session_state.result = None

if run_btn:
    if not topic.strip():
        st.error("Please enter a topic first.")
    elif missing:
        st.error("Cannot run — missing API keys. Add them to your environment/.env file.")
    else:
        app = build_app()
        initial_state = {
            "topic": topic.strip(),
            "messages": [],
            "draft": "",
            "review_feedback": "",
            "is_approved": False,
            "attempt": 0,
            "max_attempts": max_attempts,
        }

        progress_area = st.container()
        attempt_num = 0
        attempt_expander = None
        searched_this_attempt = False
        final_state = None

        with progress_area:
            st.subheader("🧵 Live progress")
            status = st.status("Starting agents…", expanded=True)

        working_state = dict(initial_state)

        try:
            for update in app.stream(initial_state, stream_mode="updates"):
                node_name, payload = next(iter(update.items()))
                working_state.update(payload)

                if node_name == "writer":
                    attempt_num = payload["attempt"]
                    searched_this_attempt = False
                    status.update(label=f"✍️ Writer drafting — attempt {attempt_num}", state="running")
                    attempt_expander = st.expander(f"Attempt {attempt_num}", expanded=True)

                elif node_name == "tools":
                    searched_this_attempt = True
                    status.update(label=f"🔎 Searching the web (attempt {attempt_num})…", state="running")

                elif node_name == "generate":
                    status.update(label=f"✍️ Writing with search results — attempt {attempt_num}", state="running")

                elif node_name == "extract_draft":
                    draft_text = payload["draft"]
                    status.update(label=f"📄 Draft {attempt_num} ready — reviewing…", state="running")
                    if attempt_expander:
                        with attempt_expander:
                            if searched_this_attempt:
                                st.caption("🔎 Used web search for current info")
                            st.markdown('<div class="step-label">Draft</div>', unsafe_allow_html=True)
                            st.markdown(f'<div class="draft-card">{draft_text}</div>', unsafe_allow_html=True)

                elif node_name == "reviewer":
                    approved = payload["is_approved"]
                    feedback = payload["review_feedback"]
                    badge = (
                        '<span class="badge-approved">✅ APPROVED</span>'
                        if approved
                        else '<span class="badge-rejected">❌ REJECTED</span>'
                    )
                    if attempt_expander:
                        with attempt_expander:
                            st.markdown('<div class="step-label">Reviewer verdict</div>', unsafe_allow_html=True)
                            st.markdown(badge, unsafe_allow_html=True)
                            st.write(feedback)
                    if approved:
                        status.update(label=f"✅ Approved on attempt {attempt_num}", state="complete")
                    elif attempt_num >= max_attempts:
                        status.update(label=f"⏹️ Stopped after {attempt_num} attempts (not approved)", state="error")

            final_state = working_state
        except Exception as e:
            st.error(f"Something went wrong while running the agents: {e}")
            final_state = None

        st.session_state.result = final_state

# ──────────────────────────────────────────────────────────────────────────
#  FINAL RESULT
# ──────────────────────────────────────────────────────────────────────────
result = st.session_state.result
if result:
    st.divider()
    st.subheader("📬 Final Post")

    badge = (
        '<span class="badge-approved">✅ Approved</span>'
        if result["is_approved"]
        else '<span class="badge-rejected">⚠️ Best effort (not approved)</span>'
    )
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(badge, unsafe_allow_html=True)
    with col2:
        st.caption(f"Attempts used: {result['attempt']}")

    st.markdown(f'<div class="post-card">{result["draft"]}</div>', unsafe_allow_html=True)

    st.download_button(
        "⬇️ Download as .txt",
        data=result["draft"],
        file_name="linkedin_post.txt",
        mime="text/plain",
    )
else:
    st.info("Enter a topic in the sidebar and click **Generate Post** to get started.")