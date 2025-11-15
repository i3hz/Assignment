import os
import gradio as gr
from typing import List, TypedDict
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END


load_dotenv()


@staticmethod
def ingest_data():

    loader = DirectoryLoader(
        './data/',
        glob="*.txt",
        loader_cls=TextLoader,
        show_progress=True
    )
    documents = loader.load()

   

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    splits = text_splitter.split_documents(documents)


    persist_directory = './chroma_db'
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    
    print(f"Loaded {len(documents)} documents.")
    print(f"Split into {len(splits)} chunks.")
    print(f"Vectorstore created at {persist_directory}")
    return vectorstore


if not os.path.exists('./chroma_db'):
    vectorstore = ingest_data()
else:
    vectorstore = Chroma(
        persist_directory='./chroma_db',
        embedding_function=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        )
    

retriever = vectorstore.as_retriever(search_kwargs={'k': 3})



class AgentState(TypedDict):
    question: str
    plan: str
    documents: List[Document]
    answer: str
    reflection: str

reflection_llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash", 
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.7,
    max_output_tokens=8000,  
    timeout=30
)
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash", 
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.7,
    max_output_tokens=8000,  
    timeout=30
)


def plan_node(state: AgentState):

    question = state['question']

    plan = (
        "1. Retrieve relevant documents from the knowledge base. "
        "2. Generate an answer based on the retrieved context. "
        "3. Reflect on the answer for relevance and groundedness."
    )
    print(f"Question: {question}")
    print(f"Plan: {plan}")
    return {"plan": plan}

def retrieve_node(state: AgentState):
    question = state['question']
    documents = retriever.invoke(question)
    print(f"Retrieved {len(documents)} documents.")
    # for doc in documents:
    #     print(doc.metadata)
    return {"documents": documents}

def answer_node(state: AgentState):

    print("---NODE: ANSWER---")
    question = state['question']
    documents = state['documents']
    
    # Format context
    context_str = "\n\n---\n\n".join([doc.page_content for doc in documents])
    
    # RAG Prompt
    prompt_template = """You are an assistant for question-answering tasks.
    Use **only** the following context to answer the question.
    If you don't know the answer from the context, just say that you don't know.
    Do not use any prior knowledge. Be concise and helpful.

    Context:
    {context}

    Question:
    {question}

    Answer:"""
    prompt = ChatPromptTemplate.from_template(prompt_template)
    
    rag_chain = prompt | llm | StrOutputParser()
    answer = rag_chain.invoke({"context": context_str, "question": question})
    print(f"Generated Answer: {answer}")
    return {"answer": answer}

def reflect_node(state: AgentState):

    question = state['question']
    answer = state['answer']
    documents = state['documents']
    context_str = "\n\n---\n\n".join([doc.page_content for doc in documents])

    # Reflection Prompt
    reflection_prompt_template = """You are an evaluator. Your task is to check if an
    AI-generated 'Answer' is relevant to the 'Question' and
    factually grounded in the provided 'Context'.

    Respond with a single word:
    - 'Relevant' if the answer is relevant and grounded.
    - 'NotRelevant' if the answer is irrelevant or not grounded in the context.

    Context:
    {context}

    Question:
    {question}

    Answer:
    {answer}
    """
    prompt = ChatPromptTemplate.from_template(reflection_prompt_template)
    
    reflect_chain = prompt | reflection_llm | StrOutputParser()
    
    reflection = reflect_chain.invoke({
        "context": context_str,
        "question": question,
        "answer": answer
    }).strip()
    
    print(f"Reflection: {reflection}")
    return {"reflection": reflection}


workflow = StateGraph(AgentState)

workflow.add_node("plan", plan_node)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("answer", answer_node)
workflow.add_node("reflect", reflect_node)

workflow.set_entry_point("plan")
workflow.add_edge("plan", "retrieve")
workflow.add_edge("retrieve", "answer")
workflow.add_edge("answer", "reflect")
workflow.add_edge("reflect", END)

app = workflow.compile()
print("---AGENT READY---")


def run_agent_workflow(question):
    if not question:
        return "### ❗ Please ask a question."
        
    inputs = {"question": question}

    try:
        final_state = app.invoke(inputs)

        # Build Markdown cleanly
        output = [
            "## Agent Workflow Complete\n",
            "### **Question**",
            final_state.get("question", "N/A"),
            "",
            "### **Plan**",
            f"```text\n{final_state.get('plan', 'N/A')}\n```",
            "",
            "### **Answer**",
            final_state.get("answer", "N/A"),
            "",
            "### **Reflection**",
            f"```text\n{final_state.get('reflection', 'N/A')}\n```",
            "",
            "---",
            "##  Retrieved Context Snippets:",
        ]

        docs = final_state.get("documents", [])

        if not docs:
            output.append("_No snippets retrieved._")
        else:
            for i, doc in enumerate(docs):
                snippet = doc.page_content[:200].replace("\n", " ") + "..."
                source = doc.metadata.get("source", "Unknown")

                output.append(
                    f"### **Snippet {i+1}** _(from `{source}`)_\n"
                    f"> {snippet}"
                )

        return "\n".join(output)

    except Exception as e:
        print(f"Error during agent run: {e}")
        return f"### ❌ Error\n```\n{e}\n```"



def main():
 
    iface = gr.Interface(
        fn=run_agent_workflow,
        inputs=gr.Textbox(
            lines=3,
            placeholder="e.g., What are the benefits of renewable energy?"
        ),
        outputs=gr.Markdown(line_breaks=True),
        title="LangGraph RAG Agent",
        description=(
            "Ask a question about AI or renewable energy. "
            "The agent will show its plan, answer, and reflection. "
            "Traces are logged to LangSmith (if configured)."
        ),
        examples=[
            ["What are the benefits of renewable energy?"],
            ["What is RAG in AI?"],
            ["What is the capital of France?"] 
        ]
    )
    
    iface.launch()

if __name__ == "__main__":
    main()