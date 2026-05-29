from langchain_core.messages import HumanMessage
from utils import *
from graph import create_graph_with_rag, create_graph_with_0_shot
from langchain_core.runnables import RunnableConfig
from langgraph.errors import GraphRecursionError
from langchain_core.callbacks import UsageMetadataCallbackHandler

langgraph_config = RunnableConfig(recursion_limit=100)

from prompts import user_prompts, ordered_prompts

import argparse

parser = argparse.ArgumentParser(description='Code Generation Agent with RAG and 0-shot modes')
parser.add_argument('--mode', type=str, default=None, choices=['rag', '0-shot'], help='Sim mode to use')
parser.add_argument('--llm', type=str, default=None, help='LLM model to use')
parser.add_argument('--embedding_model', type=str, default='qwen3-embedding:8b-fp16', help='Embedding model to use for RAG')
parser.add_argument('--chunk_size', type=int, default=1500, help='Chunk size for document splitting in RAG')
parser.add_argument('--chunk_overlap', type=int, default=300, help='Chunk overlap for document splitting in RAG')
parser.add_argument('--rag_k', type=int, default=4, help='Number of documents to retrieve in RAG')
parser.add_argument('--temperature', type=float, default=0.0, help='Temperature setting for the LLM')
parser.add_argument('--top_p', type=float, default=1.0, help='Top-p setting for the LLM')
parser.add_argument('--model_provider', type=str, default='ollama', help='Model provider for the LLM')
parser.add_argument('--db_rebuild', default=False, action='store_true', help='Rebuild the vector database for RAG')
parser.add_argument('--save_graph_png', default=False, action='store_true', help='Save the graph as a PNG file')
parser.add_argument('--filter_only_relevant_docs', default=False, action='store_true', help='Filter out not relevant and partially relevant documents')
parser.add_argument('--do_not_use_reranker', default=False, action='store_true', help='Do not use reranker in the RAG process')
parser.add_argument('--filter_docs_only_with_reranker', default=False, action='store_true', help='Filter documents only with the reranker in the RAG process')
parser.add_argument('--reranker_model', type=str, default="BAAI/bge-reranker-v2-m3", help='Reranker model to use in RAG')
parser.add_argument('--reranker_top_n', type=int, default=3, help='Top-n documents to keep after reranking in RAG')

args = parser.parse_args()

def main():
    """Main function to run the code generation agent."""
    print("--- Initializing Components ---")
    
    # parameters from args
    llm_name_from_user = args.llm
    embedding_model_name = args.embedding_model
    chunk_size = args.chunk_size
    chunk_overlap = args.chunk_overlap
    rag_k = args.rag_k
    temperature = args.temperature
    top_p = args.top_p
    model_provider = args.model_provider
    db_rebuild = args.db_rebuild
    save_graph_png = args.save_graph_png
    filter_only_relevant_docs = args.filter_only_relevant_docs
    reranker_model = args.reranker_model
    do_not_use_reranker = args.do_not_use_reranker
    filter_docs_only_with_reranker = args.filter_docs_only_with_reranker
    reranker_top_n = args.reranker_top_n
    
    do_not_use_reranker = True
    print(f"!!!WARNING!!! Do not use reranker always forced to TRUE for now!")
    
    if args.mode:
        modes = [args.mode]
    else:
        modes = ["0-shot", "rag"]

    if llm_name_from_user:
        LLM_NAMES = [llm_name_from_user]
    else:
        LLM_NAMES = [
            "hf.co/mistralai/Devstral-Small-2507_gguf:BF16",
            "qwen3-coder:30b-a3b-fp16",
            "phi4:14b-fp16",
            "gemma3:27b-it-fp16",
        ]

    if not image_exists("python-sandbox"):
        print("Python sandbox image not found.")
        
        img = build_image(path="./docker-env", image_tag="python-sandbox")
        
        if img is None:
            print("Failed to build the Docker image. Exiting.")
            exit(1)
            
    for mode in modes:

        if mode == "rag":
            retriever = create_retriever_and_embedder(embedding_model_name, collection_name="code_rag", k=rag_k, chunk_size=chunk_size, chunk_overlap=chunk_overlap, rebuild=db_rebuild)
            
            if do_not_use_reranker:
                print("NOT using reranker!")
                reranker_retriever = None
                
                retriever_and_embedder = retriever
            else:
                print(f"Using reranker: {reranker_model}, top_n: {reranker_top_n}")
                reranker_retriever = create_reranker_retriever(retriever, model_name=reranker_model, top_n=reranker_top_n)
                
                retriever_and_embedder = reranker_retriever
        else:
            retriever_and_embedder = None

        for llm_name in LLM_NAMES:
            print(f"\n=== Running in {mode} mode with LLM: {llm_name} ===\n")
            
            llm = load_llm(llm_name, temperature=temperature, top_p=top_p, model_provider=model_provider)
            
            llm_name_to_save = llm_name.split("/")[-1].replace(":", "-")
            
            if mode == "rag":
                graph = create_graph_with_rag(llm, retriever_and_embedder, save_png=save_graph_png)
                
            elif mode == "0-shot":
                graph = create_graph_with_0_shot(llm)

            if save_graph_png:
                save_graph_to_png(graph)
            else:
                print(f"save_graph_png: {save_graph_png}")

            for user_prompt_index, user_prompt in enumerate(ordered_prompts):
                print(f"\n--- Processing User Prompt ---\n")
                print(f"{user_prompt}\n")
                
                callback = UsageMetadataCallbackHandler()

                initial_state = {
                    "messages": [HumanMessage(content=user_prompt)],
                    "iterations": 0,
                    "tracebacks": [],
                    "error": "no",
                    "llm_name": llm_name_to_save,
                    "complete_llm_path": llm_name,
                    "prompt_index": user_prompt_index,
                    "agent_mode": mode,
                    "temperature": temperature,
                    "top_p": top_p,
                    "filter_only_relevant_docs": filter_only_relevant_docs,
                    "filter_docs_only_with_reranker": filter_docs_only_with_reranker,
                    "callback_handler": callback,
                }

                final_state = graph.invoke(input=initial_state, config=langgraph_config)
                
                save_state_to_json(final_state)

if __name__ == "__main__":
    main()