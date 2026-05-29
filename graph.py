from functools import partial
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import SystemMessage, HumanMessage
from schemas import *
from prompts import *
import time
import os
import subprocess
from power_profiler import Profiler

from ollama._types import ResponseError

def invoke_with_metadata(llm, all_messages, node_name, state): 
    callback = state.get("callback_handler", None)
    
    # 1. Create a profiler instance
    profiler = Profiler(cpu_tdp=385) # TDP 

    # 2. Start profiling from the "outside"
    profiler.start()
    
    start = time.perf_counter()

    response = llm.invoke(all_messages, config={"callbacks": [callback]})

    end = time.perf_counter()
    
    power_hardware_results = profiler.stop()
    
    metadatas = state.get("metadatas", [])
    complete_llm_path = state.get("complete_llm_path", "unknown")
    
    power_hardware_metadata = {"node_name": node_name, "power_metrics": power_hardware_results}
    
    tot_tokens = state.get("tot_tokens", 0)
    tot_input_tokens = state.get("tot_input_tokens", 0)
    tot_output_tokens = state.get("tot_output_tokens", 0)
    
    # print(f"Total tokens so far before this invoke: {tot_tokens}")
    # print(f"Total input tokens so far before this invoke: {tot_input_tokens}")
    # print(f"Total output tokens so far before this invoke: {tot_output_tokens}")

    invoke_time = {"node_name": node_name, "time": end - start}
    
    cur_metadata = callback.usage_metadata
    
    # print(f"LLM invocation metadata for node '{node_name}':\n\n{cur_metadata}\n\n" )
    
    cur_tot_tokens = cur_metadata[complete_llm_path]['total_tokens'] - tot_tokens
    cur_input_tokens = cur_metadata[complete_llm_path]['input_tokens'] - tot_input_tokens
    cur_output_tokens = cur_metadata[complete_llm_path]['output_tokens'] - tot_output_tokens
    
    print(f"tot_tokens_for_this_node_in_this_invoke: {cur_tot_tokens}")
    print(f"input_tokens_for_this_node_in_this_invoke: {cur_input_tokens}")
    print(f"output_tokens_for_this_node_in_this_invoke: {cur_output_tokens}")

    metadata = {"node_name": node_name, "tot_tokens_for_this_node_in_this_invoke": cur_tot_tokens, "input_tokens_for_this_node_in_this_invoke": cur_input_tokens, "output_tokens_for_this_node_in_this_invoke": cur_output_tokens}
    
    tot_tokens = cur_metadata[complete_llm_path]['total_tokens']
    tot_input_tokens = cur_metadata[complete_llm_path]['input_tokens']
    tot_output_tokens = cur_metadata[complete_llm_path]['output_tokens']

    return response, metadata, invoke_time, tot_tokens, tot_input_tokens, tot_output_tokens, power_hardware_metadata

def perform_retrieve_from_query(retriever, query):
    start = time.perf_counter()
    
    try:
        docs = retriever.invoke(query)
    except ResponseError as e:
        print(f"Ollama failed to generate embeddings for query: '{query}'")
        print(f"Error: {e}")
        # Return an empty list so the graph can continue
        docs = [] 
    
    end = time.perf_counter()
    
    retrieval_time = end - start
    
    print(f"Retrieved {len(docs)} documents.")
    
    return docs, retrieval_time

# --- GRAPH NODES ---   

def generate_queries_from_user_prompt_node(state: State, llm):
    """Reformulates the user's last message into an optimized list of search queries."""
    
    node_name = "generate_queries_from_user_prompt"
    
    metadatas = state.get("metadatas", [])
    invoke_times = state.get("invoke_times", [])
    power_hardware_metrics = state.get("power_hardware_metrics", [])
    
    print(node_name)

    system_message = generate_queries_from_user_prompt_sys_message

    all_messages = [system_message] + state["messages"]

    response, metadata, invoke_time, tot_tokens, tot_input_tokens, tot_output_tokens, power_hardware_metadata = invoke_with_metadata(llm, all_messages, node_name, state)

    metadatas.append(metadata)
    invoke_times.append(invoke_time)
    power_hardware_metrics.append(power_hardware_metadata)

    parsed_output = response['parsed']

    queries = parsed_output.queries

    print(f"Generated queries: {queries}")

    return {
        "queries_from_user_prompt": queries,
        "metadatas": metadatas,
        "invoke_times": invoke_times,
        "tot_tokens": tot_tokens,
        "tot_input_tokens": tot_input_tokens,
        "tot_output_tokens": tot_output_tokens,
        "power_hardware_metrics": power_hardware_metrics,
    }

def create_summary_for_each_query_node(query, chunks, llm, state):
    print("create_summary_for_each_query_node")
    
    node_name = "create_summary_for_each_query"

    sys_message = generate_context_for_each_query_sys_message
    
    chunks_formatted = ""
    
    for i,chunk in enumerate(chunks):
        chunks_formatted += f"<CHUNK_{i}>\n{chunk}\n</CHUNK_{i}>\n\n"
    
    human_message = HumanMessage(
        content=(
            "<CHUNKS>\n"
            f"{chunks_formatted}"
            "</CHUNKS>\n\n"
            "<QUERY>\n"
            f"{query}\n"
            "</QUERY>\n\n"
        )
    )
    
    # print()
    # print(human_message.content)
    # print()
    
    all_messages = [sys_message] + [human_message]
    
    response, metadata, invoke_time, tot_tokens, tot_input_tokens, tot_output_tokens, power_hardware_metadata = invoke_with_metadata(llm, all_messages, node_name, state)
    
    parsed_response = response['parsed']
    
    status = parsed_response.status
    context = parsed_response.context
    
    # print()
    # print("------RESULT------")
    # print(f"Status: {status}")
    # print(f"Context: {context}")
    # print("------------------\n")
    # print()

    return status, context, metadata, invoke_time, tot_tokens, tot_input_tokens, tot_output_tokens, power_hardware_metadata

def generate_queries_for_insufficient_contexts_node(state: State, llm):
    node_name = "generate_queries_for_insufficient_contexts_node"
    print(node_name)
    
    metadatas = state.get("metadatas", [])
    invoke_times = state.get("invoke_times", [])
    power_hardware_metrics = state.get("power_hardware_metrics", [])
    
    insufficient_contexts_for_each_query = state.get("insufficient_contexts_for_each_query", [])
    how_many_insufficient_retrials = state.get("how_many_insufficient_retrials", 0)
    
    print(f"how_many_insufficient_retrials: {how_many_insufficient_retrials}")
    
    how_many_insufficient_retrials += 1
    
    if how_many_insufficient_retrials >= 3:
        print("Already had 3 insufficient retrials, skipping further retrieval attempts.")
        return {"how_many_insufficient_retrials": how_many_insufficient_retrials}
    
    sys_msg = reformulate_insufficient_queries_sys_message
    
    new_queries_for_insufficient_retrievals = state.get("new_queries_for_insufficient_retrievals", [])
    
    new_queries_for_insufficient_retrieval = []
    
    print(f"len for insufficient_contexts_for_each_query: {len(insufficient_contexts_for_each_query)}")
    
    for item in insufficient_contexts_for_each_query[-1]:
        query = item['query']
        
        human_msg = HumanMessage(
            content=(
                "<QUERY>\n"
                f"{query}\n"
                "</QUERY>\n\n"
            )
        )
        
        all_messages = [sys_msg] + [human_msg]
        
        response, metadata, invoke_time, tot_tokens, tot_input_tokens, tot_output_tokens, power_hardware_metadata = invoke_with_metadata(llm, all_messages, node_name, state)
        
        metadatas.append(metadata)
        invoke_times.append(invoke_time)
        power_hardware_metrics.append(power_hardware_metadata)
        
        response_parsed = response['parsed']
        
        reformulated_queries = response_parsed.queries
        
        print()
        print(f"Reformulated queries for insufficient query '{query}':\n{reformulated_queries}")
        print()
        
        new_queries_for_insufficient_retrieval.extend(reformulated_queries)

    new_queries_for_insufficient_retrievals.append(new_queries_for_insufficient_retrieval)
    
    return {
        "new_queries_for_insufficient_retrievals": new_queries_for_insufficient_retrievals,
        "how_many_insufficient_retrials": how_many_insufficient_retrials,
        "metadatas": metadatas,
        "invoke_times": invoke_times,
        "power_hardware_metrics": power_hardware_metrics,
        "tot_tokens": tot_tokens,
        "tot_input_tokens": tot_input_tokens,
        "tot_output_tokens": tot_output_tokens,
    }

def retrieve_docs_for_user_node(state: State, retriever, llm):
    print("retrieve_docs_for_user_node")
    
    metadatas = state.get("metadatas", [])
    invoke_times = state.get("invoke_times", [])
    power_hardware_metrics = state.get("power_hardware_metrics", [])
    node_name = "retrieve_docs_for_user_node"
    
    doc_search_needed = state.get("doc_search_needed", "no")
    retrieval_times = state.get("retrieval_times", [])
    there_is_an_insufficient_query = state.get("there_is_an_insufficient_query", False)
    status_contexts_for_each_query = state.get("status_contexts_for_each_query", [])
    how_many_insufficient_retrials = state.get("how_many_insufficient_retrials", 0)
    
    state_tot_tokens = state.get("tot_tokens", 0)
    
    if there_is_an_insufficient_query and how_many_insufficient_retrials >= 3:
        print("Already had 3 insufficient retrials, skipping further retrieval attempts.")
        
        there_is_an_insufficient_query = False
        how_many_insufficient_retrials = 0

        return {"there_is_an_insufficient_query": there_is_an_insufficient_query, "how_many_insufficient_retrials": how_many_insufficient_retrials}

    if doc_search_needed == "no" and not there_is_an_insufficient_query:
        queries = state.get("queries_from_user_prompt")
        status_contexts_for_this_run = []
    elif doc_search_needed == "yes" and not there_is_an_insufficient_query:
        queries = state.get("errors_queries", [])[-1]
        status_contexts_for_this_run = []
    elif there_is_an_insufficient_query:
        queries = state.get("new_queries_for_insufficient_retrievals", [])[-1]
        there_is_an_insufficient_query = False
        status_contexts_for_this_run = status_contexts_for_each_query[-1] if len(status_contexts_for_each_query) > 0 else []
        
        print(f"there was a retrial for insufficient queries, now trying with new queries: {queries} with len status contexts for this run: {len(status_contexts_for_this_run)}")

    docs = []
    
    insufficient_contexts_for_each_query = state.get("insufficient_contexts_for_each_query", [])
    
    insufficient_contexts_for_this_run = []

    print(f"Evaluating {len(queries)} queries.")
    
    for query in queries:
        print(f"query: '{query}'")

        temp_docs, retrieval_time = perform_retrieve_from_query(retriever, query)
        docs.append({"query": query, "docs": temp_docs})

        status, context, metadata, invoke_time, tot_tokens, tot_input_tokens, tot_output_tokens, power_hardware_metadata = create_summary_for_each_query_node(query, temp_docs, llm, state)

        metadatas.append(metadata)
        invoke_times.append(invoke_time)
        power_hardware_metrics.append(power_hardware_metadata)
        
        state_tot_tokens = tot_tokens
        state_tot_input_tokens = tot_input_tokens
        state_tot_output_tokens = tot_output_tokens
        
        state["tot_tokens"] = state_tot_tokens
        state["tot_input_tokens"] = state_tot_input_tokens
        state["tot_output_tokens"] = state_tot_output_tokens

        if status == "insufficient":
            print(f"Query '{query}' resulted in insufficient context.")
            there_is_an_insufficient_query = True
            insufficient_contexts_for_this_run.append({"query": query, "status": status, "context": context})
            
            print(f"len of insufficient_contexts_for_this_run: {len(insufficient_contexts_for_this_run)}")
        else:
            status_contexts_for_this_run.append({"query": query, "status": status, "context": context})

        retrieval_time_metadata = {"query": query, "retrieval_time": retrieval_time}
        retrieval_times.append(retrieval_time_metadata)
    
    status_contexts_for_each_query.append(status_contexts_for_this_run)
    
    if there_is_an_insufficient_query:
        insufficient_contexts_for_each_query.append(insufficient_contexts_for_this_run)

    print(f"len for insufficient_contexts_for_each_query: {len(insufficient_contexts_for_each_query)}")

    context = state.get("contexts", [])
    context.append(docs)

    return {
        "status_contexts_for_each_query": status_contexts_for_each_query,
        "retrieval_times": retrieval_times,
        "insufficient_contexts_for_each_query": insufficient_contexts_for_each_query,
        "there_is_an_insufficient_query": there_is_an_insufficient_query,
        "how_many_insufficient_retrials": how_many_insufficient_retrials,
        "metadatas": metadatas,
        "invoke_times": invoke_times,
        "power_hardware_metrics": power_hardware_metrics,
        "tot_tokens": state_tot_tokens,
        "tot_input_tokens": state_tot_input_tokens,
        "tot_output_tokens": state_tot_output_tokens,
    }

def process_chunks_to_create_context(state: State, llm):
    print("process_chunks_to_create_context")
    
    node_name = "process_chunks_to_create_context"
    
    metadatas = state.get("metadatas", [])
    invoke_times = state.get("invoke_times", [])
    power_hardware_metrics = state.get("power_hardware_metrics", [])

    status_contexts_for_each_query = state.get("status_contexts_for_each_query", [])
    last_status_contexts_for_each_query = status_contexts_for_each_query[-1] if len(status_contexts_for_each_query) > 0 else []
    
    system_message = synthesizer_system_message

    contexts_formatted = ""
    for i, item in enumerate(last_status_contexts_for_each_query):
        context = item['context']
        status = item['status']
        
        if status != "insufficient":
            contexts_formatted += f"<context_block_{i}>\n{context}\n</context_block_{i}>\n\n"
        else:
            print(f"Skipping context block {i} due to insufficient status.")

    human_message = HumanMessage(
        content=(
            "<CONTEXTS>\n"
            f"{contexts_formatted}\n"
            "</CONTEXTS>\n\n"
        )
    )
    
    # print(f"human message:\n{human_message.content}\n")

    all_messages = [system_message] + [human_message]

    response, metadata, invoke_time, tot_tokens, tot_input_tokens, tot_output_tokens, power_hardware_metadata = invoke_with_metadata(llm, all_messages, node_name, state)

    metadatas.append(metadata)
    invoke_times.append(invoke_time)
    power_hardware_metrics.append(power_hardware_metadata)

    context_filtered_and_formatted = response.content if response.content != "" else "No relevant context found."
    
    # print("CONTEXT FILTERED AND FORMATTED:\n\n")
    # print(context_filtered_and_formatted)
    # print()

    contexts_filtered_and_formatted = state.get("contexts_filtered_and_formatted", [])
    contexts_filtered_and_formatted.append(context_filtered_and_formatted)

    return {
        "contexts_filtered_and_formatted": contexts_filtered_and_formatted,
        "metadatas": metadatas,
        "invoke_times": invoke_times,
        "tot_tokens": tot_tokens,
        "tot_input_tokens": tot_input_tokens,
        "tot_output_tokens": tot_output_tokens,
        "power_hardware_metrics": power_hardware_metrics,
    }

def generate_code_node(state: State, llm):
    """Generates Python code based on the user's request and retrieved context."""
    print("generate_code_node")
    
    node_name = "generate_code"
    
    imports = state.get("imports", [])
    codes = state.get("codes", [])
    descriptions = state.get("descriptions", [])
    
    metadatas = state.get("metadatas", [])
    invoke_times = state.get("invoke_times", [])
    
    mode = state.get("agent_mode", "rag")
    
    power_hardware_metrics = state.get("power_hardware_metrics", [])

    system_message = generate_code_sys_prompt
    
    if mode == "0-shot":
        print("0-shot mode sys.")
        system_message = generate_code_sys_prompt_0_shot
    elif mode == "rag":
        print("RAG mode sys.")
        system_message = generate_code_sys_prompt
    
    message = state.get("messages")[-1].content
    
    if mode == "rag":
        contexts_filtered_and_formatted = state.get("contexts_filtered_and_formatted", [])
        
        human_message = HumanMessage(
            content=(
                "<CONTEXT>\n"
                f"{contexts_filtered_and_formatted}\n"
                "</CONTEXT>\n\n"
                "<USER_REQUEST>\n"
                f"{message}\n"
                "</USER_REQUEST>\n\n"
            )
        )
    elif mode == "0-shot":
        human_message = HumanMessage(
            content=(
                "<USER_REQUEST>\n"
                f"{message}\n"
                "</USER_REQUEST>\n\n"
            )
        )

    all_messages = [system_message] + [human_message]

    response, metadata, invoke_time, tot_tokens, tot_input_tokens, tot_output_tokens, power_hardware_metadata = invoke_with_metadata(llm, all_messages, node_name, state)

    metadatas.append(metadata)
    invoke_times.append(invoke_time)
    power_hardware_metrics.append(power_hardware_metadata)
    
    parsed_output = response['parsed']

    last_import = parsed_output.import_code
    last_code = parsed_output.code
    last_description = parsed_output.description

    print(f"Generated code:\n\n{last_import}\n\n{last_code}\n\n")
    
    imports.append(last_import)
    codes.append(last_code)
    descriptions.append(last_description)

    return {
        "imports": imports,
        "codes": codes,
        "descriptions": descriptions,
        "metadatas": metadatas,
        "invoke_times": invoke_times,
        "tot_tokens": tot_tokens,
        "tot_input_tokens": tot_input_tokens,
        "tot_output_tokens": tot_output_tokens,
        "power_hardware_metrics": power_hardware_metrics
    }

def code_check_node(state: State):
    """
    Check code and capture the output on success.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New keys added to state: error, tracebacks, execution_traces
    """

    print("code_check_node")

    # State
    error = "no" # Default to no error
    tracebacks = state.get("tracebacks", [])
    execution_traces = state.get("execution_traces", []) # <<< NEW: Get the list of successful traces

    # Get solution components
    imports = state["imports"]
    codes = state["codes"]
    
    if not imports or not codes:
        print("Warning: No imports or codes to execute!")
        traceback = "Imports and/or code are missing. You must generate both of them first!"
        tracebacks.append(traceback)
        execution_traces.append("No execution traces available because imports and/or code are missing.")
        
        return {
            "error": "yes", # Or you could set an error here if code is expected
            "tracebacks": tracebacks,
            "execution_traces": execution_traces,
        }
        
    last_import = imports[-1]
    last_code = codes[-1]

    # --- 1. Save the code to a local file ---
    full_code = last_import + "\n" + last_code
    
    os.makedirs("temp", exist_ok=True)
    temp_filename = "temp/temp.py"
    
    with open(temp_filename, "w") as f:
        f.write(full_code)

    # --- 2. Run the file using Docker ---
    try:
        host_script_path = os.path.abspath(temp_filename)
        container_script_path = "/sandbox/script.py"
        
        docker_command = [
            "docker", "run",
            "--rm",
            "-v", f"{host_script_path}:{container_script_path}:ro",
            "python-sandbox",
            "python", container_script_path
        ]

        result = subprocess.run(
            docker_command,
            capture_output=True,
            text=True,
            timeout=20
        )
        
        if result.returncode == 0:
            error = "no"
            successful_output = result.stdout
            execution_traces.append(successful_output) # Append stdout on success
            print("---CODE EXECUTED SUCCESSFULLY---")
            print(f"Output:\n{successful_output}")
        else:
            error = "yes"
            error_output = result.stderr
            tracebacks.append(error_output) # Append stderr on failure
            print(f"---CODE EXECUTION FAILED---\n{error_output}")

    except Exception as e:
        error_string = traceback.format_exc()
        print(f"---DOCKER EXECUTION FAILED---\n{error_string}")
        print(full_code)
        error = "yes"
        tracebacks.append(error_string)
    
    return {
        "error": error,
        "tracebacks": tracebacks,
        "execution_traces": execution_traces, 
    }
    
def code_error_diagnosis_node(state: State, llm):
    print("code_error_diagnosis")
    
    node_name = "code_error_diagnosis"
    
    metadatas = state.get("metadatas", [])
    invoke_times = state.get("invoke_times", [])
    power_hardware_metrics = state.get("power_hardware_metrics", [])
    agent_mode = state.get("agent_mode", "rag")

    imports = state.get("imports", [])
    codes = state.get("codes", [])
    descriptions = state.get("descriptions", [])
    
    last_description = descriptions[-1]
    last_import = imports[-1]
    last_code = codes[-1]

    tracebacks = state.get("tracebacks")
    last_traceback = tracebacks[-1] if tracebacks else "No tracebacks available!"
    
    system_message = code_diag_sys_message

    human_message = HumanMessage(
        content=(
            "<DESCRIPTION>\n"
            f"{last_description}\n"
            "</DESCRIPTION>\n\n"
            "<IMPORTS>\n"
            f"{last_import}\n"
            "</IMPORTS>\n\n"
            "<CODE>\n"
            f"{last_code}\n"
            "</CODE>\n\n"
            "<TRACEBACK>\n"
            f"{last_traceback}\n"
            "</TRACEBACK>\n\n"
        )
    )

    all_messages = [system_message] + [human_message]

    response, metadata, invoke_time, tot_tokens, tot_input_tokens, tot_output_tokens, power_hardware_metadata = invoke_with_metadata(llm, all_messages, node_name, state)

    metadatas.append(metadata)
    invoke_times.append(invoke_time)
    power_hardware_metrics.append(power_hardware_metadata)
    
    # print(response['parsed'])
    
    response_parsed = response['parsed']
    
    errorClassification = response_parsed.errorClassification
    rootCauseAnalysis = response_parsed.rootCauseAnalysis
    potentialRippleEffects = response_parsed.potentialRippleEffects
    pathToResolution = response_parsed.pathToResolution

    print()
    print(f"Error Classification: {errorClassification}")
    print()
    print(f"Root Cause Analysis: {rootCauseAnalysis}")
    print()
    print(f"Potential Ripple Effects: {potentialRippleEffects}")
    print()
    print(f"Path to Resolution: {pathToResolution}")
    print()
    
    errors_classification = state.get("errors_classification", [])
    errors_classification.append(errorClassification)
    
    if errorClassification == "library_specific" and agent_mode == "rag":
        doc_search_needed = "yes"
    else:
        doc_search_needed = "no"
        # contexts_filtered_and_formatted = state.get("contexts_filtered_and_formatted", [])
        # contexts_filtered_and_formatted.append("No documentation needed.")
    
    root_causes_analyses = state.get("root_causes_analyses", [])
    root_causes_analyses.append(rootCauseAnalysis)
    
    ripple_effects = state.get("ripple_effects", [])
    ripple_effects.append(potentialRippleEffects)
    
    paths_to_resolution = state.get("paths_to_resolution", [])
    paths_to_resolution.append(pathToResolution)

    return {
        "doc_search_needed": doc_search_needed,
        "errors_classification": errors_classification,
        "root_causes_analyses": root_causes_analyses,
        "ripple_effects": ripple_effects,
        "paths_to_resolution": paths_to_resolution,
        "metadatas": metadatas,
        "invoke_times": invoke_times,
        "tot_tokens": tot_tokens,
        "tot_input_tokens": tot_input_tokens,
        "tot_output_tokens": tot_output_tokens,
        "power_hardware_metrics": power_hardware_metrics
    }
    
def from_diag_to_queries_node(state: State, llm):
    print("from_diag_to_queries_node")
    node_name = "from_diag_to_queries"
    
    imports = state.get("imports", [])
    codes = state.get("codes", [])
    last_import = imports[-1] if imports else "No imports available."
    last_code = codes[-1] if codes else "No code available."
    full_code = last_import + "\n" + last_code
    
    metadatas = state.get("metadatas", [])
    invoke_times = state.get("invoke_times", [])
    power_hardware_metrics = state.get("power_hardware_metrics", [])
    
    errors_classification = state.get("errors_classification", [])
    last_error_classification = errors_classification[-1] if errors_classification else "No error classification available."
    root_causes_analyses = state.get("root_causes_analyses", [])
    last_root_cause_analysis = root_causes_analyses[-1] if root_causes_analyses else "No root cause analysis available."
    ripple_effects = state.get("ripple_effects", [])
    last_ripple_effect = ripple_effects[-1] if ripple_effects else "No ripple effect analysis available."
    paths_to_resolution = state.get("paths_to_resolution", [])
    last_path_to_resolution = paths_to_resolution[-1] if paths_to_resolution else "No path to resolution available."
    
    system_message = from_diag_to_queries_sys_message
    
    human_message = HumanMessage(
        content=(
            "<CURRENT_CODE>\n"
            f"{full_code}\n"
            "</CURRENT_CODE>\n\n"
            "<ERROR_CLASSIFICATION>\n"
            f"{last_error_classification}\n"
            "</ERROR_CLASSIFICATION>\n\n"
            "<ROOT_CAUSE_ANALYSIS>\n"
            f"{last_root_cause_analysis}\n"
            "</ROOT_CAUSE_ANALYSIS>\n\n"
            "<RIPPLE_EFFECTS>\n"
            f"{last_ripple_effect}\n"
            "</RIPPLE_EFFECTS>\n\n"
            "<PATH_TO_RESOLUTION>\n"
            f"{last_path_to_resolution}\n"
            "</PATH_TO_RESOLUTION>\n\n"
        )
    )
    
    all_messages = [system_message] + [human_message]
    
    response, metadata, invoke_time, tot_tokens, tot_input_tokens, tot_output_tokens, power_hardware_metadata = invoke_with_metadata(llm, all_messages, node_name, state)
    
    response_parsed = response['parsed']
    
    queries = response_parsed.search_queries
    
    print(f"Generated queries for doc search: {queries}")
    
    metadatas.append(metadata)
    invoke_times.append(invoke_time)
    power_hardware_metrics.append(power_hardware_metadata)
    
    errors_queries = state.get("errors_queries", [])
    errors_queries.append(queries)
    
    return {
        "errors_queries": errors_queries,
        "metadatas": metadatas,
        "invoke_times": invoke_times,
        "tot_tokens": tot_tokens,
        "tot_input_tokens": tot_input_tokens,
        "tot_output_tokens": tot_output_tokens,
        "power_hardware_metrics": power_hardware_metrics
    }

def fix_code_node(state: State, llm):
    """
    Fixes the code based on the error message.

    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to call
    """
    print("fix_code_node")
    
    node_name = "fix_code"
    
    mode = state.get("agent_mode", "rag")
    
    if mode == "0-shot":
        print("0-shot mode sys.")
        system_message = fix_code_sys_message_0_shot
    elif mode == "rag":
        print("RAG mode sys.")
        system_message = fix_code_sys_message
    
    imports = state.get("imports", [])
    codes = state.get("codes", [])
    descriptions = state.get("descriptions", [])
    
    metadatas = state.get("metadatas", [])
    invoke_times = state.get("invoke_times", [])
    power_hardware_metrics = state.get("power_hardware_metrics", [])

    last_import = imports[-1]
    last_code = codes[-1]
    last_description = descriptions[-1]

    iterations = state.get("iterations", 0)
    
    tracebacks = state["tracebacks"] if state["tracebacks"] else "No tracebacks available"
    
    last_traceback = tracebacks[-1]
    
    errors_classification = state.get("errors_classification", [])
    last_error_classification = errors_classification[-1] if errors_classification else "No error classification available."
    root_causes_analyses = state.get("root_causes_analyses", [])
    last_root_cause_analysis = root_causes_analyses[-1] if root_causes_analyses else "No root cause analysis available."
    ripple_effects = state.get("ripple_effects", [])
    last_ripple_effect = ripple_effects[-1] if ripple_effects else "No ripple effect analysis available."
    paths_to_resolution = state.get("paths_to_resolution", [])
    last_path_to_resolution = paths_to_resolution[-1] if paths_to_resolution else "No path to resolution available."
    
    last_diagnosis = ""
    last_diagnosis += f"- Root Cause Analysis: {last_root_cause_analysis}\n"
    last_diagnosis += f"- Potential Ripple Effects: {last_ripple_effect}\n"
    last_diagnosis += f"- Path to Resolution: {last_path_to_resolution}\n"

    if mode == "rag":
        context_for_code_fix = state.get("contexts_filtered_and_formatted", [])
        
        doc_search_needed = state.get("doc_search_needed", "no")
        
        if doc_search_needed == "yes" and len(context_for_code_fix) > 0:
            context_for_code_fix = context_for_code_fix[-1]
            print(f"Context for code fix available.\n")
            
            # print(f"Context for code fix:\n{context_for_code_fix}\n")
            # print()
            # print()
        else:
            print(f"No context for code fix available.\n")
            context_for_code_fix = "No additional context needed."

        human_message = HumanMessage(
            content=(
                "<CONTEXT>\n"
                f"{context_for_code_fix}\n"
                "</CONTEXT>\n\n"
                
                "<IMPORTS>\n"
                f"{last_import}\n"
                "</IMPORTS>\n\n"

                "<CODE>\n"
                f"{last_code}\n"
                "</CODE>\n\n"

                "<TRACEBACK>\n"
                f"{last_traceback}\n"
                "</TRACEBACK>\n\n"
                
                "<DIAGNOSIS>\n"
                f"{last_diagnosis}\n"
                "</DIAGNOSIS>\n\n"
                
                "Given the above context, code, traceback, and diagnosis, please provide the necessary fixes.\n"
            )
        )
    elif mode == "0-shot":
        human_message = HumanMessage(
            content=(
                "<DESCRIPTION>"
                f"{last_description}"
                "</DESCRIPTION>\n\n"

                "<IMPORTS>\n"
                f"{last_import}\n"
                "</IMPORTS>\n\n"

                "<CODE>\n"
                f"{last_code}\n"
                "</CODE>\n\n"

                "<TRACEBACK>\n"
                f"{last_traceback}\n"
                "</TRACEBACK>\n\n"
                
                "<DIAGNOSIS>\n"
                f"{last_diagnosis}\n"
                "</DIAGNOSIS>\n\n"
                
                "Given the above context, code, traceback, and diagnosis, please provide the necessary fixes.\n"
            )
        )
    
    # print(f"Human message for fix_code:\n{human_message.content}\n")
    all_messages = [system_message, human_message]

    response, metadata, invoke_time, tot_tokens, tot_input_tokens, tot_output_tokens, power_hardware_metadata = invoke_with_metadata(llm, all_messages, node_name, state)

    metadatas.append(metadata)
    invoke_times.append(invoke_time)
    power_hardware_metrics.append(power_hardware_metadata)

    output_parsed = response['parsed']
    
    last_import = output_parsed.import_code
    last_code = output_parsed.code
    last_reasoning = output_parsed.reasoning

    imports.append(last_import)
    codes.append(last_code)
    descriptions.append(last_description)
    
    fix_reasoning = state.get("fix_reasoning", [])
    fix_reasoning.append(last_reasoning)

    print()
    print(f"Description: {last_description}")
    print()
    print(f"Imports: {last_import}")
    print()
    print(f"Code:\n{last_code}")
    print()
    print(f"Reasoning for the fix:\n{last_reasoning}")
    print()
    
    return {
        "imports": imports,
        "codes": codes,
        "descriptions": descriptions,
        "fix_reasoning": fix_reasoning,
        "doc_search_needed": "no",
        "fixing": "no",
        "iterations": iterations + 1,
        "metadatas": metadatas,
        "invoke_times": invoke_times,
        "tot_tokens": tot_tokens,
        "tot_input_tokens": tot_input_tokens,
        "tot_output_tokens": tot_output_tokens,
        "power_hardware_metrics": power_hardware_metrics
    }
    
def code_qa_validator_node(state: State, llm):
    # checks for generated code, error or not and then semantically validates or grades the code
    
    print("code_qa_validator_node")
    
    node_name = "code_qa_validator"
    
    messages = state.get("messages", [])
    last_user_prompt = messages[-1].content if messages else "No user prompt available."
    metadatas = state.get("metadatas", [])
    invoke_times = state.get("invoke_times", [])
    power_hardware_metrics = state.get("power_hardware_metrics", [])
    
    imports = state.get("imports", [])
    codes = state.get("codes", [])
    
    last_import = imports[-1]
    last_code = codes[-1]
    
    full_code = last_import + "\n" + last_code
    
    error = state.get("error", "no")

    tracebacks = state.get("tracebacks", [])
    
    execution_traces = state.get("execution_traces", [])
    
    if error == "yes":
        last_traceback = tracebacks[-1] if tracebacks else "No tracebacks available!"
        last_execution_trace = "No execution traces available because the code did not execute successfully."
    else:
        last_traceback = "Empty traceback because code executed successfully."
        last_execution_trace = execution_traces[-1] if execution_traces else "No execution traces available."
    
    system_message = final_qa_validator_sys_message
    
    human_message = HumanMessage(
        content=(
            "<USER_PROMPT>\n"
            f"{last_user_prompt}"
            "</USER_PROMPT>\n\n"

            "<CODE>\n"
            f"{full_code}\n"
            "</CODE>\n\n"
            
            "<ERROR_FLAG>\n"
            f"{error}\n"
            "</ERROR_FLAG>\n\n"

            "<TRACEBACK>\n"
            f"{last_traceback}\n"
            "</TRACEBACK>\n\n"
            
            "<EXECUTION_OUTPUT>\n"
            f"{last_execution_trace}\n"
            "</EXECUTION_OUTPUT>\n"
        )
    )
    
    # print()
    # print(human_message.content)
    # print()
    
    all_messages = [system_message] + [human_message]
    
    response, metadata, invoke_time, tot_tokens, tot_input_tokens, tot_output_tokens, power_hardware_metadata = invoke_with_metadata(llm, all_messages, node_name, state)
    
    metadatas.append(metadata)
    invoke_times.append(invoke_time)
    power_hardware_metrics.append(power_hardware_metadata)
    
    parsed_output = response['parsed']
    
    compliance_status = parsed_output.compliance_status
    assessment_summary = parsed_output.assessment_summary
    
    print(f"Compliance Status: {compliance_status}")
    print(f"Assessment Summary: {assessment_summary}")
    
    return {
        "compliance_status": compliance_status,
        "assessment_summary": assessment_summary,
        "metadatas": metadatas,
        "invoke_times": invoke_times,
        "tot_tokens": tot_tokens,
        "tot_input_tokens": tot_input_tokens,
        "tot_output_tokens": tot_output_tokens,
        "power_hardware_metrics": power_hardware_metrics
    }

# ### Edges

def decide_to_go_to_qa_val(state: State):
    """
    Determines whether to proceed to code QA validation or go back to code diagnosis.
    
    Args:
        state (dict): The current graph state
    Returns:
        str: Next node to call
    """
    print("decide_to_go_to_qa_val")

    error = state["error"]
    iterations = state["iterations"]

    max_iterations = 5

    if error == "no" or iterations == max_iterations:
        print("Decision: code_qa_validator")
        return "code_qa_validator"
    else:
        print("Decision: code_diag")
        return "code_diag"
    
def decide_insufficient_retrieval_or_process_context(state: State):
    
    """
    Determines whether to perform another retrieval due to insufficient context or proceed to process context.

    Args:
        state (dict): The current graph state
    Returns:
        str: Next node to call
    """
    
    print("decide_insufficient_retrieval_or_process_context")

    there_is_an_insufficient_query = state.get("there_is_an_insufficient_query", False)

    if there_is_an_insufficient_query:
        print("Decision: generate_queries_for_insufficient_contexts")
        return "generate_queries_for_insufficient_contexts"
    else:
        print("Decision: process_chunks_to_create_context")
        return "process_chunks_to_create_context"

def decide_to_fix_or_query(state: State):
    """
    Determines whether to fix existing code or query for more information.

    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to call
    """
    print("decide_to_fix_or_query")

    if state.get("doc_search_needed", "no") == "yes":
        print("Decision: from_diag_to_queries")
        return "from_diag_to_queries"
    else:
        print("Decision: fix_code")
        return "fix_code"

def decide_to_generate_or_fix(state: State):
    """
    Determines whether to generate new code or fix existing code.

    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to call
    """
    
    print("decide_to_generate_or_fix")

    if state.get("doc_search_needed", "no") == "yes":
        print("Decision: fix_code")
        return "fix_code"
    else:
        print("Decision: generate_code")
        return "generate_code"

# --- GRAPH BUILDER ---

def create_graph_with_rag(llm, retriever, save_png=True):
    """Builds and compiles the LangGraph state machine."""
    
    llm_with_structure_for_prompt_queries = llm.with_structured_output(schema=FromUserPromptToListOfQueries, include_raw=True)
    llm_with_structure_for_code_gen = llm.with_structured_output(schema=CodeFormatter, include_raw=True)
    llm_with_structure_for_code_diag = llm.with_structured_output(schema=CodeDiagnosis, include_raw=True)
    llm_with_structure_for_code_qa = llm.with_structured_output(schema=FinalQAValidationReport, include_raw=True)
    llm_with_structure_for_distilled_context = llm.with_structured_output(schema=DistilledContext, include_raw=True)
    llm_with_structure_for_code_fix = llm.with_structured_output(schema=CodeHealerOutput, include_raw=True)
    llm_with_structure_for_diag_to_queries = llm.with_structured_output(schema=FromDiagToQueries, include_raw=True)
    
    graph_builder = StateGraph(State)

    # Bind models and retriever to the node functions
    generate_queries_from_user_prompt_with_llm = partial(generate_queries_from_user_prompt_node, llm=llm_with_structure_for_prompt_queries)
    retrieve_docs_for_user_with_retriever = partial(retrieve_docs_for_user_node, retriever=retriever, llm=llm_with_structure_for_distilled_context)
    process_chunks_to_create_context_with_llm = partial(process_chunks_to_create_context, llm=llm)
    generate_code_with_llm = partial(generate_code_node, llm=llm_with_structure_for_code_gen)
    code_diag_with_llm = partial(code_error_diagnosis_node, llm=llm_with_structure_for_code_diag)
    fix_code_with_llm = partial(fix_code_node, llm=llm_with_structure_for_code_fix)
    code_qa_validator_with_llm = partial(code_qa_validator_node, llm=llm_with_structure_for_code_qa)
    from_diag_to_queries_with_llm = partial(from_diag_to_queries_node, llm=llm_with_structure_for_diag_to_queries)
    generate_queries_for_insufficient_contexts_node_with_llm = partial(generate_queries_for_insufficient_contexts_node, llm=llm_with_structure_for_prompt_queries)

    graph_builder.add_node("generate_queries_from_user_prompt", generate_queries_from_user_prompt_with_llm)
    graph_builder.add_node("retrieve_docs_for_user", retrieve_docs_for_user_with_retriever)
    graph_builder.add_node("process_chunks_to_create_context", process_chunks_to_create_context_with_llm)
    graph_builder.add_node("generate_code", generate_code_with_llm)
    graph_builder.add_node("code_check", code_check_node)
    graph_builder.add_node("code_diag", code_diag_with_llm)
    graph_builder.add_node("fix_code", fix_code_with_llm)
    graph_builder.add_node("code_qa_validator", code_qa_validator_with_llm)
    graph_builder.add_node("from_diag_to_queries", from_diag_to_queries_with_llm)
    graph_builder.add_node("generate_queries_for_insufficient_contexts", generate_queries_for_insufficient_contexts_node_with_llm)
    
    graph_builder.add_edge(START, "generate_queries_from_user_prompt")
    graph_builder.add_edge("generate_queries_from_user_prompt", "retrieve_docs_for_user")
    graph_builder.add_edge("generate_queries_for_insufficient_contexts", "retrieve_docs_for_user")
    # graph_builder.add_edge("retrieve_docs_for_user", "process_chunks_to_create_context")
    # graph_builder.add_edge("process_chunks_to_create_context", "generate_code")
    graph_builder.add_edge("generate_code", "code_check")
    graph_builder.add_edge("fix_code", "code_check")
    graph_builder.add_edge("from_diag_to_queries", "retrieve_docs_for_user")
    graph_builder.add_edge("code_qa_validator", END)

    graph_builder.add_conditional_edges(
        "code_check",
        decide_to_go_to_qa_val,
        {
            "code_qa_validator": "code_qa_validator",
            "code_diag": "code_diag",
        },
    )
    
    graph_builder.add_conditional_edges(
        "retrieve_docs_for_user",
        decide_insufficient_retrieval_or_process_context,
        {
            "generate_queries_for_insufficient_contexts": "generate_queries_for_insufficient_contexts",
            "process_chunks_to_create_context": "process_chunks_to_create_context",
        },
    )
    
    graph_builder.add_conditional_edges(
        "code_diag",
        decide_to_fix_or_query,
        {
            "fix_code": "fix_code",
            "from_diag_to_queries": "from_diag_to_queries",
        },
    )
    
    graph_builder.add_conditional_edges(
        "process_chunks_to_create_context",
        decide_to_generate_or_fix,
        {
            "generate_code": "generate_code",
            "fix_code": "fix_code",
        },
    )

    compiled_graph = graph_builder.compile()
        
    return compiled_graph

def create_graph_with_0_shot(llm):
    """Builds and compiles the LangGraph state machine for 0-shot code generation."""
    
    llm_with_structure_for_code_gen = llm.with_structured_output(schema=CodeFormatter, include_raw=True)
    llm_with_structure_for_code_diag = llm.with_structured_output(schema=CodeDiagnosis, include_raw=True)
    llm_with_structure_for_code_qa = llm.with_structured_output(schema=FinalQAValidationReport, include_raw=True)
    llm_with_structure_for_code_fix = llm.with_structured_output(schema=CodeHealerOutput, include_raw=True)
    
    graph_builder = StateGraph(State)
    
    generate_code_with_llm = partial(generate_code_node, llm=llm_with_structure_for_code_gen)
    code_diag_with_llm = partial(code_error_diagnosis_node, llm=llm_with_structure_for_code_diag)
    fix_code_with_llm = partial(fix_code_node, llm=llm_with_structure_for_code_fix)
    code_qa_validator_with_llm = partial(code_qa_validator_node, llm=llm_with_structure_for_code_qa)
    
    graph_builder.add_node("generate_code", generate_code_with_llm)
    graph_builder.add_node("code_check", code_check_node)
    graph_builder.add_node("code_diag", code_diag_with_llm)
    graph_builder.add_node("fix_code", fix_code_with_llm)
    graph_builder.add_node("code_qa_validator", code_qa_validator_with_llm)
    
    graph_builder.add_edge(START, "generate_code")
    graph_builder.add_edge("generate_code", "code_check")
    graph_builder.add_edge("fix_code", "code_check")
    
    graph_builder.add_conditional_edges(
        "code_check",
        decide_to_go_to_qa_val,
        {
            "code_qa_validator": "code_qa_validator",
            "code_diag": "code_diag",
        },
    )
    
    graph_builder.add_conditional_edges(
        "code_diag",
        decide_to_fix_or_query,
        {
            "fix_code": "fix_code",
            "retrieve_docs_for_user": "fix_code", # In 0-shot, we don't retrieve docs, so we go to fix_code
        },
    )
    
    compiled_graph = graph_builder.compile()
    
    return compiled_graph