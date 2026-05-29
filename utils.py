# utils.py

import json
import bs4
from langchain.chat_models import init_chat_model
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.document_loaders import WebBaseLoader
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.storage import InMemoryStore
from langchain.retrievers import ParentDocumentRetriever
import pickle
from pathlib import Path

import re

import docker
from docker.errors import BuildError, ImageNotFound, APIError

import os
import shutil

from schemas import LangChainJSONEncoder

import unicodedata
from typing import Iterable, List


import getpass

def load_llm(llm_name, temperature=0.0, top_p=1.0, model_provider="ollama"):
    """Initializes and returns the specified chat model."""
    print(f"Loading LLM: {llm_name}")
    print("LLM parameters:")
    print(f" - Temperature: {temperature}")
    print(f" - Top P: {top_p}")
    print(f" - Model Provider: {model_provider}")

    if model_provider == "openai":
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "Missing environment variable: OPENAI_API_KEY.\n"
            )

    return init_chat_model(
        llm_name,
        model_provider=model_provider,
        temperature=temperature,
        top_p=top_p,
    )

FUNC_OR_SECT = re.compile(
    r'^(?P<title>[A-Za-z_][\w\.]*\s*\([^)]*\)|[A-Z][\w\s:/-]+)\s*(?:\[\s*source\s*\])?#\s*$'
)
PROPERTY = re.compile(r'^\s*property\s+([A-Za-z_][\w\.]*)#\s*$')
PARAM_LINE = re.compile(r'^\s*([-\*]?\s*)([A-Za-z_][\w\.]*)\s*\(([^)]+)\)\s*[–-]\s*(.+)$')

def normalize_unicode(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    return text.replace("\u00A0", " ")

def strip_source_anchors(line: str) -> str:
    # e.g., "get_pump_curve()[source]#" -> "get_pump_curve()"
    line = re.sub(r'\s*\[\s*source\s*\]\s*#\s*$', '', line)
    line = re.sub(r'#\s*$', '', line)
    return line

def normalize_headings(line: str) -> str:
    # Functions/sections → "### Title"
    if FUNC_OR_SECT.match(line.strip()):
        clean = strip_source_anchors(line).strip()
        return f"### {clean}"
    # Properties → "### property: name"
    m = PROPERTY.match(line)
    if m:
        return f"### property: {m.group(1)}"
    return line

def collapse_blank_runs(text: str) -> str:
    # trim trailing spaces
    text = re.sub(r'[ \t]+$', '', text, flags=re.MULTILINE)
    # collapse >2 blank lines to exactly 2
    text = re.sub(r'\n{3,}', '\n\n', text)
    # collapse internal multiple spaces (not at line starts)
    text = re.sub(r'(?<!\n)[ ]{2,}', ' ', text)
    return text

def join_hard_wraps(lines: List[str]) -> List[str]:
    """
    Join lines that were broken in the middle of sentences.
    Heuristics:
      - keep blank lines (paragraphs)
      - keep lines that look like headers/lists as-is
      - otherwise, if a line ends w/o terminal punctuation and the next starts lowercase/number, join with a space
    """
    out = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            out.append(line)
            i += 1
            continue

        # header/list/code heuristics: don't join
        is_blocky = (
            line.lstrip().startswith(('#', '-', '*', '>', '`')) or
            line.strip().endswith(':') or
            FUNC_OR_SECT.match(line.strip()) or
            PROPERTY.match(line.strip())
        )
        if is_blocky or i == len(lines) - 1:
            out.append(line)
            i += 1
            continue

        nxt = lines[i + 1]
        # if next is blank, leave break
        if not nxt.strip():
            out.append(line)
            i += 1
            continue

        # decide to join
        end_char = line.rstrip()[-1]
        next_first = nxt.lstrip()[:1]
        should_join = (
            end_char not in '.:;?!' and
            next_first and next_first.islower()
        )
        if should_join:
            # merge with a space
            merged = line.rstrip() + ' ' + nxt.lstrip()
            lines[i + 1] = merged
            i += 1  # reprocess merged line against the following line
        else:
            out.append(line)
            i += 1
    return out

def normalize_blocks(lines: List[str]) -> List[str]:
    """
    Normalize common blocks:
      - "Returns:" → bold label
      - "Return type:", "Type:", "Parameters:" sections
      - Parameter lines like: name (Type) – description → "- name (Type): description"
    """
    out = []
    for line in lines:
        s = line.strip()

        # Upgrade headings for known labels
        if s.lower() in {"returns:", "parameters:", "return type:", "type:"}:
            out.append(f"**{s[:-1].title()}:**")
            continue

        # Parameter bullet normalization
        m = PARAM_LINE.match(s)
        if m:
            _, name, ty, desc = m.groups()
            out.append(f"- **{name}** (*{ty}*): {desc}")
            continue

        # Single-line lightweight normalizations
        if s.endswith("[source]#"):
            s = strip_source_anchors(s)

        out.append(normalize_headings(s))
    return out

def clean_page_text(text: str) -> str:
    text = normalize_unicode(text)
    text = collapse_blank_runs(text)

    # split, normalize headings, then re-join for wrap fixing
    lines = [normalize_headings(ln) for ln in text.splitlines()]
    lines = normalize_blocks(lines)

    # re-collapse blank runs after block normalization
    text2 = "\n".join(lines)
    text2 = collapse_blank_runs(text2)

    # join hard wraps (mid-sentence breaks)
    lines2 = join_hard_wraps(text2.splitlines())
    text3 = "\n".join(lines2)

    # final tidy
    text3 = collapse_blank_runs(text3).strip()
    return text3

def preprocess_docs(docs: Iterable[Document]) -> List[Document]:
    cleaned: List[Document] = []
    for d in docs:
        cleaned.append(
            Document(
                page_content=clean_page_text(d.page_content),
                metadata=d.metadata.copy(),
            )
        )
    return cleaned

def load_wntr_docs_from_web() -> list[Document]:
    """Loads and splits documents from the web URLs."""
    print("load_wntr_docs_from_web")
    
    # Data Loading Configuration
    WEB_URLS = (
        "https://usepa.github.io/WNTR/getting_started.html",
        "https://usepa.github.io/WNTR/waternetworkmodel.html",
        "https://usepa.github.io/WNTR/model_io.html",
        "https://usepa.github.io/WNTR/controls.html",
        "https://usepa.github.io/WNTR/networkxgraph.html",
        "https://usepa.github.io/WNTR/layers.html",
        "https://usepa.github.io/WNTR/options.html",
        # "https://usepa.github.io/WNTR/libraries.html",
        "https://usepa.github.io/WNTR/hydraulics.html",
        "https://usepa.github.io/WNTR/waterquality.html",
        "https://usepa.github.io/WNTR/resultsobject.html",
        # "https://usepa.github.io/WNTR/disaster_models.html",
        "https://usepa.github.io/WNTR/resilience.html",
        # "https://usepa.github.io/WNTR/fragility.html",
        # "https://usepa.github.io/WNTR/morph.html",
        "https://usepa.github.io/WNTR/graphics.html",
        "https://usepa.github.io/WNTR/gis.html",
        "https://usepa.github.io/WNTR/advancedsim.html",

        "https://usepa.github.io/WNTR/apidoc/wntr.epanet.io.InpFile.html",
        
        "https://usepa.github.io/WNTR/apidoc/wntr.network.controls.Control.html",
        "https://usepa.github.io/WNTR/apidoc/wntr.network.controls.ControlAction.html",
        "https://usepa.github.io/WNTR/apidoc/wntr.network.controls.ControlBase.html",
        "https://usepa.github.io/WNTR/apidoc/wntr.network.elements.Valve.html",
        "https://usepa.github.io/WNTR/apidoc/wntr.network.elements.TimeSeries.html",
        "https://usepa.github.io/WNTR/apidoc/wntr.network.controls.TimeOfDayCondition.html",
        
        "https://usepa.github.io/WNTR/apidoc/wntr.network.base.Link.html",
        "https://usepa.github.io/WNTR/apidoc/wntr.network.base.LinkStatus.html",
        "https://usepa.github.io/WNTR/apidoc/wntr.network.base.LinkType.html",
        "https://usepa.github.io/WNTR/apidoc/wntr.network.base.Node.html",
        "https://usepa.github.io/WNTR/apidoc/wntr.network.base.NodeType.html",
        "https://usepa.github.io/WNTR/apidoc/wntr.network.elements.HeadPump.html",
        # "https://usepa.github.io/WNTR/apidoc/wntr.library.demand_library.html",
        # "https://usepa.github.io/WNTR/apidoc/wntr.library.model_library.html",
        "https://usepa.github.io/WNTR/apidoc/wntr.epanet.util.ControlType.html",
        "https://usepa.github.io/WNTR/apidoc/wntr.epanet.util.FlowUnits.html",
        "https://usepa.github.io/WNTR/apidoc/wntr.epanet.util.FormulaType.html",
        "https://usepa.github.io/WNTR/apidoc/wntr.epanet.util.LinkTankStatus.html",
        "https://usepa.github.io/WNTR/apidoc/wntr.epanet.util.MassUnits.html",
        "https://usepa.github.io/WNTR/apidoc/wntr.metrics.hydraulic.average_expected_demand.html",
        "https://usepa.github.io/WNTR/apidoc/wntr.metrics.hydraulic.entropy.html",
        "https://usepa.github.io/WNTR/apidoc/wntr.metrics.hydraulic.expected_demand.html",
        "https://usepa.github.io/WNTR/apidoc/wntr.metrics.hydraulic.tank_capacity.html",
        "https://usepa.github.io/WNTR/apidoc/wntr.network.elements.Curve.html",
        "https://usepa.github.io/WNTR/apidoc/wntr.network.elements.Demands.html",
        "https://usepa.github.io/WNTR/apidoc/wntr.network.elements.FCValve.html",
        "https://usepa.github.io/WNTR/apidoc/wntr.network.elements.GPValve.html",
        "https://usepa.github.io/WNTR/apidoc/wntr.network.elements.HeadPump.html",
        "https://usepa.github.io/WNTR/apidoc/wntr.network.elements.Pump.html",
        "https://usepa.github.io/WNTR/apidoc/wntr.network.elements.Junction.html",
        "https://usepa.github.io/WNTR/apidoc/wntr.network.elements.Pipe.html",
        "https://usepa.github.io/WNTR/apidoc/wntr.network.elements.Reservoir.html",
        "https://usepa.github.io/WNTR/apidoc/wntr.network.elements.Tank.html",
        "https://usepa.github.io/WNTR/apidoc/wntr.network.elements.PBValve.html",
        "https://usepa.github.io/WNTR/apidoc/wntr.network.elements.PRValve.html",
        "https://usepa.github.io/WNTR/apidoc/wntr.network.elements.PSValve.html",
        "https://usepa.github.io/WNTR/apidoc/wntr.network.elements.Pattern.html",
        "https://usepa.github.io/WNTR/apidoc/wntr.network.elements.PowerPump.html",
        "https://usepa.github.io/WNTR/apidoc/wntr.network.elements.Source.html",
        "https://usepa.github.io/WNTR/apidoc/wntr.network.elements.TCValve.html",
        "https://usepa.github.io/WNTR/apidoc/wntr.network.io.read_inpfile.html",
        "https://usepa.github.io/WNTR/apidoc/wntr.network.io.to_graph.html",
        "https://usepa.github.io/WNTR/apidoc/wntr.network.io.write_inpfile.html",
        "https://usepa.github.io/WNTR/apidoc/wntr.network.model.CurveRegistry.html",
        "https://usepa.github.io/WNTR/apidoc/wntr.network.model.LinkRegistry.html",
        "https://usepa.github.io/WNTR/apidoc/wntr.network.model.NodeRegistry.html",
        "https://usepa.github.io/WNTR/apidoc/wntr.network.model.PatternRegistry.html",
        "https://usepa.github.io/WNTR/apidoc/wntr.network.model.SourceRegistry.html",
        "https://usepa.github.io/WNTR/apidoc/wntr.network.model.WaterNetworkModel.html",
        "https://usepa.github.io/WNTR/apidoc/wntr.network.options.HydraulicOptions.html",
        "https://usepa.github.io/WNTR/apidoc/wntr.network.options.Options.html"
        "https://usepa.github.io/WNTR/apidoc/wntr.network.options.TimeOptions.html",
        "https://usepa.github.io/WNTR/apidoc/wntr.network.options.ReportOptions.html",
        "https://usepa.github.io/WNTR/apidoc/wntr.sim.core.WNTRSimulator.html",
        "https://usepa.github.io/WNTR/apidoc/wntr.sim.epanet.EpanetSimulator.html",
        "https://usepa.github.io/WNTR/apidoc/wntr.sim.core.WaterNetworkSimulator.html",
        "https://usepa.github.io/WNTR/apidoc/wntr.sim.hydraulics.get_results.html",
        "https://usepa.github.io/WNTR/apidoc/wntr.sim.results.SimulationResults.html",
        "https://usepa.github.io/WNTR/apidoc/wntr.sim.results.ResultsStatus.html",
        "https://usepa.github.io/WNTR/apidoc/wntr.graphics.color.custom_colormap.html",
        "https://usepa.github.io/WNTR/apidoc/wntr.graphics.curve.plot_fragility_curve.html",
        "https://usepa.github.io/WNTR/apidoc/wntr.graphics.curve.plot_pump_curve.html",
        "https://usepa.github.io/WNTR/apidoc/wntr.graphics.curve.plot_tank_volume_curve.html",
        "https://usepa.github.io/WNTR/apidoc/wntr.graphics.network.plot_network.html",
    )
    
    print("Loading documents from web...")
    loader = WebBaseLoader(
        web_paths=WEB_URLS,
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(class_=("bd-article"))
        ),
    )
    
    docs = loader.load()

    clean_docs = preprocess_docs(docs)

    return clean_docs

def create_retriever_and_embedder(
    embedding_model_name: str,
    cache_folder: str = "./hf_cache",
    collection_name: str = "rag",
    persist_directory: str = "./chroma_db",
    docstore_path: str = "./chroma_db/docstore.pkl",
    k: int = 10,
    chunk_size: int = 512,
    chunk_overlap: int = 128,
    child_chunk_size: int = 300,
    child_chunk_overlap: int = 50,
    rebuild: bool = False,
):
    """
    Creates a retriever from a Chroma vector store.
    - If `rebuild` is False and a DB exists, reuse it.
    - If `rebuild` is True, delete existing DB and re-embed docs.
    """
    print("create_retriever")
    print(f"embedding_model_name: {embedding_model_name}")
    print(f"cache_folder: {cache_folder}")
    print(f"collection_name: {collection_name}")
    print(f"persist_directory: {persist_directory}")
    print(f"docstore_path: {docstore_path}")
    print(f"k: {k}")
    print(f"chunk_size: {chunk_size}")
    print(f"chunk_overlap: {chunk_overlap}")
    print(f"child_chunk_size: {child_chunk_size}")
    print(f"child_chunk_overlap: {child_chunk_overlap}")
    print(f"rebuild: {rebuild}")
    
    # Handle rebuild: delete existing database and docstore if present
    if rebuild:
        persist_path = Path(persist_directory)
        if persist_path.exists():
            print(f"Rebuild flag set. Deleting existing database at {persist_directory}...")
            shutil.rmtree(persist_directory)
            print("Existing database deleted.")
        
        docstore_file = Path(docstore_path)
        if docstore_file.exists():
            print(f"Deleting existing docstore at {docstore_path}...")
            docstore_file.unlink()
            print("Existing docstore deleted.")
    
    print("Using OllamaEmbeddings")
    embedding_model = OllamaEmbeddings(model=embedding_model_name)
    
    # Create persist directory if it doesn't exist
    os.makedirs(persist_directory, exist_ok=True)
    
    # Initialize Chroma with persistence
    child_vs = Chroma(
        collection_name=collection_name,
        embedding_function=embedding_model,
        persist_directory=persist_directory,
    )
    
    # Check if we need to rebuild or if the DB is empty
    existing_count = child_vs._collection.count()
    docstore_exists = Path(docstore_path).exists()
    should_add_docs = rebuild or existing_count == 0 or not docstore_exists
    
    # Create splitters
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=child_chunk_size,
        chunk_overlap=child_chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    
    if should_add_docs:
        print(f"{'Rebuilding' if rebuild else 'Building'} vector store and docstore...")
        
        # Initialize new docstore
        docstore = InMemoryStore()
        
        # 1. Load web documentation
        print("Loading web docs...")
        web_docs = load_wntr_docs_from_web()
        
        # Create retriever
        retriever = ParentDocumentRetriever(
            vectorstore=child_vs,
            docstore=docstore,
            child_splitter=child_splitter,
            parent_splitter=parent_splitter,
            search_kwargs={"k": k}
        )
        
        # Add documents (this will chunk and embed as needed)
        print("Adding documents to vector store...")
        retriever.add_documents(web_docs)
        print(f"Added {len(web_docs)} documents to vector store.")
        
        # Persist docstore to disk
        print(f"Saving docstore to {docstore_path}...")
        with open(docstore_path, 'wb') as f:
            pickle.dump(docstore.store, f)
        print("Docstore saved.")
    else:
        print(f"Reusing existing vector store with {existing_count} documents.")
        
        # Load docstore from disk
        print(f"Loading docstore from {docstore_path}...")
        with open(docstore_path, 'rb') as f:
            store_data = pickle.load(f)
        
        docstore = InMemoryStore()
        docstore.store = store_data
        print(f"Docstore loaded with {len(store_data)} entries.")
        
        retriever = ParentDocumentRetriever(
            vectorstore=child_vs,
            docstore=docstore,
            child_splitter=child_splitter,
            parent_splitter=parent_splitter,
            search_kwargs={"k": k}
        )
    
    return retriever

def create_reranker_retriever(retriever, model_name="BAAI/bge-reranker-v2-m3", top_n=3):
    print("Using reranker in RAG process")
    print(f"Reranker model: {model_name}")
    print(f"Top-n documents to keep after reranking: {top_n}")
    
    print("Using HuggingFaceCrossEncoder...")
    model = HuggingFaceCrossEncoder(model_name=model_name)
    compressor = CrossEncoderReranker(model=model, top_n=top_n)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=retriever
    )
    
    return compression_retriever
    
def save_code_to_file(code, description, imports, llm_name, id="0", mode="rag"):
    """Saves the generated code to a specified file."""
    print("--- Saving Generated Code ---")
    
    full_code = (
        f"# Description: {description}\n\n"
        f"{imports}\n\n"
        f"{code}"
    )
    
    llm_name_to_save = llm_name.replace(":", "-")  # Ensure valid filename

    path_to_save = f"results/{llm_name_to_save}/code/{mode}/"
    os.makedirs(path_to_save, exist_ok=True)

    filename = f"generated_script_{llm_name_to_save}_{id}.py"
    file_path = os.path.join(path_to_save, filename)

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(full_code)
    
    print(f"Code successfully saved to: {file_path}")

def save_query_to_file(query, llm_name, id="0"):
    """Saves the generated query to a specified file."""
    print("--- Saving Generated Query ---")
    
    llm_name_to_save = llm_name.replace(":", "-")  # Ensure valid filename

    path_to_save = f"results/{llm_name_to_save}/queries/"
    os.makedirs(path_to_save, exist_ok=True)

    filename = f"generated_query_{llm_name_to_save}_{id}.txt"
    file_path = os.path.join(path_to_save, filename)

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(query)
    
    print(f"Query successfully saved to: {file_path}")
    
def save_retrieved_docs_to_file(docs, llm_name, id="0"):
    """Saves the retrieved documents to a specified file."""
    print("--- Saving Retrieved Documents ---")
    
    llm_name_to_save = llm_name.replace(":", "-")  # Ensure valid filename

    path_to_save = f"results/{llm_name_to_save}/retrieved_docs/"
    os.makedirs(path_to_save, exist_ok=True)

    filename = f"retrieved_docs_{llm_name_to_save}_{id}.txt"
    file_path = os.path.join(path_to_save, filename)

    with open(file_path, "w", encoding="utf-8") as f:
        for doc in docs:
            f.write(f"--- Document ---\n{doc.page_content}\n\n")
    
    print(f"Retrieved documents successfully saved to: {file_path}")

def save_times_to_csv(query_time, retrieval_time, code_gen_time, code_fix_times, llm_name, id="0", mode="rag"):
    """Saves the timing information to a CSV file."""
    import csv
    import os
    
    llm_name_to_save = llm_name.replace(":", "-")  # Ensure valid filename

    path_to_save = f"results/{llm_name_to_save}/timings/{mode}/"
    os.makedirs(path_to_save, exist_ok=True)

    filename = f"timings_{llm_name_to_save}.csv"
    file_path = os.path.join(path_to_save, filename)

    file_exists = os.path.isfile(file_path)
    
    with open(file_path, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["ID", "LLM Name", "Query Time", "Retrieval Time", "Code Gen Time", "Code Fix Times"])
        writer.writerow([id, llm_name_to_save, query_time, retrieval_time, code_gen_time, code_fix_times])
    
    print(f"Timing information successfully saved to: {file_path}")

def save_graph_to_png(compiled_graph, file_path="graph_visualization.png"):
    """
    Renders a compiled LangGraph graph to a PNG file locally.

    This function uses the Pyppeteer draw method to avoid making external API
    calls to mermaid.ink, making it suitable for offline use or environments
    where the API might be blocked.

    Args:
        compiled_graph: The compiled LangGraph object.
        file_path (str): The path where the PNG file will be saved.
    """
    print("--- Saving Graph Visualization as PNG (Local Rendering) ---")
    from langchain_core.runnables.graph_mermaid import MermaidDrawMethod
    
    try:
        # Generate the PNG image data in bytes using the local Pyppeteer method.
        # This requires the 'pyppeteer' library to be installed.
        png_bytes = compiled_graph.get_graph().draw_mermaid_png(
            draw_method=MermaidDrawMethod.PYPPETEER
        )

        # Open the file in binary write mode and save the image
        with open(file_path, "wb") as f:
            f.write(png_bytes)
        
        print(f"Graph saved successfully to {file_path}")
    except Exception as e:
        # This can still fail if pyppeteer is not installed or configured correctly.
        print(f"Error saving graph locally: {e}")
        print("\nPlease ensure you have 'pyppeteer' installed: `pip install pyppeteer`")
        pass

def save_errors_to_file(state, llm_name, id="0", mode="rag"):
    error = state.get("error", "no")
    iterations = state.get("iterations", 0)
    tracebacks = state.get("tracebacks", [])
    
    llm_name_to_save = llm_name.replace(":", "-")  # Ensure valid filename
    
    path_to_save = f"results/{llm_name_to_save}/errors/{mode}/"
    os.makedirs(path_to_save, exist_ok=True)
    filename = f"errors_{llm_name_to_save}_{id}.txt"
    file_path = os.path.join(path_to_save, filename)
    
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(f"Error: {error}\n")
        f.write(f"Iterations: {iterations}\n")
        if tracebacks:
            f.write("Tracebacks:\n")
            for tb in tracebacks:
                f.write(f"{tb}\n")

def image_exists(image_tag):
    """Checks if a Docker image exists locally."""
    try:
        client = docker.from_env()
        # Ping the Docker daemon to ensure it's running
        client.ping() 
    except (APIError, docker.errors.DockerException):
        print("Error: Could not connect to the Docker daemon. Is it running?")
        exit(1)
        
    try:
        client.images.get(image_tag)
        print(f"Image '{image_tag}' found locally.")
        return True
    except ImageNotFound:
        print(f"Image '{image_tag}' not found locally.")
        return False

def build_image(path, image_tag):
    """Builds a Docker image from a Dockerfile."""
    print(f"Building image '{image_tag}'...")
    
    try:
        client = docker.from_env()
        # Ping the Docker daemon to ensure it's running
        client.ping() 
    except (APIError, docker.errors.DockerException):
        print("Error: Could not connect to the Docker daemon. Is it running?")
        exit(1)
    
    try:
        image, build_log = client.images.build(
            path=path,
            dockerfile="Dockerfile",
            tag=image_tag,
            quiet=True,  # Suppress the build output
            rm=True  # Remove intermediate containers after a successful build
        )
        
        # Stream and print the build log
        for chunk in build_log:
            if 'stream' in chunk:
                print(chunk['stream'], end='')
        print(f"Successfully built image '{image_tag}'")
        return image
    except BuildError as e:
        print(f"Error building image: {e}")
        # Print the full build log on error
        for chunk in e.build_log:
            if 'stream' in chunk:
                print(chunk['stream'], end='')
        return None
    
def save_state_to_json(state):

    llm_name = state.get("llm_name", "unknown")
    prompt_index = state.get("prompt_index", "unknown")
    agent_mode = state.get("agent_mode", "unknown")
    temperature = state.get("temperature", 0.0)
    top_p = state.get("top_p", 1.0)
    
    temp_top_p = f"top_p_{top_p}_temp_{temperature}".replace(".", "_")

    file_path = f"final_state_prompt_{prompt_index}_{temp_top_p}.json"

    path_to_save = f"results/{llm_name}/{agent_mode}/"
    os.makedirs(path_to_save, exist_ok=True)
    
    file_path = os.path.join(path_to_save, file_path)

    # Create a shallow copy to avoid changing the original state dict
    state_to_save = state.copy()
    
    # Remove the 'callback_handler' key. 
    # Use .pop() with a default (None) to avoid an error if the key doesn't exist.
    state_to_save.pop("callback_handler", None)

    with open(file_path, 'w', encoding='utf-8') as f:
        # Dump the modified copy to the JSON file
        json.dump(state_to_save, f, indent=4, cls=LangChainJSONEncoder)

    print(f"Final state saved to {file_path}")