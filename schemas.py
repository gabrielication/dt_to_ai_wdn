# schemas.py

from typing import Optional, List, Annotated
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langchain_core.documents import Document
from pydantic import BaseModel, Field
from langgraph.graph.message import add_messages
from typing import Literal
from enum import Enum
from langchain_core.callbacks import UsageMetadataCallbackHandler

import json

class DistilledContext(BaseModel):
    """
    Defines the structured output for the RAG Context Distillation Agent.
    """
    status: Literal["sufficient", "insufficient"] = Field(
        ...,
        description="Indicates whether the provided chunks contained enough information to answer the query."
    )
    context: str = Field(
        ...,
        description="If status is 'sufficient', this contains the clean, Markdown-formatted context. If 'insufficient', it contains an explanation of what was missing."
    )

class FinalQAValidationReport(BaseModel):
    """
    Defines the structured output for the WNTR Code Compliance Validator.
    """
    compliance_status: Literal["FULL", "PARTIAL", "EXCESSIVE", "FAILED"] = Field(
        ...,
        description="A categorical assessment of how well the code meets the user's request. 'FULL' for complete, 'PARTIAL' for missing requirements, 'EXCESSIVE' for including unrequested features, and 'FAILED' for code that did not execute successfully."
    )
    assessment_summary: str = Field(
        ...,
        description="A concise, one-sentence natural language summary that justifies the compliance_status. It should explain the core reason for the assessment."
    )

class FromUserPromptToListOfQueries(BaseModel):
    """Generates one or more targeted search queries from a user's prompt."""
    
    queries: List[str] = Field(
        ...,
        min_length=1, # Enforces that the list must have at least one item.
        # max_length=3,
        description="A list of one or more specific and targeted RAG queries derived from the user's prompt. This list must not be empty."
    )

class CodeHealerOutput(BaseModel):
    """
    Defines the structured output for the CodeHealer agent.
    This model ensures that the LLM's response is a valid JSON object
    containing the corrected code and a clear explanation of the fix.
    """
    import_code: str = Field(
        description="The complete and corrected block of import statements."
    )
    code: str = Field(
        description="The complete and corrected body of the code, without imports."
    )
    reasoning: str = Field(
        description="A clear explanation of the error's root cause, the fix applied, and the justification."
    )

class CodeFormatter(BaseModel):
    """
    A structured container for code-generation.

    Fields:
        description: A one-line natural language description of the code.
        import_code: The full block of import statements required.
        code: The complete correct code (excluding imports).
    """
    description: str = Field(..., description="One phrase describing the code.")
    import_code: str = Field(..., description="All import statements required for the code snippet.")
    code: str = Field(..., description="The correct code snippet (excluding imports).")
    
class CodeDiagnosis(BaseModel):
    """A structured representation of a Python code diagnosis."""

    errorClassification: Literal['generic_python', 'library_specific'] = Field(
        ...,
        description="The classification of the error, either 'generic_python' or 'library_specific'."
    )
    
    rootCauseAnalysis: str = Field(
        ...,
        description="A detailed explanation of what went wrong, where it went wrong, and why."
    )
    
    potentialRippleEffects: str = Field(
        ...,
        description="An analysis of other parts of the code that may suffer from a similar logical error."
    )
    
    pathToResolution: str = Field(
        ...,
        description="Guiding questions and concepts to help the user investigate and find the solution on their own."
    )

# Custom JSON encoder to save Langchain states
class LangChainJSONEncoder(json.JSONEncoder):
    """
    A custom JSON encoder that handles modern LangChain/Pydantic objects.
    """
    def default(self, obj):
        if isinstance(obj, BaseMessage):
            # Use the modern .model_dump() method
            return obj.model_dump()
        if isinstance(obj, Document):
            # Use the modern .model_dump() method
            return obj.model_dump()
        # For any other object types, fall back to the default encoder
        return super().default(obj)

class FromDiagToQueries(BaseModel):
    """A structured representation of search queries for a vector store."""

    search_queries: List[str] = Field(
        ...,
        min_length=1,
        max_length=3,
        description="A list of targeted search queries designed to retrieve relevant documentation from the WNTR vector store. Maximum of 3 queries."
    )

class State(TypedDict):
    """Defines the state for the graph, tracking the conversation and generated artifacts."""
    llm_name: str # To track which LLM model was used
    complete_llm_path: str # Full path of the LLM model used (including provider and format)
    prompt_index: int # To track which prompt template was used during the testing
    agent_mode: str # "rag" or "0-shot"
    temperature: float
    top_p: float
    
    messages: Annotated[List[BaseMessage], add_messages] # Conversation history
    
    queries_from_user_prompt: List[str] # Queries generated from the user prompt
    errors_queries: List[str] # Queries generated from the error analysis
    
    there_is_an_insufficient_query: bool # flag to indicate if there was an insufficient query in the last retrieval
    how_many_insufficient_retrials: int # counter of how many times there was an insufficient query
    insufficient_contexts_for_each_query: list # status and context for each insufficient query generated
    new_queries_for_insufficient_retrievals: List[str] # new queries generated for insufficient contexts

    status_contexts_for_each_query: list # status and context for each query generated
    contexts: List[List[Document]] # Retrieved documents for each query
    contexts_filtered_and_formatted: List[str] # Filtered content formatted for LLM input
    
    fixing: str # "yes" or "no" to indicate if in fixing mode

    imports: List[str] # List of import statements generated
    codes: List[str] # List of code snippets generated
    descriptions: List[str] # List of descriptions for each code snippet

    error: str # error happened: yes/no
    iterations: int # number of fixing iterations done
    tracebacks: List[str] # List of tracebacks from code execution attempts
    
    errors_classification: List[str] # "generic_python" or "library_specific"
    root_causes_analyses: List[str] # detailed explanation of what, where, why
    ripple_effects: List[str] # analysis of other parts of the code that may suffer
    paths_to_resolution: List[str] # guiding questions and concepts to help the user investigate and
    
    doc_search_needed: str # "yes" or "no" if doc search is needed for the current error
    fix_reasoning: List[str] # reasoning for the fix applied in the current iteration
    
    metadatas: List[dict] # token metadata for each invoke in each node
    tot_tokens: int # total tokens used in the graph execution
    tot_input_tokens: int # total input tokens used in the graph execution
    tot_output_tokens: int # total output tokens used in the graph execution
    
    invoke_times: List[dict] # timing information for each invoke
    retrieval_times: List[dict] # timing information for document retrieval
    
    filter_only_relevant_docs: bool # whether to filter out not relevant and partially relevant documents
    filter_docs_only_with_reranker: bool # whether to filter documents only with the reranker
    
    execution_traces: List[str] # list of successful traces if the code executed correctly
    
    # --- Validation Report ---
    compliance_status: Optional[str] # FINAL: FULL, PARTIAL, EXCESSIVE, FAILED
    assessment_summary: Optional[str] # FINAL: summary of the compliance status
    
    power_hardware_metrics: List[dict] # list of power metrics collected during the execution
    
    callback_handler: Optional[UsageMetadataCallbackHandler] # callback handler to collect token usage metadata