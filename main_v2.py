import streamlit as st
import nbformat
from io import BytesIO
import tiktoken
import dotenv
import os
import openai
from nbconvert import HTMLExporter

from prompts import PLANNER_AGENT_MINDSET

dotenv.load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

SUGGESTOR_SYSTEM_PROMPT = """ You are an expert bioinformatitian specialised in scRNA-seq analysis. Your answers are short and specific because you are a scientist.
You take an input of a jupyter notebook and a tool description and you suggest 
the next step in the analysis and write code for it. If the description of tool is absent, make suggestion without it."""

# Placeholder function for querying the vector database
def query_vector_db(notebook_content):
    # Implement querying logic here
    # Return top 3 tools
    return  ['Tool1', 'Tool2', 'Tool3']

# OpenAI querying logic here
def get_code_suggestions(notebook_content, tool='', selected_model='gpt-3.5-turbo'):

    messages = [
        {"role": "system", "content": SUGGESTOR_SYSTEM_PROMPT}
    ]
    
    SUGGESTOR_CONTEXT = f"scRNA-seq analysis notebook right now: {str(notebook_content)}. \
        Description of a suggested tool to use: {str(tool)}. 1) Explain why this tool can be usefull. \
            2) Write code for the next step in the analysis using that tool."  # Corrected variable name

    try:
        messages.append({"role": "user", "content": SUGGESTOR_CONTEXT})
        
        completion = openai.ChatCompletion.create(
            model=selected_model,
            messages=messages
        )
        
        assistant_message = completion.choices[0].message['content']  # Corrected attribute access
        messages.append({"role": "assistant", "content": assistant_message})
        return assistant_message

    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None

def num_tokens_from_string(string: str, model_name: str = "gpt-3.5-turbo") -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model(model_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens
    
from nbformat.notebooknode import NotebookNode

def remove_code_cells(notebook: NotebookNode) -> NotebookNode:
    """
    Removes all code cells from a Jupyter notebook.

    :param notebook: The Jupyter notebook object.
    :return: A new notebook object with all code cells removed.
    """
    # Create a deep copy of the notebook to avoid modifying the original
    new_notebook = nbformat.v4.new_notebook()
    new_notebook.cells = [cell for cell in notebook.cells if cell.cell_type != 'code']
    return new_notebook

st.set_page_config(layout="wide")
st.title('scRNA-seq copilot')

# Sidebar for model selection
model_options = ["gpt-3.5-turbo", "gpt-4-1106-preview", "gpt-3", "gpt-3.5"]
selected_model = st.sidebar.selectbox("Choose the OpenAI model", model_options)

st.sidebar.write(f"You selected: {selected_model}")

# Center: upload and display the notebook
uploaded_file = st.file_uploader("Upload Jupyter Notebook", type="ipynb")
if uploaded_file is not None:
    notebook_content = nbformat.read(uploaded_file, as_version=4)
    notebook_content_nocode = remove_code_cells(notebook_content)
    st.write(f"Notebook size without code: {num_tokens_from_string(str(notebook_content_nocode))} tokens")
    # Display the notebook content (or render using nbconvert)
    st.text('<notebook was gonna be displayed here>')  # Placeholder for notebook display
    # Convert notebook to HTML for better display
    # html_exporter = HTMLExporter()
    # html_exporter.exclude_input = True  # Exclude code cells if needed
    # notebook_html, _ = html_exporter.from_notebook_node(notebook_content_nocode)
    # st.markdown(notebook_html, unsafe_allow_html=True)

# Generate next step and display suggestions
if uploaded_file and st.button('Generate next step in the analysis'):
    top_tools = query_vector_db(notebook_content_nocode)
    
    # Create three columns for suggestions
    col1, col2, col3 = st.columns(3)
    tools_columns = [col1, col2, col3]

    for index, tool in enumerate(top_tools):
        with tools_columns[index]:
            st.subheader(f'Suggestion: {tool}')
            suggestions = get_code_suggestions(notebook_content_nocode, tool, selected_model)
            st.write(suggestions)
