import streamlit as st
import nbformat
from io import BytesIO
import tiktoken
import dotenv
import os
import openai
from nbconvert import HTMLExporter
import chromadb
import pandas as pd

from prompts import PLANNER_AGENT_MINDSET

dotenv.load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

SUGGESTOR_SYSTEM_PROMPT = """ You are an expert bioinformatitian specialised in scRNA-seq analysis. Your answers are short and specific because you are a scientist.
You take an input of a jupyter notebook and a tool description and you suggest 
the next step in the analysis and write code for it. If the description of tool is absent, make suggestion without it."""

# df = pd.read_csv('tableExport-2.csv')
# # embed vectors
# chroma_client = chromadb.EphemeralClient() # Equivalent to chromadb.Client(), ephemeral.
# # Uncomment for persistent client
# # chroma_client = chromadb.PersistentClient()
# EMBEDDING_MODEL = "text-embedding-ada-002"
# # change this to biotech specialised model later
# from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
# embedding_function = OpenAIEmbeddingFunction(api_key=openai.api_key, model_name=EMBEDDING_MODEL)
# scrnatools_description_collection = chroma_client.create_collection(name='scRNA_Tools_2', embedding_function=embedding_function)
# scrnatools_description_collection.add(
#     documents = list(df['extented_desc_readme_trim']),
#     metadatas = df.drop(['extented_desc_readme_trim'], axis = 1).to_dict(orient='records'),
#     ids = list(df.Name))

# # Query DB
# def query_collection(collection, query, max_results, dataframe):
#     results = collection.query(query_texts=query, n_results=max_results, include=['distances'])
#     df = pd.DataFrame({
#                 'id':results['ids'][0],
#                 'score':results['distances'][0],
#                 'content': dataframe[dataframe.Name.isin(results['ids'][0])]['extented_desc_readme_trim'],
#                 'platform': dataframe[dataframe.Name.isin(results['ids'][0])]['Platform'],
#                 })

#     return df['content'].tolist() , df['id'].tolist()

#print(query_collection(scrnatools_description_collection, 'quality controll python', 5, df))

# Placeholder function for querying the vector database
def query_collection(collection, query, max_results , dataframe):
    #Implement querying logic here
    #Return top 3 tools
   return  ['scanpy', 'mnnpy', 'scRNASequest'], ['scanpy', 'mnnpy', 'scRNASequest']

# OpenAI querying logic here
def get_code_suggestions(notebook_content, tool='', selected_model='gpt-3.5-turbo'):

    messages = [
        {"role": "system", "content": SUGGESTOR_SYSTEM_PROMPT}
    ]
    
    SUGGESTOR_CONTEXT = f"scRNA-seq analysis notebook right now: {str(notebook_content)}. \
        Description of a suggested tool to use: {str(tool)}. Give outout in a format: 1) <why this tool is usefull>. \
            2) <Write code for the next step in the analysis using that tool given my current notebook>."  # Corrected variable name

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
    #top_tools_desc, top_tools_names = query_collection(scrnatools_description_collection, notebook_content_nocode, 3, df)
    top_tools_desc, top_tools_names = query_collection('scrnatools_description_collection', notebook_content_nocode, 3, 'df')
    #st.write(notebook_content_nocode)
    # Create three columns for suggestions
    col1, col2, col3 = st.columns(3)
    tools_columns = [col1, col2, col3]

    for index, tool_desc, tool_name in zip([0, 1, 2], top_tools_desc, top_tools_names):
        with tools_columns[index]:
            st.subheader(f'Suggestion: {tool_name}')
            suggestions = get_code_suggestions(notebook_content_nocode, tool_desc, selected_model)
            st.write(suggestions)
