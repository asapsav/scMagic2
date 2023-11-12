import pandas as pd
import requests
import chromadb
import tiktoken

# Load data
df = pd.read_csv('tableExport-2.csv')

# Function to fetch README from GitHub
def get_readme(url):
    if pd.isna(url) or url == "":
        return ""
    try:
        readme_url = url.replace('github.com', 'raw.githubusercontent.com').rstrip('/') + '/master/README.md'
        response = requests.get(readme_url)
        return response.text if response.status_code == 200 else ""
    except Exception:
        return ""

# Add README content to DataFrame
df['Readme'] = df['Code'].apply(get_readme)

# Initialize ChromaDB client
chroma_client = chromadb.EphemeralClient()

# Set up embedding function
embedding_function = chromadb.utils.embedding_functions.OpenAIEmbeddingFunction(api_key='', model_name="text-embedding-ada-002")
scrnatools_description_collection = chroma_client.create_collection(name='scRNA_Tools', embedding_function=embedding_function)

# Token calculation function
def num_tokens_from_string(string: str, model_name: str = "gpt-3.5-turbo"):
    encoding = tiktoken.encoding_for_model(model_name)
    return len(encoding.encode(string))

# Extend description with README
def make_ext_desc(row):
    desc = f"Platform: {row['Platform']}\n Description: {row['Description']}\n Categories: {row['Categories']}"
    return f"{desc}\n Readme: {row['Readme'][:22000]}" if pd.notna(row['Readme']) else desc

df['extented_desc'] = df.apply(make_ext_desc, axis=1)
df['tokens_in_ext_desc'] = df['extented_desc'].apply(num_tokens_from_string)

# Assertions and checks
assert df.isna().sum()['extented_desc'] == 0
assert df['tokens_in_ext_desc'].sum() < 100000
assert df.Name.nunique() == df.shape[0]

# Add to ChromaDB collection
scrnatools_description_collection.add(
    documents=list(df['extented_desc']),
    metadatas=df.drop(['extented_desc'], axis=1).to_dict(orient='records'),
    ids=list(df.Name)
)

# Query function
def query_collection(collection, query, max_results, dataframe):
    results = collection.query(query_texts=query, n_results=max_results, include=['distances'])
    return pd.DataFrame({
        'id': results['ids'][0],
        'score': results['distances'][0],
        'content': dataframe[dataframe.Name.isin(results['ids'][0])]['extented_desc'],
        'platform': dataframe[dataframe.Name.isin(results['ids'][0])]['Platform'],
    })

# Example query
print(query_collection(scrnatools_description_collection, 'quality control python', 5, df))
