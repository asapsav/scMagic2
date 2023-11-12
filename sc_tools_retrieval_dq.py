import pandas as pd
import openai
import chromadb
import numpy as np
import requests

df = pd.read_csv('tableExport-2.csv')

import requests
import pandas as pd

# Assuming df is your existing DataFrame

# Define a function to fetch README content
def get_readme(url):
    if pd.isna(url) or url == "":
        return ""
    try:
        # Construct URL to the raw README file
        readme_url = url.replace('github.com', 'raw.githubusercontent.com').rstrip('/') + '/master/README.md'
        response = requests.get(readme_url)
        if response.status_code == 200:
            return response.text
        else:
            return ""
    except Exception:
        # Handle exceptions
        return ""

# Counter for printing status
counter = 0

# Apply the function to each row in the DataFrame
for index, row in df.iterrows():
    df.at[index, 'Readme'] = get_readme(row['Code'])

    # Increment the counter
    counter += 1
    if counter % 100 == 0:
        print(f"Processed {counter} rows.")

chroma_client = chromadb.EphemeralClient() # Equivalent to chromadb.Client(), ephemeral.
# Uncomment for persistent client
# chroma_client = chromadb.PersistentClient()

EMBEDDING_MODEL = "text-embedding-ada-002"
# change this to biotech specialised model later

from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction


embedding_function = OpenAIEmbeddingFunction(api_key='', model_name=EMBEDDING_MODEL)

scrnatools_description_collection = chroma_client.create_collection(name='scRNA_Tools', embedding_function=embedding_function)

import tiktoken
def num_tokens_from_string(string: str, model_name: str = "gpt-3.5-turbo") -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model(model_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def make_desc(row):
    # Example function using row 'A' and 'B'
    return 'Platform: ' + row['Platform'] + \
            '\n Description: ' + row['Description'] + \
            '\n Categories: ' + row['Categories']

df['extented_desc'] = df.apply(make_desc, axis=1)

def make_desc(row):
    # Example function using row 'A' and 'B'
    return 'Platform: ' + row['Platform'] + \
            '\n Description: ' + row['Description'] + \
            '\n Categories: ' + row['Categories'] + \
            '\n Readme: ' + row['Readme']

df['extented_desc_readme'] = df.apply(make_desc, axis=1)

assert df.isna().sum()['extented_desc'] == 0 # check for nans in desc to make embedding collection

df['tokens_in_ext_desc'] = df['extented_desc'].apply(num_tokens_from_string)
df.to_csv('tableExport-2.csv', index = False)

#df['tokens_in_ext_desc'].hist()

df['extented_desc_readme_trim'] = df['extented_desc_readme'].apply(lambda x: x[:22000] if pd.notna(x) else x)

df['tokens_in_ext_desc_readme_trim'] = df['extented_desc_readme_trim'].apply(num_tokens_from_string)

#df['tokens_in_ext_desc_readme_trim'].hist(bins = 50)

assert df['tokens_in_ext_desc'].sum() < 100000
# as if Nov 7 2023 allshort  descriptions of tools is jsut 68K tokens, so
# no need for embed db truly speaking, can also just make claude calls every time need to pick a tool

assert df.Name.nunique() == df.shape[0]

df['extented_desc'].isna().sum()

df['Code'].isna().sum()

df['Readme'].isna().sum()

df['Readme'].head(20)

df['Readme'] = df['Readme'].apply(lambda x: x.replace('\n', ' ').replace('\r', ' ').replace("'", "\\'") if pd.notna(x) else x)

df.to_csv('tableExport-2.csv', index = False)
print('Saved the DF')

"""Embedd Vectors"""

# Add the content vectors
scrnatools_description_collection.add(
    documents = list(df['extented_desc']),
    metadatas = df.drop(['extented_desc'], axis = 1).to_dict(orient='records'),
    ids = list(df.Name)
)

scrnatools_description_collection.add(
    documents = list(df['extented_desc_readme_trim']),
    metadatas = df.drop(['extented_desc_readme_trim'], axis = 1).to_dict(orient='records'),
    ids = list(df.Name))

"""Query DB"""

def query_collection(collection, query, max_results, dataframe):
    results = collection.query(query_texts=query, n_results=max_results, include=['distances'])
    df = pd.DataFrame({
                'id':results['ids'][0],
                'score':results['distances'][0],
                'content': dataframe[dataframe.Name.isin(results['ids'][0])]['extented_desc'],
                'platform': dataframe[dataframe.Name.isin(results['ids'][0])]['Platform'],
                })

    return df

print(query_collection(scrnatools_description_collection, 'quality controll python', 5, df))

