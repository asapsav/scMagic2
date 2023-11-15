# scMagic2
Vector DB of software tools for scRNA-seq analysis

## How to use
1. Use `make_vectordb.py` to create and query a vector database of scRNA-esq tools
2. Run `steamlit run test_vectordb.py` to use vector db as coding recomendatin engine for your scRNA-seq analysis Jupyter Notebook

## Some opportunitiy for optimisation
- a clear metric of why it is more usefull then ~gpt-4 is not yet defined
- only tools description and readme files are embedded
- generic embedding model is used

## Roadmap
- [x] Get database of scRNA-seq tools and enrich it with tools documentation
- [x] Create vector database of scRNA-seq tools
- [x] Create a simple query engine for vector database
- [x] Create a simple web app to use vector database as coding recomendation engine
- [x] Make web app use vector DB directly
- [ ] Make live connection with [source data](https://github.com/scRNA-tools/scRNA-tools/tree/master/database)
- [ ] Optimise vector DB for scEvals by utisiling citation count, usage context from citations, full documentstion, other
 metadata, etc.
- [ ] Experiment with custom distance metrics, embedding methods and other search/ranking algorithms. 

## Acknowledgements
* data from [scrna-tool.org](https://www.scrna-tools.org)

## Miscelanious
[google doc with plan](https://docs.google.com/document/d/1Hldune730uqvTMbDne8wDymYPwhZkFch6T9RnzBfVa4/edit)