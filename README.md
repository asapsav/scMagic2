# scMagic2
Vector DB of software tools for scRNA-seq analysis

# How to use
1. Use `make_vectordb.py` to create and query a vector database of scRNA-esq tools
2. Run `steamlit run test_vectordb.py` to use vector db as coding recomendatin engine for your scRNA-seq analysis Jupyter Notebook

# Known Limitations
- a clear metric of why it is more usefull then ~gpt-4 is not yet defined
- t

# Roadmap
- [x] Get database of scRNA-seq tools and enrich it with tools documentation
- [x] Create vector database of scRNA-seq tools
- [x] Create a simple query engine for vector database
- [x] Create a simple web app to use vector database as coding recomendation engine
- [ ] Make web app use vector DB directly
- [ ] Make live connection with [source data](https://github.com/scRNA-tools/scRNA-tools/tree/master/database)
- [ ] Optimise vector DB for scEvals by utisiling citation count, usage context from citations, full documentstion, other metadata, etc.

# Acknowledgements
* data from [scrna-tool.org](https://www.scrna-tools.org)

# Miscelanious
[google doc with plan](https://docs.google.com/document/d/1Hldune730uqvTMbDne8wDymYPwhZkFch6T9RnzBfVa4/edit)