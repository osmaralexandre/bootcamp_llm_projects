# %%
# !pip install git+https://github.com/vioshyvo/mrpt/
# !pip install vectordb2
# !pip install langchain

# %%
import requests
import re
from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from vectordb import Memory

# %%
raw_text = requests.get(
    "https://raw.githubusercontent.com/abjur/constituicao/main/CONSTITUICAO.md"
).text
raw_text

# %%
chapter_exp = r"^##\s+(.*)$"
sections = re.split(chapter_exp, raw_text, flags=re.MULTILINE)
sections = [section.strip() for section in sections[1:]]
sections

# %%
chapter_pathern = [("##", "Capitulo")]
markdown_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=chapter_pathern
)
sections = markdown_splitter.split_text(raw_text)
sections

# %%
# Testando os chunks
chunk_size = 20
chunk_overlap = 3

text = "A república federativa do Brasil, formada pela união indissolúvel dos Estados e Municípios e do Distrito Federal"

splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size, chunk_overlap=chunk_overlap
)

chunks = splitter.split_text(text)

for chunk in chunks:
    print(chunk)

# %%
memory = Memory(
    chunking_strategy={
        "mode": "sliding_window",
        "window_size": 128,
        "overlap": 8,
    }
)
# %%
for i in range(0, len(sections)):
    chapter = sections[i].metadata
    text = sections[i].page_content

    metadata = {"capitulo": chapter, "origem": "constituicao federal"}

    memory.save(text, metadata)

# %%
memory.search("direitos dos trabalhadores", top_n=2)

# %%
