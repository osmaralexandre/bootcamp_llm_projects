# %%[markdown]
## Word embedings

# %%
# Imports
import pandas as pd
import tiktoken
from random import randint

# %%[markdown]
# Simple encoding by letters:

# %%
# Sorting the letters
text_letter = "Freedom is the best way to make the world better"
characters = sorted(list(set(text_letter)))
print("Vocabulary >> ", characters)
print("The size of vocabulary >> ", len(characters))

# %%
# Mapping the vocabulary in a dictionary
letter_to_index = {lt: i for i, lt in enumerate(characters)}
index_to_letter = {i: lt for i, lt in enumerate(characters)}

encode = lambda s: [letter_to_index[c] for c in s]
decode = lambda l: "".join([index_to_letter[i] for i in l])

print("letter_to_index >> ", letter_to_index)
print("index_to_letter >> ", index_to_letter)

# %%
# Encoding and decoding a sample text
sample_text = "the best way"
print("Encoding a text >> ", encode(sample_text))
print("Decoding a text >> ", decode(encode(sample_text)))

# %%[markdown]
# Simple encoding by words:

# %%
# Texts
text_word = {
    "text": [
        "Freedom is the best way to make the world better.",
        "One of the best ways to make the world better is to defend freedom.",
    ]
}
text_word = pd.DataFrame(text_word)
text_word

# %%
# Building an occurrence matrix
# That's not a good encoding because it creates a very large sparse matrix.
text_word["text"].str.get_dummies(" ")

# %%[markdown]
# An efficient tokenization:

# %%
# Getting the GPT2 encoding using tiktoken
enc = tiktoken.get_encoding("gpt2")
enc.encode("Freedom is the best way to make the world better")

# %%
# Our improvised encoding by letters
# The diffenrence in size is too big!
print(encode("Freedom is the best way to make the world better"))

# %%[markdown]
# The difference between GPT and BERT:

# %%
# Text
text = encode("Freedom is the best way to make the world better")
print("Length >> ", len(text))
print("Text encoded by letters>> ", text)

# %%
# GPT
for i in range(len(text) - 1):
    x = text[:i]

    y = text[i]
    if x != []:
        print(f"When the data is: {x}, the target is {y}")

# %%
# BERT - bidirectional sequencing
for i in range(len(text)):
    x = text
    y = text.copy()
    idx_mask = randint(0, len(text) - 1)
    y[idx_mask] = "<mask>"
    print(f"When the data is: {x}, the target is {y}")

# %%
