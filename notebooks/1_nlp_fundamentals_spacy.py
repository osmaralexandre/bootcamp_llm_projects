# %%[markdown]
## NLP with spacy

import subprocess

# %%
# Imports
import spacy

# %%
# Load the model
try:
    nlp = spacy.load("pt_core_news_lg")
except OSError:
    # If it's not installed
    subprocess.run(["python", "-m", "spacy", "download", "pt_core_news_lg"])
    nlp = spacy.load("pt_core_news_lg")

# %%
# NLP Pipeline
print(nlp.pipe_names)

# %%
# Pass the full text to the model
document = nlp(
    "Liberdade significa não somente que o indivíduo tenha tanto a oportunidade quanto o fardo da escolha; significa também que ele deve arcar com as consequências de suas ações. Liberdade e responsabilidade são inseparáveis."
)

# %%
# Length of vocabulary
len(document.vocab)

# %%
# 1. Sentence Segmentation
print("Sentences:")
for sent in document.sents:
    print(f"--> {sent}")

# %%
# 2. Tokenization
print("\nTokens:")
for sent in document.sents:
    tokens = [token.text for token in sent]
    print(tokens)

# %%
# 3. Stop Words
print("\nStop Words:")
for sent in document.sents:
    stop_words = [stop_word.text for stop_word in sent if stop_word.is_stop]
    print(stop_words)

# %%
# Tokens without Stop Words
print("\nTokens without Stop Words:")
for sent in document.sents:
    tokens = [token.text for token in sent if not token.is_stop]
    print(tokens)

# %%
# 4. Lemmatization
print("\nLemmas:")
for sent in document.sents:
    lemmas = [token.lemma_ for token in sent]
    print(lemmas)

# %%
# 5. Part-Of-Speech Tagging (POS)
print("\nPOS Tags:")
for sent in document.sents:
    pos_tags = [(token.text, token.pos_) for token in sent]
    print(pos_tags)

# %%
# 6. Dependency Parsing
print("\nDependency Parsing:")
for sent in document.sents:
    dependencies = [(token.text, token.dep_, token.head.text) for token in sent]
    print(dependencies)

# %%
# 7. Named Entity Recognition (NER)
print("\nNamed Entities:")
for ent in document.ents:
    print(ent.text, ent.label_)

# %%
# Some good methods
print("Tokens: ", [token.text for token in document])
print("Stop word: ", [token.is_stop for token in document])
print("Alphanumeric: ", [token.is_alpha for token in document])
print("Upper case: ", [token.is_upper for token in document])
print("Punctuation: ", [token.is_punct for token in document])
print("Number: ", [token.like_num for token in document])
print("Initial sentence: ", [token.is_sent_start for token in document])
print("Formato: ", [token.shape_ for token in document])

# %%
