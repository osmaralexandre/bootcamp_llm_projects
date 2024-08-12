# %%[markdown]
## NLP with NLTK

# %%
# Imports
import nltk
from nltk import ne_chunk
from nltk.corpus import stopwords
from nltk.stem import (
    LancasterStemmer,
    PorterStemmer,
    SnowballStemmer,
    WordNetLemmatizer,
)
from nltk.tag import pos_tag, pos_tag_sents
from nltk.tokenize import sent_tokenize, word_tokenize

nltk.download("stopwords")
nltk.download("punkt")
nltk.download("tagsets")
nltk.download("wordnet")
nltk.download("averaged_perceptron_tagger")
nltk.download("maxent_ne_chunker")
nltk.download("words")

# %%
# Text
text = "Freedom means not only that the individual has both the opportunity and the burden of choice; it also means that they must bear the consequences of their actions. Freedom and responsibility are inseparable."

# %%
# 1. Sentence tokenize
sentences = sent_tokenize(text, language="english")
print(type(sentences))
print(sentences)
print(f"number of sentence tokens: {len(sentences)}")

# %%
# 2. Word tokenize
words = word_tokenize(text, language="english")
print(words)
print(f"number of word tokens: {len(words)}")

# %%
# 3. Stop Words
all_stop_words = set(stopwords.words("english"))
stop_words = [word for word in words if word.lower() in all_stop_words]
print("Stop Words:", stop_words)

# %%
# Tokens without Stop Words
filtered_words = [word for word in words if word.lower() not in all_stop_words]
print("Tokens without Stop Words:", filtered_words)

# %%
# 4.1 Stemming with PorterStemmer
porter_stemmer = PorterStemmer()
porter_stemmed_words = [porter_stemmer.stem(word) for word in filtered_words]
print("Words after Stemming:", porter_stemmed_words)

# %%
# 4.2 Stemming with SnowballStemmer
snowball_stemmer = SnowballStemmer(language="english")
snowball_stemmed_words = [snowball_stemmer.stem(word) for word in filtered_words]
print("Words after Stemming:", snowball_stemmed_words)

# %%
# 4.3 Stemming with LancasterStemmer
lancaster_stemmer = LancasterStemmer()
lancaster_stemmed_words = [lancaster_stemmer.stem(word) for word in filtered_words]
print("Words after Stemming:", lancaster_stemmed_words)

# %%
# 5. Lemmatization
lemmatizer = WordNetLemmatizer()
lemmatized_words = [lemmatizer.lemmatize(word) for word in filtered_words]
print("Words after Lemmatization:", lemmatized_words)

# %%
# 6. Part-Of-Speech Tagging (POS)
tagged_words = pos_tag(words, lang="eng")
print("POS Tagging:", tagged_words)

# %%
# 7. Named Entity Recognition (NER)
named_entities = ne_chunk(tagged_words)
print("Named Entity:", named_entities)

# %%
