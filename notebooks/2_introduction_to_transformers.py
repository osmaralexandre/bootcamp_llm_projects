# %%[markdown]
## Transformers

# %%
# Imports
from transformers import pipeline

# %%
# 1) Task: text-generation
text_generator = pipeline(
    task="text-generation", model="pierreguillou/gpt2-small-portuguese"
)

text = "Liberalismo é uma corrente política e moral baseada na liberdade, consentimento dos governados e igualdade perante a lei."
results = text_generator(text, max_length=60, do_sample=True)
print(results)

# %%
# 2) Task: question-answering
qa = pipeline(
    task="question-answering",
    model="pierreguillou/bert-base-cased-squad-v1.1-portuguese",
)

text = "Os liberais procuraram e estabeleceram uma ordem constitucional que valorizava liberdades individuais importantes, como liberdade de expressão e liberdade de associação; um judiciário independente e um julgamento público por júri; assim como a abolição dos privilégios aristocráticos."
question = "Quais são as liberdades individuais importantes?"
answer = qa(question=question, context=text)
print("Question: ", question)
print("Answer: ", answer["answer"])
print("Score: ", answer["score"])

# %%
# 3) Task: fill-mask
mask = pipeline(task="fill-mask", model="neuralmind/bert-base-portuguese-cased")

text = mask("A luta pela segurança tende a ser mais forte do que o [MASK] à liberdade.")
for x in range(len(text)):
    print(text[x])

# %%
# 4) Task: summarization
summarizer = pipeline(task="summarization")

text = "Hayek nasceu na Áustria no ano de 1899 e foi uma mente brilhante que, embora seja mais conhecido por ser um grande economista, recebendo a mais alta e relevante premiação da área, o Prêmio Nobel da Economia em 1974, serviu de inspiração para inúmeros estudiosos e amantes da filosofia, psicologia, ciências sociais e direito. É considerado por muitos o maior filósofo da liberdade do século XX."
summary = summarizer(text, max_length=100, min_length=50)
print(summary)

# %%
