import ollama

res = ollama.chat(
    model='llama3.2',
    messages=[
        {'role':'user', 
         "content": 'i had an accident what should i do?'
        }
    ],
    stream=True
)
for chunk in res:
    print(chunk['message']['content'], end='', flush=True)

# res = ollama.generate(
#     model='llama3.2',
#     prompt='hello!'
# )
# print(res['response'])

