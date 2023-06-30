import openai

openai.api_key = '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'

caption = ""
with open('tested_captions/pizza_caption_v2', mode='r') as f:
    for line in f:
        caption += ' ' + line

# question = 'What is the chef is coking ?'
question = 'What is the ingredients used in the pizza ?'
prompt = f"the following text is video caption answer the following question:  {question} using the provided " \
         f"caption: {caption}"

messages = [{
    'role': 'user',
    'content': prompt
}]

chat = openai.ChatCompletion.create(
    model="gpt-3.5-turbo", messages=messages
)
reply = chat.choices[0].message.content
print(f'Question: {question}')
print(f'Answer: {reply}')
