import openai

openai.api_key = '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'

caption = ""
with open('pizza_caption_v1', mode='r') as f:
    for line in f:
        caption += ' ' + line

question = 'What is the chef cooking in the video ?'
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

print(reply)
