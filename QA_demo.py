import openai

openai.api_key = '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'


prompt = 'What is the animal given the following text a lovely fat cat sitting in the couch lovely fat cat sitting in ' \
         'the couch'

response = openai.Completion.create(
    engine='text-davinci-003',  # Specify the GPT-3 engine
    prompt=prompt,
    max_tokens=100  # Specify the desired length of the generated completion
)

completion_text = response.choices[0].text.strip()
print(completion_text)
