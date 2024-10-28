from openai import OpenAI
client = OpenAI()

completion = client.chat.completions.create(
  model="ft:gpt-4o-mini-2024-07-18:personal::ANL9adYM",
  messages=[
    {"role": "system", "content": "You are an expert chess player."},
    {"role": "user", "content": "1. e4"}
  ]
)
print(completion.choices[0].message)