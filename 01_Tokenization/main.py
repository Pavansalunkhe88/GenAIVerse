import tiktoken

# create an encoder
enc = tiktoken.encoding_for_model("gpt-4o")

# text we wants to encode
text = "Hey my name is pavan salunkhe"

# tokens and prints the tokens
tokens=enc.encode(text)
print("tokens",tokens)
# tokens [25216, 922, 1308, 382, 275, 24803, 2370, 4060, 273]

decoded=enc.decode([25216, 922, 1308, 382, 275, 24803, 2370, 4060, 273])
print(decoded)
# Hey my name is pavan salunkhe