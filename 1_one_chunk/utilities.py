import openai
import tiktoken

################################################################################################################################

# use model for generating output
# from OpenAI courses
def get_completion_from_messages(messages, 
                                 model="gpt-3.5-turbo", 
                                 temperature=0, 
                                 max_tokens=500):
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature, 
        max_tokens=max_tokens,
    )
    return response.choices[0].message["content"]

################################################################################################################################

# count tokens inside file
# https://blog.devgenius.io/how-to-get-around-openai-gpt-3-token-limits-b11583691b32
# https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
def count_tokens(filename):
    # encoding = tiktoken.get_encoding("cl100k_base")
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    with open(filename, 'r') as f:
        text = f.read()

    input_ids = encoding.encode(text)
    num_tokens = len(input_ids)
    return num_tokens

################################################################################################################################

# separate tweets into chunks
# each batch of tweets must contain system instructions at the beginning
def break_up_file_to_chunks(system_message,tweets):

    encoding = tiktoken.get_encoding("gpt2")
    with open(filename, 'r') as f:
        text = f.read()

    tokens = encoding.encode(text)
    num_tokens = len(tokens)
    
    chunks = []
    for i in range(0, num_tokens, chunk_size - overlap):
        chunk = tokens[i:i + chunk_size]
        chunks.append(chunk)
    
    return chunks

################################################################################################################################



################################################################################################################################