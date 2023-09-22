import openai
import tiktoken

################################################################################################################################

# use model for generating output
# from OpenAI courses
def get_completion_from_messages(messages, 
                                 model="gpt-4", 
                                 temperature=0, 
                                 max_tokens=4000):
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature, 
        max_tokens=max_tokens,
    )
    return response.choices[0].message["content"], response

################################################################################################################################

# count tokens inside file
# https://blog.devgenius.io/how-to-get-around-openai-gpt-3-token-limits-b11583691b32
# https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
def count_tokens_file(filename,model):
    # encoding = tiktoken.get_encoding("cl100k_base")
    encoding = tiktoken.encoding_for_model(model)
    with open(filename, 'r') as f:
        text = f.read()

    input_ids = encoding.encode(text)
    num_tokens = len(input_ids)
    return num_tokens

################################################################################################################################

# separate tweets into chunks
# each batch of tweets must contain system instructions at the beginning
def break_in_chunks(system_message,tweets,model):
    """
    Each chunk consists of system_message + some amount of tweets, T
    but we need to know how many tweets are we using because the output increases the number of tokens used by the model

    We won't actually use system_message as part of the chunk since we can call role:system later on

    <number of tweet> & <category> & <1-5 elements> could need up to 40 tokens, maybe more but these are special cases
    checked with https://platform.openai.com/tokenizer
    supposing "worst" case scenario:
    tokens_output = 40*n_tweets

    chunk = input + output = (system_message + T) + 40*number of tweets in T
    
    gpt-3.5-turbo accepts 4096 tokens. Just in case something goes wrong we reduce the number to 4k
    """     
    
    encoding = tiktoken.encoding_for_model(model)
    list_tweets = tweets.split("\n")

    text = system_message
    tokens = encoding.encode(text)
    num_tokens_sytem = len(tokens)

    text = "" # we don't want system_message inside the chunks bcs we will call it as role: system later
    last_i = 0
    chunks = []
    n_tweets_chunk = 0 # number of tweets inside current chunk
    unfinished = True

    while(unfinished):
        for i,new_tweet in enumerate(list_tweets[last_i:],start=last_i):
            new_text = text + "\n" + new_tweet # spacing tweets with a skip line
            tokens = encoding.encode(new_text)

            num_tokens = num_tokens_sytem + len(tokens) + 40*(n_tweets_chunk+1) # tokens until now + output from all previous tweets
            if(num_tokens >= 4000): # 4000 =approx 4096 == max tokens gpt3.5
                chunks.append(text)

                # breaks at point i so the ith tweet wasn't appended
                last_i = i
                print("Last tweet added to the chunk (count starts at 1): ", i) # i-1(not added)+1(starts at 1) # debug
                print("Tokens inside current chunk + system_message + output prediction: ",
                    len(encoding.encode(text)) + num_tokens_sytem + 40*n_tweets_chunk)
                text = "" 
                n_tweets_chunk = 0
                break
            else:
                text = new_text
                n_tweets_chunk = n_tweets_chunk + 1
                # print(i) # debug
                if i == (len(list_tweets)-1): # when last tweet is added
                    unfinished = False

    # append last chunk (the one that isn't cut)
    chunks.append(text)      
    print("Last tweet added to the chunk (count starts at 1): ", i+1) # debug
    print("Tokens inside last chunk + system_message + output prediction:", 
          len(encoding.encode(text))+ num_tokens_sytem + 40*(n_tweets_chunk+1))
          
    return chunks 

################################################################################################################################



################################################################################################################################