import os
from utilities import *
import openai
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file


openai.api_key  = os.environ['OPENAI_API_KEY']


delimiter = "####"
system_message = f"""
You will be provided with a number of tweets delimited with {delimiter} characters.
Classify each tweet into a primary category.
Provide your output in json format with the keys: <number of tweet> and secondary.

Primary categories: Billing, Technical Support, \
Account Management, or General Inquiry.

Billing secondary categories:
Unsubscribe or upgrade
Add a payment method
Explanation for charge
Dispute a charge

Technical Support secondary categories:
General troubleshooting
Device compatibility
Software updates

Account Management secondary categories:
Password reset
Update personal information
Close account
Account security

General Inquiry secondary categories:
Product information
Pricing
Feedback
Speak to a human

"""
user_message = f"""\
I want to delete my profile and all of my user data"""
messages =  [  
{'role':'system', 
 'content': system_message},    
{'role':'user', 
 'content': f"{delimiter}{user_message}{delimiter}"},  
] 
response = get_completion_from_messages(messages)
print(response)