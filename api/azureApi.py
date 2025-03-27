import os
import base64
from openai import AzureOpenAI

# Set up environment variables and API details
api_key ='6JSK5oHMv76xL6IAtFwVfgCRykf24MWdvp6oRpxawBk9sGyqXuQYJQQJ99BCACmepeSXJ3w3AAABACOGjB0M'
endpoint = os.getenv("ENDPOINT_URL", "https://dw-openai-dev.openai.azure.com/")
deployment = os.getenv("DEPLOYMENT_NAME", "dw-dev1-gpt-4o")
subscription_key = os.getenv("AZURE_OPENAI_API_KEY", api_key)

# Initialize Azure OpenAI Service client with key-based authentication
client = AzureOpenAI(
    azure_endpoint=endpoint,
    api_key=subscription_key,
    api_version="2024-05-01-preview",
)

# Prepare the chat prompt
chat_prompt = [
    {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": "You are an AI assistant that helps people find information."
            }
        ]
    },
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "What are the top three tourist attractions in Paris?"
            }
        ]
    }
]

try:
    # Generate chat completion
    completion = client.chat.completions.create(
        model=deployment,
        messages=chat_prompt,
        max_tokens=800,
        temperature=0.7,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None,
        stream=False
    )

    # Extract and print the assistant's response
    assistant_response = completion.choices[0].message.content
    print("\nAssistant:", assistant_response)

except Exception as e:
    print(f"\n❌ Error: {str(e)}")

