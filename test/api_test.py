# from qdrant_client import QdrantClient
#
# def qudrantTest():
#     qdrant_client = QdrantClient(
#         url="https://0a25c9cf-4685-49c7-9382-4c3510754343.europe-west3-0.gcp.cloud.qdrant.io:6333",
#         api_key=os.getenv("QDRANT_API_KEY", ""),
#     )
#
#     print(qdrant_client.get_collection('documents'))
#
# def endpointTest():
#     headers = {
#         'Content-Type': 'application/json',
#     }
#
#     json_data = {"query":"summarize the document","user_id":"muthu.subramanian@dhsit.co.uk",
#                  "profile_id":"67bd5d6b1981cea3aba6aa30","model_name":"Azure-OpenAI"}
#
#     response = requests.post('https://dhs-docwain-api.azure-api.net/ask', headers=headers, json=json_data)
#     print(response.content)
# endpointTest()
# import os
# import base64
# from openai import AzureOpenAI
#
# # Set up environment variables and API details
# api_key = os.getenv("AZUREGPT4O_API_KEY", "")
# endpoint = os.getenv("ENDPOINT_URL", "https://dw-openai-dev.openai.azure.com/")
# deployment = os.getenv("DEPLOYMENT_NAME", "dw-dev1-gpt-4o")
# subscription_key = os.getenv("AZURE_OPENAI_API_KEY", api_key)
#
# # Initialize Azure OpenAI Service client with key-based authentication
# client = AzureOpenAI(
#     azure_endpoint=endpoint,
#     api_key=subscription_key,
#     api_version="2024-05-01-preview",
# )
#
# # Prepare the chat prompt
# chat_prompt = [
#     {
#         "role": "system",
#         "content": [
#             {
#                 "type": "text",
#                 "text": "You are an AI assistant that helps people find information."
#             }
#         ]
#     },
#     {
#         "role": "user",
#         "content": [
#             {
#                 "type": "text",
#                 "text": "What are the top three tourist attractions in Paris?"
#             }
#         ]
#     }
# ]
#
# try:
#     # Generate chat completion
#     completion = client.chat.completions.create(
#         model=deployment,
#         messages=chat_prompt,
#         max_tokens=800,
#         temperature=0.7,
#         top_p=0.95,
#         frequency_penalty=0,
#         presence_penalty=0,
#         stop=None,
#         stream=False
#     )
#
#     # Extract and print the assistant's response
#     assistant_response = completion.choices[0].message.content
#     print("\nAssistant:", assistant_response)
#
# except Exception as e:
#     print(f"\n❌ Error: {str(e)}")

from src.storage.azure_blob_client import get_chat_container_client

# Get the container client
container_client = get_chat_container_client()
container_name = container_client.container_name

# Check if the container exists
if container_client.exists():
    print(f"Connected to container: {container_name}")
else:
    print(f"Container '{container_name}' does not exist.")
