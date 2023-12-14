import os
from dotenv import load_dotenv
from langchain.chat_models import AzureChatOpenAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

def openAiLlm():

    # Getting openAi key from environment
    load_dotenv()
    openai_key = os.getenv("WE_OPENAI_API_KEY")
    openai_base = os.getenv("WE_OPENAI_BASE")
    openai_deployment = os.getenv("WE_AZURE_OPEN_AI_MODEL")
    openai_chat_deployment_name = os.getenv("WE_OPENAI_CHAT_DEPLOYMENT_NAME")
    openai_chat_deployment_model_name = os.getenv(
        "WE_OPENAI_CHAT_DEPLOYMENT_NAME_GPT4")
    openai_chat_embedding_name = os.getenv("WE_OPENAI_EMBEDDING_DEPLOYMENT_NAME")
    openai_chat_gpt_4 = os.getenv("WE_OPEN_AI_CHAT_GPT4")
    open_chat_gpt_4_8k = os.getenv('WE_OPEN_AI_CHAT_GPT4_8K')

    os.environ["OPENAI_API_TYPE"] = os.getenv("WE_OPENAI_API_TYPE")
    os.environ["OPENAI_API_BASE"] = os.getenv("WE_OPENAI_API_BASE")
    os.environ["OPENAI_API_KEY"] = os.getenv("WE_OPENAI_API_KEY")
    os.environ["OPENAI_API_VERSION"] = os.getenv("WE_OPENAI_API_VERSION")
    
    # Create a ChatOpenAI instance for interactive chat using the OpenAI model
    llm = AzureChatOpenAI(temperature=0, openai_api_type="azure",
                          openai_api_key=openai_key,
                          openai_api_base=openai_base,
                          deployment_name=openai_chat_gpt_4,
                          model="gpt-3.5-turbo-16k", openai_api_version="2023-05-15")
    return llm
