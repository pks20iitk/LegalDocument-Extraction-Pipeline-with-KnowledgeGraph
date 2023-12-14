import os
from rest_framework.response import Response
from rest_framework.decorators import api_view
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from django.http import StreamingHttpResponse
import re
import tiktoken
from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain import PromptTemplate, LLMChain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainFilter
from regService.prompt.Prompt import create_prompt_with_examples
from langchain.chat_models import ChatOpenAI
from langchain.chat_models import AzureChatOpenAI
from langchain.callbacks import get_openai_callback

import json
import uuid
from azure.data.tables import TableServiceClient
import openai

import requests
from .data import user_data
import neo4j
from django.conf import settings

from neo4j import GraphDatabase

load_dotenv()

openai_gpt_key = os.getenv("LE_OPEN_AI_KEY")
openai_key = os.getenv("LE_OPENAI_API_KEY")
openai_base = os.getenv("LE_OPENAI_BASE")
openai_deployment = os.getenv("LE_AZURE_OPEN_AI_MODEL")
openai_chat_deployment_name = os.getenv("LE_OPENAI_CHAT_DEPLOYMENT_NAME")
openai_chat_deployment_model_name = os.getenv("LE_OPENAI_CHAT_DEPLOYMENT_NAME_GPT4")
openai_chat_embedding_name = os.getenv("LE_OPENAI_EMBEDDING_DEPLOYMENT_NAME")
openai_chat_gpt_4 = os.getenv("LE_OPEN_AI_CHAT_GPT4")
open_chat_gpt_4_8k = os.getenv('LE_OPEN_AI_CHAT_GPT4_8K')
type_script_backend = os.getenv('LE_TS_BACKEND_ENDPOINT')
azure_openai_version = os.getenv('LE_OPENAI_API_VERSION')

os.environ["LE_OPENAI_API_TYPE"] = os.getenv("LE_OPENAI_API_TYPE")
os.environ["LE_OPENAI_API_BASE"] = os.getenv("LE_OPENAI_API_BASE")
os.environ["LE_OPENAI_API_KEY"] = os.getenv("LE_OPENAI_API_KEY")
os.environ["LE_OPENAI_API_VERSION"] = os.getenv("LE_OPENAI_API_VERSION")

DATA_PATH = 'data-legal/'
DB_FAISS_PATH = 'vectorstore/db_faiss'

qdrant_client = QdrantClient(":memory:")
collection_name = "contract-chatbot"

azure_connection_string = os.getenv("AZURE_CONNECTION_STRING")
azure_table_name = os.getenv("AZURE_TABLE_NAME")


class CustomRecursiveCharacterTextSplitter(RecursiveCharacterTextSplitter):
    def split_documents(self, documents):
        texts = []
        for i, doc in enumerate(documents):
            chunks = super().split_documents([doc])
            for chunk in chunks:
                chunk.metadata = {'page_number': i + 1}  # Adding page number
                texts.append(chunk)
        return texts


def set_custom_prompt(url, promptName):
    """function to set the prompt for the client"""
    if promptName == '':
        resultPrompt = getContractType(url)
    else:
        resultPrompt = {}

    if resultPrompt.get('doc_type') in user_data.MSA_TYPES or promptName in user_data.MSA_TYPES:
        prompt = 'msa_abstract'
    else:
        prompt = "abstract"

    custom_prompt_template = user_data.PROMPTS[prompt]
    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['context', 'question'])
    return prompt


def set_relevancy_page_extraction_prompt(version):
    """function to extract the page number from document using prompt"""
    if version == 'V4':
        custom_prompt_for_document_relevancy = user_data.PROMPTS['document_relevancy_prompt_gpt_4']
    elif version == 'V3':
        custom_prompt_for_document_relevancy = user_data.PROMPTS['document_relevancy_prompt']
    retrieval_prompt = PromptTemplate(template=custom_prompt_for_document_relevancy,
                                      input_variables=['context', 'query'])
    return retrieval_prompt


def process_url(url: str):
    try:
        response = requests.get(url)
        if not os.path.exists(DATA_PATH):
            os.mkdir(DATA_PATH)
        with open('data-legal/contract.pdf', 'wb') as f:
            f.write(response.content)
        return True
    except e:
        return False


def saveToTable(userId, response):
    """ function to save the response to the database
        input:  userId and the response object returned from the server
        response: boolean value indicating Weather the response object was saved to database or not.
    """
    try:
        table_service = TableServiceClient.from_connection_string(azure_connection_string)
        table_client = table_service.get_table_client(azure_table_name)
        entity = {'RowKey': userId, 'response': str(response), 'PartitionKey': str(uuid.uuid4())}
        table_client.create_entity(entity)
        return True
    except Exception as ex:
        print(f"An error occurred: {ex}")
        return False


def countTokens(str):
    # Initializing token count with 0
    token_count = 0

    # Initializing tiktoken to count tokens
    enc = tiktoken.get_encoding("cl100k_base")

    # Incrementing token count by one for each token counted by encoder
    for token in enc.encode(str):
        token_count = token_count + 1

    # Returning the final token count
    return token_count


def getPageNumbers(answer, llm, compression_retriever, version):
    context = ""
    total_tokens = 0

    page_retrieval_prompt_template = set_relevancy_page_extraction_prompt(version)

    # Counting tokens for page_retrieval_prompt
    total_tokens = total_tokens + countTokens(page_retrieval_prompt_template.template)

    relevance_qa_chain = LLMChain(
        llm=llm,
        prompt=page_retrieval_prompt_template
    )
    compressed_docs = compression_retriever.get_relevant_documents(answer)
    for index, doc in enumerate(compressed_docs, start=1):
        context += f"Document {index}\n{doc.page_content}\nPage Number: {doc.metadata['page_number']}\n\n"

    # Counting tokens for context and query
    total_tokens = total_tokens + countTokens(context + '' + answer)

    relevancy_res = relevance_qa_chain.predict(context=context, query=answer)
    page_number = relevancy_res.split("Page Number: ")[-1]
    if ',' in page_number:
        page_number = page_number.split(',')
    else:
        page_number = [page_number]
    return {"pageNumber": page_number, "locationTokens": total_tokens}


# Function to format a single step
def format_step(step):
    return f"{step['Action']} - Reason: {step['Reason']}"


# Function to format the entire insight
def format_insight(result_dict):
    insight = ""
    for i, step in enumerate(result_dict["ChainOfThought"], start=1):
        insight += f"{i}. {format_step(step)}\n"
    return insight


# URL RagService endpoint function
@api_view(['GET', 'POST'])
def urlresponse(request):
    """
    POST method for the response
    # """
    # try:
    if request.method != 'POST':
        return Response({"message": "Use the POST method for the response"})

    url = request.data.get("url")
    questions = request.data.get("questions")
    prompt = request.data.get("promptName")
    isAccuracy = request.data.get("isAccuracy")
    isChainOfThought = request.data.get("isChainOfThought")
    userId = request.data.get("userId")
    isModified = request.data.get("isModified")
    isKnowledgeGraph = request.data.get('isKnowledgeGraph')
    deploymentName = openai_chat_deployment_name
    cotResult = ""
    if not url or not questions:
        return Response({"message": "Please provide valid URL and questions"})

    # Process the documents received from the user
    try:
        doc_store = process_url(url)
        if not doc_store:
            return Response({"message": "PDF document not loaded"})
    except Exception as e:
        return Response({"message": "Error loading PDF document"})

    # Loading the OpenAI GPT-3.5-Turbo-16K model
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo-16k",
        temperature=0,
        openai_api_key=openai_gpt_key,
    )

    custom_prompt_template = set_custom_prompt(url, prompt)
    # Load and process documents
    loader = DirectoryLoader(DATA_PATH, glob='*.pdf', loader_cls=PyPDFLoader)
    documents = loader.load()

    text_splitter = CustomRecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=30)
    texts = text_splitter.split_documents(documents)

    # Creating Embeddings using OpenAI
    embeddings = OpenAIEmbeddings(chunk_size=16, openai_api_key=openai_gpt_key, model="text-embedding-ada-002")

    db = FAISS.from_documents(texts, embeddings)
    db.save_local(DB_FAISS_PATH)
    search_kwargs = {
        'k': 30,
        'fetch_k': 100,
        'maximal_marginal_relevance': True,
        'distance_metric': 'cos',
    }
    retriever = db.as_retriever(search_kwargs=search_kwargs)

    # initializing the compression retriever
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": custom_prompt_template},
    )

    # get the list of question from the body
    questionList = request.data['questions']
    getLocation = request.data['IsLocation']
    response = []

    # get the response fro the question from knowledge base
    compressor = LLMChainFilter.from_llm(llm)
    compression_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=retriever)
    total_tokens = 0
    totalCost = 0
    total_token_gpt_4 = 0
    finalToken = 0

    # Iterating through the questions list
    for question in questionList:
        with get_openai_callback() as cb:
            # hack for interest rate with
            if isKnowledgeGraph == "True" and question.lower():
                llm = AzureChatOpenAI(temperature=0, openai_api_type="azure",
                                      openai_api_key=openai_key,
                                      openai_api_base=openai_base,
                                      deployment_name=openai_chat_gpt_4,
                                      model="gpt-3.5-turbo-16k", openai_api_version=azure_openai_version)
                relations, knowledgeGraphResult, questionfromKG = knowledgeGraph(question)
                prompt = PromptTemplate(template=user_data.PROMPTS['abstract'],
                                        input_variables=['context', 'question'])
                # print(prompt)
                qa = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=retriever,
                    return_source_documents=True,
                    chain_type_kwargs={"prompt": prompt},
                )
                # added hack Questions
                res = qa({'query': questionfromKG})
                total_token_gpt_4 += cb.total_tokens
                total_tokens += cb.total_tokens
                totalCost += cb.total_cost
                response.append({"displayQuestion": question, "answer": res["result"]})
                # calling knowledge graph

                # Calculating GPT-4 tokens as GPT-3 tokens
                if total_token_gpt_4 > 0:
                    gpt4TokenCost = ((0.12 / 1000) * total_token_gpt_4)
                    gpt4costToGpt3Tokens = (gpt4TokenCost / 0.004) * 1000
                    finalToken = finalToken + gpt4costToGpt3Tokens
                else:
                    finalToken = total_tokens

                # Calling user token deduction function to deduct user tokens
                deductUserTokens(userId, finalToken)

                return Response(
                    {"response": response, "totalToken": total_tokens - total_token_gpt_4, "totalCost": totalCost,
                     "totalTokenGpt4": total_token_gpt_4, "knowledgeGraphQuestion": questionfromKG,
                     "knowledgeGraphRelations": relations})
            elif question == "What is the spread and what is the initial benchmark and its relation with the spread? What is the index floor percentage in this agreement?" and isAccuracy == "True":
                llm = AzureChatOpenAI(temperature=0, openai_api_type="azure",
                                      openai_api_key=openai_key,
                                      openai_api_base=openai_base,
                                      deployment_name=openai_chat_gpt_4,
                                      openai_api_version=azure_openai_version)

                qa = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=retriever,
                    return_source_documents=True,
                    chain_type_kwargs={"prompt": custom_prompt_template},
                )
                res = qa({'query': question})

                if isChainOfThought == "True":
                    custom_prompt_template = user_data.PROMPTS['chain_of_thought_abstract']
                    custom_prompt_template = custom_prompt_template.format(context='{context}',
                                                                           question='{question}')
                    prompt = PromptTemplate(template=user_data.PROMPTS['chain_of_thought_abstract'],
                                            input_variables=['context', 'question'])
                    qa = RetrievalQA.from_chain_type(
                        llm=llm,
                        chain_type="stuff",
                        retriever=retriever,
                        return_source_documents=True,
                        chain_type_kwargs={"prompt": prompt},
                    )
                    cotResult = qa({'query': res["result"]})

                total_token_gpt_4 += cb.total_tokens
                total_tokens += cb.total_tokens
                totalCost += cb.total_cost

            else:
                qa = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=retriever,
                    return_source_documents=True,
                    chain_type_kwargs={"prompt": custom_prompt_template},
                )
                res = qa({'query': question})
                # Disabling Chain of thought while using GPT-3.5-Turbo-16K
                # if isChainOfThought == "True":
                #     custom_prompt_template = user_data.PROMPTS['chain_of_thought_abstract']
                #     custom_prompt_template = custom_prompt_template.format( context='{context}', question='{question}')
                #     prompt = PromptTemplate(template=user_data.PROMPTS['chain_of_thought_abstract'],
                #     input_variables=['context', 'question'])
                #     qa = RetrievalQA.from_chain_type(
                #     llm=llm,
                #     chain_type="stuff",
                #     retriever=retriever,
                #     return_source_documents=True,
                #     chain_type_kwargs={"prompt": prompt},
                #     )
                #     print(res["result"])
                #     cotResult = qa({'query': res["result"]})
                #     print(cotResult["result"])

                total_tokens += cb.total_tokens
                totalCost += cb.total_cost

        if getLocation == 'True' and isAccuracy == "True" and question == "What is the spread and what is the initial benchmark and its relation with the spread? What is the index floor percentage in this agreement?":
            # LLM with the gpt4 8k model for the location accuracy
            llm = AzureChatOpenAI(temperature=0, openai_api_type="azure",
                                  openai_api_key=openai_key,
                                  openai_api_base=openai_base,
                                  deployment_name=openai_chat_gpt_4,
                                  openai_api_version=azure_openai_version)
            # get the page number of the of the question in given context
            page_number = getPageNumbers(res["result"], llm, compression_retriever, "V4")
            if isChainOfThought == "True":
                custom_prompt_template = user_data.PROMPTS['chain_of_thought_abstract']
                custom_prompt_template = custom_prompt_template.format(context='{context}', question='{question}')
                prompt = PromptTemplate(template=user_data.PROMPTS['chain_of_thought_abstract'],
                                        input_variables=['context', 'question'])
                qa = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=retriever,
                    return_source_documents=True,
                    chain_type_kwargs={"prompt": prompt},
                )
                cotResult = qa({'query': res["result"]})
            # tokens for the gpt-4 calculation
            total_token_gpt_4 += cb.total_tokens
            totalCost += cb.total_cost

            # If there is no result in cot then return response without the cot key
            if cotResult:
                response.append({"displayQuestion": question, "answer": res["result"], "location": f"{page_number}",
                                 "chainofthought": cotResult["result"]})
            else:
                response.append({"displayQuestion": question, "answer": res["result"],
                                 "location": f"page {page_number['pageNumber']}"})

        elif getLocation == "True":
            # get the location of the question in the context
            page_number = getPageNumbers(question, llm, compression_retriever, "V3")
            total_tokens += cb.total_tokens
            totalCost += cb.total_cost
            if cotResult != "":
                response.append({"displayQuestion": question, "answer": res["result"],
                                 "location": f"page {page_number['pageNumber']}",
                                 "chainofthought": cotResult["result"]})
            else:
                response.append({"displayQuestion": question, "answer": res["result"],
                                 "location": f"page {page_number['pageNumber']}"})
        else:
            if cotResult != "":
                response.append(
                    {"displayQuestion": question, "answer": res["result"], "chainofthought": cotResult["result"]})
            else:
                response.append({"displayQuestion": question, "answer": res["result"]})
    # return the response
    # Save the response to the database
    if bool(isModified) == True:
        saveToTable(userId, response)
    #
    # except Exception as e:
    #     return Response({"message": f"An error occurred {e}"})

    # Calculating GPT-4 tokens as GPT-3 tokens
    if total_token_gpt_4 > 0:
        gpt4TokenCost = ((0.12 / 1000) * total_token_gpt_4)
        gpt4costToGpt3Tokens = (gpt4TokenCost / 0.004) * 1000
        finalToken = finalToken + gpt4costToGpt3Tokens
    else:
        finalToken = total_tokens

    # Calling user token deduction function, passing total_tokens if > 0 else GPT-4 tokens
    deductUserTokens(userId, finalToken)

    return Response({"response": response, "totalToken": total_tokens - total_token_gpt_4, "totalCost": totalCost,
                     "totalTokenGpt4": total_token_gpt_4})


# Function to deduct tokens from user
def deductUserTokens(userId: str, total_tokens: int):
    # Making request to TypeScript Backend to update user tokens
    try:
        # Initializing the TS backend URL
        ts_backend_url = type_script_backend + "auth/update-user-tokens"

        # Creating the payload
        payload = {
            "userId": userId,
            "token": total_tokens
        }

        # Sending request to TS backend and awaiting response
        response = requests.post(ts_backend_url, payload)
        data = response.json()
        print(data)
        return Response({"message": "Token update success"})
        # Handling exceptions
    except requests.exceptions.RequestException as e:
        return Response({"message": f"Token Update failed with error: {e}"})


# Function to update App's total token processed metric
def updateAppTokenUsage(total_tokens: int, userId: str):
    # Making request to TypeScript Backend to update user tokens
    try:
        # Initializing the TS backend URL
        ts_update_app_token_url = type_script_backend + "usage/update-app-token-usage"
        ts_update_user_token_url = type_script_backend + "usage/update-user-token-usage"

        # Creating the payload to update app token usage
        app_token_payload = {
            "tokenUsage": total_tokens
        }

        # Creating payload to update user's token usage 
        user_token_payload = {
            "userId": userId,
            "tokenUsage": total_tokens
        }

        # Sending requests to TS backend and awaiting response
        requests.post(ts_update_app_token_url, app_token_payload)
        requests.post(ts_update_user_token_url, user_token_payload)

        return Response({"message": "Token update success"})
        # Handling exceptions
    except requests.exceptions.RequestException as e:
        return Response({"message": f"Token Update failed with error: {e}"})


"""
Streaming OpenAI responses using SSE (Server Sent Events)
"""


@api_view(['GET', 'POST'])
def sse_view(request):
    # Initializing OpenAI
    openai.api_key = openai_gpt_key
    openai.api_type = "openai"
    openai.api_base = "https://api.openai.com/v1"
    openai.api_version = "2020-11-07"

    if request.method != 'POST':
        return Response({"message": "Use the POST method for the response"})

    # Initializing tiktoken with cl100k_base (gpt-3.5-turbo, gpt-4, and text-embeddings-ada-002)
    enc = tiktoken.get_encoding("cl100k_base")

    # Storing Data from request into variables
    url = request.data.get("url")
    questions = request.data.get("questions")
    userId = request.data.get("userId")
    isModified = request.data.get("isModified")
    isAccuracy = request.data.get("isAccuracy")
    isChainOfThought = request.data.get("isChainOfThought")
    isKnowledgeGraph = request.data.get('isKnowledgeGraph')

    if not url or not questions:
        return Response({"message": "Please provide valid URL and questions"})

    # Process the documents received from the user
    try:
        doc_store = process_url(url)
        if not doc_store:
            return Response({"message": "PDF document not loaded"})
    except Exception as e:
        return Response({"message": "Error loading PDF document"})

    # Load and process documents
    loader = DirectoryLoader(DATA_PATH, glob='*.pdf', loader_cls=PyPDFLoader)
    documents = loader.load()

    text_splitter = CustomRecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=30)
    texts = text_splitter.split_documents(documents)

    # get the list of question from the body
    questionList = request.data['questions']
    getLocation = request.data['IsLocation']

    # Initializing question
    interest_rate_question = "What is the spread and what is the initial benchmark and its relation with the spread? What is the index floor percentage in this agreement?"

    # This function is to generate responses from OpenAi
    def openai_response_generator():
        # Initializing prompt and other variables
        prompt = request.data.get("promptName")
        response = []
        openai.api_key = openai_gpt_key
        openai.api_type = "openai"
        openai.api_base = "https://api.openai.com/v1"
        openai.api_version = "2020-11-07"
        promptInitialized = False

        # Creating Embeddings using OpenAI
        embeddings = OpenAIEmbeddings(chunk_size=16, openai_api_key=openai_gpt_key, model="text-embedding-ada-002")

        db = FAISS.from_documents(texts, embeddings)
        db.save_local(DB_FAISS_PATH)
        search_kwargs = {
            'k': 30,
            'fetch_k': 100,
            'maximal_marginal_relevance': True,
            'distance_metric': 'cos',
        }
        retriever = db.as_retriever(search_kwargs=search_kwargs)

        # Initializing token values as 0
        total_tokens = 0
        total_token_gpt_4 = 0
        totalCost = 0
        finalToken = 0

        # Loading the OpenAI GPT-3.5-Turbo-16K model
        llm = ChatOpenAI(
            model_name="gpt-3.5-turbo-32k",
            temperature=0,
            openai_api_key=openai_gpt_key,
        )

        # get the response fro the question from knowledge base
        compressor = LLMChainFilter.from_llm(llm)
        compression_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=retriever)

        # Iterating through the questions list
        for question in questionList:
            with get_openai_callback() as cb:
                # Initializing the Prompt
                final_prompt = user_data.PROMPTS['abstract']

                # Getting relevant chunks from the document
                relevant_chunks = retriever.get_relevant_documents(question)
                final_chunk = ''
                for chunk in relevant_chunks:
                    final_chunk = final_chunk + chunk.page_content

                # Replacing template literals inside prompt with real values
                final_prompt = final_prompt.replace('{context}', final_chunk).replace('{question}', question)

                # Prompt tokens needs to be counted once
                if not promptInitialized:
                    # Incrementing token count for deduction
                    for token in enc.encode(final_prompt):
                        total_tokens = total_tokens + 1

                    # Setting prompt initialized status as True (no need to count prompt tokens again and again)
                    promptInitialized = True

            try:
                """
                    - Using OpenAI's "ChatCompletion" to generate response to the prompt
                    - Response tokens are yielded as soon as they are available from OpenAI
                    """
                if question != interest_rate_question and question.lower() != "what is interest rate?":
                    # Initializing OpenAI key
                    openai.api_key = openai_gpt_key

                    # Calling the OpenAI ChatCompletion method
                    for c in openai.ChatCompletion.create(
                            model="gpt-3.5-turbo-16k",
                            messages=[
                                {"role": "system", "content": "You are an experienced loan agreement analyst."},
                                {"role": "user", "content": final_prompt}
                            ],
                            temperature=0,
                            stream=True
                    ):
                        # Handling chunks where 'content' doesn't exist
                        if 'choices' in c and c['choices'][0]['delta'].get('content'):

                            # Incrementing token count for deduction
                            for token in enc.encode(c["choices"][0]['delta']['content']):
                                total_tokens = total_tokens + 1

                            response = {"content": c["choices"][0]['delta']['content'], "isDone": "False"}
                            final_res = json.dumps(response)

                            # Returning chunks from OpenAI as soon as they are available
                            yield final_res

                    # Generating location if location is set to True
                    if getLocation == "True" and question != interest_rate_question:
                        # Initializing GPT version
                        # If Enhance Mode is enabled use GPT-4 to get locations else use GPT-3.5-Turbo
                        gpt_version = "V3" if isAccuracy != "True" else "V4"

                        # Getting the location for the response
                        page_number_result = getPageNumbers(question, llm, compression_retriever, gpt_version)

                        # Adding tokens consumed by location function to total_tokens
                        total_tokens = total_tokens + page_number_result['locationTokens']

                        # Returning the location of the given question
                        yield json.dumps({"location": f"page {page_number_result['pageNumber']}", "isDone": "False"})

                    # Updating final tokens
                    finalToken = total_tokens

                # Triggering knowledge graph is isKnowledgeGraph is set to True
                if isKnowledgeGraph == "True" and (
                        question.lower() == "interest rate" in question.lower()):
                    relations, knowledgeGraphResult, questionfromKG = knowledgeGraph(question)

                    # Initializing the Prompt
                    final_prompt = user_data.PROMPTS['abstractWithKnowledgeGraph']

                    # Getting relevant chunks from the document
                    relevant_chunks = retriever.get_relevant_documents(questionfromKG)
                    final_chunk = ''
                    for chunk in relevant_chunks:
                        final_chunk = final_chunk + chunk.page_content

                    # Replacing template literals inside prompt with real values
                    final_prompt = final_prompt.replace('{context}', final_chunk).replace('{question}', questionfromKG)

                    # Configuring OpenAI to use Azure OpenAI
                    openai.api_key = openai_key
                    openai.api_type = "azure"
                    openai.api_base = openai_base
                    openai.api_version = "2023-05-15"

                    # Calling Azure OpenAi's ChatCompletion method
                    for c in openai.ChatCompletion.create(
                            temperature=0,
                            engine=openai_chat_gpt_4,  # engine = "deployment_name".
                            stream=True,
                            messages=[
                                {"role": "system", "content": "You are an experienced loan agreement analyst."},
                                {"role": "user", "content": final_prompt}
                            ]
                    ):
                        # Handling chunks where 'content' doesn't exist
                        if 'choices' in c and c['choices'][0]['delta'].get('content'):

                            # Incrementing token count for deduction
                            for token in enc.encode(c["choices"][0]['delta']['content']):
                                total_tokens = total_tokens + 1
                            # Returning chunks from OpenAI as soon as they are available
                            response = {"content": c["choices"][0]['delta']['content'], "isDone": "False"}
                            final_res = json.dumps(response)
                            yield final_res

                    # Converting tokens used by GPT-4 to GPT-3 tokens
                    gpt4TokenCost = ((0.12 / 1000) * total_tokens)
                    gpt4costToGpt3Tokens = (gpt4TokenCost / 0.004) * 1000
                    finalToken = finalToken + gpt4costToGpt3Tokens

                    if isChainOfThought == "True":
                        # Initializing chain of thought prompt template
                        cot_prompt_template = user_data.PROMPTS['chain_of_thought_abstract']

                        # Replacing template literals inside prompt with real values
                        cot_prompt_template = cot_prompt_template.replace('{context}', final_chunk).replace(
                            '{question}', question)

                        # Incrementing token count for deduction
                        for token in enc.encode(final_prompt):
                            total_tokens = total_tokens + 1

                        cot_response = openai.ChatCompletion.create(
                            engine=openai_chat_gpt_4,  # engine = "deployment_name".
                            messages=[
                                {"role": "system", "content": "You are an experienced loan agreement analyst."},
                                {"role": "user", "content": cot_prompt_template}
                            ]
                        )

                        # Incrementing token count for deduction
                        for token in enc.encode(cot_response['choices'][0]['message']['content']):
                            total_tokens = total_tokens + 1

                        # Calculating GPT-4 tokens to GPT-3 tokens
                        gpt4TokenCost = ((0.12 / 1000) * total_tokens)
                        gpt4costToGpt3Tokens = (gpt4TokenCost / 0.004) * 1000
                        finalToken = finalToken + gpt4costToGpt3Tokens

                        # Returning Chain Of Through response as a JSON
                        yield json.dumps({"displayQuestion": question,
                                          "chainofthought": cot_response['choices'][0]['message']['content'],
                                          "isDone": "False"})

                # Trigger for Interest rate question
                elif question == interest_rate_question and isAccuracy == "True" and getLocation == "False":
                    # Configuring OpenAI to use Azure OpenAI
                    openai.api_key = openai_key
                    openai.api_type = "azure"
                    openai.api_base = openai_base
                    openai.api_version = "2023-05-15"

                    # Calling Azure OpenAi's ChatCompletion method
                    for c in openai.ChatCompletion.create(
                            engine=openai_chat_gpt_4,  # engine = "deployment_name".
                            stream=True,
                            messages=[
                                {"role": "system", "content": "You are an experienced loan agreement analyst."},
                                {"role": "user", "content": final_prompt}
                            ]
                    ):
                        # Handling chunks where 'content' doesn't exist
                        if 'choices' in c and c['choices'][0]['delta'].get('content'):

                            # Incrementing token count for deduction
                            for token in enc.encode(c["choices"][0]['delta']['content']):
                                total_tokens = total_tokens + 1
                            # Returning chunks from OpenAI as soon as they are available
                            response = {"content": c["choices"][0]['delta']['content'], "isDone": "False"}
                            final_res = json.dumps(response)
                            yield final_res

                    # Converting tokens used by GPT-4 to GPT-3 tokens
                    gpt4TokenCost = ((0.12 / 1000) * total_tokens)
                    gpt4costToGpt3Tokens = (gpt4TokenCost / 0.004) * 1000
                    finalToken = finalToken + gpt4costToGpt3Tokens

                    if isChainOfThought == "True":

                        # Initializing chain of thought prompt template
                        cot_prompt_template = user_data.PROMPTS['chain_of_thought_abstract']

                        # Replacing template literals inside prompt with real values
                        cot_prompt_template = cot_prompt_template.replace('{context}', final_chunk).replace(
                            '{question}', question)

                        # Incrementing token count for deduction
                        for token in enc.encode(final_prompt):
                            total_tokens = total_tokens + 1

                        cot_response = openai.ChatCompletion.create(
                            engine=openai_chat_gpt_4,  # engine = "deployment_name".
                            messages=[
                                {"role": "system", "content": "You are an experienced loan agreement analyst."},
                                {"role": "user", "content": cot_prompt_template}
                            ]
                        )

                        # Incrementing token count for deduction
                        for token in enc.encode(cot_response['choices'][0]['message']['content']):
                            total_tokens = total_tokens + 1

                        # Calculating GPT-4 tokens to GPT-3 tokens
                        gpt4TokenCost = ((0.12 / 1000) * total_tokens)
                        gpt4costToGpt3Tokens = (gpt4TokenCost / 0.004) * 1000
                        finalToken = finalToken + gpt4costToGpt3Tokens

                        # Returning Chain Of Through response as a JSON
                        yield json.dumps({"displayQuestion": question,
                                          "chainofthought": cot_response['choices'][0]['message']['content'],
                                          "isDone": "False"})

                # Trigger for Interest rate question with location set to true
                elif getLocation == 'True' and isAccuracy == "True" and question == interest_rate_question:
                    # Configuring OpenAI to use Azure OpenAI
                    openai.api_key = openai_key
                    openai.api_type = "azure"
                    openai.api_base = openai_base
                    openai.api_version = "2023-05-15"

                    # Initializing interest rate response
                    interest_rate_response = []

                    # Initializing Azure OpenAI GPT-4 model
                    azure_llm = AzureChatOpenAI(temperature=0, openai_api_type="azure",
                                                openai_api_key=openai_key,
                                                openai_api_base=openai_base,
                                                deployment_name=openai_chat_gpt_4,
                                                openai_api_version=azure_openai_version)

                    # Creating Embeddings using OpenAI
                    embeddings = OpenAIEmbeddings(deployment=openai_chat_embedding_name, chunk_size=16,
                                                  openai_api_key=openai_key, openai_api_type="azure")

                    db = FAISS.from_documents(texts, embeddings)
                    db.save_local(DB_FAISS_PATH)
                    search_kwargs = {
                        'k': 30,
                        'fetch_k': 100,
                        'maximal_marginal_relevance': True,
                        'distance_metric': 'cos',
                    }
                    retriever = db.as_retriever(search_kwargs=search_kwargs)

                    # get the response fro the question from knowledge base
                    azure_compressor = LLMChainFilter.from_llm(azure_llm)
                    azure_compression_retriever = ContextualCompressionRetriever(base_compressor=azure_compressor,
                                                                                 base_retriever=retriever)

                    # Calling Azure OpenAi's ChatCompletion method
                    for c in openai.ChatCompletion.create(
                            engine=openai_chat_gpt_4,  # engine = "deployment_name".
                            stream=True,
                            messages=[
                                {"role": "system", "content": "You are an experienced loan agreement analyst."},
                                {"role": "user", "content": final_prompt}
                            ]
                    ):
                        # Handling chunks where 'content' doesn't exist
                        if 'choices' in c and c['choices'][0]['delta'].get('content'):

                            # Incrementing token count for deduction
                            for token in enc.encode(c["choices"][0]['delta']['content']):
                                total_tokens = total_tokens + 1

                            # String interest rate response if Chain Of Thought is set to True
                            if isChainOfThought:
                                interest_rate_response.append(c["choices"][0]['delta']['content'])

                            # Returning chunks from OpenAI as soon as they are available
                            yield json.dumps({"content": c["choices"][0]['delta']['content'], "isDone": "False"})

                    # Converting tokens used by GPT-4 to GPT-3 tokens
                    gpt4TokenCost = ((0.12 / 1000) * total_tokens)
                    gpt4costToGpt3Tokens = (gpt4TokenCost / 0.004) * 1000
                    finalToken = finalToken + gpt4costToGpt3Tokens

                    interest_rate = ""
                    for elements in interest_rate_response:
                        interest_rate = interest_rate + elements

                    # get the page number of the of the question in given context
                    page_number_result = getPageNumbers(interest_rate, azure_llm, azure_compression_retriever, "V4")

                    # Adding tokens consumed by location function to total_tokens
                    total_tokens = total_tokens + page_number_result['locationTokens']

                    # Returning the location of the given question
                    response = {"location": f"page {page_number_result['pageNumber']}", "isDone": "False"}
                    final_res = json.dumps(response)
                    yield final_res

                    if isChainOfThought == "True":

                        # Initializing chain of thought prompt template
                        cot_prompt_template = user_data.PROMPTS['chain_of_thought_abstract']

                        # Replacing template literals inside prompt with real values
                        cot_prompt_template = cot_prompt_template.replace('{context}', final_chunk).replace(
                            '{question}', question)

                        # Incrementing token count for deduction
                        for token in enc.encode(final_prompt):
                            total_tokens = total_tokens + 1

                        cot_response = openai.ChatCompletion.create(
                            engine=openai_chat_gpt_4,  # engine = "deployment_name".
                            messages=[
                                {"role": "system", "content": "You are an experienced loan agreement analyst."},
                                {"role": "user", "content": cot_prompt_template}
                            ]
                        )

                        # Incrementing token count for deduction
                        for token in enc.encode(cot_response['choices'][0]['message']['content']):
                            total_tokens = total_tokens + 1

                        # Calculating GPT-4 tokens to GPT-3 tokens
                        gpt4TokenCost = ((0.12 / 1000) * total_tokens)
                        gpt4costToGpt3Tokens = (gpt4TokenCost / 0.004) * 1000
                        finalToken = finalToken + gpt4costToGpt3Tokens

                        # Returning Chain Of Through response as a JSON
                        yield json.dumps({"displayQuestion": question,
                                          "chainofthought": cot_response['choices'][0]['message']['content'],
                                          "isDone": "False"})

                # Trigger for iInterest rate question when Enhanced mode is turned off
                elif isAccuracy == "False" and question == interest_rate_question:
                    # Initializing OpenAI key
                    openai.api_key = openai_gpt_key

                    # Calling the OpenAI ChatCompletion method
                    for c in openai.ChatCompletion.create(
                            model="gpt-3.5-turbo-16k",
                            messages=[
                                {"role": "system", "content": "You are an experienced loan agreement analyst."},
                                {"role": "user", "content": final_prompt}
                            ],
                            temperature=0,
                            stream=True
                    ):
                        # Handling chunks where 'content' doesn't exist
                        if 'choices' in c and c['choices'][0]['delta'].get('content'):

                            # Incrementing token count for deduction
                            for token in enc.encode(c["choices"][0]['delta']['content']):
                                total_tokens = total_tokens + 1

                            # Returning chunks from OpenAI as soon as they are available
                            yield json.dumps({"content": c["choices"][0]['delta']['content'], "isDone": "False"})

                    # Updating final tokens
                    finalToken = total_tokens

                else:
                    # Sending frontend identifier after each question is done streaming
                    yield json.dumps({"content": " ", "isDone": "True"})

            # Handling exceptions
            except Exception as e:
                yield {"message": f"An error occurred {e}"}

        # Sending frontend identifier to end streaming
        yield json.dumps({"content": " ", "isDone": "True"})

        # Return consumed token to frontend for telemetry
        consumedTokens = json.dumps({"consumedTokens": finalToken})
        yield consumedTokens
        print(consumedTokens)

        # Calling TS backend as microservice to deduct user tokens
        deductUserTokens(userId, finalToken)

        # Calling TS backend to add used tokens to app's and user's token processed metric
        updateAppTokenUsage(finalToken, userId)

        promptInitialized = False

    # Initializing res as an http event-stream and clearing cache
    res = StreamingHttpResponse(openai_response_generator(), content_type="text/event-stream")
    res['Cache-Control'] = 'no-cache'
    res['X-Accel-Buffering'] = 'no'
    res['X-Envoy-Upstream-Service-Time'] = 478

    # Returning res as a continuous http event-stream (SSE)
    return res


def set_doc_type_prompt():
    """function to set the prompt for the client"""
    custom_prompt_template = user_data.PROMPTS['set_doc_type']
    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['context', 'question'])
    return prompt


def get_type_of_contract(llm, docs):
    question = "What is the type of the contract?"
    prompt = set_doc_type_prompt()
    qa_chain = LLMChain(
        llm=llm,
        prompt=prompt
    )
    context = ""
    for doc in docs:
        context += f"{doc.page_content}\n\n"
    response = qa_chain.predict(context=context, question=question)
    return response


def process_url_type(url: str):
    try:
        response = requests.get(url)
        if response.status_code // 100 == 2:
            if not os.path.exists(DATA_PATH):
                os.mkdir(DATA_PATH)
            with open('data-legal/contract.pdf', 'wb') as f:
                f.write(response.content)
            return True
    except Exception as e:
        return {"error": e}


def getContractType(url):
    """Returns the type of the contract"""

    try:

        if not url:
            return Response({"message": "Please provide valid URL"})

        # Process the documents received from the user
        try:
            doc_store = process_url_type(url)
            if not doc_store:
                return Response({"message": "PDF document not loaded"})
        except Exception as e:

            return Response({"message": "Error loading PDF document"})

        # Load the model and create the prompts
        llm = ChatOpenAI(
            model_name="gpt-3.5-turbo-16k",
            temperature=0,
            openai_api_key=openai_gpt_key,
        )

        # Load and process documents
        loader = DirectoryLoader(DATA_PATH, glob='*.pdf', loader_cls=PyPDFLoader)
        documents = loader.load()

        type_of_contract_response = get_type_of_contract(llm, documents[:2])
    except Exception as e:
        return Response({"message": f"An error occurred {e}"})
    return {"doc_type": type_of_contract_response}


def set_question_generation_prompt(doc_type, isSuggested=False):
    """function to set the prompt for the client"""

    custom_prompt_template = user_data.PROMPTS['generate_questions']
    # checking for the suggested generation prompt
    if isSuggested == 'True':
        # get the suggestion generation prompt
        custom_prompt_template = user_data.PROMPTS['generate_suggested_question']
        custom_prompt_template = custom_prompt_template.format(doc_type=doc_type, context='{context}',
                                                               question='{question}')
        # generate the suggestion prompt template
        prompt = PromptTemplate(template=custom_prompt_template,
                                input_variables=['context', 'question'],
                                )

    else:
        custom_prompt_template = custom_prompt_template.format(doc_type=doc_type, context='{context}',
                                                               question='{question}')
        prompt = PromptTemplate(template=custom_prompt_template,
                                input_variables=['context', 'question'],
                                )
    return prompt


@api_view(['GET', 'POST'])
def getQuestions(request):
    # Initializing OpenAI GPT-3
    openai.api_key = openai_gpt_key
    openai.api_type = "openai"
    openai.api_base = "https://api.openai.com/v1"
    openai.api_version = "2020-11-07"

    """Returns the various question based on the contract type"""
    # Load and process documents
    getTypeOfContract = getContractType(request.data.get("url"))
    #  send loan_agreement abstraction questions as response
    if getTypeOfContract['doc_type'] in user_data.CREDIT_TYPES or any(
            substring in getTypeOfContract['doc_type'].lower() for substring in ['loan', 'loan agreement']):
        return Response(
            {"questions": user_data.LOAN_AGREEMENT_QUESTIONS, "document_type": getTypeOfContract['doc_type']})

    elif getTypeOfContract['doc_type'] in user_data.MSA_TYPES:
        return Response(
            {"questions": user_data.MSA_AGREEMENT_QUESTIONS, "document_type": getTypeOfContract['doc_type']})
    loader = DirectoryLoader(DATA_PATH, glob='*.pdf', loader_cls=PyPDFLoader)
    documents = loader.load()
    text_splitter = CustomRecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=30)
    texts = text_splitter.split_documents(documents)

    # Creating Embeddings using OpenAI
    embeddings = OpenAIEmbeddings(chunk_size=16, openai_api_key=openai_gpt_key, model="text-embedding-ada-002")

    db = FAISS.from_documents(texts, embeddings)

    # Creating the LLM using OpenAI
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo-16k",
        temperature=0,
        openai_api_key=openai_gpt_key,
    )

    prompt_template = set_question_generation_prompt(doc_type=getTypeOfContract['doc_type'])
    search_kwargs = {
        'k': 30,
        'fetch_k': 100,
        'maximal_marginal_relevance': True,
        'distance_metric': 'cos',
    }
    retriever = db.as_retriever(search_kwargs=search_kwargs)
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt_template},
    )
    # get the response from the question from knowledge base
    question = "Generate the top 5 legal questions based on the document"
    res = qa({'query': question})
    questions = res["result"].split('\n')
    data = [x.split('Question: ')[-1].split('Key: ')[-1] for x in questions if
            len(x.split('Question: ')[-1]) > 0 or len(x.split('Key: ')[-1]) > 0]
    return Response({"questions": [{"Question": data[i], "key": data[i + 1]} for i in range(0, len(data), 2)],
                     "document_type": getTypeOfContract['doc_type']})


@api_view(['GET', 'POST'])
def getSuggestedQuestion(request):
    # Initializing OpenAI GPT-3
    openai.api_key = openai_gpt_key
    openai.api_type = "openai"
    openai.api_base = "https://api.openai.com/v1"
    openai.api_version = "2020-11-07"

    # get the requested data from the request object
    question = request.data.get('question')
    answer = request.data.get('answer')
    url = request.data.get('url')
    isSuggested = request.data.get('isSuggested')
    # get the type of contract
    typeOfContract = getContractType(url)
    # check for the value of question and answer is empty string if the values are empty generate the default questions of the type of document
    if question == "" or answer == "":
        # check for the type of contract contains the credit types of loan agreement
        if typeOfContract['doc_type'] in user_data.CREDIT_TYPES or any(
                substring in typeOfContract['doc_type'].lower() for substring in ['loan', 'loan agreement']):
            # return the response with questions in it.
            return Response({"questions": user_data.DEFAULT_CREDIT_AGREEMENT_QUESTIONS})
    # perform the load and split the document
    loader = DirectoryLoader(DATA_PATH, glob='*.pdf', loader_cls=PyPDFLoader)
    documents = loader.load()
    text_splitter = CustomRecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=30)
    texts = text_splitter.split_documents(documents)
    # Creating Embeddings using OpenAI
    embeddings = OpenAIEmbeddings(chunk_size=16, openai_api_key=openai_gpt_key, model="text-embedding-ada-002")
    db = FAISS.from_documents(texts, embeddings)

    # Creating the LLM using OpenAI
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo-16k",
        temperature=0,
        openai_api_key=openai_gpt_key,
    )

    # get the required prompt for the suggestion questions
    prompt_template = set_question_generation_prompt(typeOfContract['doc_type'], isSuggested)

    search_kwargs = {
        'k': 30,
        'fetch_k': 100,
        'maximal_marginal_relevance': True,
        'distance_metric': 'cos',
    }
    # get the db reteriever from the search kwargs
    retriever = db.as_retriever(search_kwargs=search_kwargs)
    # generate a qa object
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt_template},
    )
    # get the question for the prompt
    emptyQuestion = ""
    if question == "":
        emptyQuestion = "generate at most 5 relevant legal questions for the document with the given context"

    else:
        emptyQuestion = f"Generate other at most 5 relevant legal Questions based on this question: {question}  and this answer for the current question provided: {answer} to generate more related questions based on answer and questions"
    # get the response from the qa
    res = qa({'query': emptyQuestion})
    # clean the response
    questions = res["result"].split('\n')
    # print(data)
    pattern = r'^Question \d+: '
    pattern2 = r'^Question:'

    # Remove the "Question number:" part from each string using re.sub
    cleaned_questions = [re.sub(pattern, '', question) for question in questions]
    cleaned_questions = [re.sub(pattern2, '', question) for question in cleaned_questions]
    # return the cleaned_questions list.
    return Response({"questions": cleaned_questions})


def generate_system_message() -> str:
    return """
You are a data scientist working for a company that is building a graph database. Your task is to extract information from data and convert it into a graph database.
Provide a set of Nodes in the form [ENTITY_ID, TYPE, PROPERTIES] and a set of relationships in the form [ENTITY_ID_1, RELATIONSHIP, ENTITY_ID_2, PROPERTIES].
It is important that the ENTITY_ID_1 and ENTITY_ID_2 exists as nodes with a matching ENTITY_ID. If you can't pair a relationship with a pair of nodes don't add it.
When you find a node or relationship you want to add try to create a generic TYPE for it that  describes the entity you can also think of it as a label.

Example:
Data: Interest Rate: shall mean, for any Interest Period, (x) the Spread plus the
Benchmark for such Interest Period or (y) when applicable pursuant to this Agreement or any other
Loan Document, the Default Rate.
Spread: shall mean 4.75%.
Benchmark: shall mean, initially, the London interbank offered rate for U.S. dollars
with a 1-month tenor; provided that if a Benchmark Transition Event or an Early Opt-in Election,
as applicable, and its related Benchmark Replacement Date have occurred with respect to the then current Benchmark, then “Benchmark” means the applicable Benchmark Replacement to the
extent that such Benchmark Replacement has replaced such prior benchmark rate pursuant to
Section 2.10 hereof. Lender shall determine the Benchmark (and the applicable Reference Time)
as in effect from time to time, and each such determination by Lender shall be conclusive and
binding absent manifest error. While the Benchmark remains the London interbank offered rate
for U.S. dollars with a 1-month tenor, Lender shall determine the same in accordance with the
defined term “LIBOR.”
LIBOR: shall mean, with respect to any Interest Period, a rate per annum (expressed
as a percentage per annum rounded upwards, if necessary, to the nearest one hundredth (1/100th)
of one percent (1%)) for deposits in U.S. Dollars for a one (1) month period that appears on Reuters
Screen LIBOR01 Page as of 11:00 a.m., London time. Notwithstanding the
foregoing, in no event shall LIBOR be an amount less than the Index Floor.
Index Floor: shall mean one-quarter of one percent (0.25%) per annum.
Nodes: ["Interest Rate", "FinancialTerm", {"name":"Interest Rate"}], ["Spread", "FinancialTerm", {"value": "4.75%", "name": "Spread"}], ["Benchmark", "FinancialTerm", {"name": "Benchmark"}], ["LIBOR", "FinancialTerm", {"name": "LIBOR"}], ["IndexFloorRate", "FinancialTerm", {"name": "Index Floor Rate", "value": "0.25%"}]
Relationships: ["Interest Rate", "SUM_OF", "Spread", {}], ["Interest Rate", "SUM_OF", "Benchmark", {}], ["Benchmark", "Initial Benchmark", "LIBOR", {}],  ["LIBOR", "MINIMUM_VALUE", "Index Floor Rate", {}]
"""


def generate_prompt(data) -> str:
    return f"""
Data: {str(data)}
"""


def generate_prompt_for_question(entity_relationship: str, question: str) -> str:
    return f"""
You are a graph data scientist working for a company that is building a graph database. Your task is to extract information from questions and convert it to cypher query.
The following is the nodes and relationships of the graph database:

{entity_relationship}

Example:\
Question: How is the interest rate calculated?\
Cypher: MATCH (a:FinancialTerm)-[b:SUM_OF]->(c:FinancialTerm), (d:FinancialTerm)-[e:Initial_Benchmark]->(f:LIBOR), (g:FinancialTerm)-[h:MINIMUM_VALUE]->(i:FinancialTerm)  RETURN a,b,c,d,e,f,g,h,i\

Now do the same for the following questions:

The following is the question:
{question}\

Return the cypher code to query the graph database. Only write the code.
"""


def text2triplets(text: str):
    entity_rel_output = openai.ChatCompletion.create(
        engine=openai_deployment,  # engine = "deployment_name".
        messages=[
            {"role": "system", "content": generate_system_message()},
            {"role": "user", "content": generate_prompt(text)}
        ]
    )
    entity_rel_output = entity_rel_output['choices'][0]['message']['content']
    return entity_rel_output


def question2cypher(entity_relationship: str, question: str) -> str:
    cypher_output = openai.ChatCompletion.create(
        engine=openai_deployment,
        messages=[
            {"role": "system",
             "content": generate_prompt_for_question(question=question, entity_relationship=entity_relationship)},
            {"role": "user", "content": question}
        ]
    )
    cypher_output = cypher_output['choices'][0]['message']['content']
    return cypher_output


def generate_prompt_cypher2text(cypher: str, context: str):
    return f"""
You are a graph data scientist working for a company that is building a graph database. Your task is to convert cypher query to natural language text.
The following is the cypher query:\
{cypher}\

Using the logic above, extract the answer to the question from the below context:\
{context}\
"""


def generate_prompt_relation2text(relations: str, context: str):
    return f"""
You are a graph data scientist working for a company that is building a graph database. Your task is to convert relations query to natural language text. use the financial term , relation and initial benchmark terms to generate the relations in the result. and also include the relevant terms which is present in the relations into the result
The following is the relation query:\
{relations}\

Using the logic above, extract the answer to the question from the below context:\
{context}\
"""


def cypher2text(cypher: str, question: str, text: str) -> str:
    print("cypher_question:   ", question)
    openai.api_key = openai_gpt_key
    output = openai.ChatCompletion.create(
        model='gpt-3.5-turbo-16k',
        temperature=0,
        messages=[
            {"role": "system", "content": generate_prompt_relation2text(relations=cypher, context=text)},
            {"role": "user", "content": question}
        ]
    )
    output = output['choices'][0]['message']['content']
    return output


# function to create the and store to the session
def createAndStoreNode(query):
    query.run("CREATE (query:HelloWorld {message: 'Hello World'})")


# def generate_response(prompt_with_examples, new_question):
#     # Use OpenAI API for few-shot learning
#     response = openai.Completion.create(
#         engine="text-davinci-003",
#         prompt=prompt_with_examples + new_question,
#         max_tokens=256
#     )
#
#     # Get the generated response
#     generated_text = response['choices'][0]['text']
#     return generated_text


def generate_response(prompt_with_examples, question):
    print("question_generate_response:",question)

    # Use OpenAI API for few-shot learning
    openai.api_key = openai_gpt_key
    openai.api_type = "openai"
    openai.api_base = "https://api.openai.com/v1"
    openai.api_version = "2020-11-07"

    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt_with_examples + question,
        temperature=0,
        max_tokens=256,
        top_p=0.2,
        frequency_penalty=0,
        presence_penalty=0
    )
    # Get the generated response
    generated_text = response.choices[0].text
    generated_text.replace("?\n", "")
    print("GenetatedCypher_query:", generated_text)
    return generated_text


# query to get the result
# def getResult(query):
#     print("query", query)
#     result = query.run("""MATCH (a:FinancialTerm)-[b:SUM_OF]->(c:FinancialTerm)
# OPTIONAL MATCH (c:FinancialTerm)-[d:Initial_Benchmark]->(e:FinancialTerm)
# OPTIONAL MATCH (a:FinancialTerm)-[f:SUM_OF]->(g:FinancialTerm {name: "Spread"})
# OPTIONAL MATCH (l:FinancialTerm)-[r:MINIMUM_VALUE]->(ifr:FinancialTerm {name: "Index Floor Rate"})
# OPTIONAL MATCH (ir:FinancialTerm)-[calculated_from:CALCULATED_FROM]->(dr:FinancialTerm {name: "Default Rate"})
# RETURN a, b, c, d, e, g, l, r, ifr, ir, calculated_from, dr
# """)
#     print(result.data())
#     return result.data()

def getResult(query, question):
    print("question_get_result", question)
    generated_cypherGraph_query = generate_response(create_prompt_with_examples(), question)
    result = query.run(str(generated_cypherGraph_query))
    print("results_data", result.data())
    return result.data()


def generate_questions_from_kg_relations(relations):
    return f"""Given a financial agreement that includes terms like 'Spread,' 'Initial Benchmark,' 'Index Floor Rate,' 'Interest Rate,' and their respective relationships represented as {relations}, generate a question to extract relevant information.
     - generate single questions combination of all smaller questions based on the relations and also form the questions for the values for the relations in the same question. sub questions like what is value of spread?,  what is value of index floor rate? etc. combine all the sub questions to form a single question.
       For instance, you can ask questions like:

'What is the spread and what is the initial benchmark and its relation with the spread? What is the index floor percentage in this agreement?' 

- a single question in which all the values can be extracted from the relations provided use the relations to generate the question similar to above.

Now, let's generate a single question based on the relations provided use the relations to generate a single questions that gets results from the context: """


# neo4j integration to generate the knowledge graph
def knowledgeGraph(question):
    # Initializing OpenAI GPT-3 we can use GPT-4 also
    print("Qknowledge_graph:", question)
    openai.api_key = openai_gpt_key
    openai.api_type = "openai"
    openai.api_base = "https://api.openai.com/v1"
    openai.api_version = "2020-11-07"

    neo4j_url = settings.NEO4J_DATABASES['default']['URL']
    neo4j_username = settings.NEO4J_DATABASES['default']['USER']
    neo_4j_password = settings.NEO4J_DATABASES['default']['PASSWORD']
    driver = neo4j.GraphDatabase.driver(neo4j_url, auth=(neo4j_username, neo_4j_password))
    with open('text.txt', encoding='utf-8') as f:
        text = f.read()
    with GraphDatabase.driver(neo4j_url, auth=(neo4j_username, neo_4j_password)) as driver:
        with driver.session() as session:
            # Writing to the database
            # Reading the database and returning the results
            relations = []
            cypherOutput = session.read_transaction(getResult, question)
            for item in cypherOutput:
                for key, value in item.items():
                    if isinstance(value, tuple):
                        source_entity, relation, target_entity = value
                        source_name = source_entity['name']
                        target_name = target_entity['name']
                        relations.append((source_name, relation, target_name))
            answer = cypher2text(cypher=relations, question=question, text=text)
            openai.api_key = openai_gpt_key
            output = openai.ChatCompletion.create(
                model='gpt-3.5-turbo-16k',
                temperature=0,
                messages=[
                    {"role": "system", "content": generate_questions_from_kg_relations(relations=relations)},
                ])
            question = output['choices'][0]['message']['content']
            return relations, answer, question
