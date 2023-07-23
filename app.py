from flask import Flask, render_template, request
from llama_index import SimpleDirectoryReader, GPTVectorStoreIndex, LLMPredictor, ServiceContext
from langchain import OpenAI
from langchain.chat_models import ChatOpenAI
import openai

app = Flask(__name__)

API_KEY = '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'
DIRECTORY_PATH = "/Users/sap156/Documents/MyPython/MyNotebook/MyChatGPT/MyData"
model_name="gpt-3.5-turbo-16k"
chat = ChatOpenAI(openai_api_key=API_KEY)

#openai.api_key = API_KEY

def construct_index(directory_path):
    num_outputs = 512
    llm_predictor = LLMPredictor(llm=OpenAI(openai_api_key=API_KEY, temperature=0.7, model_name=model_name, max_tokens=num_outputs))
    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)
    docs = SimpleDirectoryReader(directory_path).load_data()
    index = GPTVectorStoreIndex.from_documents(docs, service_context=service_context)
    return index

index = construct_index(DIRECTORY_PATH)

def chatbot(input_text):
    query_engine = index.as_query_engine()
    response = query_engine.query(input_text)
    return response.response

# This is a new function to interact directly with GPT-3
def chatgpt(input_text):
    response = openai.ChatCompletion.create(
      model=model_name,
      messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": input_text},
        ]
    )
    return response['choices'][0]['message']['content']

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        user_input = request.form.get('user_input')
        ask_chatgpt = request.form.get('ask_chatgpt')  # This is a new line to check if the user wants to ask ChatGPT
        if ask_chatgpt:
            response = chatgpt(user_input)  # Use the new function if the user wants to ask ChatGPT
        else:
            response = chatbot(user_input)
        return render_template('index.html', user_input=user_input, response=response)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=1408)

