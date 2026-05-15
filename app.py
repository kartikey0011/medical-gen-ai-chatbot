from flask import Flask, render_template, jsonify, request
from langchain_groq import ChatGroq
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os


app = Flask(__name__)


load_dotenv()



chatModel = ChatGroq(model="llama-3.1-8b-instant"
)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(chatModel, prompt)




@app.route("/")
def index():
    return render_template('index.html')



@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]

    response = chatModel.invoke(msg)

    return str(response.content)



if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 5000, debug= True)
