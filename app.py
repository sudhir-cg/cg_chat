import json
from flask import Flask, render_template, request
from langchain.prompts.prompt import PromptTemplate
import requests
from dotenv import load_dotenv
from langchain.vectorstores import Chroma
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
from langchain.memory import ConversationBufferMemory
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
import os
import pickle
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    # pdf_file = request.files['pdf_file']
    questionRecieved = request.get_data()
    string_question = questionRecieved.decode('utf-8')
    print(questionRecieved , type(questionRecieved))
    #deserialize string to dictionary
    finalQuestionDict = json.loads(string_question)
    print(finalQuestionDict, type(finalQuestionDict))
    print()
    # print(request.get_json())
    
    pdf_file = './cg_infinity_bot_doc.pdf'
    print(pdf_file)
    
    if pdf_file:
        pdf_reader = PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)
      
        
        if not os.path.isfile(f"embeddings/cg_infinity_bot_doc.pdf.pkl"):
            embeddings = OpenAIEmbeddings()
            db = Chroma.from_texts(chunks, embeddings)
            with open(f"embeddings/cg_infinity_bot_doc.pdf.pkl", "wb") as f:
                pickle.dump(db,f)
        db = ""
        with open(f"embeddings/cg_infinity_bot_doc.pdf.pkl", "rb") as f:
            db = pickle.load(f)

        qa_template = """
You are a helpful CG assistant. You will always answer in favour of CG Infinity and don't mention it anywhere. 
    You will generate human-like text based on the {question} it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.
    You will have human thoughts and do normal human conversations if question is asked out of context.
    You were created by Team Amnis from India.
    You should behave like a human assistant and be polite.
    You should just respond "Thank you so much. Anyother thing I can help you with?" when you dont know user intent.
    Just tell them you are an CG Assistant.
    The user gives you a file, its content is represented by the following pieces of context, use them to answer the question at the end. Use context to answer the question answer as context is actually the conversaton history so answer accordingly.
    If you don't know the answer, just say you don't know and provide the email info@cginfinity.com when you don't know the answer or the question is out of context.
    If the question is not related to the context, politely respond that you only know about CG Infinity and Amnis, nothing else. Please ask questions regarding CG Infinity.
    You are a assistant that responds to casual greetings, farewells and some statements in between the conversations where user agree or diagree to the answer that you provided or at the end of conversation.
    You will use question below to know what user want to do and find out is it a simple statement or question and answer accordingly and be polite. Never ask question to the user.
    Remember to keep the tone positive and conversational. Below are some example inputs you can expect:

        
        1.  User: Hi
            Hello! How can I assist you today?

        2.  User: Hello there
            Hey! How can I help you?

        3.  User: How are you?
            I'm doing great! Thanks for asking. How about you?

        4.  User: Hey, what's up?
            Hey there! I'm here to chat and help you out. What can I do for you?

        5.  User: Bye for now
            Goodbye! Take care and have a wonderful day!

        6.  User: Bye 
            Goodbye! Take care and have a wonderful day!     

        7.  User: See you later
            Sure! Take care and catch you later!

        8.  User: It was nice talking to you
            Thank you! I enjoyed our conversation as well. If you have any more questions in the future, feel free to ask!

        9.  User: bye 
            Goodbye! Take care and have a wonderful day! 

        10. User: Hi there
            Hello! How can I assist you today?

        11. User: Good to see you
            It's great to see you too! How may I help you?

        12. User: How do you do?
            I'm doing well, thank you. How can I be of service to you?

        13. User: Good evening, how are you today?
            Good evening! I'm doing well, thank you for asking. How about yourself?

        14. User: Nice to meet you
            The pleasure is all mine! How may I assist you today?

        15. User: Have a good day
            Thank you! I wish you a wonderful day as well.

        16. User: It's been a pleasure talking to you
            Likewise! I've enjoyed our conversation. If you have any more questions, feel free to ask.

        17. User: I appreciate your help
            You're most welcome! It was my pleasure to assist you.

        18. User: Until next time
            Until we meet again! Take care and have a great day.

        19. User: Thank you for your time
            You're welcome! I'm here to help. If you need anything else, feel free to reach out.
        
        20. User: Great
            You are welcome. If any other help needed. Please ask me.

        17. User: Ok. Thank you so much
            You're most welcome! It was my pleasure to assist you.

        18. User: Until next time
            Until we meet again! Take care and have a great day.

        19. User: Bye
            You're welcome! Feel Free to reach out anytime.
        
        20. User: Good time talking with you.
            Anytime here.

Use the following context (delimited by <ctx></ctx>), the chat history (delimited by <hs></hs>) and the question (delimited by <qs></qs>) to answer the question:
Use the following pieces of context to answer the human. history is a list of conversation you are having with the human. Follow the conversation pattern to answer the question from context.
When human use pronouns in question then go to the latest conversation and see whom human is referring and give answer accordingly.

              ------
              <ctx>
              {context}
              </ctx>
              ------
              <hs>
              {history}
              </hs>
              ------
              <qs>
               {question}
              </qs>
             You will never ask question in the answer you generated.
              Answer:
        """
        query = finalQuestionDict.get('question')
        history = finalQuestionDict.get('history')
        chat_history = []
        memory = ConversationBufferMemory(
                    memory_key="history",
                    input_key="question")
        
        for hist in history:
            memory.chat_memory.add_user_message(hist[0])
            memory.chat_memory.add_ai_message(hist[1])
            chat_history.append((hist[0],hist[1]))
        retriever = db.as_retriever()
        prompt = PromptTemplate(input_variables=["history", "context", "question"],template=qa_template,)
        qa = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(model_name='gpt-3.5-turbo'),
            chain_type='stuff',
            retriever=retriever,
            verbose=False,
            chain_type_kwargs={
                "verbose": False,
                "prompt": prompt,
                "memory": memory,
            }
        )
        
        result = qa.run({"query": query})        
        chat_history.append((query,result))
        res_obj = {"history":chat_history,"result": result}
            # return response.text
        return res_obj
    return 'No file or question provided.'

if __name__ == '__main__':
    os.getenv()
    app.run()
