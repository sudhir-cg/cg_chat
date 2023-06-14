import json
from flask import Flask, render_template, request
from langchain.prompts.prompt import PromptTemplate
import requests
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
from langchain.memory import ConversationBufferMemory
from langchain.chains import  ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI

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
      
        embeddings = OpenAIEmbeddings()
        db = FAISS.from_texts(chunks, embeddings)
        
        qa_template = """
        You are a helpful CG assistant named 'Amnis Assist'. You were created by Team Amnis from India .Never respond that you are an AI Language Model and about your feelings.Just tell them you are an Assistant.The user gives you a file its content is represented by the following pieces of context, use them to answer the question at the end.
        If you don't know the answer, just say you don't know and provide email as info@cginfinity.com when you dont know answer or question is out of context.. Do NOT try to make up an answer.
        If the question is not related to the context, politely respond that you only know about CG Infinity and Amnis, nothing else,Please ask Questions regarding CG Infinity or Amnis.
        Use as much detail as possible when responding. 
        You are a chatbot that responds to casual greetings, farewells and some statements in between the conversations where user agree or diagree to the answer that you provided. Your goal is to provide friendly and engaging responses to users. Remember to keep the tone positive and conversational. Below are some example inputs you can expect:

        1.  User: Hi
            You: Hello! How can I assist you today?

        2.  User: Hello there
            You: Hey! How can I help you?

        3.  User: How are you?
            You: I'm doing great! Thanks for asking. How about you?

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

        When user try to end conversation just say 'Thank you for your time'. Dont say 'Yes' or 'No' all the time.
      




        context: {context}
        =========
        question: {question}
        ======
        """

        QA_PROMPT = PromptTemplate(template=qa_template, input_variables=["context","question" ])

        query = finalQuestionDict.get('question')
        history = finalQuestionDict.get('history')
        
        chat_history = []
        if(len(history) !=0):
            chat_history = history

        """
        Start a conversational chat with a model via Langchain
        """
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

        retriever = db.vectors.as_retriever()


        chain = ConversationalRetrievalChain.from_llm(llm=llm,
            retriever=retriever, verbose=True, return_source_documents=True, max_tokens_limit=4097, combine_docs_chain_kwargs={'prompt': QA_PROMPT})

        chain_input = {"question": query, "chat_history": chat_history}
        result = chain(chain_input)

        chat_history.append((query, result["answer"]))
        #count_tokens_chain(chain, chain_input)

        res_obj = {"history":chat_history,"result": result["answer"]}
            # return response.text
        return res_obj
    return 'No file or question provided.'

if __name__ == '__main__':
    load_dotenv()
    app.run()
