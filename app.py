
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter  
#from langchain.embeddings import HuggingFaceEmbeddings, SentenceTransformerEmbeddings
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
#from langchain.llms import OpenAI
from htmlTemplate import css, bot_template, user_template 
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
#from langchain.llms import HuggingFaceHub
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI


def get_pdf_text(pdf_docs):
	text=""
	for pdf in pdf_docs:
		pdf_reader = PdfReader(pdf)
		for page in pdf_reader.pages:
			text += page.extract_text()
	return text
			

def get_text_chunks(text):
	text_splitter = CharacterTextSplitter(
		separator ="\n",
		chunk_size =1000,
		chunk_overlap =200,
		length_function =len
		)
	chunks =text_splitter.split_text(text)
	return chunks



def get_vectorstore(text_chunks):
	# embeddings = HuggingFaceEmbeddings(model_name ="sentence-transformers/all-mpnet-base-v2")
	embeddings = OpenAIEmbeddings()
	vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
	return vectorstore



def get_conversation_chain(vectorstore):
	# llm= HuggingFaceHub(repo_id="google/flan-t5-small")
	llm = ChatOpenAI()
	memory = ConversationBufferMemory(memory_key='chat_history', return_messages = True)
	conversation_chain= ConversationalRetrievalChain.from_llm(llm = llm, retriever= vectorstore.as_retriever(),
	memory = memory)
	return conversation_chain
				
def handle_userinput(user_question):
	response = st.session_state.conversation({'question': user_question})
	st.session_state.chat_history = response['chat_history']
	for i, message in enumerate(st.session_state.chat_history):
		if i%2 == 0:
			st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True) 
		else:
			st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

def main():
	load_dotenv()
	st.set_page_config(page_title="Chatbot for solving queries",page_icon=":books:")
	st.write(css, unsafe_allow_html=True)
    
	if "conversation" not in st.session_state:
		st.session_state.conversation = None
  	
	if "chat_history" not in st.session_state:
		st.session_state.chat_history = None
    		
	user_question = st.text_input("Ask questions here!")
	if user_question:
		handle_userinput(user_question)
		
    
     	
	with st.sidebar:
		# PDF Upload Section
		st.subheader("Your documents")
		pdf_docs=st.file_uploader("Upload your pdfs", accept_multiple_files=True)
		if st.button("Press"):
			with st.spinner("Processing"):
        		# Combining all texts from the uploaded PDF
				raw_text = get_pdf_text(pdf_docs)
				#st.write(raw_text)
        		# Get the text Chunks
				text_chunks = get_text_chunks(raw_text)
				#st.write(text_chunks)
        		# Create vector Store
        		
				vectorstore = get_vectorstore(text_chunks)
				
        		# Create conversation
				st.session_state.conversation= get_conversation_chain(vectorstore)        		
	
	# st.session_state.conversation        		
        		
        	          		        		
if __name__ == '__main__':
	main()
