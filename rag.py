import tiktoken 
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA 
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os 

#환경변수 로드
load_dotenv(r'C:\Users\2020A00155\Desktop\OCRproject\OCR-Business-Assistant\python_test\.env')  # 환경 변수 로드
api_key = os.getenv("api_key")

#gpt 토큰 단위로 인코딩
tokenizer = tiktoken.get_encoding("cl100k_base")

#토큰 수 세기 
def tiktoken_len(text):
    tokens = tokenizer.encode(text)
    return len(tokens)

#파일 불러오기 후 페이지 별로 리스트 형태로 분할
loader = PyPDFLoader(r"C:\Users\2020A00155\Desktop\OCRproject\OCR-Business-Assistant\media\특허와논문정보를활용한OCR기술발전동향예측에관한연구.pdf")
pages = loader.load_and_split()

#분할된 페이지를 청크 단위로 분할 document 형태 
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, 
chunk_overlap = 50, length_function = tiktoken_len)
texts = text_splitter.split_documents(pages)

#임베딩 모델
embeddings_model = OpenAIEmbeddings(openai_api_key = api_key)

#임베딩화 시킨 벡터를 vector store에 임시 저장
db = Chroma.from_documents(texts,embeddings_model)

# qurey = "ocr과 ai"
# docs = db.similarity_search(qurey)
# print(docs[0].page_content) 
#similarity_search_With_score(qurey, k = 유사도가 높은 문서를 상위 몇 개 반환할 건지 ) 유사도 점수 가져오는 함수 

openai = ChatOpenAI(model_name = "gpt-4o-mini",openai_api_key = api_key ,streaming =True, callbacks = [StreamingStdOutCallbackHandler()],temperature = 0)
qa = RetrievalQA.from_chain_type(
    llm =openai,
    chain_type = "stuff",
    #mmr:연관성이 높은 풀 중에 최대한 다양하게 소스를 조합
    retriever =db.as_retriever(search_type = "mmr",search_kwargs= {'k':3, 'fetch_k':10}),
     return_source_documents=True)

qurey = "AI-OCR문자인식기술의중요한부분에 대해 설명해줘"
result = qa(qurey)
print(result)

#pip install Chromadb transformers sentence_transformers openai langchain pypdf
#pip install -U langchain-openai