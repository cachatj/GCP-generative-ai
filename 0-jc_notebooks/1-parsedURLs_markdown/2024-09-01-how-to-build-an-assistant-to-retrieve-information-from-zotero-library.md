

Zotero is a fantastic library tool widely used by researchers and students to
manage documents, PDFs, links, and more. However, when your document
collection grows, it becomes challenging to quickly locate specific
information for references or research, especially among numerous books and
links.

That’s where an AI assistant comes in handy! Imagine an AI that scans your
library, providing answers to your queries on a particular topic and even
supplying direct links to the relevant content. This not only saves a
considerable amount of time but proves especially beneficial for subjects you
haven’t explored in a while.

In this article, I’ll guide you through a tutorial on building such an agent
using a vector database combined with a large language model within a RAG
system. The best part? No need for a GPU to run the system.

Let’s break down the steps we’ll cover:

  1. Extracting data from Zotero using the Zotero API
  2. Creating a vector database
  3. Utilizing langchain to develop a chatbot that retrieves information from the vector database
  4. Building a local web application like ChatGPT for seamless operation.

* * *

# **Data preprocessing**

To get data about your library, you need to use the ZoteroAPI, and the library
pyzotero.

## [GitHub - urschrei/pyzotero: Pyzotero: a Python client for the Zotero
APIPyzotero: a Python client for the Zotero API. Contribute to
urschrei/pyzotero development by creating an account
on…github.com](https://github.com/urschrei/pyzotero?source=post_page-----
dfc07c648619--------------------------------)

To connect to the Zotero database, ensure that you obtain the Zotero API key
and the library ID, as explained in the README. With these credentials, we can
retrieve information about your collections and documents from the Zotero
database. By incorporating the path to Zotero storage on your local PC, we can
effortlessly generate a dataframe containing the paths to PDFs or links
associated with specific topics you wish to include in your chatbot. The code
to do that is provided in this link.

## [zoterollm/zoterollm/pyzotero.py at main · vankhoa21991/zoterollmContribute
to vankhoa21991/zoterollm development by creating an account on
GitHub.github.com](https://github.com/vankhoa21991/zoterollm/blob/main/zoterollm/pyzotero.py?source=post_page
-----dfc07c648619--------------------------------)

* * *

# Vector database

## Document loader

As our documents content different types of data such as pdf or urls,
langchain has a variaty loader that can support all types of data in image
below, just with a few line of code and you can read all your data.

In this code, suppose that you have a list of pdf paths and a list of URLs,
you can load them into a single document list.

```
documents = []  
    for pdf_file in pdf_files:  
        loader_pdf = PyPDFLoader(pdf_file)  
        documents.extend(loader_pdf.load())  
      
    loader = WebBaseLoader(web_links)  
    documents.extend(loader_web.load())

```
## Splitter

There are different kind of splitters in langchain that helps you split the
documents created above. In simple terms, text splitters work like this:

  1. Break the text into small, meaningful parts (usually sentences).
  2. Combine these small parts into a larger one until it reaches a certain size (measured by some function).
  3. Once it reaches that size, treat it as its own piece of text and start a new one with some overlap (to maintain context between parts).

You can customize your text splitter in two ways:

  1. How the text is divided.
  2. How the size of each chunk is determined.

## [Recursively split by character | 🦜️🔗 LangchainThis text splitter is the recommended one for generic text. It ispython.langchain.com](https://python.langchain.com/docs/modules/data_connection/document_transformers/recursive_text_splitter?source=post_page-----dfc07c648619--------------------------------)

```
splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)  
    splitter.split_documents(documents)

```
## Embeddings

The base `HuggingFaceEmbedding` class is a generic wrapper around any
HuggingFace model for embeddings.

## [MTEB Leaderboard - a Hugging Face Space by mtebDiscover amazing ML apps
made by the
communityhuggingface.co](https://huggingface.co/spaces/mteb/leaderboard?source=post_page
-----dfc07c648619--------------------------------)

In this tutorial, I use a well known all-MiniLM-L6-v2.

```
embeddings = HuggingFaceEmbeddings(  
                model_name="sentence-transformers/all-MiniLM-L6-v2",  
                model_kwargs={'device': 'cpu'})

```
## FAISS

FAISS, created by Facebook AI, is a library that facilitates efficient
similarity searches. With FAISS, we can index a collection of vectors. By
using another vector, known as the query vector, we can search for the most
similar vectors within the index.

> [Facebook AI Similarity Search
> (Faiss)](https://engineering.fb.com/2017/03/29/data-infrastructure/faiss-a-
> library-for-efficient-similarity-search/) is a library for efficient
> similarity search and clustering of dense vectors. It contains algorithms
> that search in sets of vectors of any size, up to ones that possibly do not
> fit in RAM. It also contains supporting code for evaluation and parameter
> tuning.

However, FAISS does more than just help us build an index and conduct
searches. It significantly enhances search speed, achieving remarkable
performance levels using the embeddings and the split documents.

```
db = FAISS.from_documents(texts, embeddings)  
    db.save_local(output_dir)  
      
    query = "query = What is LLama 2?"  
    docs = db.similarity_search(query, k=5)

```
You can store this database locally for future use. Later on, you can read the
database and use it to retrieve a document based on a query. Additionally, the
database can be employed to retrieve documents that are related to the given
query.

* * *

# QA chain

## Large Language Model

This tutorial will utilize two versions of the llama 2 model: one is the 4-bit
quantized version, and the other is a CPU-only version quantized by llamacpp.
Although other models like phi-2 or mistral are available, we will focus on
llama for this tutorial.

## [meta-llama/Llama-2-7b-chat-hf at mainWe're on a journey to advance and
democratize artificial intelligence through open source and open
science.huggingface.co](https://huggingface.co/meta-llama/Llama-2-7b-chat-
hf/tree/main?source=post_page-----
dfc07c648619--------------------------------)

```
class LLModel:  
     def __init__(self):  
      pass  
      
     def load_llm_4bit(self, model_id, hf_auth, device='cuda:0'):  
      bnb_config = transformers.BitsAndBytesConfig(  
       load_in_4bit=True,  
       bnb_4bit_quant_type='nf4',  
       bnb_4bit_use_double_quant=True,  
       bnb_4bit_compute_dtype=bfloat16  
      )  
      
      config = transformers.AutoConfig.from_pretrained(model_id, trust_remote_code=True)  
      config.init_device = "cuda"  
      
      model = transformers.AutoModelForCausalLM.from_pretrained(  
       model_id, config=config, torch_dtype=torch.bfloat16, trust_remote_code=True,  
       use_auth_token=hf_auth,  
       quantization_config=bnb_config,  
      )  
      model.eval()  
      
      tokenizer = transformers.AutoTokenizer.from_pretrained(  
       model_id,  
       use_auth_token=hf_auth  
      )  
      streamer = TextStreamer(tokenizer, skip_prompt=True)  
      
      generate_text = transformers.pipeline(  
       model=model,  
       tokenizer=tokenizer,  
       # return_full_text=True,  # langchain expects the full text  
       task='text-generation',  
       # we pass model parameters here too  
       temperature=0.1,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max  
       max_new_tokens=256,  # max number of tokens to generate in the output  
       repetition_penalty=1.1,  # without this output begins repeating  
       torch_dtype=torch.float16,  
       device_map='auto',  
       # streamer=streamer,  
       # device='cuda:0'  
      )  
      
      llm = HuggingFacePipeline(pipeline=generate_text)  
      return llm  
      
     def load_llm_cpp(self):  
      model_path = '/data/llm/Llama-2-7b-chat-hf/llama-2-7b-chat-hf.Q4_K_M.gguf'  
      callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])  
      llm = LlamaCpp(model_path=model_path, # model available here: https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/tree/main  
          temperature=0.1,  
             repetition_penalty=1.1,  
            n_batch=256,  
            n_gpu_layers=0,  
           callbacks = callback_manager,  
            verbose=True  
           )  
      
      return llm

```
## Promt template

Ensuring your model produces accurate responses hinges on defining the prompt
effectively. In this scenario, I aim to utilize the retrieval document as
context and include the conversation history in the query. To achieve this, I
introduce two extra input variables in addition to the query within the prompt
template, as illustrated in the following code.

```
template = """Respond with "I will help you with your question using the information i have in these context."   
        Use the provided context to answer the user's question.   
        If you don't know the answer, respond with "I do not know".  
        Previous conversation: {past}  
        Context: {context}  
        Question: {question}  
        My answer is:"""  
      
    prompt = PromptTemplate(  
        template=template,  
        input_variables=['context', 'question', 'past'])

```
## Question answering

If you’ve come this far, here’s what I’ll do next. I’ll check the vector
database for related content to your query and combine that with the Language
Model to give you a complete answer. If you used the gguf version of the model
you don’t need any gpu to have the model works.

```
from langchain.chains import LLMChain  
    query = "What is LLama 2 ?"  
      
    # load vector database and get similar results  
    vectordb = VectorStoreFAISS(output_dir="../vectordb/faiss")  
    db = vectordb.load_vector_store()  
    docs = db.similarity_search(query, k=5)  
      
    # load LLM model  
    llmodel = LLModel()  
    model = llmodel.load_llm_4bit(model_id='/data/llm/Llama-2-7b-chat-hf/', hf_auth=None, device='cuda:0')  
    llm_chain = LLMChain(prompt=prompt_template, llm=model)  
      
    context = [doc.to_json()['kwargs']['page_content'] for doc in docs]  
    context = "\n".join(context)  
      
    response = llm_chain.run(question=query,  
                             context=context,  
                             name="Super chatbot",  
                             past="")

```
Response:

> Llama 2 is an open-source language model developed by Meta AI. It was
> released on July 18, 2023, and is available for free, including for
> commercial use. The model is trained on 2 trillion tokens of text data and
> comes in 4 variants, ranging from 7–70 billion parameters. Llama 2 is
> intended for use in English, with almost 90% of the pre-training data being
> in English.

# Webapp with streamlit

I utilize Streamlit to create a basic web application similar to ChatGPT. It’s
designed to provide answers while considering the ongoing conversation.

As you can see, the chatbot can give me the correct answer together with the
link to the related document as well as my local storage. In the second
question, I didn’t mention the context on purpose, but it still able to
understand the model I want to mention here is the language model.

# Conclusion

As we conclude this tutorial, we’ve navigated through the process of
extracting information from your Zotero, constructing a robust vector
database, and employing a powerful language model for effective retrieval of
answers and sources from queries. For the full implementation, check out the
complete code on GitHub via this link. If you have any questions, don’t
hesitate to reach out!

## 