# import io
import json
import os
import re
# from functools import partial
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union

# import time
import numpy as np
import pandas as pd
import rapidfuzz as rf

from pandas_llm import PandasLLM

try:
    from pydantic.v1 import Extra
except:
    from pydantic import Extra

from langchain.chains.base import Chain
from langchain.chains.combine_documents.reduce import ReduceDocumentsChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.embeddings.openai import OpenAIEmbeddings
# from langchain.callbacks.manager import (AsyncCallbackManagerForChainRun,
#                                          CallbackManagerForChainRun)
from langchain_core.callbacks.manager import (AsyncCallbackManagerForChainRun,
                                              CallbackManagerForChainRun)
# from langchain.schema import Document
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseLanguageModel
# from langchain.prompts import PromptTemplate
# from langchain.prompts.base import BasePromptTemplate
from langchain_core.prompts.base import BasePromptTemplate
from langchain_core.prompts.prompt import PromptTemplate
# from langchain.vectorstores.base import VectorStore
from langchain_core.vectorstores import VectorStore
# from langchain_community.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from parsers_loaders import DocAI_Loader, PyPDF2_TesseractOCR_Loader
from prompts import (QA_PROMPT_REDUCE_TEMPLATE, QA_PROMPT_TEMPLATE,
                     SUMMARY_PROMPT_TEMPLATE)
from vector_stores import (DEFAULT_FAISS_DIRECTORY, DEFAULT_FAISS_INDEX,
                           DEFAULT_MILVUS_CONNECTION, EMBEDDINGS_DIM,
                           DaiVectorStore, PDFQnA_COLLECTION_NAME, get_index)

# from langchain_openai.embeddings import OpenAIEmbeddings


# from langchain.embeddings.huggingface import HuggingFaceEmbeddings


def summarise(text, summ_chain: LLMChain):
    if len(text) <= 100:
        return text
    summary_json = summ_chain.invoke({summ_chain.input_keys[0]: text})
    summary_json = summary_json[list(summary_json.keys())[0]]
    try:
        sum_dict = json.loads(summary_json)
        summary = sum_dict.get('summary', None) or sum_dict.get('Summary', None) or sum_dict.get('3. Summary', None) or sum_dict.get('3. Summary', None)
        kwords = sum_dict.get("Keywords", None) or sum_dict.get("keywords", [])
        if summary is None:
            # print(sum_dict.keys())
            return summary_json.strip('{}\n')
        if kwords is not None and kwords != '':
            return f"{summary}\nKeywords: {kwords}"
        return summary
    except:
        summary = re.split(r"summary|Summary", summary_json, re.I)[-1].strip("\"").strip("}{").strip(":").strip("\n").strip(' ').strip('"')
        kwords = re.split(r"keywords|Keywords", summary_json, re.I)[-1].strip("\"").strip("}{").strip(":").strip("\n").strip(' ').strip('"')
        if summary is None:
            # print("Couldn't parse JSON and split string")
            return summary_json.strip('}{\n ')
        if kwords is not None and kwords != '':
            return f"{summary}\nKeywords: {kwords}"
        return summary
    # return summary_json

def file_data_to_docs(
        fname_to_id: Optional[dict[str, str]],
        filepaths: Optional[Union[str, List[str]]] = None, 
        filenames: Optional[Union[str, List[str]]] = None, 
        data:Optional[Union[str,bytes]] = None, 
        use_ocr: bool = False,
        use_docai: bool = False,
        chunk_size: int = 1500,
        user_id: Optional[str] = None
    ) -> Tuple[List[Document], List[Document] ,List[Document]]:
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=100, add_start_index=True)
    
    if isinstance(filepaths, str):
        filepaths = [filepaths]
    if isinstance(filenames, str):
        filenames = [filenames]
    if fname_to_id is None:
        fname_to_id = {fname: fname.replace(' ','').replace("'","") for fname in filenames}
    if user_id is None:
        user_id = -1
    if use_docai:
        loader = DocAI_Loader(filenames=filenames, file_uris=filepaths, user_id=user_id, output_tables_folder=os.environ.get("TABLES_FOLDERPATH", "table_files"))
    else:
        loader = PyPDF2_TesseractOCR_Loader(filepaths[0], data=data, filename=filenames[0], user_id=user_id, use_ocr=use_ocr, use_docai=use_docai)
    docs = loader.load()
    # Find the total number of words
    # doc_words = {}
    # meta_docs = [doc for doc in docs if doc.metadata['page'] == 0]
    # for md in meta_docs:
    #     nwords = re.findall(r"(\d+) words", md.page_content, flags=re.I)[0]
    #     doc_words[md.metadata['filename']] = int(nwords)
    print("Document parsing complete! Summarising & splitting...")
    #for doc in docs:
    #    print(doc)
    #    print('+'*100)
    sum_chain = LLMChain(llm=ChatOpenAI(temperature=0), prompt=PromptTemplate.from_template(SUMMARY_PROMPT_TEMPLATE))
    sum_docs = []
    mtdocs = []
    tdocs = []
    for doc in docs:
        doc.metadata['id'] = fname_to_id[doc.metadata['filename']]
        if doc.metadata['page'] == 0 or doc.metadata.get('start_index', 0) < 0:
            # Ignore the metadata doc and the per-page table docs
            # if doc.metadata.get('start_index',-1) == -2:
            #     print(doc)
            #     print()
            mtdocs.append(doc)
        else:
            # if doc_words[doc.metadata['filename']] >= 75_000:
            summary = summarise(doc.page_content, sum_chain)
            # else:
            #     summary = doc.page_content
            # print(f"Page {doc.metadata['page']} Summary:\n{summary}\n")
            meta = doc.metadata
            meta['start_index'] = -1
            sdoc = Document(
                page_content=summary,
                metadata=meta
            )
            sum_docs.append(sdoc)
            tdocs.append(doc)
    tdocs = text_splitter.split_documents(tdocs)
    
    return tdocs, sum_docs, mtdocs

def get_pdfs_index(file_id: Optional[str]=None, filepath:Optional[str]=None, data:Optional[Union[str,bytes]]=None, chunk_size=1500,
              embs_model_name="sentence-transformers/all-mpnet-base-v2", collection_name=PDFQnA_COLLECTION_NAME, 
              index_name='milvus',drop_old=False, filename:Optional[str]=None, connection_args=None, 
              faiss_folderpath: str=DEFAULT_FAISS_DIRECTORY, faiss_index_name: str = DEFAULT_FAISS_INDEX
             ):

    # from langchain.vectorstores import Chroma, Milvus
    from vector_stores import DaiChroma, DaiMilvus

    embs = OpenAIEmbeddings() #HuggingFaceEmbeddings(model_name=embs_model_name)

    if filepath is None and data is None:
        return get_index(
            index_name=index_name, 
            collection_name=collection_name,
            embs=embs,
            connection_args=connection_args,
            faiss_folderpath=faiss_folderpath,
            faiss_index_name=faiss_index_name
        )
    if filename is None:
        filename = filepath.split('/')[-1]
    if file_id is None:
        file_id = filename.replace(' ','').replace("'","").lower()
    fname_to_id = {filename: file_id}
    docs, sum_docs, mtdocs = file_data_to_docs(fname_to_id, filepaths=[filepath], data=data, filenames=[filename], chunk_size=chunk_size, use_ocr=False, use_docai=True)
    
    # vector_index = Chroma(collection_name='temp', embedding_function=embs)
    if index_name.lower() == 'chroma':
        vector_index = DaiChroma.from_documents(docs, collection_name=collection_name, embedding=embs)
        vector_index.add_documents(sum_docs)
        vector_index.add_documents(mtdocs)
    elif index_name.lower() == 'milvus':
        if connection_args is None:
            connection_args = DEFAULT_MILVUS_CONNECTION
        index_params = {"metric_type": "L2", "index_type": "HNSW", "params": {"M": 8, "efConstruction": 32}}
        search_params = {"metric_type": "L2", "params": {"ef": 10, "radius": 1.5, "range_filter": 0.0}}
        vector_index = DaiMilvus.from_documents(docs+sum_docs+mtdocs, collection_name=collection_name, embedding=embs, connection_args=connection_args,
                                             search_params=search_params, index_params=index_params, drop_old=drop_old)
        # vector_index.add_documents(sum_docs)
        # vector_index.add_documents(mtdocs)
    elif index_name.lower() == 'faiss':
        from langchain.vectorstores import FAISS
        if faiss_folderpath is not None and faiss_index_name is not None and os.path.isdir(faiss_folderpath):
            try:
                vector_index = FAISS.load_local(faiss_folderpath, embs, faiss_index_name)
                vector_index.add_documents(docs+sum_docs+mtdocs)
                # vector_index.add_documents(sum_docs, create_embeddings=False)
            except:
                vector_index = FAISS.from_documents(docs+sum_docs+mtdocs, embedding=embs)
        # raise NotImplementedError("FAISS index has not been implemented")
        else:
            vector_index = FAISS.from_documents(docs, embedding=embs)
    else:
        raise NotImplementedError(f"{index_name} index has not been implemented")
    return vector_index

def update_data(vector_store: DaiVectorStore, file_ids: List[str], filenames :List[str], 
                user_id: str,
                filepaths: Optional[List[str]]=None,
                data: Optional[Union[bytes, str]]=None, chunk_size:int=1500,
                use_ocr: bool=False, use_docai:bool=False, 
                faiss_folderpath=DEFAULT_FAISS_DIRECTORY, faiss_index_name=DEFAULT_FAISS_INDEX):
    

    from google.api_core.exceptions import GoogleAPICallError

    fname_to_id = {filenames[i]: file_ids[i] for i in range(len(filenames))}
    
    try:
        docs, sum_docs, mtdocs = file_data_to_docs(fname_to_id=fname_to_id, filepaths=filepaths, data=data, filenames=filenames, user_id=user_id, chunk_size=chunk_size, use_ocr=use_ocr, use_docai=use_docai)
    except GoogleAPICallError as g:
        print(f"There was a problem with the DocumentAI call due to error\n{g}\n{g.reason} of type {type(g)}")
        return False, f"Couldn't create docs from files: {filenames} due to {e} of {type(e)}"
    except Exception as e:
        print(f"Couldn't create docs from files: {filenames} due to {e} of type {type(e)}")
        return False, f"Couldn't create docs from files: {filenames} due to {e}"
    
    print("All new docs created! Inserting docs in vector store...")
    
    # First delete the existing docs so that there are no 'old' documents
    
    store_type = str(type(vector_store)).lower()
    ids_to_delete = []
    if "milvus" in store_type.lower():
        sfile_ids = [f"'{fn.lower()}'" for fn in file_ids]
        sfile_ids = '[' + ','.join(sfile_ids) + ']'
        ids_to_delete = vector_store.query(
            expr=f"id in {sfile_ids} and user_id == '{user_id}'",
            return_ids=True,
            return_data=False,
            k=None
        )
        
        try:
            vector_store.add_documents(docs+sum_docs+mtdocs)
            # vector_store.add_documents(sum_docs)
            # vector_store.add_documents(mtdocs)
        except Exception as e:
            return False, f"Couldn't add new docs of file, {filenames}, to vector store due to error\n{e}\n of type {type(e)}"
        #print(ids)
        # print(ids_to_delete)
    elif "faiss" in store_type.lower():
        ids = [f"{doc.metadata['user_id']}-{doc.metadata['source']}-{doc.metadata['start_index']}" for doc in docs]
        try:
            deleted = vector_store.delete(ids)
            if deleted:
                print("Previous file docs deleted!")
        except ValueError as ve:
            return False, f"Couldn't delete previous file docs for files: {filenames} from faiss vector store due to:\n{ve}"
        try:
            vector_store.add_documents(docs+sum_docs+mtdocs, ids=ids)
        except:
            return False, f"Couldn't add new docs for file: {filenames} to vector store"
        #print(ids)
        vector_store.save_local(folder_path=faiss_folderpath, index_name=faiss_index_name)
        return
    elif "chroma" in store_type.lower():
        try:
            sfile_ids = [f.lower() for f in file_ids]
            vector_store.delete(
                filter={'id': {"$in": sfile_ids}, 'user_id': user_id}
            )
            print("Previous file docs deleted!")
            vector_store.add_documents(docs+sum_docs+mtdocs)
        except BaseException as be:
            return False, f"There was a problem in adding new file docs files: {filenames} due to:\n{be}\n of type {type(be)}"
    if len(ids_to_delete) > 0: # Delete all the ids to be deleted after new docs have been added
        _ = vector_store.delete(
            ids=ids_to_delete
        )
    #     print("Previous file docs deleted!")
    return True, f"Files {filenames} added successfully"
    # if not filename in self.user_files_loaded[user_id]:
    #     self.user_files_loaded[user_id].add(filename)


def query_table(table_filepath: str, query: str, paraphraser_chain: Optional[LLMChain] = None) -> str:
    # from typing import Iterable
    if not os.path.exists(table_filepath):
        return ""
    try:
        if table_filepath.endswith('.csv'):
            df = pd.read_csv(table_filepath, sep='|')
        elif table_filepath.endswith('.xlsx'):
            df = pd.read_excel(table_filepath)
        elif table_filepath.endswith('.parq'):
            df = pd.read_parquet(table_filepath)
        elif table_filepath.endswith('.json'):
            df = pd.read_json(table_filepath)
        
        if query is None or len(query) <= 5:
            return df.to_csv(sep='|', index=False)
            # return df.to_json(orient='records')
        
        if paraphraser_chain is not None:
            query = paraphraser_chain.run(query)
    except:
        return ""

    # For now, a temporary stop-gap measure until paraphraser_chain is ready
    query = re.sub(r"mentioned .*|present .*", "", query, flags=re.I).strip()

    custom_prompt = """1. Do NOT write code to create data.
    2. If you can't come up with a suitable Python code snippet to fulfill the request, reply with ''.
    3. The Python code snippet should ONLY be about creating a query using the given column names.
    4. Do NOT write Python code to read or write a file. IGNORE all filenames."""
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        conv_df = PandasLLM(data=df, force_sandbox=True, verbose=True, custom_prompt=custom_prompt)
    try:
        result = conv_df.prompt(query)
    except:
        return ""
    code = conv_df.code_block

    # print(f"Code Generated for query,'{query}' :\n`{code}`")
    
    if result is None:
        return ""
    if isinstance(result, float) and np.isnan(result):
        return ""
    if isinstance(result, str) and "try later" in result:
        return ""
    
    if isinstance(result, pd.DataFrame) or isinstance(result, pd.Series):
        if len(result.index) > 0 and len(result.columns) > 0:
            return result.to_json(orient='records')
        return ""
    
    if isinstance(result, Iterable):
        if len(result) == 0:
            return ""
        return f"{result}"
    try:
        result = f"{result}"
    except:
        return ""
    if result.lower() in ['nan', 'error', 'na', 'inf']:
        return ""
    return result
    
class UserFiles:
    def __init__(self, vector_index):
        try:
            from vector_stores import DaiMilvus
        except ImportError:
            raise Exception("Need to install pymilvus using `pip install pymilvus`")

        if not isinstance(vector_index, DaiMilvus):
            self.user_files = {}
            return
        self.user_files: Dict[str, Set] = {}
        ents_count = vector_index.col.num_entities
        if ents_count == 0:
            self.user_files = {}
        else:
            all_ents = vector_index.col.query(expr="",output_fields=["user_id", "filename", "id"], limit=ents_count)
            for e in all_ents:
                if not e['user_id'] in self.user_files:
                    self.user_files[e['user_id']] = set()
                self.user_files[e['user_id']].add((e.get('id', e['filename']), e['filename']))
                    

    def add_file(self, file_id, filename, user_id):
        if not user_id in self.user_files:
            self.user_files[user_id] = set()
        self.user_files[user_id].add((file_id, filename))

    def remove_file(self, file_id, user_id):
        if not user_id in self.user_files:
            print(f"User ID {user_id} has no file stored against it.")
            return
        # if filename.lower() in self.user_files[user_id]:
        #     self.user_files[user_id].remove(filename.lower())
        self.user_files[user_id] = set([ui for ui in self.user_files[user_id] if not file_id in ui])

    def get_files(self, user_id: str = '', return_only_fnames=True, return_only_ids=False):
        if user_id == '':
            if return_only_fnames:
                return {k: [ui[1] for ui in self.user_files[k]] for k in self.user_files.keys()}
            if return_only_ids:
                return {k: [ui[0] for ui in self.user_files[k]] for k in self.user_files.keys()}
            return {k: [f"{ui[0]}--{ui[1]}" for ui in self.user_files[k]] for k in self.user_files.keys()}
        if user_id in self.user_files:
            if return_only_fnames:
                return [ui[1] for ui in self.user_files[user_id]]
            if return_only_ids:
                return [ui[0] for ui in self.user_files[user_id]]
            return [f"{ui[0]}--{ui[1]}" for ui in self.user_files[user_id]] #list(self.user_files[user_id])
        return []

def _match_pattern(regex:str, query:str) -> str:
    try:
        match = re.findall(rf"{regex}", query, flags=re.IGNORECASE)[0]
    except IndexError:
        match = ""
    return match

def _split_query(query:str) -> Tuple[str]:
    fname = _match_pattern("FILENAMES?: ?(.+)", query).strip(' ,').split("USER")[0] #PDFQnA._get_filename(query)
    user_id = _match_pattern("USER ID: ?(\S+)", query).strip(' ,')
    query = re.split(r"USER ID:|FILENAMES?:|PAGES?:", query)[0].replace("QUESTION:","").strip(' ,')
    return query, user_id, fname


class PDFQnA(Chain):

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    input_key: str = "query"
    output_answer_key = "answer"
    output_source_key = "sources"
    output_context_key = "context"
    #output_keys: List[str] = ["answer","sources"]
    user_files: UserFiles
    qa_chain: Optional[Chain] = None
    vector_store: DaiVectorStore
    search_args: dict = {"k": 5} #, "fetch_k": 10}
    _meta_fname_keyname: str = "filename"
    _meta_uid_keyname: str = "user_id"
    _meta_source_keyname: str = "source"
    _meta_page_keyname: str = "page"
    _score_thresh: float = 1.0
    
    @property
    def input_keys(self) -> List[str]:
        """Expect input key.
        :meta private:
        """
        return [self.input_key] #, "filenames", "user_id"]
    
    @property
    def output_keys(self) -> List[str]:
        """Exepected output keys.
        :meta private:
        """
        return [self.output_answer_key] #, self.output_source_key]
    

    @classmethod
    def from_llm_and_vector_index(cls, vector_index:VectorStore, user_files:UserFiles, llm: Optional[BaseLanguageModel]=None, 
                                    chain_type: str = "stuff", verbose: bool = False,
                                    search_kwargs: dict = {"k": 5}, **kwargs):
        
        if llm is not None:
            output_key = "answer"
            if chain_type == 'stuff':
                qa_prompt = kwargs.get("qa_prompt", QA_PROMPT_TEMPLATE)
                if isinstance(qa_prompt,str):
                    qa_prompt = PromptTemplate.from_template(qa_prompt)
                elif not isinstance(qa_prompt, BasePromptTemplate):
                    raise TypeError(f"qa_prompt must either be type str or BasePromptTemplate - Not {type(qa_prompt)}")
                qa_chain = load_qa_with_sources_chain(llm, chain_type="stuff",prompt=qa_prompt, verbose=verbose)
            elif chain_type == "map_reduce":
                map_reduce_chain_kwargs = {}
                
                qa_prompt = kwargs.get("qa_prompt", None)
                if qa_prompt is not None:
                    if isinstance(qa_prompt,str):
                        qa_prompt = PromptTemplate.from_template(qa_prompt)
                    elif not isinstance(qa_prompt, BasePromptTemplate):
                        raise TypeError(f"qa_prompt must either be type str or BasePromptTemplate - Not {type(qa_prompt)}")
                    map_reduce_chain_kwargs["question_prompt"] = qa_prompt
                
                # If combine_prompt is given, change it
                combine_prompt = kwargs.get("qa_combine_prompt", None)
                if combine_prompt is not None:
                    if isinstance(combine_prompt, str):
                        combine_prompt = PromptTemplate.from_template(combine_prompt)
                    elif not isinstance(combine_prompt, BasePromptTemplate):
                        raise TypeError(f"combine_prompt must either be type str or BasePromptTemplate - Not {type(qa_prompt)}")
                    map_reduce_chain_kwargs['combine_prompt'] = combine_prompt

                map_reduce_chain_kwargs['token_max'] = kwargs.get('qa_token_max', 3000) or kwargs.get('token_max', 3000)

                qa_chain = load_qa_with_sources_chain(llm, chain_type="map_reduce", verbose=verbose, **map_reduce_chain_kwargs)
            elif chain_type in ["map_rerank", "refine"]:
                qa_chain = load_qa_with_sources_chain(llm, chain_type=chain_type, verbose=verbose)
            else:
                raise ValueError(
                    f"Got unsupported chain type: {chain_type}.",
                    f"Should be one of ['stuff', 'map_reduce', 'refine', 'map_rerank']."
                )
        else:
            output_key = "context"
            qa_chain = None

        return cls(qa_chain=qa_chain, vector_store=vector_index, output_answer_key=output_key, 
                   search_args=search_kwargs, verbose=verbose, user_files=user_files)
    
    @classmethod
    def from_llm_and_store_type(cls, user_files, vector_store_type: str = "milvus", llm:Optional[BaseLanguageModel] = None, 
                        embeddings:Embeddings = OpenAIEmbeddings(),
                        chain_type: str = "stuff", verbose: bool = False, search_kwargs: dict = {"k":6}, **kwargs):
            
        if vector_store_type.lower() == "chroma":
            # from langchain.vectorstores import Chroma
            from vector_stores import DaiChroma

            collection_name = kwargs.get('chroma_collection_name',PDFQnA_COLLECTION_NAME)
            if collection_name is None:
                raise ValueError("You must pass the collection_name argument for using chromadb vector store")
            
            vector_index = DaiChroma(collection_name, embedding_function=embeddings, client=kwargs.get('chroma_client', None),
                                  persist_directory=kwargs.get('chroma_persist_directory', None), 
                                  client_settings=kwargs.get('chroma_client_settings', None))
        elif vector_store_type.lower() == "milvus":
            # from langchain.vectorstores import Milvus
            from vector_stores import DaiMilvus
            collection_name = kwargs.get('milvus_collection_name', PDFQnA_COLLECTION_NAME)
            if collection_name is None:
                raise ValueError("You must pass the collection_name argument for using Milvus vector store")
            connection_args = kwargs.get("milvus_connection_args", DEFAULT_MILVUS_CONNECTION)
            search_params = kwargs.get("milvus_search_params", None)
            index_params = kwargs.get("milvus_index_params", None)
            vector_index = DaiMilvus(auto_id=True,collection_name=collection_name, embedding_function=embeddings, search_params=search_params, index_params=index_params, connection_args=connection_args)
        elif vector_store_type.lower() == "faiss":
            from langchain.docstore.in_memory import InMemoryDocstore
            from langchain.vectorstores.faiss import (FAISS,
                                                      dependable_faiss_import)

            # raise NotImplementedError("FAISS index has not been implemented")
            
            faiss_folderpath = kwargs.get("faiss_folder_path", DEFAULT_FAISS_DIRECTORY)
            faiss_index_name = kwargs.get("faiss_index_name", DEFAULT_FAISS_INDEX)
            
            if faiss_folderpath is not None and faiss_index_name is not None and \
                os.path.isdir(faiss_folderpath) and \
                os.path.exists(os.path.join(faiss_folderpath,f"{faiss_index_name}.faiss")) and \
                os.path.exists(os.path.join(faiss_folderpath,f"{faiss_index_name}.pkl")):
                try:
                    vector_index = FAISS.load_local(faiss_folderpath, embeddings, faiss_index_name)
                except:
                    faiss = dependable_faiss_import()
                    vector_index = FAISS(embedding_function=embeddings.embed_query, docstore=InMemoryDocstore(), 
                                         index=faiss.IndexFlatL2(EMBEDDINGS_DIM), index_to_docstore_id={}, normalize_L2=True)
            else:
                faiss = dependable_faiss_import()
                vector_index = FAISS(embedding_function=embeddings.embed_query, docstore=InMemoryDocstore(), 
                                     index=faiss.IndexFlatL2(EMBEDDINGS_DIM), index_to_docstore_id={}, normalize_L2=True)
        else:
            raise NotImplementedError(f"{vector_store_type} has not been implemented. Only Milvus, Chroma and FAISS vector stores have been implemented so far.")
        if llm is None:
            qa_chain = None
            output_key = "context"
        else:
            output_key = "answer"
            # chain_type = kwargs.get("qa_chain_type")
            if chain_type == 'stuff':
                qa_prompt = kwargs.get("qa_prompt", QA_PROMPT_TEMPLATE)
                if isinstance(qa_prompt,str):
                    qa_prompt = PromptTemplate.from_template(qa_prompt)
                elif not isinstance(qa_prompt, BasePromptTemplate):
                    raise TypeError(f"qa_prompt must either be type str or BasePromptTemplate - Not {type(qa_prompt)}")
                qa_chain = load_qa_with_sources_chain(llm, chain_type="stuff",prompt=qa_prompt, verbose=verbose)
            elif chain_type == "map_reduce":
                map_reduce_chain_kwargs = {}
                
                qa_prompt = kwargs.get("qa_prompt", None)
                if qa_prompt is not None:
                    if isinstance(qa_prompt,str):
                        qa_prompt = PromptTemplate.from_template(qa_prompt)
                    elif not isinstance(qa_prompt, BasePromptTemplate):
                        raise TypeError(f"qa_prompt must either be type str or BasePromptTemplate - Not {type(qa_prompt)}")
                    map_reduce_chain_kwargs["question_prompt"] = qa_prompt
                
                # If combine_prompt is given, change it
                combine_prompt = kwargs.get("qa_combine_prompt", None)
                if combine_prompt is not None:
                    if isinstance(combine_prompt, str):
                        combine_prompt = PromptTemplate.from_template(combine_prompt)
                    elif not isinstance(combine_prompt, BasePromptTemplate):
                        raise TypeError(f"combine_prompt must either be type str or BasePromptTemplate - Not {type(qa_prompt)}")
                    map_reduce_chain_kwargs['combine_prompt'] = combine_prompt

                map_reduce_chain_kwargs['token_max'] = kwargs.get('qa_token_max', 3000) or kwargs.get('token_max', 3000)

                qa_chain = load_qa_with_sources_chain(llm, chain_type="map_reduce", verbose=verbose, **map_reduce_chain_kwargs)


        return cls(qa_chain=qa_chain, vector_store=vector_index, output_answer_key=output_key, 
                   search_args=search_kwargs, verbose=verbose, user_files=user_files)
    
    def _split_sources(self, answer: str) -> Dict[str, str]:
        """Split sources from answer."""
        
        if re.search(r"SOURCES:\s?", answer): #, re.IGNORECASE
            try:
                answer, sources = re.split(
                    r"SOURCES:\s?|QUESTION:\s", answer
                )[:2] #, flags=re.IGNORECASE
                sources = re.split(r"\n", sources)[0].strip()
            except:
                sources = ""
        else:
            sources = ""
        sources = sources.split('_')
        if len(sources) > 1:
            sources = '_'.join(sources[1:])
        else:
            sources = sources[-1]
        return {self.output_answer_key:answer, self.output_source_key:sources}
    
    '''
    def prep_inputs(self, inputs: Union[Dict[str, Any], Any]) -> Dict[str, str]:
        if not isinstance(inputs, dict):
            _input_keys = set(self.input_keys)
            if self.memory is not None:
                # If there are multiple input keys, but some get set by memory so that
                # only one is not set, we can still figure out which key it is.
                _input_keys = _input_keys.difference(self.memory.memory_variables)
            if len(_input_keys) != 1:
                raise ValueError(
                    f"A single string input was passed in, but this chain expects "
                    f"multiple inputs ({_input_keys}). When a chain expects "
                    f"multiple inputs, please call it by passing in a dictionary, "
                    "eg `chain({'foo': 1, 'bar': 2})`"
                )
            inputs = {list(_input_keys)[0]: inputs}
        else:
            #######################################################################
            # This `else` Block Is My Own Changes Made to Incorporate user_id and filenames
            #######################################################################
            _input_keys = []
            for ik in self.input_keys:
                if not ik in self.memory.memory_variables:
                    _input_keys.append(ik)
            
            if len(_input_keys) == 0:
                raise ValueError("No agent inputs found in input!")
            
            query = inputs.pop('query',None) or inputs[_input_keys[0]]
            user_id = inputs.pop('user_id', "-1")
            filenames = inputs.pop('filenames',[])
            filenames = ','.join(filenames)
            if user_id is not None:
                query = f"QUESTION: {query} USER ID: {user_id}"
            else:
                query = f"QUESTION: {query}"
            
            if filenames is not None and filenames != "":
                query = f"{query} FILENAMES: {filenames}"

            # if filenames is not None and filenames != "":
            #     query = re.sub(r"this file|this document|current document|current file|this pdf|current pdf", f"the {filename} file", query, flags=re.I)
            inputs[_input_keys[0]] = query
            #######################################################################
            # END OF CHANGES
            #######################################################################
        
        
        if self.memory is not None:
            external_context = self.memory.load_memory_variables(inputs)
            inputs = dict(inputs, **external_context)
        self._validate_inputs(inputs)
        return inputs
    '''

    def _call(self, inputs: Dict[str, Any], run_manager: CallbackManagerForChainRun | None = None) -> Dict[str, Any]:
        query = inputs[self.input_key]
        #print(query)
        
        pages = re.findall(r"pages? (\d+ ?[(,|and) ?\d]+)", query, flags=re.I)
        if len(pages) > 0:
            pages = [p.strip()  for ps in pages for p in re.split(',|and', ps) if p.strip() != '']
        
        query, user_id, ufnames = _split_query(query)
        ufnames = re.split("PAGES?", ufnames, flags=re.IGNORECASE)[0].strip()
        if len(ufnames.split(',')) == 0:
            # If no filename was given as input, then use re to try and find the filename in the user query
            ufname = _match_pattern("mentioned in (.+)\.pdf|present in (.+)\.pdf|in the (.+)\.pdf file|from (.+)\.pdf|mentioned in (.+) file|the (.+) file|from (.+) file|mentioned in (.+)[\.\?\!]|from the (.+)[\.\?\!]|present in the (.+)[\.\?\!]", query)
            ufnames = ''.join(ufname)
        ufnames = ufnames.lower()
        store_type = str(type(self.vector_store)).lower()
        
        fnames = []
        cur_user_files = self.user_files.get_files(user_id)
        if len(cur_user_files) == 0:
            return {self.output_answer_key: f"Sorry! User {user_id} has not uploaded any file so I cannot answer question. Please try again with a different user id."}
        for ufname in ufnames.split(','):
            fname = rf.process.extract(ufname,cur_user_files,scorer=rf.fuzz.partial_ratio, processor=lambda s: s.lower(),limit=1, score_cutoff=50)
            if len(fname) > 0:
                fname = fname[0][0]
                fnames.append(fname.lower())
        # print(fname)
        #print(pages)
        #print("Inside PDF Q/A after splitting: ", query)
        
        if "chroma" in store_type:
            # file_docs = []
            filters = {}
            if fnames is not None and len(fnames) > 0 and fname != "":
                # query = re.sub(r"this file|file|this document|document|current document|current file|this pdf|current pdf", fname, query, re.I)
                filters[self._meta_fname_keyname] = {"$in": fnames}
            if user_id is not None and user_id != "":
                filters[self._meta_uid_keyname] = user_id
            if pages is not None and len(pages) > 0:
                filters[self._meta_page_keyname] = {"$in": pages}
                
            if len(filters.keys()) == 0:
                filters = None
            
            docs = self.vector_store.max_marginal_relevance_search(query, filter=filters, **self.search_args)
            #docs = self.vector_store.similarity_search(query, filter=filters, **self.search_args)
            # docs += file_docs

            # Remove all duplicate docs
            ndocs = []
            for i in range(len(docs)):
                if not docs[i] in ndocs:
                    ndocs.append(docs[i])
                # else:
                #     del docs[i]
            del docs
            docs = ndocs
            # print(docs)
            
            
            ndocs = []
            used_filters = {}
            for doc in docs:
                fname = doc.metadata[self._meta_fname_keyname]
                pno = doc.metadata[self._meta_page_keyname]
                filters = {"filename": fname, "user_id": user_id, "page": pno}
                if not str(filters) in used_filters:
                    used_filters.add(str(filters))
                    # tfilters = filters
                    # First query for any 'table docs' - doc that contains a list of tables present on this page
                    filters['start_index'] = -2
                    tab_doc = self.vector_store.query(filter=filters, k=1)
                    if len(tab_doc) > 0:
                        tab_doc = tab_doc[0]
                        fresult = ""
                        for tab_fpath in tab_doc.page_content.split(','):
                            result = query_table(tab_fpath, query)
                            if result is not None and isinstance(result, str) and len(result.strip()) > 0:
                                run_manager.on_text(f"Answer from querying table {tab_fpath}: {result}", color="green", end="\n", verbose=self.verbose)
                                fresult += result + ' '
                        if fresult.strip() != "":
                            return {self.output_answer_key: f"{fresult.strip()}\n\nSOURCE: {tab_doc.metadata.get(self._meta_source_keyname, '')}"}
                    # Now, query for more of the regular docs from the same page to get added context
                    filters['start_index'] = {"$gte", 0}
                    adocs = self.vector_store.query(filter=filters, k=None)
                    #adocs = self.vector_store.similarity_search(query, filter=filters, fetch_k=self.search_args.get("fetch_k",10), k=3)
                    #print(adocs)
                    if not doc in adocs:
                        adocs.append(doc)
                    adocs = sorted(adocs, key=lambda a: (a.metadata.get('start_index',0))) #+a.metadata.get('page_start_index',0))
                    all_text = [adoc.page_content for adoc in adocs]
                    # doc.metadata['source'] = f"{fname}: Pg. {pno}"
                    ndocs.append(Document(page_content='\n'.join(all_text), metadata=doc.metadata))
                
            del docs
            docs = ndocs #docs + ndocs
        elif "milvus" in store_type:
            if len(pages) > 0:
                pages = '[' + ", ".join(pages) + ']'
            else:
                pages = ""

            if len(fnames) > 0:
                fnames = [f"'{fn}'" for fn in fnames]
                fnames = '['+ ','.join(fnames) + ']'
            else:
                fnames = ''

            search_args = self.search_args
            expr = None
            
            expr = ["start_index > -1"]
            if fnames is not None and fnames != "":
                expr.append(f"{self._meta_fname_keyname} in {fnames}")
            if user_id is not None and user_id != "":
                expr.append(f"{self._meta_uid_keyname} == '{user_id}'")
            if pages is not None and pages != "":
                expr.append(f"{self._meta_page_keyname} in {pages}")
            
            if len(expr) == 0:
                expr = None
            elif len(expr) == 1:
                expr = expr[0]
            elif len(expr) > 1:
                expr = " and ".join(expr)
            
            docs = self.vector_store.max_marginal_relevance_search(query, expr=expr, **search_args)
            # docs = self.vector_store.similarity_search(query, expr=expr, **search_args) # Return 'k' docs
            
            ndocs = []
            used_expr = set({})
            # used_tables_expr = set({})
            nsources = {}
            for doc in docs:
                fname = doc.metadata[self._meta_fname_keyname]
                pno = doc.metadata[self._meta_page_keyname]
                expr = f"filename == '{fname}' and user_id == '{user_id}' and page == {pno}"
                texpr = expr+f" and start_index == -2"
                # Don't make the same queries more than once
                source = doc.metadata.get(self._meta_source_keyname,'')
                nsources[source] = nsources.pop(source, 0) + 1
                if not expr in used_expr:
                    used_expr.add(expr)
                    tab_doc = self.vector_store.query(expr=texpr, k=1)
                    if len(tab_doc) > 0:
                        tab_doc = tab_doc[0]
                        fresult = ""
                        for tab_fpath in tab_doc.page_content.split(','):
                            result = query_table(tab_fpath, query)
                            if result is not None and isinstance(result, str) and len(result.strip()) > 1:
                                run_manager.on_text(f"Answer from querying table {tab_fpath}: {result}", color="green", end="\n", verbose=self.verbose)
                                fresult += result + ' '
                        if fresult.strip() != "":
                            return {self.output_answer_key: f"{fresult.strip()}\n\nSOURCE: {tab_doc.metadata.get(self._meta_source_keyname, '')}"}
                    # used_tables_expr.add(texpr)
                
                    adocs = self.vector_store.query(expr=expr+f" and start_index >= 0", k=None)
                    #adocs = self.vector_store.similarity_search(query, expr=expr+f" and start_index >= 0", k=3)
                    if not doc in adocs:
                        adocs.append(doc)
                    adocs = sorted(adocs, key=lambda a: (a.metadata.get('start_index',0))) #+a.metadata.get('page_start_index',0))
                    all_text = [adoc.page_content for adoc in adocs]
                    ndocs.append(Document(page_content='\n'.join(all_text), metadata=adocs[0].metadata))
                
            del docs
            docs = ndocs #docs + ndocs
            #docs = list(set(docs))
        else:
            raise TypeError(f"Vector Store of type {store_type} is not supported.")
        
        # Remove all duplicate docs
        ndocs = []
        for i in range(len(docs)):
            if not docs[i] in ndocs:
                ndocs.append(docs[i])
            # else:
            #     del docs[i]
        del docs
        docs = ndocs
            
        if len(docs) == 0:
            return {self.output_answer_key:"No documents found that match your query."}#, self.output_source_key: ""}
        
        if self.qa_chain is None:
            texts = [re.sub(r"\n{2,}","\n",doc.page_content) for doc in docs]
            str_docs = [f"{texts[i]}\n\nSOURCE: {doc.metadata[self._meta_source_keyname]}" for i,doc in enumerate(docs)]
            str_docs = "\n\n".join(str_docs)
            return {self.output_answer_key: str_docs}
        
        answer = self.qa_chain.invoke({'question':query,'input_documents':docs}, return_only_outputs=True, callbacks=run_manager.get_child())
        #answer = {self.output_answer_key: answer['output_text']}
        answer = self._split_sources(answer[list(answer.keys())[0]])
        answer = {self.output_answer_key:f"{answer[self.output_answer_key]}\n\nSOURCES: {answer[self.output_source_key]}"}
        return answer
    
    async def _acall(self, inputs: Dict[str, Any], run_manager: AsyncCallbackManagerForChainRun | None = None) -> Dict[str, Any]:
        query = inputs[self.input_key]
        #print(query)
        # query = re.sub(r"QUESTION:", "", query, re.IGNORECASE).strip()
        #print("Inside PDF Q/A before splitting: ", query)
        # pages = re.findall(r"pages?:? ?(\d+)", query, re.I) #PDFQnA._match_pattern("PAGE:? ?(\d+)", query)
        pages = re.findall(r"pages? (\d+ ?[(,|and) ?\d]+)", query)
        if len(pages) > 0:
            pages = [p.strip()  for ps in pages for p in re.split(',|and', ps) if p.strip() != '']
        
        query, user_id, ufnames = _split_query(query)
        # ufnames = re.split("PAGES?", ufnames, flags=re.IGNORECASE)[0].strip()
        if len(ufnames) == 0:
            # If no filename was given as input, then use re to try and find the filename in the user query
            ufnames = _match_pattern("mentioned in (.+)\.pdf|present in (.+)\.pdf|in the (.+)\.pdf file|from (.+)\.pdf|mentioned in (.+) file|the (.+) file|from (.+) file|mentioned in (.+)[\.\?\!]|from the (.+)[\.\?\!]|present in the (.+)[\.\?\!]", query)
            ufnames = ''.join(ufnames)
        ufnames = ufnames.lower()
        
        fnames = []
        cur_user_files = self.user_files.get_files(user_id)
        if len(cur_user_files) == 0:
            return {self.output_answer_key: f"Sorry! User {user_id} has not uploaded any file so I cannot answer the question. Please try again with a different user id or upload a file before asking the question."}
        if len(ufnames) > 0:
            for ufname in ufnames.split(','):
                fname = rf.process.extract(ufname,cur_user_files,scorer=rf.fuzz.partial_ratio, processor=lambda s: s.lower(),limit=1, score_cutoff=70)
                if len(fname) > 0:
                    fname = fname[0][0]
                    fnames.append(fname.lower())
        
        
        # print(fname)
        #print(pages)
        #print("Inside PDF Q/A after splitting: ", query)
        store_type = str(type(self.vector_store)).lower()
        if "chroma" in store_type:
            # file_docs = []
            filters = {}
            if fnames is not None and len(fnames) > 0 and fname != "":
                # query = re.sub(r"this file|file|this document|document|current document|current file|this pdf|current pdf", fname, query, re.I)
                filters[self._meta_fname_keyname] = {"$in": fnames}
            if user_id is not None and user_id != "":
                filters[self._meta_uid_keyname] = user_id
            if pages is not None and len(pages) > 0:
                filters[self._meta_page_keyname] = {"$in": pages}
                
            if len(filters.keys()) == 0:
                filters = None
            #docs = await self.vector_store.asimilarity_search(query, filter=filters, **self.search_args)
            docs = await self.vector_store.amax_marginal_relevance_search(query, filter=filters, **self.search_args)
            # docs += file_docs

            # Remove all duplicate docs
            ndocs = []
            for i in range(len(docs)):
                if not docs[i] in ndocs:
                    ndocs.append(docs[i])
                # else:
                #     del docs[i]
            del docs
            docs = ndocs
            # print(docs)
            
            
            ndocs = []
            used_filters = {}
            for doc in docs:
                fname = doc.metadata[self._meta_fname_keyname]
                pno = doc.metadata[self._meta_page_keyname]
                filters = {"filename": fname, "user_id": user_id, "page": pno}
                if not str(filters) in used_filters:
                    used_filters.add(str(filters))
                    # tfilters = filters
                    # First query for any 'table docs' - doc that contains a list of tables present on this page
                    filters['start_index'] = -2
                    tab_doc = self.vector_store.query(filter=filters, k=1)
                    if len(tab_doc) > 0:
                        tab_doc = tab_doc[0]
                        fresult = ""
                        for tab_fpath in tab_doc.page_content.split(','):
                            result = query_table(tab_fpath, query)
                            if result is not None and isinstance(result, str) and len(result.strip()) > 0:
                                run_manager.on_text(f"Answer from querying table {tab_fpath}: {result}", color="green", end="\n", verbose=self.verbose)
                                fresult += result + ' '
                        if fresult.strip() != "":
                            return {self.output_answer_key: f"{fresult.strip()}\n\nSOURCE: {tab_doc.metadata.get(self._meta_source_keyname, '')}"}
                    # Now, query for more of the regular docs from the same page to get added context
                    filters['start_index'] = {"$gte", 0}
                    #adocs = await self.vector_store.asimilarity_search(query, filter=filters, fetch_k=self.search_args.get("fetch_k",10), k=3)
                    adocs = self.vector_store.query(filter=filters, k=None)
                    #print(adocs)
                    if not doc in adocs:
                        adocs.append(doc)
                    adocs = sorted(adocs, key=lambda a: (a.metadata.get('start_index',0))) #+a.metadata.get('page_start_index',0))
                    all_text = [adoc.page_content for adoc in adocs]
                    # doc.metadata['source'] = f"{fname}: Pg. {pno}"
                    ndocs.append(Document(page_content='\n'.join(all_text), metadata=doc.metadata))
                
            del docs
            docs = ndocs #docs + ndocs
        elif "milvus" in store_type:
            if len(pages) > 0:
                pages = '[' + ", ".join(pages) + ']'
            else:
                pages = ""

            if len(fnames) > 0:
                fnames = [f"'{fn}'" for fn in fnames]
                fnames = '['+ ','.join(fnames) + ']'
            elif len(ufnames) > 0:
                return {self.output_answer_key: f"No docs found that match your query - User {user_id} has not uploaded any of the given filenames: {ufnames.split(',')}"}
            else:
                fnames = ''
            search_args = self.search_args
            expr = None
            # file_docs = []
            # if fnames is not None and fnames != "":
            #     # query = re.sub(r"this file|this document|current document|current file|this pdf|current pdf", fname, query, flags=re.I)
            #     expr = f"{self._meta_fname_keyname} in {fname}"
            #     if user_id is not None and user_id != "":
            #         expr += f" and {self._meta_uid_keyname} == '{user_id}'"
            #     if pages is not None and pages != "":
            #         expr += f" and {self._meta_page_keyname} in {pages}"
            #     expr += " and start_index > -1"
                # file_docs += self.vector_store.similarity_search(query, expr=expr, k=4) # Return k docs
                # search_args['k'] = max(search_args['k'] - 4, 4)
            
            # if page is not None and page != "":
            #     expr = f"{self._meta_page_keyname} == {page}"
            #     if user_id is not None and user_id != "":
            #         expr += f" and {self._meta_uid_keyname} == '{user_id}'"
            #     file_docs += self.vector_store.similarity_search(query, expr=expr, k=2) # Return k docs
            
            expr = ["start_index > -1"]
            if fnames is not None and fnames != "" and len(fnames) > 0:
                expr.append(f"{self._meta_fname_keyname} in {fnames}")
            if user_id is not None and user_id != "":
                expr.append(f"{self._meta_uid_keyname} == '{user_id}'")
            if pages is not None and pages != "":
                expr.append(f"{self._meta_page_keyname} in {pages}")
            
            if len(expr) == 0:
                expr = None
            elif len(expr) == 1:
                expr = expr[0]
            elif len(expr) > 1:
                expr = " and ".join(expr)
            
            # docs = await self.vector_store.asimilarity_search(query, expr=expr, **search_args) # Return 'k' docs
            docs = await self.vector_store.amax_marginal_relevance_search(query, expr=expr, **search_args)
            ndocs = []
            used_expr = set({})
            # used_tables_expr = set({})
            for doc in docs:
                fname = doc.metadata[self._meta_fname_keyname]
                pno = doc.metadata[self._meta_page_keyname]
                expr = f"filename == '{fname}' and user_id == '{user_id}' and page == {pno}"
                texpr = expr+f" and start_index == -2"
                # Don't make the same queries more than once
                if not expr in used_expr:
                    used_expr.add(expr)
                    tab_doc = self.vector_store.query(expr=texpr, k=1)
                    if len(tab_doc) > 0:
                        tab_doc = tab_doc[0]
                        # fresult = ""
                        tables = []
                        for tab_fpath in tab_doc.page_content.split(','):
                            csv = query_table(tab_fpath, query="")
                            tables.append(csv)
                        
                        #     result = query_table(tab_fpath, query)
                        #     if result is not None and isinstance(result, str) and len(result.strip()) > 0:
                        #         await run_manager.on_text(f"Answer from querying table {tab_fpath}: {result}", color="green", end="\n", verbose=self.verbose)
                        #         fresult += result + ' '
                        # if fresult.strip() != "" and str(fresult.strip()) != '[]' and len(fresult) > 0:
                        #     return {self.output_answer_key: f"{fresult.strip()}\n\nSOURCE: {tab_doc.metadata.get(self._meta_source_keyname, '')}"}
                        tab_doc.page_content = '\n\n'.join(tables)
                    else:
                        tab_doc = None
                    # used_tables_expr.add(texpr)
                    adocs = self.vector_store.query(expr=expr+f" and start_index >= 0", k=None)
                    # adocs = [tab_doc] + adocs
                    if not tab_doc is None:
                        adocs.insert(0, tab_doc)
                    #adocs = await self.vector_store.asimilarity_search(query, expr=expr+f" and start_index >= 0", k=3)
                    if not doc in adocs:
                        adocs.append(doc)
                    adocs = sorted(adocs, key=lambda a: (a.metadata.get('start_index',0))) #+a.metadata.get('page_start_index',0))
                    all_text = [adoc.page_content for adoc in adocs]
                    ndocs.append(Document(page_content='\n'.join(all_text), metadata=adocs[0].metadata))
                
            del docs
            docs = ndocs #docs + ndocs
            #docs = list(set(docs))
        else:
            raise TypeError(f"Vector Store of type {store_type} is not supported.")
        
        # Remove all duplicate docs
        ndocs = []
        for i in range(len(docs)):
            if not docs[i] in ndocs:
                ndocs.append(docs[i])
            # else:
            #     del docs[i]
        del docs
        docs = ndocs
            
        if len(docs) == 0:
            return {self.output_answer_key:"No documents found that match your query."}#, self.output_source_key: ""}
        
        if self.qa_chain is None:
            texts = [re.sub(r"\n{2,}","\n",doc.page_content) for doc in docs]
            str_docs = [f"{texts[i]}\n\nSOURCE: {doc.metadata[self._meta_source_keyname]}" for i,doc in enumerate(docs)]
            str_docs = "\n\n".join(str_docs)
            return {self.output_answer_key: str_docs}
        
        answer = await self.qa_chain.ainvoke({'question':query,'input_documents':docs}, return_only_outputs=True, callbacks=run_manager.get_child())
        #answer = {self.output_answer_key: answer['output_text']}
        answer = self._split_sources(answer[list(answer.keys())[0]])
        answer = {self.output_answer_key:f"{answer[self.output_answer_key]}\n\nSOURCES: {answer[self.output_source_key]}"}
        return answer

class PDFQnAFile(Chain):
    
    class Config:
        """Configuration for this pydantic object."""

        extra = "forbid" #Extra.forbid
        arbitrary_types_allowed = True

    input_key: str = "query"
    output_answer_key = "answer"
    output_source_key = "sources"
    #output_keys: List[str] = ["answer","sources"]
    vector_store: VectorStore
    reduce_chain: Chain
    user_files: UserFiles
    _meta_fname_keyname: str = "filename"
    _meta_uid_keyname: str = "user_id"
    _meta_source_keyname: str = "source"
    _meta_page_keyname: str = "page"
    
    
    @property
    def input_keys(self) -> List[str]:
        """Expected input key.
        :meta private:
        """
        return [self.input_key]
    
    @property
    def output_keys(self) -> List[str]:
        """Exepected output keys.
        :meta private:
        """
        return [self.output_answer_key] #, self.output_source_key]
    
    @classmethod
    def from_llm_and_vector_index(cls, vector_index:VectorStore, user_files:UserFiles, llm: BaseLanguageModel,
                                  verbose: bool = False, **kwargs):
        

        store_type = str(type(vector_index)).lower()
        if not any([x in store_type for x in ["milvus", "faiss", "chroma"]]):
            raise TypeError("Vector Index can only be either of Milvus, FAISS or Chroma")

        # If combine_prompt is given, change it
        combine_prompt = kwargs.get("qa_combine_prompt", QA_PROMPT_REDUCE_TEMPLATE)
        
        if isinstance(combine_prompt, str):
            combine_prompt = PromptTemplate.from_template(combine_prompt)
        elif not isinstance(combine_prompt, BasePromptTemplate):
            raise TypeError(f"combine_prompt must either be type str or BasePromptTemplate - Not {type(combine_prompt)}")
        
        
        reduce_chain = LLMChain(llm=llm, prompt=combine_prompt, verbose=verbose)

        combine_documents_chain = StuffDocumentsChain(
            llm_chain=reduce_chain, 
            document_prompt=PromptTemplate.from_template("{page_content}"),
            document_variable_name="doc_summaries",
            verbose=verbose
        )

        # Combines and iteravely reduces the mapped documents
        reduce_documents_chain = ReduceDocumentsChain(
            # This is final chain that is called.
            combine_documents_chain=combine_documents_chain,
            # If documents exceed context for `StuffDocumentsChain`
            collapse_documents_chain=combine_documents_chain,
            # The maximum number of tokens to group documents into.
            token_max=kwargs.get('token_max',2800),
        )
        

        return cls(reduce_chain=reduce_documents_chain, vector_store=vector_index, verbose=verbose, user_files=user_files)
    
    
    @classmethod
    def from_llm_and_store_type(cls, llm:BaseLanguageModel, user_files:UserFiles, vector_store_type: str = "milvus",
                        embeddings:Embeddings = OpenAIEmbeddings(),
                        search_kwargs: dict = {"k":6}, verbose: bool = False, **kwargs):
            
        if vector_store_type.lower() == "chroma":
            # from langchain.vectorstores import Chroma
            from vector_stores import DaiChroma

            collection_name = kwargs.get('chroma_collection_name',PDFQnA_COLLECTION_NAME)
            if collection_name is None:
                raise ValueError("You must pass the collection_name argument for using chromadb vector store")
            
            vector_index = DaiChroma(collection_name, embedding_function=embeddings, client=kwargs.get('chroma_client', None),
                                  persist_directory=kwargs.get('chroma_persist_directory', None), 
                                  client_settings=kwargs.get('chroma_client_settings', None))
        elif vector_store_type.lower() == "milvus":
            # from langchain.vectorstores import Milvus
            from vector_stores import DaiMilvus
            collection_name = kwargs.get('milvus_collection_name', PDFQnA_COLLECTION_NAME)
            if collection_name is None:
                raise ValueError("You must pass the collection_name argument for using Milvus vector store")
            connection_args = kwargs.get("milvus_connection_args", DEFAULT_MILVUS_CONNECTION)
            search_params = kwargs.get("milvus_search_params", None)
            index_params = kwargs.get("milvus_index_params", None)
            vector_index = DaiMilvus(auto_id=True,collection_name=collection_name, embedding_function=embeddings, search_params=search_params, index_params=index_params, connection_args=connection_args)
        elif vector_store_type.lower() == "faiss":
            from langchain.docstore.in_memory import InMemoryDocstore
            from langchain.vectorstores.faiss import (FAISS,
                                                      dependable_faiss_import)

            # raise NotImplementedError("FAISS index has not been implemented")
            
            faiss_folderpath = kwargs.get("faiss_folder_path", DEFAULT_FAISS_DIRECTORY)
            faiss_index_name = kwargs.get("faiss_index_name", DEFAULT_FAISS_INDEX)
            
            if faiss_folderpath is not None and faiss_index_name is not None and \
                os.path.isdir(faiss_folderpath) and \
                os.path.exists(os.path.join(faiss_folderpath,f"{faiss_index_name}.faiss")) and \
                os.path.exists(os.path.join(faiss_folderpath,f"{faiss_index_name}.pkl")):
                try:
                    vector_index = FAISS.load_local(faiss_folderpath, embeddings, faiss_index_name)
                except:
                    faiss = dependable_faiss_import()
                    vector_index = FAISS(embedding_function=embeddings.embed_query, docstore=InMemoryDocstore(), 
                                         index=faiss.IndexFlatL2(EMBEDDINGS_DIM), index_to_docstore_id={}, normalize_L2=False)
            else:
                faiss = dependable_faiss_import()
                vector_index = FAISS(embedding_function=embeddings.embed_query, docstore=InMemoryDocstore(), 
                                     index=faiss.IndexFlatL2(EMBEDDINGS_DIM), index_to_docstore_id={}, normalize_L2=False)
        else:
            raise NotImplementedError(f"{vector_store_type} has not been implemented. Only Milvus, Chroma and FAISS vector stores have been implemented so far.")
        
        # If combine_prompt is given, change it
        combine_prompt = kwargs.get("qa_combine_prompt", QA_PROMPT_REDUCE_TEMPLATE)
        
        if isinstance(combine_prompt, str):
            combine_prompt = PromptTemplate.from_template(combine_prompt)
        elif not isinstance(combine_prompt, BasePromptTemplate):
            raise TypeError(f"combine_prompt must either be type str or BasePromptTemplate - Not {type(combine_prompt)}")
        
        
        reduce_chain = LLMChain(llm=llm, prompt=combine_prompt, verbose=verbose)

        combine_documents_chain = StuffDocumentsChain(
            llm_chain=reduce_chain, 
            document_prompt=PromptTemplate.from_template("{page_content}"),
            document_variable_name="doc_summaries",
            verbose=verbose,

        )

        # Combines and iteravely reduces the mapped documents
        reduce_documents_chain = ReduceDocumentsChain(
            # This is final chain that is called.
            combine_documents_chain=combine_documents_chain,
            # If documents exceed context for `StuffDocumentsChain`
            collapse_documents_chain=combine_documents_chain,
            # The maximum number of tokens to group documents into.
            token_max=kwargs.get('token_max',4000),
        )
                    
        return cls(reduce_chain=reduce_documents_chain, vector_store=vector_index, 
                   search_args=search_kwargs, verbose=verbose, user_files=user_files)
    
    def _split_sources(self, answer: str, alt_source: Optional[str] = None) -> Dict[str, str]:
        """Split sources from answer."""
        if re.search(r"SOURCES:\s?", answer): #, re.IGNORECASE
            try:
                answer, sources = re.split(
                    r"SOURCES:\s?|QUESTION:\s", answer
                )[:2] #, flags=re.IGNORECASE
                sources = re.split(r"\n", sources)[0].strip()
            except:
                sources = alt_source if alt_source is not None else ""
        else:
            sources = alt_source if alt_source is not None else ""
        
        sources = sources.split('_')
        if len(sources) > 1:
            sources = '_'.join(sources[1:])
        else:
            sources = sources[-1]
        return {self.output_answer_key:answer, self.output_source_key:sources}
    '''
    def prep_inputs(self, inputs: Union[Dict[str, Any], Any]) -> Dict[str, str]:
        if not isinstance(inputs, dict):
            _input_keys = set(self.input_keys)
            if self.memory is not None:
                # If there are multiple input keys, but some get set by memory so that
                # only one is not set, we can still figure out which key it is.
                _input_keys = _input_keys.difference(self.memory.memory_variables)
            if len(_input_keys) != 1:
                raise ValueError(
                    f"A single string input was passed in, but this chain expects "
                    f"multiple inputs ({_input_keys}). When a chain expects "
                    f"multiple inputs, please call it by passing in a dictionary, "
                    "eg `chain({'foo': 1, 'bar': 2})`"
                )
            inputs = {list(_input_keys)[0]: inputs}
        else:
            #######################################################################
            # This `else` Block Is My Own Changes Made to Incorporate user_id and filenames
            #######################################################################
            _input_keys = []
            for ik in self.input_keys:
                if not ik in self.memory.memory_variables:
                    _input_keys.append(ik)
            
            if len(_input_keys) == 0:
                raise ValueError("No agent inputs found in input!")
            
            query = inputs.pop('query',None) or inputs[_input_keys[0]]
            user_id = inputs.pop('user_id', "-1")
            filenames = inputs.pop('filenames',[])
            filenames = filenames[0]
            if user_id is not None:
                query = f"QUESTION: {query} USER ID: {user_id}"
            else:
                query = f"QUESTION: {query}"
            
            if filenames is not None and filenames != "":
                query = f"{query} FILENAME: {filenames}"

            # if filenames is not None and filenames != "":
            #     query = re.sub(r"this file|this document|current document|current file|this pdf|current pdf", f"the {filename} file", query, flags=re.I)
            inputs[_input_keys[0]] = query
            #######################################################################
            # END OF CHANGES
            #######################################################################
        
        
        if self.memory is not None:
            external_context = self.memory.load_memory_variables(inputs)
            inputs = dict(inputs, **external_context)
        self._validate_inputs(inputs)
        return inputs
    '''

    def _call(self, inputs: Dict[str, Any], run_manager: CallbackManagerForChainRun | None = None) -> Dict[str, Any]:
        query = inputs[self.input_key]
        
        query, user_id, ufname = _split_query(query)
        if len(ufname) == 0:
            # If no filename was given as input, then use re to try and find the filename in the user query
            ufname = _match_pattern("mentioned in (.+)\.pdf|present in (.+)\.pdf|in the (.+)\.pdf file|from (.+)\.pdf|mentioned in (.+) file|the (.+) file|from (.+) file|mentioned in (.+)[\.\?\!]|from the (.+)[\.\?\!]|present in the (.+)[\.\?\!]", query)
            ufname = ''.join(ufname)
        ufname = ufname.lower()
        ufnames = ufname.split(',')
        if len(ufnames) > 1:
            qufname = _match_pattern("mentioned in (.+)\.pdf|present in (.+)\.pdf|in the (.+)\.pdf file|from (.+)\.pdf|mentioned in (.+) file|the (.+) file|from (.+) file|mentioned in (.+)[\.\?\!]|from the (.+)[\.\?\!]|present in the (.+)[\.\?\!]", query)
            if len(qufname) > 0:
                qufname = ''.join(qufname).lower()
                ufname = rf.process.extract(qufname, ufnames, scorer=rf.fuzz.partial_ratio, limit=1, score_cutoff=50)
                if len(ufname) > 0:
                    ufname = ufname[0][0]
                else:
                    ufname = ufnames[0]
                # ufname = qufname + ',' + ufname
            else:
                ufname = ufnames[0]
        else:
            ufname = ufnames[0]
        
        
        # print(f"User {user_id} given filename: {ufname}")
        if ufname is None or ufname == '':
            return {self.output_answer_key: "No docs found that match your query - this tool requires at least one filename that has been uploaded but none was provided! Please provide a filename and try again."}
        
        cur_user_files = self.user_files.get_files(user_id)
        if len(cur_user_files) == 0:
            return {self.output_answer_key: f"Sorry! User {user_id} has not uploaded any file so I cannot answer question. Please try again with a different user id."}
        fname = rf.process.extract(ufname, cur_user_files, scorer=rf.fuzz.partial_ratio, processor=lambda s: s.lower(), limit=1, score_cutoff=50)
        # print(fname)
        if len(fname) == 0:
            return {self.output_answer_key: f"Sorry! No docs found that match your query - User {user_id} has not uploaded any file with name {ufname}. Please try again with a different filename! I'm happy to help!"}
        fname = fname[0][0]
        #print("Actual Filename:",fname)
        store_type = str(type(self.vector_store)).lower()
        if "chroma" in store_type or "faiss" in store_type:
            #query = re.sub(r"this file|this document|current document|current file|this pdf|current pdf", fname, query, re.I)
            filters = {self._meta_fname_keyname: fname.lower(), "start_index": -1}
            tab_filters = {self._meta_fname_keyname: fname.lower(), "start_index": -2}
            if user_id is not None and user_id != "":
                filters[self._meta_uid_keyname] = user_id
                tab_filters[self._meta_uid_keyname] = user_id
            docs =  self.vector_store.query(filter=filters, k=None)
            tidocs =  self.vector_store.query(filter=tab_filters, k=None)
            for tidoc in tidocs:
                for tab_fpath in tidoc.page_content.split(','):
                    # if "list" in query.lower():
                    #     tdoc = Document(
                    #         page_content=f"Following are the items listed in this table present on this page: {query_table(tab_fpath, query)}",
                    #         metadata=tidoc.metadata
                    #     )
                    # else:
                    #     tdoc = Document(
                    #         page_content=f"Following are the individual records from a table present on this page: {query_table(tab_fpath, query='')}",
                    #         metadata=tidoc.metadata
                    #     )
                    tdoc = Document(
                        page_content=f"Following is a table present on this page:\n{query_table(tab_fpath, query='')}",
                        metadata=tidoc.metadata
                    )
                    docs.append(tdoc)
        elif "milvus" in store_type:
            #query = re.sub(r"this file|this document|current document|current file|this pdf|current pdf", fname, query, re.I)
            expr = f"{self._meta_fname_keyname} == '{fname}' and start_index <= -1"
            # tab_expr = f"{self._meta_fname_keyname} like '{fname}%' and start_index == -2"
            if user_id is not None and user_id != "":
                expr += f" and {self._meta_uid_keyname} == '{user_id}'"
                # tab_expr += f" and {self._meta_uid_keyname} == '{user_id}'"
            docs = []
            idocs = self.vector_store.query(expr=expr, k=None) # Return k docs
            # tidocs = self.vector_store.query(expr=tab_expr, k=None)
            for tidoc in idocs:
                if tidoc.metadata.get('start_index', -1) == -1:
                    docs.append(tidoc)
                else:
                    if "list" in query.lower():
                        for tab_fpath in tidoc.page_content.split(','):
                            # if "list" in query.lower():
                            #     tdoc = Document(
                            #         page_content=f"Following are the items listed in this table present on this page:\n{query_table(tab_fpath, query)}",
                            #         metadata=tidoc.metadata
                            #     )
                            # else:
                            #     tdoc = Document(
                            #         page_content=f"Following are the records/rows from a table present on this page:\n{query_table(tab_fpath, query='')}",
                            #         metadata=tidoc.metadata
                            #     )
                            tdoc = Document(
                                page_content=f"Following is a table present on {tidoc.metadata['source']}:\n{query_table(tab_fpath, query='')}",
                                metadata=tidoc.metadata
                            )
                            docs.append(tdoc)
        
        if len(docs) == 0:
            return {self.output_answer_key:"No documents found that match your query."} #, self.output_source_key: ""}
        #docs = sorted(docs, key=lambda a: (a.metadata.get('start_index',0)+1)*a.metadata.get("page",1))

        docs = sorted(docs, key=lambda a: a.metadata.get('page'))
        # run_manager.on_text("Retrieved docs:", color="green", end="\n", verbose=self.verbose)
        # run_manager.on_text("\n-----\n".join([doc.page_content for doc in docs]), color="blue", end="\n", verbose=self.verbose)
        
        answer = self.reduce_chain.invoke({"input_documents": docs, "question":query}, callbacks=run_manager.get_child(), return_only_outputs=True)
        
        if isinstance(answer, dict):
            answer = answer[list(answer.keys())[0]]
        
        #answer = self.qa_chain({'question':query,'input_documents':docs}, return_only_outputs=True, callbacks=run_manager.get_child())
        #answer = {self.output_answer_key: answer['output_text']}
        answer = self._split_sources(answer, fname)
        # print(answer[self.output_source_key])
        answer = {self.output_answer_key:f"{answer[self.output_answer_key]}\n\nSOURCES: {answer[self.output_source_key]}"}
        return answer

    async def _acall(self, inputs: Dict[str, Any], run_manager: AsyncCallbackManagerForChainRun | None = None) -> Dict[str, Any]:
        query = inputs[self.input_key]
        
        query, user_id, ufname = _split_query(query)
        if len(ufname) == 0:
            # If no filename was given as input, then use re to try and find the filename in the user query
            ufname = _match_pattern("mentioned in (.+)\.pdf|present in (.+)\.pdf|in the (.+)\.pdf file|from (.+)\.pdf|mentioned in (.+) file|the (.+) file|from (.+) file|mentioned in (.+)[\.\?\!]|from the (.+)[\.\?\!]|present in the (.+)[\.\?\!]", query)
            ufname = ''.join(ufname)
            if ufname.lower().strip() in ['this', 'that', 'following']:
                ufname = ''
        ufname = ufname.lower()
        ufnames = ufname.split(',')
        if len(ufnames) > 1:
            qufname = _match_pattern("mentioned in (.+)\.pdf|present in (.+)\.pdf|in the (.+)\.pdf file|from (.+)\.pdf|mentioned in (.+) file|the (.+) file|from (.+) file|mentioned in (.+)[\.\?\!]|from the (.+)[\.\?\!]|present in the (.+)[\.\?\!]", query)
            if len(qufname) > 0:
                qufname = ''.join(qufname).lower()
                ufname = rf.process.extract(qufname, ufnames, scorer=rf.fuzz.partial_ratio, limit=1, score_cutoff=70)
                if len(ufname) > 0:
                    ufname = ufname[0][0]
                else:
                    ufname = ufnames[0]
                # ufname = qufname + ',' + ufname
            else:
                ufname = ufnames[0]
        else:
            ufname = ufnames[0]
        
        
        # print(f"User {user_id} given filename: {ufname}")
        if ufname is None or ufname == '':
            return {self.output_answer_key: "No docs found that match your query - this tool requires at least one filename that has been uploaded but none was provided! Please provide a filename and try again."}
        
        cur_user_files = self.user_files.get_files(user_id)
        if len(cur_user_files) == 0:
            return {self.output_answer_key: f"Sorry! User {user_id} has not uploaded any file so I cannot answer question. Please try again with a different user id."}
        fname = rf.process.extract(ufname, cur_user_files,scorer=rf.fuzz.partial_ratio, processor=lambda s: s.lower(),limit=1, score_cutoff=70)
        # print(fname)
        if len(fname) == 0:
            return {self.output_answer_key: f"Sorry! No docs found that match your query - User {user_id} has not uploaded any file with name {ufname}. Please try again with a different filename! I'm happy to help!"}
        fname = fname[0][0]
        #print("Actual Filename:",fname)
        store_type = str(type(self.vector_store)).lower()
        if "chroma" in store_type or "faiss" in store_type:
            #query = re.sub(r"this file|this document|current document|current file|this pdf|current pdf", fname, query, re.I)
            filters = {self._meta_fname_keyname: fname.lower(), "start_index": -1}
            tab_filters = {self._meta_fname_keyname: fname.lower(), "start_index": -2}
            if user_id is not None and user_id != "":
                filters[self._meta_uid_keyname] = user_id
                tab_filters[self._meta_uid_keyname] = user_id
            docs =  self.vector_store.query(filter=filters, k=None)
            tidocs =  self.vector_store.query(filter=tab_filters, k=None)
            for tidoc in tidocs:
                for tab_fpath in tidoc.page_content.split(','):
                    # if "list" in query.lower():
                    #     tdoc = Document(
                    #         page_content=f"Following are the items listed in this table present on this page: {query_table(tab_fpath, query)}",
                    #         metadata=tidoc.metadata
                    #     )
                    # else:
                    #     tdoc = Document(
                    #         page_content=f"Following are the individual records from a table present on this page: {query_table(tab_fpath, query='')}",
                    #         metadata=tidoc.metadata
                    #     )
                    tdoc = Document(
                        page_content=f"Following is a table present on this page:\n{query_table(tab_fpath, query='')}",
                        metadata=tidoc.metadata
                    )
                    docs.append(tdoc)
        elif "milvus" in store_type:
            #query = re.sub(r"this file|this document|current document|current file|this pdf|current pdf", fname, query, re.I)
            expr = f"{self._meta_fname_keyname} == '{fname}' and start_index <= -1"
            # tab_expr = f"{self._meta_fname_keyname} like '{fname}%' and start_index == -2"
            if user_id is not None and user_id != "":
                expr += f" and {self._meta_uid_keyname} == '{user_id}'"
                # tab_expr += f" and {self._meta_uid_keyname} == '{user_id}'"
            docs = []
            idocs = self.vector_store.query(expr=expr, k=None) # Return k docs
            # tidocs = self.vector_store.query(expr=tab_expr, k=None)
            for tidoc in idocs:
                if tidoc.metadata.get('start_index', -1) == -1:
                    docs.append(tidoc)
                else:
                    if "list" in query.lower():
                        for tab_fpath in tidoc.page_content.split(','):
                            # if "list" in query.lower():
                            #     tdoc = Document(
                            #         page_content=f"Following are the items listed in this table present on this page:\n{query_table(tab_fpath, query)}",
                            #         metadata=tidoc.metadata
                            #     )
                            # else:
                            #     tdoc = Document(
                            #         page_content=f"Following are the records/rows from a table present on this page:\n{query_table(tab_fpath, query='')}",
                            #         metadata=tidoc.metadata
                            #     )
                            tdoc = Document(
                                page_content=f"Following is a table present on {tidoc.metadata['source']}:\n{query_table(tab_fpath, query='')}",
                                metadata=tidoc.metadata
                            )
                            docs.append(tdoc)
        
        if len(docs) == 0:
            return {self.output_answer_key:"No documents found that match your query."} #, self.output_source_key: ""}
        #docs = sorted(docs, key=lambda a: (a.metadata.get('start_index',0)+1)*a.metadata.get("page",1))

        docs = sorted(docs, key=lambda a: a.metadata.get('page'))
        # run_manager.on_text("Retrieved docs:", color="green", end="\n", verbose=self.verbose)
        # run_manager.on_text("\n-----\n".join([doc.page_content for doc in docs]), color="blue", end="\n", verbose=self.verbose)
        
        answer = await self.reduce_chain.ainvoke({"input_documents": docs, "question":query}, callbacks=run_manager.get_child(), return_only_outputs=True)
        
        if isinstance(answer, dict):
            answer = answer[list(answer.keys())[0]]
        
        #answer = self.qa_chain({'question':query,'input_documents':docs}, return_only_outputs=True, callbacks=run_manager.get_child())
        #answer = {self.output_answer_key: answer['output_text']}
        answer = self._split_sources(answer, fname)
        # print(answer[self.output_source_key])
        answer = {self.output_answer_key:f"{answer[self.output_answer_key]}\n\nSOURCES: {answer[self.output_source_key]}"}
        return answer


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    search_kwargs = {'k': 8}
    llm = ChatOpenAI(temperature=0,model_name=os.environ.get("OPENAI_MODEL_NAME", "gpt-4"),max_tokens=700)
    vector_index = get_pdfs_index(index_name="milvus")

    pdf_qna = PDFQnA.from_llm_and_vector_index(llm=llm, vector_index=vector_index, chain_type="stuff", search_kwargs=search_kwargs, verbose=True)
    user_id = "u1"
    while True:
        filename = input("Please input the name of the filename(s) you want to query:")