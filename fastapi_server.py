import json
import os
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from fastapi import BackgroundTasks, FastAPI
from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain
# from langchain.callbacks import get_openai_callback
from langchain_community.callbacks.manager import get_openai_callback
from langchain_core.prompts import PromptTemplate
# from langchain_community.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, conlist
# import redis
from redis.client import Redis

# from create_graph_py2neo import read_items3 as read_items
# from create_graph_py2neo import read_records2 as read_records
from create_graph import CSV2KG
from fileio import (download_file_as_bytes_gcs, download_file_to_path,
                    download_from_gcs)
from pandas_llm import PandasLLM

# from pydantic.v1 import BaseModel as BaseModelv1


try:
    from dotenv import load_dotenv
    load_dotenv()
except:
    pass

from agent import QnA_Agent as KGQnA_Agent
from image_search import ImageQuery, ImageSearch, ImagesUpdate
from pdfqna_agent import QnA_Agent as PDFQnA_Agent
from QAutoComplete import AutoComplete

VERBOSE = True
PRINT_STATS = True
DOWNLOAD_FILES = False


REPHRASE_PROMPT = """Given the input question, rephrase the question while making certain that it retains the meaning of the original question.
Try to rephrase the question so that the output is easier to understand.
Question: {question}
Output:"""



class TasksTracker:
    def __init__(self, redis_client: Redis, ttl: Optional[int] = None):
        self.counter = 0
        self.redis_client = redis_client
        self.ttl = ttl
    
    def add_task(self, prefix: str) -> str:
        task_id = f'{prefix}_{self.counter}'
        self.redis_client.hset(task_id, mapping={'is_complete': 0, 'message': f'Task {task_id} in progress.', 'is_successful': 0})
        self.counter += 1
        return task_id
    
    def finish_task(self, task_id: int, is_successful: bool, message: str):
        is_successful = 0 if not is_successful else 1
        self.redis_client.hset(task_id, mapping={'is_complete': 1, 'message': message, 'is_successful': int(is_successful)})
        if self.ttl:
            self.redis_client.expire(task_id, self.ttl)

    def get_task(self, task_id: str) -> dict:
        return self.redis_client.hgetall(task_id)

    def get_counter(self) -> int:
        return self.counter


def load_kgqna(data_path: str, download_to_datapath: bool=True) -> Tuple[KGQnA_Agent, AutoComplete]:
    
    sents_path = os.environ.get("SENTS_PATH","sents_data.pkl")
    items_df_path = 'ItemsInformationUpdated.parq'
    if download_to_datapath:
        bucket_name = os.environ.get("GCS_UPLOAD_BUCKET", "qna-staging")
        prefix_path = os.environ.get("GCS_DOWNLOAD_FOLDER_PREFIX", "models/")
        errors = download_from_gcs(
            destination_directory=data_path,
            bucket_name=bucket_name, 
            prefix_path=prefix_path, # Only download sents_data.pkl for AutoComplete
            workers=4,
            include_files=[sents_path, items_df_path],
            is_folder=False
        )
        if len(errors) > 1:
            print(errors)
            raise Exception(f"There were errors downloading all objects from gs://{bucket_name}/{prefix_path} to {data_path}")
        data_path = os.path.join(data_path, prefix_path)
    
    # ner_model = os.path.join(data_path,os.environ.get("NER_MODEL_FOLDER","Bert_NER"))
    # intcls_model = os.path.join(data_path,os.environ.get("INT_MODEL_FOLDER","Bert_intentClassifier"))
    # config_fname = os.path.join(data_path,"props_slabels_int2q.json")
    image_search = ImageSearch(drop_old_collection=False)
    image_query = ImageQuery()
    agent = KGQnA_Agent.initialize(image_search=image_search, image_query=image_query, verbose=VERBOSE)

    # agent = KGQnA_Agent.initialize(qna,verbose=VERBOSE,qna_args={'k':10,'max_distance':1.5,'limit':200}) #'k':20,'max_distance':1.5,'limit':100
    print("LangChain KGQnA Agent Initialized!")
    sents_data_path = os.path.join(data_path,sents_path)
    spr = AutoComplete(use_clm=False,sents_data_path=sents_data_path) #,model_path=clm_model,checkpoint=checkpoint
    return agent, spr


redis_url = os.environ.get('REDIS_URL', 'localhost')
redis_port = os.environ.get('REDIS_PORT', '6379')
# redis_client: Redis = Redis.from_url(f"redis://{redis_url}:{redis_port}/0")
redis_client = Redis(host=redis_url, port=int(redis_port), decode_responses=True)
# redis_client = redis.from_url(f"redis://{redis_url}:{redis_port}/0")
task_counter = TasksTracker(redis_client, 7200) # Every task_id lasts for 2 hours (ttl=2hrs=120mins=7200s)
data_path = os.environ.get("KGQNA_DATA_PATH","./data/models")
Path(data_path).mkdir(exist_ok=True, parents=True)
tables_path = os.environ.get("TABLES_FOLDERPATH","./data/table_files")
Path(tables_path).mkdir(exist_ok=True, parents=True)

reph_chain = LLMChain(llm=ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0), prompt = PromptTemplate.from_template(REPHRASE_PROMPT))
pdfqna_agent: PDFQnA_Agent = PDFQnA_Agent.initialize(verbose=VERBOSE, rephraser=reph_chain, drop_old_collection=False)
print("LangChain PDFQnA Agent Initialized!")
kgqna_agent, spr = load_kgqna(data_path=data_path, download_to_datapath=DOWNLOAD_FILES)
if DOWNLOAD_FILES:
    download_file_to_path("gs://qna-staging/models/Copy of USDA.parq", os.path.join(data_path, 'Copy of USDA.parq'))


app = FastAPI(
    title="OneDiamond Server",
    version="1.0",
    description="An API server with endpoints for uploading file and answering queries"
)

# For all Upload Endpoints: csv_upload, uploadfile and images_upload
class TaskOutput(BaseModel):
    task_id: str = Field(title='Task ID', description="The generated task id of the task running in the background.")


##################################################################
# PDF Upload
##################################################################
# gs://trillo-onediamond-staging/public/QnAFiles/DAS_AppForm.pdf
# gs://trillo-onediamond-staging/public/QnAFiles/DAS_AppForm_1710420583493.pdf
# gs://trillo-onediamond-staging/public/QnAFiles/patient_questionnaire pdf_48210488_1710497462021.pdf
# gs://trillo-onediamond-staging/public/QnAFiles/etcetc123_1710497142092.pdf
class FileProcessingRequest(BaseModel):
    filepaths: conlist(str, min_length=1) = Field(title='Filepaths', description='The full Google Cloud Storage Blob URIs to the input PDF files stored in the GCS Bucket: gs://{bucket_name}/.../{filename}', examples=[["gs://qna-staging/docs/input/Smithfield Case Ready Manual.pdf"]])
    file_ids: conlist(str, min_length=1) = Field(title = "File IDs", description="The unique ID assigned to each file to be uploaded.")
    user_id: Union[int, str] = Field(title="User ID", description="The ID of the user that uploads the file.", examples=["u1", "noone@yahaha.com"])
    
# class FileProcessedOutput(BaseModel):
#     is_successful: bool = Field(title='Is Successful', description='A boolean response to determine whether the file upload was successful')
#     message: str = Field(title="Message", description="A message describing how the file uplaod went. Useful to determine what went wrong if file upload was not successful")
#     filenames: List[str] = Field(title="Filenames", description="The names of the files just uploaded")
    
def upload_file_bkground(task_id: str, filepaths: List[str], file_ids: List[str], user_id: str, chunk_size: int = 600) -> None:
    filenames = [filepath.split('/')[-1] for filepath in filepaths]
    status, message = pdfqna_agent.update_data(filenames=filenames, filepaths=filepaths, file_ids=file_ids, user_id=user_id, chunk_size=chunk_size, use_ocr=True, use_docai=True)
    if status:
        message = f"Task {task_id} completed - {message}"
    else:
        message = f"Task {task_id} failed - {message}"
    # redis_client.hmset(task_id, {'status': status})
    task_counter.finish_task(task_id, status, message)
    # redis_client.hmset(task_id, {'completed': True, 'message': message})
    # return status, message

@app.post(path="/uploadfile", response_model=TaskOutput)
async def upload_file(request: FileProcessingRequest, background_tasks: BackgroundTasks):
    '''
    Given a list of `filepaths` (Google Cloud Storage Blob URI) and the `user_id`, add the files to the vector store

    Uses DocumentAI to parse the files stored at `filepaths`.
    '''
    if isinstance(request, dict):
        filepaths = request['filepaths']
        user_id = request['user_id']
        file_ids = request['file_ids']
    else:
        filepaths = request.filepaths #request['filepath']
        user_id = request.user_id #request['user_id']
        file_ids = request.file_ids
    
    task_id = task_counter.add_task(prefix='updf')
    if len(file_ids) != len(filepaths):
        task_counter.finish_task(task_id, False, "Files Upload Failed - `file_ids` and `filepaths` must be of equal length.")
    else:
        background_tasks.add_task(upload_file_bkground, task_id=task_id, filepaths=filepaths, file_ids=file_ids, user_id=user_id)
    
    output = TaskOutput(task_id=task_id)
    # output = FileProcessedOutput(is_successful=status, message=message, filenames=filenames)
    return output

###############################################################
# CSV Uploads
###############################################################
class CSVUploadInput(BaseModel):
    filepath: str = Field(description="The full GSC path to the CSV file to be uploaded", examples=["gs://trillo-onediamond-staging/public/UploadReports/2023 Store by Store Report.csv","gs://trillo-onediamond-staging/public/UploadReports/Final Stores with Valid Column Names.csv", "gs://trillo-onediamond-staging/public/UploadReports/Final Item with valid column Names.csv"])
    datatype: int = Field(default=1, description="0  means the file is an items file, 1 means that it is a store-by-store records file and 2 means that it is a store file.",ge=0,lt=3)
    # stores_filepath: str | None = Field(
    #     default=None, 
    #     description="If the file uploaded is a records file, then to get the values from the `ChainName` column, this file may be needed. If the `ChainName` column is already provided then this is not needed", 
    #     examples=['gs://trillo-onediamond-staging/public/Final Stores with Valid Column Names.csv']
    # )

def csv_upload_bk2(
    task_id: str,
    items_or_records: Union[bool,int] = 0, 
    filepath: Optional[str] = None, 
    csv_string: Optional[str] = None, 
    stores_filepath: Optional[str] = None,
    sep: str = ',',
    is_gcs_path: bool = True
) -> Tuple[bool, str]:
    '''
    Upload a CSV file: either an items information one or a records
    
    if items_or_records == 0: 'items';
    else: 'records'
    '''
    neo_host = os.environ.get('GL_HOST',"dai-mlops-neo4j-service")
    neo_port = int(os.environ.get('GL_PORT',7687))
    
    if neo_host is None:
        raise Exception("GL_HOST environment variable not found! Can't continue")
    
    neo_passwd = os.environ.get('GL_PASSWORD',None)
    neo_user = os.environ.get('GL_USER','neo4j')

    csv2kg = CSV2KG(url=f"bolt://{neo_host}:{neo_port}", username=neo_user, password=neo_passwd)
    if filepath is not None:
        try:
            if is_gcs_path and filepath.startswith("gs://"):
                bytes_data = download_file_as_bytes_gcs(filepath)
                try:
                    csv_string = bytes_data.decode('utf-8')
                except:
                    csv_string = bytes_data.decode('latin-1')
            else:
                with open(filepath) as f:
                    csv_string = f.read()
        except Exception as e:
            task_counter.finish_task(task_id, False, f"Could not read from the given `filepath`: {filepath} due to error\n{e}.")
            # return False, f"Could not read from the given `filepath`: {filepath} due to error\n{e}."
        # filepath = None
    elif csv_string is None: # if both csv_string and filepath are None, then we raise an error
        task_counter.finish_task(task_id, False, "File Upload Failed: Both `filepath` and `csv_string` can not be None, at least 1 of these must be provided.")
        # return False, "File Upload Failed: Both `filepath` and `csv_string` can not be None, at least 1 of these must be provided."
    if items_or_records or items_or_records != 0: # Means it is a records file
        try:
            stores_csv_string = ""
            if stores_filepath is not None:
                if is_gcs_path and filepath.startswith("gs://"):
                    stores_bytes_data = download_file_as_bytes_gcs(stores_filepath)
                    try:
                        stores_csv_string = stores_bytes_data.decode('utf-8')
                    except:
                        stores_csv_string = stores_bytes_data.decode('latin-1')
                else:
                    with open(stores_filepath) as f:
                        stores_csv_string = f.read()
            csv2kg.read_records(filepath=None, data=csv_string, sep=sep, stores_data=stores_csv_string)
            task_counter.finish_task(task_id, True, message=f"Records File {filepath} Uploaded Successfully")
            # return True, f"Records File {filepath} Uploaded Successfully"
        except Exception as e:
            task_counter.finish_task(task_id, False, message=f"Records File {filepath} Upload Failed Due To Error {e}")
            # return False, f"Records File {filepath} Upload Failed Due To Error {e}"
    else:
        try:
            csv2kg.read_items(filepath=None, data=csv_string, sep=sep)
            task_counter.finish_task(task_id, True, message=f"Items File {filepath} Uploaded Successfully")
            # return True, f"Items File {filepath} Uploaded Successfully"
        except Exception as e:
            task_counter.finish_task(task_id, False, message=f"Items File {filepath} Upload Failed Due To Error {e}")
            # return False, f"Items File {filepath} Upload Failed Due To Error {e}"

def csv_upload_bk(
        task_id: str,
        filepath: str,
        datatype: int = 1,
        is_gcs_path: bool = True,
        sep: str = ','
):
    '''
    Upload a CSV file: either an items information one or a store by store records file 
    or one containing Stores/Customer information
    
    if datatype == 0: 'items';
    elif datatype == 1: 'records';
    elif datatype == 2: 'stores'
    '''
    neo_host = os.environ.get('GL_HOST',"dai-mlops-neo4j-service")
    neo_port = int(os.environ.get('GL_PORT',7687))
    
    if neo_host is None:
        raise Exception("GL_HOST environment variable not found! Can't continue")
    
    neo_passwd = os.environ.get('GL_PASSWORD',None)
    neo_user = os.environ.get('GL_USER','neo4j')

    csv2kg = CSV2KG(url=f"bolt://{neo_host}:{neo_port}", username=neo_user, password=neo_passwd)
    try:
        if is_gcs_path and filepath.startswith("gs://"):
            bytes_data = download_file_as_bytes_gcs(filepath)
            try:
                csv_string = bytes_data.decode('utf-8')
            except:
                csv_string = bytes_data.decode('latin-1')
        else:
            with open(filepath) as f:
                csv_string = f.read()
    except Exception as e:
        task_counter.finish_task(task_id, False, f"Could not read from the given `filepath`: {filepath} due to error\n{e}.")
    if datatype == 0:
        try:
            csv2kg.read_items(filepath=None, data=csv_string, sep=sep)
            task_counter.finish_task(task_id, True, message=f"Items File {filepath} Uploaded Successfully")
            
        except Exception as e:
            task_counter.finish_task(task_id, False, message=f"Items File {filepath} Upload Failed Due To Error {e}")
    elif datatype == 1:
        try:
            csv2kg.read_records(filepath=None, data=csv_string, sep=sep)
            task_counter.finish_task(task_id, True, message=f"Records File {filepath} Uploaded Successfully")
        except Exception as e:
            task_counter.finish_task(task_id, False, message=f"Records File {filepath} Upload Failed Due To Error {e}")
    elif datatype == 2:
        try:
            csv2kg.read_stores(filepath=None, data=csv_string, sep=sep)
            task_counter.finish_task(task_id, True, message=f"Stores File {filepath} Uploaded Successfully.")
        except:
            task_counter.finish_task(task_id, False, message=f"Stores File {filepath} Upload Failed Due To Error {e}")
    else:
        task_counter.finish_task(task_id, False, message=f"datatype {datatype} is not invalid. `datatype` must be either of 0 (items), 1(records) or 2(stores)")
@app.post(path="/csv_upload", response_model=TaskOutput)
async def csv_upload(request: CSVUploadInput, background_tasks: BackgroundTasks) -> TaskOutput:
    task_id = task_counter.add_task(prefix='csv')
    
    background_tasks.add_task(csv_upload_bk, 
                              datatype=request.datatype,
                              task_id=task_id,
                              filepath=request.filepath
                            )
    
    
    # task_id = f"csv_{task_counter.get_counter()}"
    return TaskOutput(task_id=task_id)

###############################################################
# Image Upload
###############################################################
class ImagesUploadInput(BaseModel):
    # folderpath: str = Field(description='The full path to the GCS folder containing the images.', examples=['gs://trillo-onediamond-staging/public/ItemImages'])
    imgs_to_add: List[str] = Field(default=[],description="The list of paths to images that need to be added.", examples=[["gs://trillo-onediamond-staging/public/ItemImages/1710153317479_Troyer.png"]])
    imgs_to_delete: List[str] = Field(default=[],description="The list of paths to images that do not exist and need to be deleted.", examples=[["gs://trillo-onediamond-staging/public/ItemImages/1710149953861_item-cases-ytd_(5).png"]])

class ImagesUploadOutput(BaseModel):
    is_successful: bool = Field(description="Was the images upload successful")
    message: str = Field(description="")

def image_upload_bkground(task_id: int, imgs_to_add: List[str], imgs_to_delete: List[str]):
    # df_path = os.path.join(os.environ.get("KGQNA_DATA_PATH","./data/models"), 'ItemsInformationUpdated.parq')
    image_uploader = ImagesUpdate()
    try:
        image_uploader.add_images(imgs_to_add)
        add_status = True
    except Exception as e:
        add_status = False
        print(f"Failed to add images due to error {e}")
    try:
        image_uploader.delete_images(imgs_to_delete)
        del_status = True
    except Exception as e:
        del_status = False
        print(f"Failed to delete images due to error {e}")
    # status, message = image_uploader.update_images(folderpath)
    if add_status and del_status:
        message = f"Added images {','.join(imgs_to_add)} and deleted images {','.join(imgs_to_delete)} successfully!"
    elif not add_status and del_status:
        message = f"Failed to add images {','.join(imgs_to_add)} and succeeded in deleting images {','.join(imgs_to_delete)}"
    elif not del_status and add_status:
        message = f"Added images {','.join(imgs_to_add)} but failed to delete images {','.join(imgs_to_delete)}"
    elif not del_status and not add_status:
        message = f"Failed to add images {','.join(imgs_to_add)} and failed to delete images {','.join(imgs_to_delete)}"
    if add_status and del_status:
        message = f"Task {task_id} completed - {message}"
    else:
        message = f"Task {task_id} failed - {message}"
    task_counter.finish_task(task_id, add_status and del_status, message)

def __images_upload(imgs_to_add: List[str], imgs_to_delete: List[str]):
    # df_path = os.path.join(os.environ.get("KGQNA_DATA_PATH","./data/models"), 'ItemsInformationUpdated.parq')
    image_uploader = ImagesUpdate()
    try:
        add_status, failed = image_uploader.add_images(imgs_to_add)
        # add_status = True
    except Exception as e:
        add_status = False
        failed = imgs_to_add
        print(f"Failed to add images due to error {e}")
    try:
        del_status = image_uploader.delete_images(imgs_to_delete)
        # del_status = True
    except Exception as e:
        del_status = False
        print(f"Failed to delete images due to error {e}")
    # status, message = image_uploader.update_images(folderpath)
    if add_status and del_status:
        message = f"Added images {imgs_to_add} and deleted images {imgs_to_delete} successfully!"
    elif not add_status and del_status:
        message = f"Failed to add images {failed} and succeeded in deleting images {imgs_to_delete}"
    elif not del_status and add_status:
        message = f"Added images {','.join(imgs_to_add)} but failed to delete images {imgs_to_delete}"
    elif not del_status and not add_status:
        message = f"Failed to add images {','.join(failed)} and failed to delete images {imgs_to_delete}"
    return add_status and del_status, message

@app.post('/images_upload', response_model=ImagesUploadOutput)
async def image_upload(request: ImagesUploadInput) -> ImagesUploadOutput:
    # , background_tasks: BackgroundTasks
    # task_id = task_counter.add_task(prefix='img')
    status, message = __images_upload(request.imgs_to_add, request.imgs_to_delete)
    # background_tasks.add_task(image_upload_bkground, 
    #                           task_id=task_id, 
    #                           imgs_to_add=request.imgs_to_add,
    #                           imgs_to_delete=request.imgs_to_delete
    #                         )
    return ImagesUploadOutput(is_successful=status, message=message) #TaskOutput(task_id=task_id)

##########################################################################
# KGQnA Agent (Predict_One)
##########################################################################
class KGAgentInput(BaseModel):
    question: str = Field(default='', title='Question', description="The input question to the KGQnA Agent", maxLength=1000, examples=["What is the price of perdue chicken breast?"])
    query: str = Field(default='', title='Query', description="The input query given for autocompletion", maxLength=500)


class KGAgentOutput(BaseModel):
    heading: str
    rows: str
    suggestions: List[str]
    images_urls: List[str]
    llm_response: str

class PredictOneInput(BaseModel):
    input_data: KGAgentInput = Field(title='Input Data', description='The input data to the agent or autocomplete')
    top_k: int = Field(default=10, title="Num. Suggestions", description="The number of suggestions to return from the autocomplete model")
    user_id: str = Field(default='', title="User ID", description="The User ID of the user that asks the query - needed for storing the memory for each user separately")
    conversation_id: str = Field(default='', title='Conversation ID', description="The ID of the particular conversation/session between user `user_id` and agent.")
    filters: Dict[str, str] | None = Field(default=None, title='Filter Dictionary', description='Any filters that need to be applied to the query - Only brand is supported right now')
    # image_path: str | None = Field(default=None, title='Image Path', description='GCS path to any image that is to be queried', examples=["gs://trillo-onediamond-staging/public/ItemImages/1711384920664_285912_1.png"])

class PredictOneOutput(BaseModel):
    prediction: KGAgentOutput


@app.post(path='/predict_one')
async def predict_one(request: PredictOneInput) -> PredictOneOutput:
    '''
    Given either a `question` or `query`, `question` is sent to the KG QnA Agent and `query` is sent to the Autocomplete model. 

    The Agent returns its output inside `prediction` in either of 3 forms:
    
    1) As a table:
    
        `heading` -> The string preceding the table output
        
        `rows` -> A '|' separated string of tabular results from the KG search/query
        
        `suggestions` -> An empty list
        
        `llm_response` -> An empty string

        `images_urls` -> An empty list
        
    2) As a list of suggestions if the KG search/query fails:
    
        `heading` -> An error message if something goes wrong
        
        `rows` ->  An empty string
        
        `suggestions` -> A list of suggestions for an alternate query that can be used (can be empty sometimes)
        
        `llm_response` -> An empty string

        `images_urls` -> An empty list
        
    3) A natural language or 'free text' response from the agent:
    
        `heading` -> An empty string
        
        `rows` ->  An empty string
        
        `suggestions` -> An empty list
        
        `llm_response` -> The full text response to return from the user

        `images_urls` -> An empty list

    4) A list of image urls when asked for images

        `heading` -> A string

        `rows` -> An empty string

        `suggestions` -> An empty list

        `llm_response` -> An empty list

        `images_urls` -> A list of strings
    
    The Autocomplete returns its response inside `prediction` only in the `suggestions` key:

    `heading` -> An empty string
    
    `rows` ->  An empty string
    
    `suggestions` -> A list of 'top_k' or fewer suggestions(strings) from the autocomplete model
    
    `llm_response` -> An empty string

    `images_urls` -> An empty list
    '''
    request = request.model_dump()
    input_data = request['input_data']
    user_id = f"{request['user_id']}_{request.get('conversation_id','1')}".strip('_')
        
    top_k, score_cutoff = request['top_k'], 5
    if "query" in input_data.keys():
        query = input_data['query']
        if isinstance(query,str) and len(query) > 2:
            acres = spr.generate_suggestions(query,top_k=top_k,score_cutoff=score_cutoff)
            res = {'heading': "", 'rows': "", 'suggestions': acres['suggestions'], 'llm_response': "", "images_urls": []}
        elif "question" in input_data.keys():
            question = input_data['question']
            # if request['filters'] is not None and len(request['filters']) > 0:
            #     question += " Use KG QnA tool."
            with get_openai_callback() as cb:
                try:
                    res = kgqna_agent.invoke({'input': question, 'user_id': user_id.strip('_'), 'filters': request['filters']}) #, 'image_path': request.get('image_path', None)})
                    if not 'sorry' in res['llm_response'].lower() and not 'sorry' in res['heading'].lower():
                        spr.add_data(question, write_to_file=False)
                except Exception as e:
                    print(e)
                    e_message = str(e)
                    if "Could not parse LLM output:" in e_message and re.search(r"\"action_input\": ", e_message):
                        res = {
                            'heading': "",
                            'rows': "",
                            'suggestions': [],
                            'llm_response': re.split(r"\"action_input\": ", e_message)[-1].strip("\"' "),
                            'images_urls': []
                        }
                    else:
                        res = {
                            'heading':f"Sorry, there was a problem while answering your query.",
                            "rows":"",
                            "suggestions":[],
                            "llm_response":f"Sorry, there was a problem while answering your query.",
                            'images_urls': []
                        }
                if PRINT_STATS:
                    print('--'*50)
                    print(f"Total Tokens: {cb.total_tokens}")
                    print(f"Prompt Tokens: {cb.prompt_tokens}")
                    print(f"Completion Tokens: {cb.completion_tokens}")
                    print(f"Total Cost (USD): ${cb.total_cost}")
                    print('--'*50)
            
        else:
            res = {'heading': "", 'rows': "", 'suggestions': [], 'llm_response': ""}
    elif "question" in input_data.keys():
        question = input_data['question']
        with get_openai_callback() as cb:
            try:
                res = kgqna_agent.invoke({'input': question, 'user_id': user_id.strip('_'), 'filters': request['filters']}) #, 'image_path': request.get('image_path', None)})
                if not 'sorry' in res['llm_response'].lower() and not 'sorry' in res['heading'].lower():
                    spr.add_data(question, write_to_file=False)
            except Exception as e:
                print(e)
                e_message = str(e)
                if "could not parse llm output:" in e_message.lower() and re.search(r"\"action_input\": ", e_message):
                    res = {
                        'heading': "",
                        'rows': "",
                        'suggestions': [],
                        'llm_response': re.split(r"\"action_input\": ", e_message)[-1].strip("\"' "),
                        'images_urls': []
                    }
                else:
                    res = {
                        'heading':f"Sorry, there was some problem answering your query: '{e}'",
                        "rows":"",
                        "suggestions":[],
                        "llm_response":f"Sorry, there was some problem answering your query: '{e}'",
                        'images_urls': []
                    }
            if PRINT_STATS:
                print('--'*50)
                print(f"Total Tokens: {cb.total_tokens}")
                print(f"Prompt Tokens: {cb.prompt_tokens}")
                print(f"Completion Tokens: {cb.completion_tokens}")
                print(f"Total Cost (USD): ${cb.total_cost}")
                print('--'*50)
    else:
        res = {'heading': "", 'rows': "", 'suggestions': [], 'llm_response': "", 'images_urls': []}
    prediction = KGAgentOutput(
        heading = res.get('heading',''),
        rows = res.get('rows',''),
        suggestions = res.get('suggestions',[]),
        llm_response = res.get('llm_response',''),
        images_urls = res.get('images_urls', [])
    )
    return PredictOneOutput(prediction=prediction)

##########################################################################
# PDFQnA Agent
##########################################################################
class PDFAgentInput(BaseModel):
    query: str = Field(title="Query", description="The input query/question to the PDF QnA Agent", maxLength=1000, examples=["What is the case weight of ribeye pork chops?"])
    user_id: Union[int, str] = Field(title="User ID", description="The ID of the user that uploads the file", examples=["nooneyahaa.com","u1", 1398])
    conversation_id: str = Field(default='', title='Conversation ID', description="The ID of the particular conversation/session between user `user_id` and agent.")
    filenames: List[str] = Field(title="Filenames", default=[], description="Optionally provide a list of the names of the filenames to be queried", examples=[[], ["Smithfield Case Ready Manual.pdf", "Universal Declaration of Human Rights.pdf"]])

class PDFAgentOutput(BaseModel):
    output: str

@app.post(path="/pdfqna_agent", response_model=PDFAgentOutput)
async def pdfqna_predict(request: PDFAgentInput):
    '''
    Given the input `query` and `user_id` and optionally `filenames`, PDFQnA Agent responds with a simple string output in `output`
    '''
    request: dict = request.model_dump()
    request['user_id'] = f"{request['user_id']}_{request.pop('conversation_id','')}".strip('_')
    with get_openai_callback() as cb:
        try:
            res = await pdfqna_agent.ainvoke(request)    
        except Exception as e:
            print(e)
            res = {'output': f"Sorry, there was a problem answering your query! Try asking something else. I'm always happy to assist!"}

        if PRINT_STATS:
            print('--'*50)
            print(f"Total Tokens: {cb.total_tokens}")
            print(f"Prompt Tokens: {cb.prompt_tokens}")
            print(f"Completion Tokens: {cb.completion_tokens}")
            print(f"Total Cost (USD): ${cb.total_cost}")
            print('--'*50)
    return PDFAgentOutput(**res)

#################################################################
# Feedback
#################################################################
class FeedbackInput(BaseModel):
    question: str = Field(title='Question/Query', description="The user's input query.", examples=["What is the case weight of ribeye pork in g?"])
    response: str = Field(title='Response', description="The agent's response which the user had an issue with. It is a JSON-formatted string of the output given by the agent.", examples=["The available information does not have anything about the case weight of ribeye pork in grams"])
    comment: str = Field(default='',title='User Comment/Feedback', description="The user's comments on the agent's response.", examples=["The information is avaialable but it couldn't find it."])

@app.post(path='/feedback_store')
async def store_feedback(request: FeedbackInput) -> Dict[str, str]:
    
    json_store = os.environ.get("JSON_RESPONSE_FILEPATH", "feedback_store.json")
    feedback_data = []
    if os.path.isfile(json_store):
        # if the file already exists, then read the data from it
        with open(json_store, 'r') as f:
            feedback_data = json.load(f)
    if isinstance(feedback_data, dict):
        feedback_data = [feedback_data]
    
    assert isinstance(feedback_data, list), f"feedback_data must be list, not {type(feedback_data)}"
    # request = request.model_dump()
    feedback_data.append(request.model_dump())
    print(feedback_data)
    with open(json_store, 'w') as f:
        json.dump(feedback_data, f, indent=4)
    print("User Feedback Added!")
    return {"status": "Feedback saved!"}

#################################################################
# Task Status
#################################################################
class TaskStatus(BaseModel):
    is_complete: int | bool
    is_successful: bool
    # message: str

@app.post(path='/status/{task_id}', response_model=TaskStatus)
def status(task_id: str) -> TaskStatus:
    status = task_counter.get_task(task_id)
    print(status.get("message","No Message"))
    if 'is_complete' in status and 'is_successful' in status:
        return TaskStatus(is_complete=status['is_complete'], is_successful=status['is_successful'])
    print('This Task does not currently exist. Either it ended 2 or more hours ago or it was never started.')
    return TaskStatus(is_complete=1, is_successful=False)
    

#################################################################
# Deleta Conversation
#################################################################
class DeleteConversationInput(BaseModel):
    user_id: str | int = Field(description='The user id of the user whose conversation is to be deleted')
    conversation_ids: List[Union[str, int]] = Field(description='The conversation id of the conversation which is to be deleted.')
    pdf_or_pred: int = Field(default=0, description='If pdf_or_pred == 0: delete the PDF Chat History Else: delete the predict_one(KGQnA Agent) Chat History')

@app.post(path='/delete_conversation')
def delete_conversation(request: DeleteConversationInput) -> Dict[str, str]:
    # request = request.model_dump()
    if len(request.conversation_ids) > 0:
        for conversation_id in request.conversation_ids:
            conv_key = f"{request.user_id}_{conversation_id}".strip('_')
            if request.pdf_or_pred:
                kgqna_agent.memory.clear(conv_key)
            else:
                pdfqna_agent.memory.clear(conv_key)
        return {"status": f"Conversations {request.conversation_ids} for user {request.user_id} deleted!"}
    if request.pdf_or_pred:
        kgqna_agent.memory.clear(request.user_id)
    else:
        pdfqna_agent.memory.clear(request.user_id)
    return {"status": f"All Conversations for user {request.user_id} deleted!"}

#################################################################
# Delete Files
#################################################################
class DeleteFilesInput(BaseModel):
    file_ids: conlist(str, min_length=1) = Field(description="The IDs of the files to be deleted.")
    user_id: str = Field(description="The `user_id` against which the the files were uploaded.")

@app.post(path='/delete_files')
def delete_files(request: DeleteFilesInput) -> Dict[str, str]:
    pdfqna_agent.delete_files(request.file_ids, request.user_id)
    return {"status": f"Files {request.file_ids} against user {request.user_id} deleted successfully!"}

#################################################################
# List User Files
#################################################################
@app.get(path="/list_files")
def get_files(user_id: str | None = None) -> Dict[str, List[str]]:
    if user_id is None or user_id == '':
        return pdfqna_agent.user_files.get_files(return_only_fnames=False)
    else:
        return {user_id: pdfqna_agent.user_files.get_files(user_id, return_only_fnames=False)}


#################################################################
# USDA CSV QnA
#################################################################
class USDAQueryInput(BaseModel):
    query: str = Field(title="Query", description="The input query/question")

class USDAQueryOutput(BaseModel):
    output: str = Field(title="Result", description="The result of the query")

@app.post(path="/usda_query")
def usda_query(request: USDAQueryInput) -> USDAQueryOutput:
    custom_prompt = """1. Do NOT write code to create data.
    2. If you can't come up with a suitable Python code snippet to fulfill the request, reply with ''.
    3. The Python code snippet should ONLY be about creating a query using the given column names.
    4. Always return your answer in either of the following data types: pd.Dataframe, int, str or float.
    5. Do NOT write Python code to read or write a file."""

    df = pd.read_parquet(os.path.join(data_path, "Copy of USDA.parq"))

    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        conv_df = PandasLLM(data=df, force_sandbox=True, verbose=True, custom_prompt=custom_prompt)
    try:
        result = conv_df.prompt(request.query)
    except:
        result = ''
    if result is None:
        result = ""
    if isinstance(result, float) and np.isnan(result):
        result = ""
    if isinstance(result, str) and "try later" in result:
        result = ""
    
    if isinstance(result, pd.DataFrame) or isinstance(result, pd.Series):
        if len(result) > 0:
            result = result.to_csv(sep='|') #result.to_json(orient='records')
        else:
            result = ""
    
    if isinstance(result, np.ndarray):
        result = result.tolist()
        result = '\n'.join(result)

    if isinstance(result, Iterable) and not isinstance(result, str):
        if len(result) == 0:
            result = ""
        result = list(result)
        result = '\n'.join(result)
    
    if not isinstance(result, str):
        try:
            result = f"{result}"
        except:
            result = ""
    if result.lower() in ['nan', 'error', 'na', 'inf']:
        result = ""

    return USDAQueryOutput(output=result)

if __name__ == "__main__":
    import os

    import uvicorn

    uvicorn.run(app, 
                host=os.environ.get("APP_HOST","localhost"), 
                port=int(os.environ.get("APP_PORT", 8000)),
            )
