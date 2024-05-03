import asyncio
import base64
import json
import os
import re
# from io import BytesIO
from typing import Any, Dict, List, Tuple, Union

# from langchain.chains import LLMChain
# from langchain.chains.base import Chain
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
# from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain.prompts.base import BasePromptTemplate
from langchain.schema import Document
from PIL import Image

from fileio import download_file_as_bytes_gcs, download_from_gcs
from KGQnA import DaiNeo4jGraph
from vector_stores import (DEFAULT_FAISS_DIRECTORY, DEFAULT_FAISS_INDEX,
                           DEFAULT_MILVUS_CONNECTION,
                           IMGSEARCH_COLLECTION_NAME, create_collection,
                           get_index)

IMG_SEARCH_PROMPT = "Give a brief description of this image."

def store_attrs(obj: Any, params: Dict[str, Any]) -> None:
    for para,argval in params.items():
        if isinstance(argval,dict):
            for k,v in argval.items():
                setattr(obj,k,v)
        else:
            setattr(obj,para,argval)
    return

def is_number(text: str) -> bool:
    return all([text[i] in "0123456789." for i in range(len(text))])

class ImageSearch:
    """A simple tool that given a query (about images) as input, 
    returns the image paths that match the query"""

    def __init__(self, index_type: str = "milvus", **kwargs) -> None:    
        if index_type.lower() == 'milvus':
            col_created = create_collection(
                collection_name=kwargs.get("collection_name", IMGSEARCH_COLLECTION_NAME),
                drop_old=kwargs.get("drop_old_collection", False)
            )
            if col_created:
                print("New collection created!")
        self.index_type = index_type
        self.vector_index = get_index(
            index_name=index_type,
            embs=HuggingFaceEmbeddings(model_name=kwargs.get('embs_model_name', "sentence-transformers/all-mpnet-base-v2")),
            collection_name=kwargs.get("collection_name", IMGSEARCH_COLLECTION_NAME), 
            connection_args=kwargs.get("connection_args", DEFAULT_MILVUS_CONNECTION)
        )
        self.search_args = {'k': 50}
        # self.use_gpt4 = use_gpt4 #kwargs.get("use_openai", True) or kwargs.get("use_gpt4")
        
        # Store all the params passed in kwargs to the object
        store_attrs(self, kwargs)
        
    def search(self, query: str) -> Dict[str, Any]:
        docs = self.vector_index.similarity_search(query, **self.search_args)
        if len(docs) == 0:
            res = {
                'heading': "There were no images with matching description to return.",
                'rows': "",
                'suggestions': [],
                'llm_response': "",
                'images_urls': []
            }
        else:
            res = {
                'heading': "Following are the images:",
                'rows': "",
                'suggestions': [],
                'llm_response': "\n".join([doc.page_content for doc in docs]),
                'images_urls': [doc.metadata['source'].replace('gs://','https://storage.googleapis.com/') for doc in docs]
            } 
        return json.dumps(res)
    
    async def asearch(self, query: str) -> Dict[str, Any]:
        docs = await self.vector_index.asimilarity_search(query, **self.search_args)
        if len(docs) == 0:
            res = {
                'heading': "There were no images with a matching description to return.",
                'rows': "",
                'suggestions': [],
                'llm_response': "",
                'images_urls': []
            }
        else:
            res = {
                'heading': "Following are the images:",
                'rows': "",
                'suggestions': [],
                'llm_response': "\n".join([doc.page_content for doc in docs]),
                'images_urls': [doc.metadata['source'].replace('gs://','https://storage.googleapis.com/') for doc in docs]
            } 
        return json.dumps(res)        

class ImagesUpdate:

    def __init__(self, index_type: str = 'milvus', **kwargs):
        assert index_type.lower() in ['chroma', 'milvus'], "Only Chroma and Milvus vector stores are supported"
        if index_type.lower() == 'milvus':
            col_created = create_collection(
                collection_name=kwargs.get("collection_name", IMGSEARCH_COLLECTION_NAME),
                drop_old=kwargs.get("drop_old_collection", False)
            )
            if col_created:
                print("New collection created!")
        self.index_type = index_type.lower()
        self.vector_index = get_index(
            index_name=index_type,
            embs=HuggingFaceEmbeddings(model_name=kwargs.get('embs_model_name', "sentence-transformers/all-mpnet-base-v2")),
            collection_name=kwargs.get("collection_name", IMGSEARCH_COLLECTION_NAME), 
            connection_args=kwargs.get("connection_args", DEFAULT_MILVUS_CONNECTION)
        )
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "Could not import openai python package. "
                "Please install it with `pip install openai`."
            )
        self.openai_client = OpenAI()
        neo_host = os.environ.get('GL_HOST',"dai-mlops-neo4j-service")
        neo_port = int(os.environ.get('GL_PORT',7687))
        
        if neo_host is None:
            raise Exception("GL_HOST environment variable not found! Can't continue")
        
        neo_passwd = os.environ.get('GL_PASSWORD',None)
        neo_user = os.environ.get('GL_USER','neo4j')
        self.graph = DaiNeo4jGraph(
            url=f"bolt://{neo_host}:{neo_port}",
            username=neo_user,
            password=neo_passwd
        )

    def __img2text(self, image: Union[Image.Image, bytes], prompt: str = IMG_SEARCH_PROMPT) -> str:
        response = self.openai_client.chat.completions.create(
            model="gpt-4-vision-preview",
            n=1,
            temperature=0.1,
            messages=[
                {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt}, #"Whats in this image?"
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image}",
                        },
                    },
                ],
                }
            ],
            max_tokens=200,
        )
        return response.choices[0].message.content
        
    def get_item_details(self, image_path: str):
        imno = image_path.split('/')[-1].split('.')[0].split('_')
        if "Flyer" in image_path:
            im_type = "flyer"
        elif "ItemImages" in image_path:
            im_type = "item picture or item specifications sheet"
        else:
            im_type = "image"
        if len(imno) > 1:
            imno = imno[1]
        else:
            imno = imno[0]
        # print(imno)
        if is_number(imno):
            try:
                item = self.graph.query(
                    """MATCH (v:Vendor)-[:SOLD_ITEM]-(i:Item {number: $ino})
RETURN i.description AS item_description, i.category AS item_category, i.brand AS item_brand, i.upc AS item_upc, i.ven_item_number AS ven_item_number, v.number AS vendor_number, v.name AS vendor_name LIMIT 1;""",
                    {"ino": int(imno)},
                    return_dict=True,
                )
                if len(item) > 0 and isinstance(item, List):
                    item = item[0]
                else:
                    return ""
                if not isinstance(item, Dict):
                    return ""
                # print(item)
            except Exception as e:
                print(e)
                return ""
            item_details = f"""
This {im_type} is regarding the following item: {item['item_description']} with item number {imno} which is sold by vendor {item['vendor_name']} with vendor number {item['vendor_number']}.
It belongs to the {item['item_category']} category, has brand name: {item['item_brand']}, has vendor item number {item['ven_item_number']} and has UPC {item['item_upc']}."""
            return item_details
        return ""

    def add_images(self, image_paths: List[str]) -> Tuple[bool, List[str]]:
        '''
        Given an input list of image paths, read and load the image descriptions into the vector store
        '''
        if len(image_paths) == 0:
            return True, []
        failed = []
        docs = []
        for img_name in image_paths:
            try:
                img = download_file_as_bytes_gcs(img_name)
            except Exception as e:
                failed.append(img_name)
                print(f"Failed to download image {img_name} due to error {e}")
                continue
            item_details = self.get_item_details(img_name)
            if len(item_details) == 0:
                img = base64.b64encode(img).decode('utf-8')
                img_text = self.__img2text(img)
            else:
                img_text = ""
            docs.append(
                Document(
                    page_content=f"Image: {img_name}\n{img_text}{item_details}",
                    metadata={
                        'source': img_name
                    }
                )
            )
        self.delete_images(image_paths)
        self.vector_index.add_documents(docs)
        return len(failed) == 0, failed

    def delete_images(self, image_paths: List[str]) -> bool:
        if len(image_paths) == 0:
            return True
        if self.index_type == 'milvus':
            sources = [f"\"{fn}\"" for fn in image_paths]
            sources = '[' + ','.join(sources) + ']'
            ids_to_delete = self.vector_index.query(
                expr=f"source in {sources}",
                return_ids=True,
                return_data=False,
                k=None
            )
            if len(ids_to_delete) > 0: # Delete all the ids
                _ = self.vector_index.delete(
                    ids=ids_to_delete
                )
                print("Previous image docs deleted!")
        elif self.index_type == 'chroma':
            self.vector_index.delete(
                filter={'source': {'$in': image_paths}}
            )
        return True

class ImageQuery:
    """
    A simple querying tool that, given the path to an image
    and a question about the image, uses GPT-4-Vision to answer that query
    """

    def __init__(self, **kwargs):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "Could not import openai python package. "
                "Please install it with `pip install openai`."
            )
        try:
            from google.cloud import storage
        except ImportError:
            raise ImportError(
                "Could not import the google.storage package."
                "Please install it with `pip install google-cloud-storage`."
            )
        self.openai_client = OpenAI()
        self._gcs_client = storage.Client()

    def __query_img(self, image: bytes, query: str):
        response = self.openai_client.chat.completions.create(
            model="gpt-4-vision-preview",
            n=1,
            temperature=0.2,
            messages=[
                {
                "role": "user",
                "content": [
                    {"type": "text", "text": query}, #"Whats in this image?"
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image}",
                        },
                    },
                ],
                }
            ],
            max_tokens=300,
        )
        return response.choices[0].message.content

    def query(self, query_imgpath) -> dict[str, str]:
        query, img_path = re.split(r"IMAGE:",query_imgpath, flags=re.I)
        query = query.strip().lower()
        img_path = img_path.strip()
        print(query, img_path)
        img = download_file_as_bytes_gcs(img_path,self._gcs_client)
        img = base64.b64encode(img).decode('utf-8')
        ans = self.__query_img(img, query)
        return {"llm_response": ans}

    async def aquery(self, query_imgpath) -> dict[str, str]:
        # loop = asyncio.get_event_loop()
        query, img_path = re.split(r"IMAGE:",query_imgpath, flags=re.I)
        query = query.strip().lower()
        img_path = img_path.strip()
        # img = await loop.run_in_executor(None, download_file_as_bytes_gcs, img_path, self._gcs_client)
        img = await asyncio.to_thread(download_file_as_bytes_gcs, gcs_uri=img_path, client=self._gcs_client)
        # img = await asyncio.coroutine(download_file_as_bytes_gcs)(img_path,self._gcs_client)
        
        img = base64.b64encode(img).decode('utf-8')
        # ans = await loop.run_in_executor(None, self.__query_img, img, query)
        ans = await asyncio.to_thread(self.__query_img, image=img, query=query)
        # ans = await asyncio.coroutine(self.__query_img)(img, query)
        return {"llm_response": ans}