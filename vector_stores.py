import logging
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

from langchain.embeddings.base import Embeddings
# from langchain.vectorstores.base import VectorStore
# from langchain.vectorstores.chroma import Chroma
# from langchain.vectorstores.milvus import Milvus
from langchain_community.vectorstores.chroma import Chroma
from langchain_community.vectorstores.milvus import Milvus
# from langchain.schema import Document
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

# from langchain.vectorstores.chroma import _results_to_docs

logger = logging.getLogger(__name__)


PDFQnA_COLLECTION_NAME = "PDF_QnA"
IMGSEARCH_COLLECTION_NAME = "ImgSearch"

DEFAULT_MILVUS_CONNECTION = {
    "host": os.environ.get("DEFAULT_MILVUS_CONNECTION","localhost"),
    "port": os.environ.get("DEFAULT_MILVUS_PORT","19530"),
    "user": "",
    "password": "",
    "secure": False,
}

DEFAULT_FAISS_INDEX = os.environ.get("DEFAULT_FAISS_INDEX_NAME","PDF_QnA")
DEFAULT_FAISS_DIRECTORY = os.environ.get("DEFAULT_FAISS_DIRECTORY","../faiss_index")

EMBEDDINGS_DIM = os.environ.get("EMBEDDINGS_DIM",1536) # OpenAI
IMGSEARCH_EMBEDDINGS_DIM = 768 #Huggingface SentenceTransformer

def get_schema(collection_name: str):
    from pymilvus import CollectionSchema, DataType, FieldSchema
    if collection_name == PDFQnA_COLLECTION_NAME:
        fields = [
            FieldSchema(
                # The primary field
                name="pk",
                description="primary key",
                dtype=DataType.INT64,
                is_primary=True,
                auto_id=True
            ),
            FieldSchema(
                # The text field
                name="text",
                description="The text field to return",
                dtype=DataType.VARCHAR,
                max_length=65_530
            ),
            FieldSchema(
                # The vector field
                name="vector",
                description="The vector field to use for searching",
                dtype=DataType.FLOAT_VECTOR,
                dim=EMBEDDINGS_DIM
            ),
            FieldSchema(
                name="filename",
                description="Name/Path of the file a piece of text belongs to",
                dtype=DataType.VARCHAR,
                max_length=200,
                default_value=""
            ),
            FieldSchema(
                name="page",
                description="Page number of the file a piece of text belongs to",
                dtype=DataType.INT64,
                default_value=-1
            ),
            FieldSchema(
                name="source",
                description="Source of the text which a piece of text comes from: includes filename and page number",
                dtype=DataType.VARCHAR,
                max_length=300,
                default_value=""
            ),
            FieldSchema(
                name="id",
                description="The unique id of every file uploaded",
                dtype=DataType.VARCHAR,
                max_length=300,
                default_value=""
            ),
            FieldSchema(
                name="start_index",
                description="Index where the text starts from relative to the page.",
                dtype=DataType.INT64,
                default_value=0
            ),
            FieldSchema(
                name="user_id",
                description="ID of the user that uploads the file containing a text",
                dtype=DataType.VARCHAR,
                max_length=140,
                default_value="-1"
            )
        ]
    elif collection_name == IMGSEARCH_COLLECTION_NAME:
        fields = [
            FieldSchema(
                # The primary field
                name="pk",
                description="primary key",
                dtype=DataType.INT64,
                is_primary=True,
                auto_id=True
            ),
            FieldSchema(
                # The text field
                name="text",
                description="The text field to return",
                dtype=DataType.VARCHAR,
                default_value="",
                max_length=65_530
            ),
            FieldSchema(
                # The vector field
                name="vector",
                description="The vector field to use for searching",
                dtype=DataType.FLOAT_VECTOR,
                dim=IMGSEARCH_EMBEDDINGS_DIM
            ),
            FieldSchema(
                # The name/path/url of the image to return
                name="source",
                description="The name/path/url of the image",
                dtype=DataType.VARCHAR,
                max_length=1000,
                default_value=""
            )
        ]
    schema = CollectionSchema(
        fields=fields,
        description="PDFQnA Search Index",
        enable_dynamic_field=True
    )
    return schema


def create_collection(collection_name=PDFQnA_COLLECTION_NAME, index_type="milvus", **kwargs):
    
    def create_collection_milvus(connection_args:Optional[Dict[str,str]]=None, drop_old=False, **kwargs):
        from pymilvus import Collection, connections, utility
        
        if connection_args is None:
            connection_args = DEFAULT_MILVUS_CONNECTION
            alias = "default"
        else:
            alias = f"S{connection_args['host']}-{connection_args['user']}"
            
        conn = connections.connect(**connection_args)
        coll_exists = utility.has_collection(collection_name, using=alias)
        if coll_exists:
            if drop_old:
                utility.drop_collection(collection_name)
                print(f"{collection_name} collection dropped!")
            else:
                coll = Collection(collection_name, using=alias)
                col_schema = coll.schema
                vec_field = [f for f in col_schema.fields if f.name == 'vector'][0]
                curr_embeddings_dim = vec_field.params['dim']
                if (curr_embeddings_dim == EMBEDDINGS_DIM and collection_name == PDFQnA_COLLECTION_NAME) or \
                    (curr_embeddings_dim == IMGSEARCH_EMBEDDINGS_DIM and collection_name == IMGSEARCH_COLLECTION_NAME):
                    return False # Return False since a new collection has not been created
                else:
                    # if the embeddings size in the current collection is different from what
                    # the embedding size will now be, then drop the existing collection
                    utility.drop_collection(collection_name)
                    print(f"{collection_name} collection dropped!")
        
        schema = get_schema(collection_name)
        
        coll = Collection(
            name=collection_name,
            schema=schema,
            using=alias,
            shards_num=2
        )
        return True # a new collection was created
    
    def create_collection_chroma(**kwargs):
        raise NotImplementedError
    
    if index_type == 'milvus':
        return create_collection_milvus(**kwargs) #.get("connection_args",None)
    elif index_type == 'chroma':
        return create_collection_chroma()
    else:
        raise NotImplementedError

def get_index(index_name: str, embs: Embeddings, collection_name: str = PDFQnA_COLLECTION_NAME, connection_args: dict=DEFAULT_MILVUS_CONNECTION, 
              faiss_folderpath: str=DEFAULT_FAISS_DIRECTORY, faiss_index_name=DEFAULT_FAISS_INDEX) -> VectorStore:
    if index_name.lower() == 'chroma':
        return DaiChroma(collection_name=collection_name, embedding_function=embs)
    elif index_name.lower() == "milvus":
        if connection_args is None:
            connection_args = DEFAULT_MILVUS_CONNECTION
        index_params = {"metric_type": "L2", "index_type": "HNSW", "params": {"M": 8, "efConstruction": 32}}
        search_params = {"metric_type": "L2", "params": {"ef": 10, "radius": 1.5, "range_filter": 0.0}}
        return DaiMilvus(auto_id=True,collection_name=collection_name, embedding_function=embs, index_params=index_params, 
                        search_params=search_params, connection_args=connection_args)
    elif index_name.lower() == 'faiss':
        # from langchain.vectorstores import FAISS
        from langchain.docstore.in_memory import InMemoryDocstore
        from langchain.vectorstores.faiss import FAISS, dependable_faiss_import

        # raise NotImplementedError("FAISS index has not been implemented")
        if faiss_folderpath is not None and faiss_index_name is not None and \
            os.path.isdir(faiss_folderpath) and \
            os.path.exists(os.path.join(faiss_folderpath,f"{faiss_index_name}.faiss")) and \
            os.path.exists(os.path.join(faiss_folderpath,f"{faiss_index_name}.pkl")):
            try:
                return FAISS.load_local(faiss_folderpath, embs, faiss_index_name)
            except:
                pass
        faiss = dependable_faiss_import()
        # import faiss
        index = faiss.IndexFlatL2(EMBEDDINGS_DIM)
        # index = faiss.IndexHNSWFlat(EMBEDDINGS_DIM, 8)
        # docstore=InMemoryDocstore()
        # docstore
        return FAISS(embedding_function=embs.embed_query, docstore=InMemoryDocstore(), index=index,
                        index_to_docstore_id={}, normalize_L2=True)
    else:
        raise NotImplementedError(f"{index_name} index has not been implemented")

class DaiVectorStore(VectorStore, ABC):

    @abstractmethod
    def query(self, k: int = 4, **kwargs) -> List[Document]:
        pass

    @abstractmethod
    def delete(self) -> None:
        pass
    
class DaiMilvus(DaiVectorStore, Milvus):
    """
    Subclass of LangChain's own implementation of pymilvus integration.

    I created this to define some particular methods of pymilvus that were
    not implemented by LangChain i.e. Collection.query() and Collection.delete()
    """
    @property
    def count(self):
        return self.col.num_entities

    def query(
        self,
        expr: str,
        k: Optional[int] = 4,
        return_ids: bool = False,
        return_data: bool = True,
        output_fields: Optional[List] = None,
        **kwargs
    ) -> List[Union[Document, int]]:
        """Perform a query using a boolean search expression and return the documents that match

        For more information about the query parameters, you can go to the pymilvus
        documentation found here:
        https://milvus.io/docs/query.md

        Args:
            expr (str): Filtering expression used to perform the query
            k (int, optional): The number of results to return. Defaults to 4.
            **kwargs: Collection.query keyword arguments - offset, partition_names, consistency_level 

        """

        if self.col is None:
            logger.debug("No existing collection to query")
            return []
        
        # Determine result metadata fields.
        if output_fields is None:
            if not (return_data or return_ids):
                raise ValueError("At least one of return_data and return_ids should be True if output_fields is None")
            output_fields = []
            if return_data:
                output_fields = self.fields[:]
                output_fields.remove(self._vector_field)
            
            if return_ids:
                output_fields = output_fields + [self._primary_field]
        
        if len(output_fields) == 0:
            raise Exception("Output fields cannot be an empty list!")
        
        res = self.col.query(
            expr=expr,
            limit=k,
            output_fields=output_fields,
            **kwargs,
        )
        if len(res) == 0:
            return []
        # Organize results.
        ret = []
        if return_data:
            for result in res:
                meta = {x: result.get(x) for x in output_fields}
                doc = Document(page_content=meta.pop(self._text_field,""), metadata=meta)
                
                ret.append(doc)
        elif return_ids:
            for result in res:
                i = result.get(self._primary_field)
                ret.append(i)
        elif not return_data and not return_ids:
            for result in res:
                meta = {x: result.get(x) for x in output_fields}
                ret.append(meta)
                
        return ret
    
    def delete(
        self,
        ids: List[int] = None,
        **kwargs
    ) -> None:
        """Perform a delete operation on the collection using a boolean expression. Returns nothing

        For more information about the query parameters, you can go to the pymilvus
        documentation found here:
        https://milvus.io/docs/delete_data.md

        Args:
            ids (List[int], optional): List of ids to select the entities to delete. 
                                        Defaults to None i.e. delete everything
            partition_name (str, optional): Name of partition of collection to perform deletion in. Defaults to None
        
        """
        if self.col is None:
            logger.debug("No existing collection to perform deletion")
            return
        
        if ids is None:
            expr = ""
        else:
            sids = ', '.join([str(i) if isinstance(i, int) else f"'{i}'" for i in ids]) 
            expr = f"{self._primary_field} in [{sids}]"
        res = self.col.delete(expr, **kwargs)
        return res.delete_count == len(ids)

def _results_to_docs(results: Any) -> List[Document]:
    return [
        # TODO: Chroma can do batch querying,
        # we shouldn't hard code to the 1st result
        Document(page_content=result[0], metadata=result[1] or {})
        for result in zip(
            results["documents"][0],
            results["metadatas"][0],
        )
    ]
    
class DaiChroma(DaiVectorStore, Chroma):
    """Subclass of LangChain's own implementation of chromadb integration.

    I created this to define some particular methods of chromaDB that did not
    have satisfactory implementations by LangChain i.e. Collection.get() and Collection.delete()
    """
    @property
    def count(self):
        return self._collection.count()

    def query(
        self,
        filter: Dict[str, Any] = None,
        ids: Optional[List[int]] = None,
        k: int = 4,
        where_document: Optional[Dict[str, str]] = None,
        output_fields: List[str] = ["metadatas", "documents"],
        **kwargs
    ) -> List[Document]:
        
        res = self._collection.get(
            ids=ids,
            where=filter,
            limit=k,
            where_document=where_document,
            include=output_fields,
            **kwargs
        )

        return _results_to_docs(res)
    
    def delete(
        self,
        filter: Dict[str, str],
        ids: Optional[List[int]] = None,
        where_document: Optional[Dict[str, str]] = None
    ) -> None:
        
        self._collection.delete(
            ids=ids,
            where=filter,
            where_document=where_document
        )
