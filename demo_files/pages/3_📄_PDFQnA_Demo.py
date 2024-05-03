import os
import json
import time
import requests
import streamlit as st
from typing import Union, Dict, Optional, List

from fileio import upload_file_to_bucket


API_URL = os.environ.get("API_URL", "localhost")
API_PORT = os.environ.get("API_PORT", 8000)


@st.cache_data(max_entries=5, ttl=180)
def answer_query_invoke(query: str, user_id: Union[str, int]="-1", filename: str = "") -> Union[str, Dict]:
    try:
        data = {"query": query, "user_id": user_id, "filenames": []}
        if filename is None or filename == "":
            data['filenames'] = []
        else:
            data['filenames'] = [filename]
        # data['filenames'] = ["Universal Declaration"]#,"Custom Lipari"] # , "Smithfield Case Manual"
        print(data)
        r = requests.post(f"http://{API_URL}:{API_PORT}/pdfqna_agent", json=data)
        if r.ok or r.status_code == requests.codes.ok:
            answer = r.json()['output']

            return answer
        else:
            st.error(f"Received invalid status code: {r.status_code}")
            return None
    except (ConnectionRefusedError,requests.ConnectionError) as ce:
        print(ce)
        st.error(f"Failed to connect to {API_URL}:{API_PORT}/pdfqna_agent!",icon="🚨")
        return None
    except requests.HTTPError as he:
        st.error(f"There was an HTTP error in sending request to the API: {he}",icon="🚨")
        return None
    except Exception as e:
        st.error(f"Could not answer question due to error: {e}",icon="🚨")
        return None


def upload_file_invoke(blob_uri: str, user_id: Union[str, int]="-1", chunk_size: int = 600):
    try:
        data = {
            "filepaths": [blob_uri],
            "user_id": user_id
        }
        print(data)
        r = requests.post(f"http://{API_URL}:{API_PORT}/uploadfile", json=data)
        if r.ok or r.status_code == requests.codes.ok:
            r = r.json()

            return r.get("is_successful", True), r.get("message", "")
        # st.error(f"Received invalid status code: {r.status_code}")
        return False, f"Received invalid status code: {r.status_code}"
    except (ConnectionRefusedError,requests.ConnectionError) as ce:
        print(ce)
        # st.error(f"Failed to connect to {API_URL}:{API_PORT}/predict_one !",icon="🚨")
        return False,f"Failed to connect to {API_URL}:{API_PORT}/uploadfile !"
    except requests.HTTPError as he:
        # st.error(f"There was an HTTP error in sending request to the API: {he}",icon="🚨")
        return False,f"There was an HTTP error in sending request to the API: {he}"
    except Exception as e:
        # st.error(f"Could not answer question due to error: {e}",icon="🚨")
        return True,f"Could not answer question due to error: {e}"
    
if __name__ == "__main__":
    
    st.set_page_config(page_title='PDF QnA System Demo',page_icon='📄',layout='wide' )
    streamlit_style = """
            <style>
            @import url('https://fonts.googleapis.com/css2?family=Ubuntu:wght@100&display=swap');

            html, body, [class*="css"]  {
                font-family: 'Ubuntu', serif;
                font-size: 16px;
            }
            </style>
            """
    
    st.markdown(streamlit_style, unsafe_allow_html=True)
    
    if not "file_name" in st.session_state:
        st.session_state.file_name = []

    if not "file_uploader_key" in st.session_state:
        st.session_state["file_uploader_key"] = 0
    
    
    with st.sidebar:
        uid = st.selectbox("Select current user ID", ("None","User 1", "User 2", "User 3"), index=1)
        if uid == "None":
            uid = "-1"
        else:
            uid = uid[0].lower() + uid.split(' ')[-1]

    container = st.container()
    file = st.file_uploader("Choose a file", accept_multiple_files=False, type=['pdf'], key=st.session_state["file_uploader_key"])
    if file is not None:
        #print(st.session_state.file_name, file.name)
        # for file in files:
        if not file.name in st.session_state.file_name:
            blob_uri = upload_file_to_bucket(
                blob_name=f"{os.environ.get('GCS_UPLOAD_FILE_PREFIX','docs/inputs/')}{file.name.split('/')[-1]}",
                data=file.getvalue(),
                bucket_name=os.environ.get("GCS_UPLOAD_BUCKET","qna-staging") # The bucket in which this file will be uploaded
            )
            is_successful, message = upload_file_invoke(blob_uri=blob_uri, user_id=uid)
            if is_successful:
                st.session_state.file_name.append(file.name)
            else:
                st.session_state["file_uploader_key"] = (st.session_state["file_uploader_key"] + 1) % 10
                if message is not None:
                    st.error(f"File upload failed with following error: {message}", icon="🚨")
                #time.sleep(20)
                #st.rerun()
        
    query = container.text_input("Enter your query:", value="")
    if container.button("Submit") and query is not None and len(query) > 5:
        curr_file = ""
        if len(st.session_state.file_name) > 0:
            curr_file = st.session_state.file_name[-1]
        answer = answer_query_invoke(query, user_id=uid, filename=curr_file)
        if answer is not None:
            st.write(answer)
