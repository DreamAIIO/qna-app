import os
from typing import List

import pandas as pd
import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "localhost")
API_PORT = os.environ.get("API_PORT", 8000)


@st.cache_data(max_entries=5, ttl=180)
def answer_query_invoke(query: str) -> tuple:
    try:
        data = {"input_data": {"question": query, "query": ""}}
        r = requests.post(f"http://{API_URL}:{API_PORT}/predict_one", json=data)
        r.raise_for_status()
        print(f'RETURNED STATUS CODE = {r.status_code}')
        if r.ok or r.status_code == requests.codes.ok:
            prediction = r.json()['prediction']
            if isinstance(prediction['rows'],str):
                result = prediction['heading'],[p.split('|') for p in prediction['rows'].split('\n') if p != ''],prediction['suggestions'],prediction['llm_response'], prediction['images_urls']
            elif isinstance(prediction['rows'],list):
                result = prediction['heading'],[p.split('|') for p in prediction['rows'] if p != ''],prediction['suggestions'],prediction['llm_response'], prediction['images_urls']
            else:
                result = prediction['heading'],prediction['rows'],prediction['suggestions'],prediction['llm_response'], prediction['images_urls']
            
            print(f'RESULT = {result}')

            return result
        else:
            st.error(f"Received invalid status code: {r.status_code}")
            return None,None,None,None,None
    except (ConnectionRefusedError,requests.ConnectionError) as ce:
        print(ce)
        st.error(f"Failed to connect to {API_URL}:{API_PORT}/kgqna_agent/invoke !",icon="🚨")
        return None,None,None,None,None
    except requests.HTTPError as he:
        st.error(f"There was an HTTP error in sending request to the API: {he}",icon="🚨")
        return None,None,None,None,None
    except Exception as e:
        st.error(f"Could not answer question due to error: {e}",icon="🚨")
        return None,None,None,None,None

@st.cache_data(max_entries=5, ttl=900)
def autocomplete_invoke(query: str, top_k: int = 10) -> list | None:
    try:
        data = {
            "input_data": {
                "query": query,
                "question": ""
            }
        }

            # "top_k": top_k,
        print(data)
        r = requests.post(f"http://{API_URL}:{API_PORT}/predict_one", json=data)
        if r.ok or r.status_code == requests.codes.ok:
            r = r.json()['prediction']

            return r.get("suggestions", [])
        st.error(f"Received invalid status code: {r.status_code}")
        return None
    except (ConnectionRefusedError,requests.ConnectionError) as ce:
        print(ce)
        st.error(f"Failed to connect to {API_URL}:{API_PORT}/predict_one !",icon="🚨")
        return None
    except requests.HTTPError as he:
        st.error(f"There was an HTTP error in sending request to the API: {he}",icon="🚨")
        return None
    except Exception as e:
        st.error(f"Could not answer question due to error: {e}",icon="🚨")
        return None

    

if __name__ == "__main__":
    st.set_page_config(page_title='KG QnA System Demo',page_icon='🤖',layout='wide' )
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
    question = st.text_input('Enter any question: ',value='What is the price of perdue chicken breast?',max_chars=300)
    if st.button('Answer Question'):
        #print("KGQnA_Demo",question)
        heading,rows,suggestions,llm_response,images_urls = answer_query_invoke(question)
        if heading is None:
            print("ERROR!")
        elif len(llm_response) > 0:
            st.markdown('**Answer:**')
            st.text('.\n'.join(llm_response.split('. ')))
        elif len(rows) == 0:
            st.write(heading)
            #st.text(heading)
            if len(suggestions) > 0:
                st.table(suggestions)
            elif len(images_urls) > 0:
                st.table(images_urls)
        elif len(rows) > 0:
            rows = pd.DataFrame(
                rows[1:],
                columns=rows[0]
            )

            st.markdown('**Answer:**')
            st.text(heading)
            st.dataframe(rows,use_container_width=True)
    if st.button('Autocomplete'):
        suggestions = autocomplete_invoke(question)
        if suggestions is None:
            print("ERROR!")
        elif len(suggestions) == 0:
            st.write("No suggestions available!")
        else:
            st.table(suggestions)
