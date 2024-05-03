import streamlit as st

st.set_page_config(
    page_title="Introduction",
    page_icon="🙂",
    layout="centered",
    menu_items={"About": "This is a Demo App containing demos for **PDFQnA** _(upload a PDF file and then query that file)_ and **KGQnA** (existing knowledge base and returning structured responses) created using Streamlit"}
)

st.write("# Welcome to the QnA Demo Introduction Page!")
st.markdown(
    """
This app contains the following two demos: PDFQnA and KGQnA.
**👈 Select a demo from the sidebar** and try it out!

- ## PDFQnA
To run, just upload a file, select the user id from the sidebar and after the file has been uploaded, ask any question from the file!
Any files uploaded earlier will remain saved against the user for future sessions.

- ## KGQnA
Just enter your query and click Submit! If an answer can be found you'll get one!
"""
)