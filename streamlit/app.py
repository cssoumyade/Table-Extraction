import os
import re
import numpy as np
import csv
import pandas as pd
from tqdm import tqdm
from PIL import Image
import cv2
from time import strftime
import pytesseract
import streamlit as st
from helper import *  



def main():
    st.set_page_config(layout="wide")
    st.text("")
    st.text("")
    st.title("Table Extraction from Images")
    image_url = "https://images.unsplash.com/photo-1497366216548-37526070297c?ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&ixlib=rb-1.2.1&auto=format&fit=crop&w=1050&q=80"
    st.image(image_url, width=500)
    
    st.sidebar.header("About")
    with open('about.txt', 'r') as f:
        about_txt = f.read()
    st.sidebar.write(about_txt)

    st.text("")
    st.text("")
    st.text("")

    st.subheader("Image Uploader :")
    img_path = st.file_uploader("Upload an image file (must be in jpg/png)", type=['jpg', 'png'])

    if img_path is not None:
        csv, tab = final(img_path, show_table=True)
        df = pd.read_csv(csv)
        st.subheader("Table Image")
        tab = Image.fromarray(tab)
        st.image(tab)
        st.subheader("Retrived Text")
        st.write(df)
    pass

if __name__=="__main__":
    main()