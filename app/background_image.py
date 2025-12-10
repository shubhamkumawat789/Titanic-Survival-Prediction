# background_image.py
import base64

def set_background(image_file):
    """
    This function sets the background of a Streamlit app to an image.
    
    Parameters:
    image_file (str): The path to the image file.
    """
    with open(image_file, "rb") as f:
        img_data = f.read()
    b64_encoded = base64.b64encode(img_data).decode()
    style = f"""
    <style>
    .stApp {{
        background-image: url(data:image/png;base64,{b64_encoded});
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    
    /* Make the content area semi-transparent for better readability */
    .main .block-container {{
        background-color: rgba(255, 255, 255, 0.85);
        padding: 2rem;
        border-radius: 10px;
        margin-top: 2rem;
        margin-bottom: 2rem;
    }}
    
    /* Style sidebar */
    .css-1d391kg {{
        background-color: rgba(255, 255, 255, 0.9);
    }}
    
    /* Style buttons for better visibility */
    .stButton>button {{
        background-color: rgba(76, 175, 80, 0.9);
        color: white;
        font-weight: bold;
    }}
    
    /* Make metrics stand out */
    .css-1xarl3l {{
        background-color: rgba(255, 255, 255, 0.9);
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }}
    
    /* Style dataframes */
    .dataframe {{
        background-color: rgba(255, 255, 255, 0.9);
    }}
    
    /* Adjust headers for better visibility */
    h1, h2, h3, h4, h5, h6 {{
        color: #1f3a93;
    }}
    </style>
    """
    
    return style