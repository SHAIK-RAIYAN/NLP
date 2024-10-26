# Custom CSS styling with complete border and background blur
st.markdown(
    """
    <style>
    /* Full-page background image */
    body {
        background-image: url('https://images.unsplash.com/photo-1511974035430-5de47d3b95da');
        background-size: cover;
        background-attachment: fixed;
        color: white;  /* Default text color */
    }
    
    /* Outer container with border and blur */
    .outer-container {
        max-width: 800px;
        margin: 20px auto;
        padding: 30px;
        border-radius: 15px;
        border: 2px solid rgba(255, 255, 255, 0.3);
        backdrop-filter: blur(15px);  /* Background blur */
        background: rgba(0, 0, 0, 0.6);  /* Dark overlay with transparency */
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
    }

    /* Title styling */
    .title {
        font-size: 2.5em;
        font-weight: bold;
        color: white;
        text-align: center;
        padding: 0.5em 0;
    }
    
    /* Button styling */
    .stButton>button {
        background-color: #007ACC;
        color: white;
        font-size: 16px;
        font-weight: bold;
        padding: 8px 16px;
        border-radius: 8px;
        transition: 0.3s;
        margin-top: 10px;
    }
    .stButton>button:hover {
        background-color: #005B99;
        color: #fff;
    }
    
    /* Dropdown styling */
    .stSelectbox {
        color: white;
        font-size: 1.1em;
        font-weight: 500;
        padding: 10px 0;
    }
    
    /* Text styling */
    .section-text {
        color: white;
        font-size: 1.2em;
        font-weight: 500;
        margin-top: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Wrap main content with `outer-container` div
st.markdown("<div class='outer-container'><div class='title'>Audio Translation and Emotion Detection System</div>", unsafe_allow_html=True)

# Continue with the rest of the code ...
