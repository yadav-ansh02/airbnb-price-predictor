import base64
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


# ==========================================
# Page Configuration
# ==========================================
st.set_page_config(
    page_title="Airbnb Price Predictor",
    page_icon="🏠",
    layout="wide"
)


# ==========================================
# File Paths
# ==========================================
ROOT = Path(__file__).resolve().parent

MODEL_PATH = ROOT / "AirBnb_price_predictor.pkl"
BACKGROUND_PATH = ROOT / "Bg_image.webp"


# ==========================================
# Load Model
# ==========================================
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


try:
    model = load_model()

except FileNotFoundError as e:
    st.error(f"Required file not found: {e.filename}")
    st.stop()

except Exception as e:
    st.error("Could not load the saved model.")
    st.exception(e)
    st.stop()


# ==========================================
# Background Image
# ==========================================
if BACKGROUND_PATH.exists():

    with open(BACKGROUND_PATH, "rb") as image_file:
        encoded_image = base64.b64encode(
            image_file.read()
        ).decode()

    background = f"""
        url("data:image/webp;base64,{encoded_image}")
    """

else:
    background = "none"


# ==========================================
# Custom CSS
# ==========================================
st.markdown(
    f"""
    <style>

    /* Main background */
    .stApp {{
        background-image:
            linear-gradient(
                rgba(0, 0, 0, 0.55),
                rgba(0, 0, 0, 0.55)
            ),
            {background};

        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}

    /* Main container */
    .block-container {{
        max-width: 1100px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }}

    /* Title */
    h1 {{
        color: white;
        text-align: center;
        font-size: 3rem;
    }}

    /* Description */
    .description {{
        color: white;
        text-align: center;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }}

    /* Input labels */
    div[data-testid="stSelectbox"] label,
    div[data-testid="stSlider"] label {{
        color: white;
        font-weight: 600;
    }}

    /* Button */
    div[data-testid="stButton"] button {{
        width: 100%;
        font-weight: bold;
        font-size: 1rem;
        padding: 0.7rem;
    }}

    </style>
    """,
    unsafe_allow_html=True
)


# ==========================================
# Title
# ==========================================
st.title("🏠 Airbnb Price Predictor")

st.markdown(
    """
    <div class="description">
        Enter the details of your Airbnb listing to estimate the nightly price.
    </div>
    """,
    unsafe_allow_html=True
)


# ==========================================
# Input Fields
# ==========================================

# First row
col1, col2, col3 = st.columns(3)


with col1:

    property_type = st.selectbox(
        "Property Type",
        [
            "Apartment",
            "Bed & Breakfast","Boat","Boutique hotel","Bungalow","Cabin","Camper/RV","Casa particular",
            "Castle","Cave","Chalet","Condominium","Dorm","Earth House","Guest suite","Guesthouse","Hostel","House",
            "Hut","In-law","Island","Lighthouse","Loft","Other","Parking Space","Serviced apartment",
            "Tent","Timeshare","Tipi","Townhouse","Train","Treehouse","Vacation home","Villa","Yurt"
        ]
    )


with col2:

    city = st.selectbox(
        "City",
        [
            "NYC",
            "LA",
            "Chicago",
            "SF",
            "DC",
            "Boston"
        ]
    )


with col3:

    accommodates = st.slider(
        "Guests",
        min_value=1,
        max_value=16,
        value=2
    )


# Second row
col4, col5, col6, col7 = st.columns(4)


with col4:

    bedrooms = st.slider(
        "Bedrooms",
        min_value=0.0,
        max_value=10.0,
        value=1.0,
        step=0.5
    )


with col5:

    review_score = st.slider(
        "Guest Rating",
        min_value=20,
        max_value=100,
        value=90
    )


with col6:

    cleaning_fee = st.selectbox(
        "Cleaning Fee",
        [False, True],
        format_func=lambda x: "Yes" if x else "No"
    )


with col7:

    instant_bookable = st.selectbox(
        "Instant Booking",
        ["f", "t"],
        format_func=lambda x: "Yes" if x == "t" else "No"
    )


# ==========================================
# Prediction Button
# ==========================================

st.write("")

if st.button("💰 Predict Price"):

    # Create input DataFrame
    input_data = pd.DataFrame(
        {
            "property_type": [property_type],
            "accommodates": [accommodates],
            "cleaning_fee": [cleaning_fee],
            "city": [city],
            "instant_bookable": [instant_bookable],
            "review_scores_rating": [review_score],
            "bedrooms": [bedrooms]
        }
    )

    try:

        # The saved model is a fitted pipeline that includes its preprocessor.
        predicted_log_price = model.predict(input_data)[0]
        prediction = np.expm1(predicted_log_price)

        # Display result
        st.success(
            f"### Estimated Nightly Price: ${prediction:,.0f} USD"
        )

        st.info(
            "Estimated price before taxes and platform fees."
        )

    except Exception as e:

        st.error("Prediction failed.")

        st.exception(e)


# streamlit run app.py
