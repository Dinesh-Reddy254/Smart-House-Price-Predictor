
import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import time

# ---------------------------------------------------------
# PAGE CONFIG (must be the first Streamlit command)
# ---------------------------------------------------------
st.set_page_config(
    page_title="Smart House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# HERO HEADER + BANNER
# ---------------------------------------------------------
st.image(
    "https://images.unsplash.com/photo-1560518883-ce09059eeffa",
    use_container_width=True
)

st.markdown("""
<div style="
padding:25px;
border-radius:12px;
background:linear-gradient(90deg,#1f2937,#111827);
color:white;
text-align:center;">
<h1>🏠 Smart House Price Predictor</h1>
<p>AI-powered real estate valuation platform</p>
</div>
""", unsafe_allow_html=True)

st.divider()

# ---------------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------------
st.markdown("""
<style>
.prediction-box{
    background:#0E1117;
    padding:30px;
    border-radius:12px;
    text-align:center;
}
.metric-card{
    padding:15px;
    border-radius:10px;
    background:#1f2937;
    color:white;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------------
model = pickle.load(open("model/house_price_model.pkl","rb"))

# ---------------------------------------------------------
# SIDEBAR INPUTS
# ---------------------------------------------------------
st.sidebar.header("Enter Property Details")

area = st.sidebar.number_input("Area (sq ft)", 500, 10000, 2000)
bedrooms = st.sidebar.number_input("Bedrooms", 1, 10, 3)
bathrooms = st.sidebar.number_input("Bathrooms", 1, 10, 2)
stories = st.sidebar.number_input("Stories", 1, 5, 2)
parking = st.sidebar.number_input("Parking Spaces", 0, 5, 1)

mainroad = st.sidebar.selectbox("Main Road", ["Yes","No"])
guestroom = st.sidebar.selectbox("Guest Room", ["Yes","No"])
basement = st.sidebar.selectbox("Basement", ["Yes","No"])
airconditioning = st.sidebar.selectbox("Air Conditioning", ["Yes","No"])
prefarea = st.sidebar.selectbox("Preferred Area", ["Yes","No"])

furnishing = st.sidebar.selectbox(
    "Furnishing Status",
    ["Furnished","Semi-Furnished","Unfurnished"]
)

# ---------------------------------------------------------
# CONVERT INPUTS
# ---------------------------------------------------------
mainroad = 1 if mainroad == "Yes" else 0
guestroom = 1 if guestroom == "Yes" else 0
basement = 1 if basement == "Yes" else 0
airconditioning = 1 if airconditioning == "Yes" else 0
prefarea = 1 if prefarea == "Yes" else 0

semi = 1 if furnishing == "Semi-Furnished" else 0
unfurn = 1 if furnishing == "Unfurnished" else 0

# ---------------------------------------------------------
# CREATE INPUT DATAFRAME
# ---------------------------------------------------------
input_data = pd.DataFrame({
    'area':[area],
    'bedrooms':[bedrooms],
    'bathrooms':[bathrooms],
    'stories':[stories],
    'mainroad':[mainroad],
    'guestroom':[guestroom],
    'basement':[basement],
    'hotwaterheating':[0],
    'airconditioning':[airconditioning],
    'parking':[parking],
    'prefarea':[prefarea],
    'furnishingstatus_semi-furnished':[semi],
    'furnishingstatus_unfurnished':[unfurn]
})

# ---------------------------------------------------------
# MAIN TABS
# ---------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "💰 Prediction", "ℹ️ About"])

# ---------------------------------------------------------
# DASHBOARD TAB
# ---------------------------------------------------------
with tab1:

    st.subheader("Property Overview")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Area", f"{area} sq ft")
        st.metric("Bedrooms", bedrooms)
        st.metric("Bathrooms", bathrooms)
        st.metric("Stories", stories)

    with col2:
        st.metric("Parking", parking)
        st.metric("Main Road", "Yes" if mainroad else "No")
        st.metric("Air Conditioning", "Yes" if airconditioning else "No")
        st.metric("Preferred Area", "Yes" if prefarea else "No")

# ---------------------------------------------------------
# PREDICTION TAB
# ---------------------------------------------------------
with tab2:

    st.subheader("Predict Property Value")

    if st.button("🚀 Predict House Price"):

        with st.spinner("Analyzing property features..."):
            time.sleep(1)

            prediction = model.predict(input_data)[0]

        st.markdown(f"""
        <div class="prediction-box">
        <h2>Estimated Property Value</h2>
        <h1>₹{prediction:,.2f}</h1>
        </div>
        """, unsafe_allow_html=True)

        # Property classification
        if prediction < 3000000:
            st.info("🏠 Budget Property")
        elif prediction < 6000000:
            st.warning("🏠 Mid-range Property")
        else:
            st.success("🏠 Premium Property")

        st.divider()

        # Feature importance chart
        st.subheader("Feature Influence on Price")

        try:
            importance = model.coef_
            features = input_data.columns

            fig, ax = plt.subplots()
            ax.barh(features, importance)
            ax.set_title("Feature Impact")

            st.pyplot(fig)

        except:
            st.write("Feature impact not available.")

# ---------------------------------------------------------
# ABOUT TAB
# ---------------------------------------------------------
with tab3:

    st.subheader("About This Project")

    st.write("""
This application predicts house prices using a Machine Learning model.

### Features
- Property data dashboard
- Price prediction engine
- Feature influence visualization

### Technologies Used
- Python
- Pandas
- Scikit-learn
- Streamlit
""")

# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------
st.divider()

st.caption("Built by Dinesh Reddy")
st.caption("Tech Stack: Python • Scikit-learn • Streamlit")

