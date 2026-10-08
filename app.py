import pandas as pd
import streamlit as st 
import joblib
import time

# Page config 
st.set_page_config(
    page_title="Audi Price Predictor",
    layout="wide"  #uses full browser width instead of a narrow centred column
)

# CUSTOM CSS
st.markdown("""
            <style>
            /* style the metric cards */
            div[data-testid = "stMetric"]{
            background-color: #F5F5F5;
            border: 1px solid #E0E0E0;
            border-radius: 10px;
            padding: 15px;
            }
            /* Styling the main title */
            h1 {
                color : #BB0A30;
            }
            /* Styling the sidebar background */
            section[data-testid = "stSidebar"]{
                background-color: #FAFAFA;
            }
            </style>
            """,unsafe_allow_html=True)

# LOAD MODEL ARTIFACTS
# @st.cache_resource - tells st to load this once and keep it in memory on every rerun

@st.cache_resource
def load_artifacts():
    regressor = joblib.load('model/svr_model_audi_cars.pkl')
    ct = joblib.load('model/ct_svr.pkl')
    sc_X = joblib.load('model/sc_X_svr.pkl')
    sc_y = joblib.load('model/sc_y_svr.pkl')
    return regressor, ct, sc_X, sc_y

regressor, ct, sc_X, sc_y = load_artifacts()

# HEADER

col_logo,col_title = st.columns([1,6]) #divide one in 1 part and the other in 6 parts 1:6 ratio

with col_title:
    st.title("Audi Price Predictor")
    st.caption("Instant, data-driven price estimates powered by Support Vector Regression")
st.divider()

# SIDEBAR - All our inputs will be here 
st.sidebar.header("📋 Car Details")
model = st.sidebar.selectbox(
    "Model",
    ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'Q2', 'Q3', 'Q5',
     'Q7', 'Q8', 'R8', 'RS3', 'RS4', 'RS5', 'RS6', 'RS7', 'S3', 'S4',
     'S5', 'S8', 'SQ5', 'SQ7', 'TT']
)
year = st.sidebar.slider("Year", min_value=1997, max_value=2026, value=2018)

transmission = st.sidebar.selectbox("Transmission", ['Manual', 'Automatic', 'Semi-Auto'])

mileage = st.sidebar.number_input("Mileage (miles)", min_value=0, value=15000, step=500)

fuelType = st.sidebar.selectbox("Fuel Type", ['Petrol', 'Diesel', 'Hybrid'])

tax = st.sidebar.number_input("Road Tax (£)", min_value=0, value=150, step=10)

mpg = st.sidebar.number_input("MPG", min_value=0.0, value=50.0, step=0.5)

engineSize = st.sidebar.number_input("Engine Size (L)", min_value=0.0, value=1.4, step=0.1)

predict_clicked = st.sidebar.button("🔍 Predict price",use_container_width=True,type='primary') # stretch to the full width of the container

#  TABS — organizes the main area into separate views
tab1,tab2 = st.tabs(["💰 Predict price","ℹ️ About this Model"])
with tab1: 
    if predict_clicked:
        # A spinner basically gives visual feedback during processing
        with st.spinner("Calculating estimate: "):
            time.sleep(0.4)
# Main Area - results appear after the button is clicked 
            new_car = pd.DataFrame({
                'model': [model],
                'year': [year],
                'transmission': [transmission],
                'mileage': [mileage],
                'fuelType': [fuelType],
                'tax': [tax],
                'mpg': [mpg],
                'engineSize': [engineSize]
            })
 # Pipeline
            new_car_encoded = ct.transform(new_car)
            new_car_scaled = sc_X.transform(new_car_encoded)
            pred_scaled = regressor.predict(new_car_scaled)
            pred_price = sc_y.inverse_transform(pred_scaled.reshape(-1, 1))[0][0]
        st.success("Estimate Ready!")
        
        # Big headline number 
        st.markdown(f"""
                    <div style = "text-align: center; padding: 20px; background-color:#BB0A30; border-radius:12px;
                    margin-bottom:20px;">
                    <span style = "color:white; font-size:18px;">Estimated Market Price</span><br>
                    <span style = "color:white; font-size:48px; font-weight:bold;">£{pred_price:,.0f}</span>
                    </div>
                    """,unsafe_allow_html=True)

# Splitting the width of the main area into 3 (supporting metric rows)
        col1,col2,col3,col4 = st.columns(4)
        # st.metric gives a nice big number display
        # Supporting Metric Rows
        col1.metric("Model", model)
        col2.metric("Year", year) 
        col3.metric("Mileage",f"{mileage} mi")
        col4.metric("Engine",f"{engineSize}L")
        
        st.divider()
        #  Expander hides detail until the user wants to check it 
        # keeps the main view clean
        with st.expander("📄 View full input summary"):
            st.dataframe(new_car,use_container_width=True,hide_index=True)
    
    else: 
        st.info("👈 Enter the car's details in the sidebar and click **Predict Price** to get started.")
    
with tab2:
    st.subheader("About this Model")
    st.write("""
    This price predictor uses a **Support Vector Regression (SVR)** model trained on
    historical Audi listing data, achieving a test R² score of **0.95** — meaning the
    model explains 95% of the variation in used Audi prices based on the features below.
    """)
    st.write("**Features used in prediction:**")
    st.markdown("""
    - Model
    - Year
    - Transmission type
    - Mileage
    - Fuel type
    - Road tax
    - Fuel economy (MPG)
    - Engine size
    """)

