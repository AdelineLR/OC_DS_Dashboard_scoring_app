import streamlit as st
from streamlit_shap import st_shap 
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt 
import os
import json
import pickle
import shap
from st_aggrid import AgGrid, GridOptionsBuilder
from preprocess import Preprocess
import boto3

shap.initjs()

#--------Page Configuration ---------
st.set_page_config(page_title="Credit scoring", layout="wide")

#--------Navigation menu ----------
st.sidebar.image("/app/app/images/pret_a_depenser.png")
debug_mode = st.sidebar.checkbox("Mode debug") 
menu = st.sidebar.selectbox(
    "Navigation",
    options=["Accueil", "Informations principales", "Analyse du score crédit",  "Comparaison"]
)

#-------Load and preprocess data-----
# Initialize class
preprocessor = Preprocess()

@st.cache_data
def get_aggregated_data():
    return preprocessor.aggregate_tables()

@st.cache_data
def generate_histograms(data, histogram_color="#1f77b4"):
    """!
    @brief Generate histograms for numerical features in the dataset.

    @param data (pd.DataFrame): The input data.
    @param histogram_color (str): Default color for the histograms.
    @param hue (str): Column name for grouping (categorical variable).

    @return dict: A dictionary of Plotly histogram figures.
    """
    histograms = {}
    features = [col for col in data.columns if col != 'SK_ID_CURR']
    for column in features:
        if data[column].dtype in ['int64', 'float64']:  # Vérifier que c'est une colonne numérique
            fig = px.histogram(data, x=column, nbins=30, title=f"Histogramme de {column}")
            fig.update_traces(marker_color=histogram_color)
            fig.update_layout(xaxis_title=column, yaxis_title="Fréquence")
            histograms[column] = fig
    return histograms

# Preprocessing the data
# Load data only once at the start
# test
AWS_REGION = "eu-west-3"
BUCKET_NAME = "p8bucket"
s3_client = boto3.client('s3', region_name=AWS_REGION)
response = s3_client.list_objects(Bucket=BUCKET_NAME)
st.write(response)
if 'data' not in st.session_state:
    st.session_state.data = None
    st.session_state.histograms = None

# Ajout d'un bouton pour déclencher le chargement et le prétraitement
if st.button("Charger et Prétraiter les Données"):
    with st.spinner("Chargement et Prétraitement en cours..."):
        st.session_state.data = get_aggregated_data(debug_mode)
        st.session_state.histograms = generate_histograms(st.session_state.data, histogram_color="#1f77b4")
        st.success("Données chargées et prétraitées avec succès!")

# Affichage des résultats si les données sont chargées
if st.session_state.data is not None:
    st.write("Données prétraitées :")
    st.write(st.session_state.data)
    st.write("Histogrammes générés :")
    st.pyplot(st.session_state.histograms)
## end test
# if 'data' not in st.session_state:
#     with st.spinner("Chargement et Prétraitement en cours..."):
#         st.session_state.data = get_aggregated_data(debug_mode)
#         st.session_state.histograms = generate_histograms(st.session_state.data, histogram_color="#1f77b4")

#---------Settings------------
# features to display
feature_filter = ['DAYS_EMPLOYED', 
                    'DAYS_BIRTH', 
                    'AMT_CREDIT', 
                    'AMT_INCOME_TOTAL', 
                    'INCOME_PER_PERSON',
                    'PAYMENT_RATE',
                    'CREDIT_INCOME_PERCENT']

# features for API
feature_names = ['EXT_SOURCE_2', 'PAYMENT_RATE', 'EXT_SOURCE_3', 'DAYS_BIRTH',
                'DAYS_EMPLOYED', 'DAYS_REGISTRATION', 'DAYS_ID_PUBLISH',
                'ANNUITY_INCOME_PERCENT', 'INSTAL_DBD_MEAN', 'DAYS_LAST_PHONE_CHANGE',
                'AMT_ANNUITY', 'ACTIVE_DAYS_CREDIT_UPDATE_MEAN',
                'REGION_POPULATION_RELATIVE', 'INSTAL_DAYS_ENTRY_PAYMENT_MAX',
                'CLOSED_DAYS_CREDIT_MAX', 'ACTIVE_DAYS_CREDIT_ENDDATE_MIN',
                'INSTAL_AMT_PAYMENT_MIN', 'PREV_APP_CREDIT_PERC_VAR',
                'BURO_DAYS_CREDIT_VAR', 'INSTAL_DBD_SUM', 'INSTAL_DBD_MAX',
                'CLOSED_DAYS_CREDIT_ENDDATE_MAX', 'BURO_AMT_CREDIT_MAX_OVERDUE_MEAN',
                'INCOME_PER_PERSON', 'ACTIVE_DAYS_CREDIT_MAX',
                'CLOSED_AMT_CREDIT_SUM_MEAN', 'PREV_HOUR_APPR_PROCESS_START_MEAN',
                'INSTAL_DAYS_ENTRY_PAYMENT_MEAN',
                'POS_NAME_CONTRACT_STATUS_Active_MEAN', 'TOTALAREA_MODE',
                'CLOSED_DAYS_CREDIT_UPDATE_MEAN', 'ACTIVE_AMT_CREDIT_SUM_DEBT_MEAN',
                'BURO_DAYS_CREDIT_MEAN', 'PREV_CNT_PAYMENT_MEAN',
                'INSTAL_DAYS_ENTRY_PAYMENT_SUM', 'CLOSED_AMT_CREDIT_SUM_SUM',
                'INSTAL_AMT_PAYMENT_MEAN', 'PREV_APP_CREDIT_PERC_MEAN',
                'POS_MONTHS_BALANCE_SIZE', 'INSTAL_DPD_MEAN', 'PREV_AMT_ANNUITY_MIN',
                'PREV_AMT_ANNUITY_MEAN', 'PREV_NAME_TYPE_SUITE_Unaccompanied_MEAN',
                'BURO_DAYS_CREDIT_ENDDATE_MIN', 'HOUR_APPR_PROCESS_START',
                'INSTAL_AMT_INSTALMENT_MAX', 'INSTAL_PAYMENT_PERC_VAR',
                'PREV_NAME_YIELD_GROUP_middle_MEAN', 'PREV_RATE_DOWN_PAYMENT_MEAN',
                'APPROVED_AMT_DOWN_PAYMENT_MAX'
                ]

#--------Home section ---------
if menu == "Accueil":
    st.header("Bienvenue sur le Dashboard d'aide à la décision pour l'octroi d'un prêt")
    st.write("Ce dashboard permet de visualiser les principales informations des clients, d'analyser la probabilité d'octroi d'un prêt, et de comparer leurs caractéristiques.")

    with st.container():
        # Check if sk_id_curr is in session_state
        if 'sk_id_curr' not in st.session_state:
            curr_value = 100002
        else:
            curr_value = st.session_state.sk_id_curr

        # Select customer id (input for SK_ID_CURR with six-digit constraint)
        st.session_state.sk_id_curr = st.number_input(
            label="Veuillez saisir l'identifiant du client",
            value=curr_value,
            placeholder="Enter SK_ID_CURR (six digits)",
            min_value=100002,  # Minimum six-digit number
            max_value=999999,  # Maximum six-digit number
            step=1
        )

# ------------- Client Data  --------------

elif menu == "Informations principales":
    st.header(f"Informations principales sur le client, id={st.session_state.sk_id_curr}")
    feature_descriptions = {
    'TARGET': 'est TARGET',
    }

    with st.container():
        col1, col2 = st.columns([1,3])

        # Filter data and display client information
        with col1:
            st.write(f"### Données client")

            # Filter the data based on SK_ID_CURR
            features = feature_filter + ['SK_ID_CURR']
            filtered_data = st.session_state.data.loc[
                                                      st.session_state.data['SK_ID_CURR'] == st.session_state.sk_id_curr, 
                                                      features
                                                      ]
            if not filtered_data.empty:
                # Get list of columns excluding 'SK_ID_CURR'
                features = [col for col in filtered_data.columns if col != 'SK_ID_CURR']

                # Reshape the DataFrame using melt
                melted_data = pd.melt(filtered_data, id_vars='SK_ID_CURR', value_vars=features)
                melted_data = melted_data.drop(columns='SK_ID_CURR')

                # Display the table with AgGrid
                grid_options = GridOptionsBuilder.from_dataframe(melted_data)
                grid_options.configure_selection('single', use_checkbox=True)  # Single row selection
                grid_options.configure_pagination(paginationPageSize=10)
                # grid_options.configure_column('value', editable=True)  # Make the 'value' column editable
                grid_options = grid_options.build()

                grid_response = AgGrid(melted_data, gridOptions=grid_options, height=500, width='100%',use_checkbox=True)

                # Display variable description when a user clicks on a feature name
                selected_rows = grid_response['selected_rows']

                if selected_rows is not None and len(selected_rows)>0:  # Check if there is at least one selected row
                    selected_variable = selected_rows['variable'].iloc[0]  # Get the 'variable' column value

                    # Check if the selected variable has a description
                    if selected_variable in feature_descriptions:
                        description = feature_descriptions[selected_variable]
                    else:
                        description = "Description not available."
                    st.write(description)
                else:
                    st.write("Sélectionner une variable pour voir sa description et le graphique associé.")
            else:
                st.warning(f"Aucune donnée trouvée pour SK_ID_CURR: {st.session_state.sk_id_curr}")
        
        
        with col2:
            # Plot the histogram
            st.write(f"### Positionnement du client")
            if selected_variable in filtered_data.columns:
                value = filtered_data.iloc[0][selected_variable]
                fig = st.session_state.histograms[selected_variable]
                fig.add_shape(
                    type="line",
                    x0=value,
                    x1=value,
                    y0=0,
                    y1=1,
                    line=dict(color="orange", width=3, dash="dash"),
                    name=f"Client id {st.session_state.sk_id_curr}", 
                    xref="x",
                    yref="paper"
                )

                st.plotly_chart(fig)


# Section : Analyse des scores
elif menu == "Analyse du score crédit":
    st.header("Analyse des scores clients")

    def request_prediction(endpoint, data):
        payload = {"features": data.iloc[0].to_dict()}

        response = requests.post(endpoint, json=payload)

        if response.status_code == 200:
            print("Prediction:", response.json())
        else:
            print("Error:", response.status_code, response.text)

        return response.json()
    

    # Source : https://github.com/REPNOT/streamviz/blob/main/streamviz.py + https://plotly.com/python/gauge-charts/
    def gauge(gVal, gTitle="", gthreshold=0.5, gMode='gauge+number', gSize="FULL", gTheme="Black",
            grLow=.29, grMid=.69, gcLow='#FF1708', gcMid='#FF9400', 
            gcHigh='#1B8720', xpLeft=0, xpRight=1, ypBot=0, ypTop=1, 
            arBot=None, arTop=1, pTheme="streamlit", cWidth=True, sFix=None):


        if sFix == "%":
            gaugeVal = round((gVal * 100), 1)
            top_axis_range = (arTop * 100)
            bottom_axis_range = arBot
            low_gauge_range = (grLow * 100)
            mid_gauge_range = (grMid * 100)
            threshold = (gthreshold * 100)

        else:
            gaugeVal = gVal
            top_axis_range = arTop
            bottom_axis_range = arBot
            low_gauge_range = grLow
            mid_gauge_range = grMid
            threshold = gthreshold

        if gSize == "SML":
            x1, x2, y1, y2 =.25, .25, .75, 1
        elif gSize == "MED":
            x1, x2, y1, y2 = .50, .50, .50, 1
        elif gSize == "LRG":
            x1, x2, y1, y2 = .75, .75, .25, 1
        elif gSize == "FULL":
            x1, x2, y1, y2 = 0, 1, 0, 1
        elif gSize == "CUST":
            x1, x2, y1, y2 = xpLeft, xpRight, ypBot, ypTop   

        if gaugeVal <= low_gauge_range: 
            gaugeColor = gcLow
        elif gaugeVal >= low_gauge_range and gaugeVal <= mid_gauge_range:
            gaugeColor = gcMid
        else:
            gaugeColor = gcHigh

        fig1 = go.Figure(go.Indicator(
            mode = gMode,
            value = gaugeVal,
            domain = {'x': [x1, x2], 'y': [y1, y2]},
            number = {"suffix": sFix},
            title = {'text': gTitle},
            gauge = {
                'axis': {'range': [bottom_axis_range, top_axis_range]},
                'bar' : {'color': gaugeColor},
                'threshold' : {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': threshold}
            }
        ))

        config = {'displayModeBar': False}
        fig1.update_traces(title_font_color=gTheme, selector=dict(type='indicator'))
        fig1.update_traces(number_font_color=gTheme, selector=dict(type='indicator'))
        fig1.update_traces(gauge_axis_tickfont_color=gTheme, selector=dict(type='indicator'))
        fig1.update_layout(margin_b=5)
        fig1.update_layout(margin_l=20)
        fig1.update_layout(margin_r=20)
        fig1.update_layout(margin_t=50)

        fig1.update_layout(margin_autoexpand=True)

        st.plotly_chart(
            fig1, 
            use_container_width=cWidth, 
            theme=pTheme, 
            **{'config':config}
        )


    # API url
    endpoint = "http://ec2-3-226-55-161.compute-1.amazonaws.com/predict"

    input_data = st.session_state.data.loc[
                                            st.session_state.data['SK_ID_CURR'] == st.session_state.sk_id_curr, 
                                            feature_names
                                            ]
    input_data = input_data.fillna(0)

    result = request_prediction(endpoint, input_data)

    prediction = result['prediction']
    probability = round(result['probabilities']['class_0'],2)
    shap_response = result['shap_values']

    # Create an object SHAP Explanation
    base_value = shap_response["base_value"]  
    shap_values = np.array(shap_response["shap_values"])
    input_data = np.array(shap_response["input_data"])

    shap_explanation = shap.Explanation(
        values=shap_values,
        base_values=base_value,
        data=input_data
    )

    shap_explanation.feature_names = feature_names


    with st.container():
        st.write(f"### Statut de la demande de crédit :{prediction}")

        gauge(probability, gTitle="Probability", gMode='gauge+number', gSize="FULL", gTheme="Black",
        grLow=.34, grMid=.64, gcLow='#FF1708', gcMid='#FF9400', 
        gcHigh='#1B8720', xpLeft=0, xpRight=1, ypBot=0, ypTop=1, 
        arBot=None, arTop=1, pTheme="streamlit", cWidth=True, sFix=None)

        col1, col2 =st.columns([2,2])
        with col1:
            st.image('images/shap_summary_plot.png')

        with col2:
            st_shap(shap.plots.waterfall(shap_explanation))

