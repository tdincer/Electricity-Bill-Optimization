import streamlit as st
import altair as alt
import pandas as pd
import numpy as np
from datetime import time
from optimizebill import electricbilloptimizer


def plot_components(df, placeholder):
    df = df.melt(id_vars='Timestamp', var_name='Components', value_name='Power')

    c = alt.Chart(df).transform_joinaggregate(
        order='sum(Power)',
        groupby=['Components']).mark_area().encode(
        x='Timestamp:T',
        y='Power:Q',
        color=alt.Color('Components:N', legend=alt.Legend(orient="bottom")),
        order=alt.Order('order:Q', sort='descending'),
        tooltip=['Components', 'hoursminutes(Timestamp)', 'Power'])

    placeholder.altair_chart(c, use_container_width=True)
    return placeholder


st.set_page_config(layout="wide")
st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 400px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 400px;
        margin-left: -400px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title('EV Charging: Bill Optimizer')
st.markdown("""
""")

# Sidebar
st.sidebar.markdown("## Input Parameters")

ncars = st.sidebar.selectbox(
    "Number of Electric Vechicles", (1, 2, 3, 4), index=3
)

edemand = []
with st.sidebar.form(key='columns_in_form'):
    st.write('Energy Demand')
    cols = st.beta_columns(ncars)
    np.random.seed(15)
    edemand = np.random.uniform(15, 35, ncars)
    for i, col in enumerate(cols):
        cols[i].text_input(label='Vehicle ' + str(i + 1), value=edemand[i])
    submitted = st.form_submit_button('Submit')


period = st.sidebar.slider("Charging Period:", max_value=time(23, 45), value=(time(10, 00), time(23, 45)))
period_min = int((period[0].hour + period[0].minute / 60) / 0.25)
period_max = int((period[1].hour + period[1].minute / 60) / 0.25 + 1)

power = st.sidebar.slider('Charger Power (kW):', 0., 10., (0., 7.), step=0.0001)

# Main Content
main_placeholder = st.empty()

buildingload = './data/buildingload.csv'
if buildingload is not None:
    buildingload = pd.read_csv(buildingload)
    buildingload['Timestamp'] = pd.to_datetime(buildingload.Timestamp)
    buildingload = buildingload.rename(columns={'BuildingLoad[kW]': 'BuildingLoad'})

    main_placeholder = plot_components(buildingload, main_placeholder)


if st.button('Optimize'):
    with st.spinner('Smart martians are optimizing your electricity bill...'):
        eoptimize = electricbilloptimizer(ncars=ncars, car_edemand=[float(x) for x in edemand],
                                          charger_min=power[0], charger_max=power[1], arrival_time=period_min,
                                          departure_time=period_max)

        eoptimize.optimize()

        if eoptimize.res['success']:
            st.success('Optimization Status: Successful!')
        else:
            st.error('Optimization Status: Failed! - ' + eoptimize.res['message'])

        st.write('Optimized Electricity Bill: $%.2f' % eoptimize.res['fun'])
        dispatch = eoptimize.make_output()
        dispatch['BuildingLoad'] = buildingload['BuildingLoad']

        main_placeholder.empty()
        main_placeholder = plot_components(dispatch, main_placeholder)

        dispatch['Timestamp'] = dispatch['Timestamp'].dt.time
        st.write(dispatch)
