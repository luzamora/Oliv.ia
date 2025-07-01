import streamlit as st
import pandas as pd
import plotly.express as px
from PIL import Image
import os
import ast
import numpy as np

# General app config
st.set_page_config(page_title="Oliv.ia", page_icon="logomini.png", layout="wide")
# Sidebar custom style
st.markdown("""
<style>
    section[data-testid="stSidebar"] {
        background-color: #d7e8bc;
        width: 280px;
    }
</style>
""", unsafe_allow_html=True)

# Load main data
@st.cache_data
def load_data():
    df_bus = pd.read_parquet("datasets/yelp_business.parquet")
    attributes = pd.read_parquet("datasets/attributes_multihot.parquet")
    attributes.columns = attributes.columns.str.strip()
    attributes.index = df_bus.index[:len(attributes)]

    def to_list(x):
        if isinstance(x, (list, np.ndarray)):
            return list(x)
        return []

    df_bus['categories_list'] = df_bus['categories'].apply(to_list)

    df = pd.concat([df_bus, attributes], axis=1)
    return df, attributes

with st.spinner("Loading data..."):
    df, attributes = load_data()


# === SIDEBAR ===
with st.sidebar:
    logo = Image.open("img/logo.png")
    st.image(logo, width=250)
    st.markdown("**Filters**")

    states = sorted(df['state'].dropna().unique().tolist())
    states.insert(0, "All")
    selected_state = st.selectbox("State", states)

    stars_slider = st.select_slider("Rating for wordclouds", options=[1, 2, 3, 4, 5])

# Filter by state
filtered_df = df.copy() if selected_state == "All" else df[df['state'] == selected_state]

# === TITLE ===
st.title("Data Exploration - Oliv.ia")

# === MAP AND TOP STATES ===
if 'categories_list' in df.columns:
    with st.spinner("Calculating main category by state..."):
        total_by_state = df.groupby('state').size().reset_index(name='Total Restaurants')

        cat_exp = df.explode('categories_list').dropna(subset=['categories_list'])
        top_category = (
            cat_exp.groupby('state')['categories_list']
            .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else "No Category")
            .reset_index(name='Category')
        )
        map_df = pd.merge(total_by_state, top_category, on='state', how='inner')

        top5_states = total_by_state.sort_values(by='Total Restaurants', ascending=False).head(5)

        # Top 5 states chart
        fig_top5 = px.bar(
            top5_states,
            x='Total Restaurants',
            y='state',
            orientation='h',
            labels={'state': 'State', 'Total Restaurants': 'Total Restaurants'},
            title='Top 5 states with more restaurants',
            color='state',
            color_discrete_sequence=px.colors.sequential.YlGn[2:]
        )
        fig_top5.update_layout(
            height=600,
            plot_bgcolor='#ffffff',
            paper_bgcolor='#ffffff',
            font=dict(color='#556b2f'), showlegend=False,
            margin=dict(l=10, r=10, t=50, b=10)
        )

        # Map by category
        oliva_palette = ['#6b8e23', '#8fbc8f', '#556b2f', '#b2d3c2', '#cddbb0', '#e6edd9', '#78866b']
        fig_map = px.choropleth(
            map_df,
            locations='state',
            locationmode="USA-states",
            color='Category',
            hover_name='Category',
            hover_data={'Total Restaurants': True, 'state': False},
            scope="usa",
            title="Most frequent category by state and total number of restaurants",
            color_discrete_sequence=oliva_palette
        )
        fig_map.update_layout(
            height=600,
            plot_bgcolor='#ffffff',
            paper_bgcolor='#ffffff',
            font=dict(color='#556b2f'), showlegend=True,
            margin=dict(l=10, r=10, t=50, b=10)
        )

        # Show charts
        col_map_l, col_map_r = st.columns([1.5, 3])
        with col_map_l:
            st.plotly_chart(fig_top5, use_container_width=True)
        with col_map_r:
            st.plotly_chart(fig_map, use_container_width=True)


# === PRELOADED WORDCLOUDS ===
st.subheader(f"Word clouds for restaurants with {stars_slider} stars")
col1, col2 = st.columns(2)

with col1:
    st.markdown("##### Category Cloud (based on rating)")
    img_cat = f"img/wordcloud_categorias_{stars_slider}.png"
    with st.spinner("Loading category cloud..."):
        if os.path.exists(img_cat):
            st.image(Image.open(img_cat))
            with open(img_cat, "rb") as f:
                st.download_button("Download Category WordCloud", f, file_name=os.path.basename(img_cat))
        else:
            st.warning("Category image not found.")

with col2:
    st.markdown("##### Review Word Cloud (based on rating)")
    img_res = f"img/wordcloud_resenas_{stars_slider}.png"
    with st.spinner("Loading review cloud..."):
        if os.path.exists(img_res):
            st.image(Image.open(img_res))
            with open(img_res, "rb") as f:
                st.download_button("Download Review WordCloud", f, file_name=os.path.basename(img_res))
        else:
            st.warning("Review image not found.")


# === SUMMARY METRICS ===
total_businesses = len(filtered_df)
avg_stars = round(filtered_df['stars'].mean(), 2) if not filtered_df.empty else "N/A"
top_city = filtered_df['city'].value_counts().idxmax() if not filtered_df.empty else "N/A"
cat_exp_state = filtered_df.explode('categories_list').dropna(subset=['categories_list'])
top_category = cat_exp_state['categories_list'].mode().iloc[0] if not cat_exp_state.empty else "N/A"

st.subheader(f"General summary for {selected_state}")
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Total Businesses", f"{total_businesses:,}")
kpi2.metric("Average Stars", f"{avg_stars}")
kpi3.metric("City with Most Businesses", top_city)
kpi4.metric("Most Common Category", top_category)


# === BUSINESSES BY CITY AND STARS ===
st.subheader(f"Businesses by city in {selected_state}")
with st.spinner("Generating chart..."):
    if not filtered_df.empty:
        cities = filtered_df['city'].value_counts().head(20)
        fig_cities = px.bar(
            x=cities.index,
            y=cities.values,
            labels={'x': 'City', 'y': 'Number of Businesses'},
            color=cities.index,
            color_discrete_sequence=['#203500', '#283e06', '#31470b', '#3b5110', '#445a14', '#586e26', '#778c43', '#96ac60', '#b7cd7f']
        )
        fig_cities.update_layout(showlegend=False)
        st.plotly_chart(fig_cities, use_container_width=True)

        # CSV download button
        csv = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download filtered data", csv, "filtered_data.csv", "text/csv")

        # Stars distribution
        st.subheader(f"Stars distribution in {selected_state}")
        stars_state = filtered_df['stars'].value_counts().sort_index()
        fig_stars = px.bar(
            x=stars_state.index,
            y=stars_state.values,
            labels={'x': 'Stars', 'y': 'Number of Businesses'},
            title=f"Businesses by stars in {selected_state}",
            color=stars_state.index.astype(str),
            color_discrete_sequence=['#203500', '#283e06', '#31470b', '#3b5110', '#445a14', '#586e26', '#778c43', '#96ac60', '#b7cd7f']
        )
        fig_stars.update_layout(
            plot_bgcolor='#ffffff',
            paper_bgcolor='#ffffff',
            font=dict(color='#556b2f'), showlegend=False,
            margin=dict(l=40, r=40, t=50, b=30)
        )
        st.plotly_chart(fig_stars, use_container_width=True)

        # Top categories in the state
        st.subheader(f"Top 10 categories in {selected_state}")
        cat_exp_state = filtered_df.explode('categories_list').dropna(subset=['categories_list'])
        top_categories_state = cat_exp_state['categories_list'].value_counts().head(10)

        df_top_cat = top_categories_state.reset_index()
        df_top_cat.columns = ['Category', 'Count']

        fig_top_cat = px.bar(
            data_frame=df_top_cat,
            x='Count',
            y='Category',
            orientation='h',
            labels={'Count': 'Number of Businesses', 'Category': 'Category'},
            title=f"Most frequent categories in {selected_state}",
            color_discrete_sequence=['#6b8e23']
        )

        fig_top_cat.update_layout(
            height=500,
            plot_bgcolor='#ffffff',
            paper_bgcolor='#ffffff',
            font=dict(color='#556b2f'), showlegend=False
        )
        st.plotly_chart(fig_top_cat, use_container_width=True)

    else:
        st.warning("No data available for the current filters.")


# === ATTRIBUTES BY STATE ===
st.subheader(f"Most common attributes in {selected_state}")

# Filter by state
atrib_df = df.copy() if selected_state == "All" else df[df['state'] == selected_state]

# Select only attribute columns
at_cols = attributes.columns.tolist()
at_bin = atrib_df[at_cols].applymap(lambda x: 1 if str(x).strip().lower() in ['true', 'yes', '1', 'y'] else 0)
attributes_sum = at_bin.sum().sort_values(ascending=False).head(15)

# Attributes chart
fig_attributes = px.bar(
    x=attributes_sum.values,
    y=attributes_sum.index,
    orientation='h',
    labels={'x': 'Number of Businesses', 'y': 'Attribute'},
    title=f"Most frequent attributes in {selected_state}",
    color=attributes_sum.index,
    color_discrete_sequence=['#203500', '#283e06', '#31470b', '#3b5110', '#445a14', '#586e26', '#778c43', '#96ac60', '#b7cd7f']
)
fig_attributes.update_layout(
    height=500,
    plot_bgcolor='#ffffff',
    paper_bgcolor='#ffffff',
    font=dict(color='#556b2f'),
    showlegend=False
)
st.plotly_chart(fig_attributes, use_container_width=True)


# === FOOTER ===
st.markdown("---")
st.markdown("If you want to know more about the dataset, check the full EDA notebook [here](link-to-your-notebook).")



