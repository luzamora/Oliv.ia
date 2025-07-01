import streamlit as st
import pandas as pd
from PIL import Image
from Olivia import Olivia
from streamlit_folium import st_folium
import folium
from folium.plugins import MarkerCluster
import warnings
import os
import json
import numpy as np

warnings.filterwarnings('ignore')

st.set_page_config(page_title="Oliv.ia", page_icon="img/logomini.png", layout="wide")

# Función principal de la app (todo dentro de esta función)
def Recommendator():
    # CSS para el fondo de la barra lateral
    st.markdown("""
        <style>
            [data-testid="stSidebar"] {
                background-color: #e0f8bc !important;
            }
        </style>
    """, unsafe_allow_html=True)

    @st.cache_resource()
    def load_all_data():
        usecols_business = ['business_id', 'name', 'state', 'latitude', 'longitude', 'categories', 'stars', 'review_count', 'address', 'city', 'postal_code','attributes_text']
        business_data = pd.read_parquet("datasets/yelp_business.parquet", columns=usecols_business)
        review_data = pd.read_parquet("datasets/yelp_reviews.parquet")
        attributes = pd.read_parquet("datasets/attributes_multihot.parquet")
        categories = pd.read_parquet("datasets/categories_multihot.parquet")
        review_emb = pd.read_parquet("datasets/restaurant_reviews_pooled_embeddings.parquet").to_numpy()
        feature_emb = pd.read_parquet("datasets/features_embeddings.parquet").to_numpy()
        return business_data, review_data, attributes, categories, review_emb, feature_emb

    business_data, review_data, attributes, categories, review_emb, feature_emb = load_all_data()

    @st.cache_data()
    def load_photos():
        return pd.read_parquet("datasets/yelp_photos.parquet")

    df_photos = load_photos()

    cache_path = "summary_cache.json"
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            resumen_dict = json.load(f)
    else:
        resumen_dict = {}

    @st.cache_resource()
    def load_full_model():
        modelo = Olivia(
            business_data=business_data,
            review_data=review_data,
            attributes=attributes,
            categories=categories,
            summary_dict=resumen_dict
        )
        modelo.load_features_embeddings(feature_emb)
        modelo.load_review_embeddings(review_emb)
        return modelo

    modelo = load_full_model()

    @st.cache_data(show_spinner="Generating recommendations...")
    def obtener_recomendaciones_cache(query, estado, resumen_model):
        modelo.reccomend(
            input_query=query,
            target_state=estado,
            summarizer=resumen_model
        )
        resumenes = modelo.get_summaries()

        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(resumenes, f, indent=2, ensure_ascii=False)

        return modelo.get_reccomendations(), resumenes

    # Estado de sesión
    if "ubicaciones" not in st.session_state:
        st.session_state["ubicaciones"] = pd.DataFrame([], columns=business_data.columns)
    if "ver_detalles_de" not in st.session_state:
        st.session_state["ver_detalles_de"] = None
    if "consulta_anterior" not in st.session_state:
        st.session_state["consulta_anterior"] = None
    if "estado_anterior" not in st.session_state:
        st.session_state["estado_anterior"] = None

    # sidebar
    with st.sidebar:
        logo = Image.open("img/logo.png")
        st.image(logo, width=300)
        st.subheader("Find your ideal restaurant")

        estados = sorted(business_data['state'].dropna().unique().tolist())
        estados.insert(0, "All")
        estado_seleccionado = st.selectbox("Where are you?", estados)

        consulta = st.text_input("What are you looking for?")

        col1, col2 = st.columns([2.5, 1.8])
        with col1:
            buscar = st.button("Search")
        with col2:
            volver = st.button("Back")

        st.subheader("Summary settings")
        modelo_summarizer = st.selectbox(
            "Select the summary model:",
            ["gemini", "bart", "sumy"],
            index=0,
            key="modelo_summarizer"
        )

        st.markdown("---")

        if volver:
            if st.session_state.get("ver_detalles_de") is not None:
                st.session_state["ver_detalles_de"] = None
            else:
                st.session_state["ubicaciones"] = pd.DataFrame([], columns=business_data.columns)
                st.session_state["consulta_anterior"] = None
                st.session_state["estado_anterior"] = None

    # --- Buscar después del selector ---
    if buscar:
        if (
            consulta != st.session_state.get("consulta_anterior") or
            estado_seleccionado != st.session_state.get("estado_anterior")
        ):
            with st.spinner("Generating recommendations..."):
                if estado_seleccionado != "All" and business_data[business_data["state"] == estado_seleccionado].empty:
                    st.warning(f"No hay restaurantes en el estado '{estado_seleccionado}'.")
                else:
                    modelo.reccomend(consulta, target_state=estado_seleccionado, summarizer=modelo_summarizer)
                    df_recomendaciones = modelo.get_reccomendations()
                    with open(cache_path, "w", encoding="utf-8") as f:
                        json.dump(modelo.get_summaries(), f, indent=2, ensure_ascii=False)

                    if 'categories' in df_recomendaciones.columns:
                        df_recomendaciones['categories'] = df_recomendaciones['categories'].astype(str).apply(
                            lambda x: x.strip("[]").replace("'", "").replace(" ", ", ")
                        )


                    st.session_state["ubicaciones"] = df_recomendaciones.reset_index(drop=True)
                    st.session_state["ver_detalles_de"] = None
                    st.session_state["consulta_anterior"] = consulta
                    st.session_state["estado_anterior"] = estado_seleccionado

                    st.success("showing recommendations...")
        else:
            st.info("You are already seeing the results of that search in that state.")

    # --- MAPA ---
    if not st.session_state["ubicaciones"].empty:
        df_mapa = st.session_state["ubicaciones"].copy()
    else:
        if estado_seleccionado != "All":
            df_estado = business_data[business_data["state"] == estado_seleccionado]
            if not df_estado.empty:
                df_mapa = df_estado.sample(n=1, random_state=42).copy()
            else:
                df_mapa = pd.DataFrame(columns=business_data.columns)
        else:
            df_mapa = pd.DataFrame(columns=business_data.columns)

    df_mapa["latitude"] = pd.to_numeric(df_mapa["latitude"], errors="coerce")
    df_mapa["longitude"] = pd.to_numeric(df_mapa["longitude"], errors="coerce")
    df_mapa = df_mapa.dropna(subset=["latitude", "longitude"])

    if df_mapa.empty:
        centro_lat = 39.50
        centro_lon = -98.35
        zoom = 4
    else:
        if st.session_state["ver_detalles_de"]:
            seleccionado = df_mapa[df_mapa["name"] == st.session_state["ver_detalles_de"]]
            if not seleccionado.empty:
                centro_lat = seleccionado.iloc[0]["latitude"]
                centro_lon = seleccionado.iloc[0]["longitude"]
                zoom = 14
            else:
                centro_lat = df_mapa["latitude"].mean()
                centro_lon = df_mapa["longitude"].mean()
                zoom = 6 if estado_seleccionado != "All" else 4
        else:
            centro_lat = df_mapa["latitude"].mean()
            centro_lon = df_mapa["longitude"].mean()
            zoom = 6 if estado_seleccionado != "All" else 4

    m = folium.Map(
        location=[centro_lat, centro_lon],
        zoom_start=zoom
    )

    if not st.session_state["ubicaciones"].empty:
        marker_cluster = MarkerCluster().add_to(m)
        for _, row in st.session_state["ubicaciones"].iterrows():
            popup_text = f"""
            <b>{row['name']}</b><br>
            Address: {row['address']}, {row['city']}, {row['state']}<br>
            Categories: {row['categories']}<br>
            Rated: {row['stars']}
            """
            folium.Marker(
                location=[row["latitude"], row["longitude"]],
                popup=folium.Popup(popup_text, max_width=300),
                tooltip=row["name"], 
                icon=folium.Icon(color='green', icon='cutlery', prefix='fa')
            ).add_to(marker_cluster)

    st_folium(m, width=1400, height=450, returned_objects=[])

    # --- DETALLES ---
    if st.session_state["ver_detalles_de"]:
        selected = st.session_state["ubicaciones"][st.session_state["ubicaciones"]["name"] == st.session_state["ver_detalles_de"]]

        if not selected.empty:
            row = selected.iloc[0]

            with st.container():
                col_text, col_img = st.columns([3, 1])

                with col_text:
                    st.markdown(f"## About *{row.get('name', 'Restaurante')}*")
                    st.markdown(f"### {row.get('name', 'Not available')}")
                    st.markdown(f"**Address**: {row.get('address', '')}, {row.get('city', '')}, {row.get('state', '')} {row.get('postal_code', '')}")
                    st.markdown(f"**Rated**: {row.get('stars', 'Undefined')}")
                    st.markdown(f"**Categories**: {row.get('categories', 'Undefined')}")
                    st.markdown(f"**Amenities**: {row.get('attributes_text', 'Undefined')}")
                    if 'review_summary' in row and pd.notna(row['review_summary']):
                        st.markdown("#### Olivia thinks that:")
                        st.markdown(f"_{row['review_summary']}_")

                with col_img:
                    try:
                        photo_id = modelo.get_random_photo(row['business_id'], df_photos)
                        if photo_id is not None:
                            photo_code = photo_id.values[0]
                            photo_path = f"yelp-photos/{photo_code}.jpg"
                            if os.path.exists(photo_path):
                                st.image(photo_path, width=300)
                            else:
                                st.image("img/no_image_available.jpg", width=300)
                        else:
                            st.image("img/no_image_available.jpg", width=300)
                    except Exception:
                        st.image("img/no_image_available.jpg", width=300)

                st.divider()
        else:
            st.warning(f"The restaurant  '{st.session_state['ver_detalles_de']}' is not in the current results.")
            st.session_state["ver_detalles_de"] = None



    # --- TOP 5 RECOMENDADOS (formato fila horizontal) ---
    if not st.session_state["ubicaciones"].empty and st.session_state["ver_detalles_de"] is None:
        top5 = st.session_state["ubicaciones"].head(5)

        st.markdown("## Top 5 Recommended Restaurants")
        st.divider()
        for i, row in top5.iterrows():
            cols = st.columns([1, 3, 1, 2, 3, 1])  # Una columna extra para el botón
            with cols[0]:
                try:
                    photo_id = modelo.get_random_photo(row['business_id'], df_photos)
                    if photo_id is not None:
                        photo_path = f"yelp-photos/{photo_id.values[0]}.jpg"
                        if os.path.exists(photo_path):
                            st.image(photo_path, width=75)
                        else:
                            st.image("img/no_image_available.jpg", width=75)
                    else:
                        st.image("img/no_image_available.jpg", width=75)
                except:
                    st.image("img/no_image_available.jpg", width=75)
            with cols[1]:
                st.write(row['name'])
            with cols[2]:
                st.markdown(f"<div style='font-size:28px; font-weight:bold;'>{row['stars']}</div>", unsafe_allow_html=True)
            with cols[3]:
                st.write(f"{row['city']}, {row['state']}")
            with cols[4]:
                cat = row['categories']
                if len(cat) > 60:
                    cat = cat[:60] + "..."
                st.write(cat)
            with cols[5]:
                if st.button("Details", key=f"ver_mas_linea_{row['business_id']}"):
                    st.session_state["ver_detalles_de"] = row["name"]
                    st.rerun()
            st.markdown("---")



# Definimos las páginas para la navegación: función + archivo
pages = [
    Recommendator, 
    "pages/EDA.py",
    "pages/Catalogue.py"
]

pg = st.navigation(pages, position="top")

pg.run()
