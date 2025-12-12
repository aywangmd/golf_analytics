import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from auth import get_user_shots
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely import wkt
from shapely.affinity import translate, rotate

st.set_page_config(page_title='Golf Course Visualization', page_icon='⛳')
st.markdown('# Golf Course Visualization')

if 'user_id' not in st.session_state or not st.session_state.user_id:
    st.warning('Please login to access this page.')
    st.stop()

# # course selection....will uncomment if we make QGIS for the other courses
# course = st.selectbox('Select a Golf Course:', ('Clifton Park', 'Elkridge', 'Pine Ridge'),index=0,)

hole_number = st.number_input('Select Hole Number:', min_value=1, max_value=18, step=1)
num_simulations = 1
show_hazards = st.checkbox('Show Hazards', value=True)

# course coordinates DO NOT TOUCH!!!
course_data = {
    'Clifton Park': {
        'center': (39.32318903778765, -76.58700524418536),
        'holes': {
            1: {
                'tee': (39.32592099564656, -76.5843713165854),
            },
            2: {
                'tee': (39.32592099564656, -76.5843713165854),
            }
        }
    }
}

# doesn't work when deployed
# st.pydeck_chart(
#     pdk.Deck(
#         map_style='mapbox://styles/mapbox/satellite-streets-v11',
#         initial_view_state={
#             'latitude': course_data['Clifton Park']['holes'][hole_number]['tee'][0],
#             'longitude': course_data['Clifton Park']['holes'][hole_number]['tee'][1],
#             'zoom': 18,
#             'pitch': 60,
#         },
#     )
# )

def geoplot_single(data, name):
    if 'geometry' not in data.columns:
        data['geometry'] = data['WKT'].apply(wkt.loads)
    
    gdf = gpd.GeoDataFrame(data, geometry='geometry', crs="EPSG:4326")

    fig, ax = plt.subplots(figsize=(8, 8))
    gdf.plot(ax=ax, color='green', edgecolor='black', alpha=0.5)

    for idx, row in gdf.iterrows():
        centroid = row.geometry.centroid
        ax.text(centroid.x, centroid.y, str(idx), fontsize=9, ha='center', color='black')

    ax.set_title(name)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    st.pyplot(fig)

greens = gpd.read_file('StreamlitApp/pages/data/greens.csv')
# geoplot_single(greens, 'Greens')
greens['hole'] = [1, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 2, 3, 5, 4, 7, 6]
greens = greens[['hole'] + [col for col in greens.columns if col != 'hole']]

fairways = gpd.read_file('StreamlitApp/pages/data/fairways.csv')
# geoplot_single(fairways, 'Fairways')
fairways['hole'] = [1, 3, 2, 5, 4, 6, 8, 9, 11, 12, 13, 16, 17, 18, 18, 2]
fairways = fairways[['hole'] + [col for col in fairways.columns if col != 'hole']]

bunkers = gpd.read_file('StreamlitApp/pages/data/bunkers.csv')
# geoplot_single(bunkers, 'Bunkers')
bunkers['hole'] = [1, 1, 1, 2, 8, 8, 9, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 13, 14, 14, 14, 15, 15, 16, 16, 16, 16, 17, 17, 17, 17, 17, 18, 18, 18, 2, 3, 3, 5, 5, 5, 5, 5, 4, 7, 6, 6, 6, 6]
bunkers = bunkers[['hole'] + [col for col in bunkers.columns if col != 'hole']]

tees = gpd.read_file("StreamlitApp/pages/data/tees.csv")
# geoplot_single(tees, 'Tees')
tees['hole'] = [1, 3, 3, 2, 5, 5, 4, 4, 7, 7, 6, 8, 9, 10, 11, 11, 11, 12, 13, 13, 14, 14, 15, 15, 16, 17, 18, 18]
tees = tees[['hole'] + [col for col in tees.columns if col != 'hole']]

# plot hole all tgt
def geoplot_hole_latlon(hole_number):
    fig, ax = plt.subplots(figsize=(8, 8))
    
    hole_greens = greens[greens['hole'] == hole_number]
    hole_fairways = fairways[fairways['hole'] == hole_number]
    hole_bunkers = bunkers[bunkers['hole'] == hole_number]
    hole_tees = tees[tees['hole'] == hole_number]
    
    if not hole_fairways.empty:
        hole_fairways.plot(ax=ax, color='palegreen', edgecolor='black', alpha=0.5, label='Fairways')
    if not hole_greens.empty:
        hole_greens.plot(ax=ax, color='darkgreen', edgecolor='black', alpha=0.7, label='Greens')
    if not hole_bunkers.empty:
        hole_bunkers.plot(ax=ax, color='yellow', edgecolor='black', alpha=0.5, label='Bunkers')
    if not hole_tees.empty:
        hole_tees.plot(ax=ax, color='blue', edgecolor='black', alpha=0.5, label='Tees')
    
    ax.set_title(f'Hole {hole_number}')
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    st.pyplot(fig)
    
geoplot_hole_latlon(hole_number)

def geoplot_hole(hole_number):
    fig, ax = plt.subplots(figsize=(8, 8))
    
    hole_greens = greens[greens['hole'] == hole_number].copy()
    hole_fairways = fairways[fairways['hole'] == hole_number].copy()
    hole_bunkers = bunkers[bunkers['hole'] == hole_number].copy()
    hole_tees = tees[tees['hole'] == hole_number].copy()
    
    tee_centroid = hole_tees.geometry.union_all().centroid
    
    # center
    hole_tees['geometry'] = hole_tees.translate(xoff=-tee_centroid.x, yoff=-tee_centroid.y)
    hole_fairways['geometry'] = hole_fairways.translate(xoff=-tee_centroid.x, yoff=-tee_centroid.y)
    hole_bunkers['geometry'] = hole_bunkers.translate(xoff=-tee_centroid.x, yoff=-tee_centroid.y)
    hole_greens['geometry'] = hole_greens.translate(xoff=-tee_centroid.x, yoff=-tee_centroid.y)
    
    # rotate
    green_centroid = hole_greens.geometry.union_all().centroid
    dx = green_centroid.x
    dy = green_centroid.y
    angle = np.degrees(np.arctan2(dx, dy))  # green is "up"
    
    hole_tees['geometry'] = hole_tees.rotate(angle, origin=(0,0))
    hole_fairways['geometry'] = hole_fairways.rotate(angle, origin=(0,0))
    hole_bunkers['geometry'] = hole_bunkers.rotate(angle, origin=(0,0))
    hole_greens['geometry'] = hole_greens.rotate(angle, origin=(0,0))
    
    latitude_to_yards = 69 * 1760  # approx yards per degree latitude
    longitude_to_yards = 69 * 1760 * np.cos(np.radians(tee_centroid.y))  # approx yards per degree longitude at given latitude
    hole_tees['geometry'] = hole_tees.scale(xfact=longitude_to_yards, yfact=latitude_to_yards, origin=(0,0))
    hole_fairways['geometry'] = hole_fairways.scale(xfact=longitude_to_yards, yfact=latitude_to_yards, origin=(0,0))
    hole_bunkers['geometry'] = hole_bunkers.scale(xfact=longitude_to_yards, yfact=latitude_to_yards, origin=(0,0))
    hole_greens['geometry'] = hole_greens.scale(xfact=longitude_to_yards, yfact=latitude_to_yards, origin=(0,0))
    
    if not hole_fairways.empty:
        hole_fairways.plot(ax=ax, color='palegreen', edgecolor='black', alpha=0.5, label='Fairways')
    if not hole_greens.empty:
        hole_greens.plot(ax=ax, color='darkgreen', edgecolor='black', alpha=0.7, label='Greens')
    if not hole_bunkers.empty:
        hole_bunkers.plot(ax=ax, color='yellow', edgecolor='black', alpha=0.5, label='Bunkers')
    if not hole_tees.empty:
        hole_tees.plot(ax=ax, color='blue', edgecolor='black', alpha=0.5, label='Tees')
    
    ax.set_title(f'Hole {hole_number}')
    plt.xlabel("Yards Left/Right of Line of Play")
    plt.ylabel("Yards from Tee")
    plt.axis('equal')
    st.pyplot(fig)

geoplot_hole(hole_number)
