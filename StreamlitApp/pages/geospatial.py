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

st.set_page_config(page_title='Golf Course Simulation', page_icon='⛳')
st.markdown('# Golf Course Simulation')

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
                'green': (39.32237917682804, -76.58753409610676),
                'fairway': [
                    (39.3202193906142, -76.57967450138088),
                    (39.3212193906142, -76.58067450138088),
                    (39.3222193906142, -76.58567450138088),
                    (39.32318903778765, -76.58700524418536),
                    (39.3222193906142, -76.58567450138088),
                    (39.3212193906142, -76.58067450138088),
                    (39.3202193906142, -76.57967450138088)
                ],
                'hazards': [
                    {
                        'type': 'bunker',
                        'coordinates': [
                            (39.32218903778765, -76.58600524418536),
                            (39.32228903778765, -76.58600524418536),
                            (39.32228903778765, -76.58610524418536),
                            (39.32218903778765, -76.58610524418536)
                        ]
                    },
                    {
                        'type': 'water',
                        'coordinates': [
                            (39.32118903778765, -76.58300524418536),
                            (39.32138903778765, -76.58300524418536),
                            (39.32138903778765, -76.58320524418536),
                            (39.32118903778765, -76.58320524418536)
                        ]
                    }
                ]
            }
        }
    }
}


user_shots = get_user_shots(st.session_state.user_id)
if not user_shots:
    st.warning('No shot data available. Please log some shots first.')
    st.stop()

if len(user_shots) > 0:
    shots_df = pd.DataFrame(user_shots, columns=[
        'id', 'user_id', 'Shot Type', 'Carry (yards)', 'Club Speed (MPH)',
        'Ball Speed (MPH)', 'Launch Angle (Deg)', 'Spin Rate (RPM)',
        'Face Angle (Deg)', 'Face to Path (Deg)', 'Club Path (Deg)',
        'Attack Angle (Deg)', 'Launch Direction (Deg)', 'timestamp'
    ])
else:
    shots_df = pd.DataFrame()

numeric_columns = [
    'Carry (yards)', 'Club Speed (MPH)', 'Ball Speed (MPH)',
    'Launch Angle (Deg)', 'Spin Rate (RPM)', 'Face Angle (Deg)',
    'Face to Path (Deg)', 'Club Path (Deg)', 'Attack Angle (Deg)',
    'Launch Direction (Deg)'
]

for col in numeric_columns:
    shots_df[col] = pd.to_numeric(shots_df[col], errors='coerce')

le_shot = LabelEncoder()
shots_df["shot_type_encoded"] = le_shot.fit_transform(shots_df['Shot Type'])
X = shots_df[numeric_columns]
y = shots_df["shot_type_encoded"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

def predict_next_shot(distance, lat, lon, shot_number):
    if shot_number == 1:  # assume first will always be a drive
        return 'Drive'
    elif distance > 50:  
        return 'Iron Shot' if distance > 100 else 'Approach'
    elif distance > 20:  
        return 'Chip' # change to chip/pitch?
    else:  
        return 'Putt'

def calculate_shot_endpoint(start_lat, start_lon, distance, angle):
    distance_deg = distance * 0.00001
    end_lat = start_lat + (distance_deg * np.sin(np.radians(angle)))
    end_lon = start_lon + (distance_deg * np.cos(np.radians(angle)))
    return end_lat, end_lon

def simulate_round(hole_data):
    shots = []
    current_lat, current_lon = hole_data["tee"]
    green_lat, green_lon = hole_data["green"]
    
    shot_number = 1
    
    while True:
        distance = np.sqrt((current_lat - green_lat)**2 + (current_lon - green_lon)**2) * 111000  # meters
        distance_yards = distance * 1.09361 # yards
        
        if distance_yards < 2:  # 2 yds tolerance for holing out
            break
            
        shot_type = predict_next_shot(distance_yards, current_lat, current_lon, shot_number)
        shot_type_stats = shots_df[shots_df['Shot Type'] == shot_type]['Carry (yards)'].mean()
        carry_distance = float(shot_type_stats)
        
        if shot_type == 'Putt':
            carry_distance = min(distance_yards, 20)  # assume putts don't go past the hole (lol)
        elif shot_type == 'Chip':
            carry_distance = min(distance_yards, 50)
        
        angle = np.arctan2(green_lat - current_lat, green_lon - current_lon)
        next_lat, next_lon = calculate_shot_endpoint(current_lat, current_lon, carry_distance, np.degrees(angle))
        
        shots.append({
            'lat': current_lat,
            'lon': current_lon,
            'lat': next_lat,
            'lon2': next_lon,
            'shot_type': shot_type,
            'shot_number': shot_number,
            'distance_to_hole': round(distance_yards, 1),
            'carry': round(carry_distance, 1)
        })
        
        current_lat, current_lon = next_lat, next_lon
        shot_number += 1
    
    return pd.DataFrame(shots)

# CAN CHANGE; CLIFTON HOLE 1 RN
hole_data = course_data['Clifton Park']['holes'][1]

all_simulations = []
for _ in range(num_simulations):
    simulation = simulate_round(hole_data)
    all_simulations.append(simulation)

# VISUALIZATION BELOW
layers = [
    # fairway
    pdk.Layer(
        'PolygonLayer',
        data=[{
            'coordinates': hole_data['fairway'],
            'color': [152, 251, 152, 120]
        }],
        get_polygon='coordinates',
        get_fill_color='color',
        get_line_color=[0, 0, 0],
        line_width_min_pixels=2,
    ),

    # green
    pdk.Layer(
        'PolygonLayer',
        data=[{
            'coordinates': [
                (hole_data['green'][0] - 0.00005, hole_data['green'][1] - 0.00005),
                (hole_data['green'][0] + 0.00005, hole_data['green'][1] - 0.00005),
                (hole_data['green'][0] + 0.00005, hole_data['green'][1] + 0.00005),
                (hole_data['green'][0] - 0.00005, hole_data['green'][1] + 0.00005)
            ],
            'color': [0, 100, 0, 120]
        }],
        get_polygon='coordinates"',
        get_fill_color='color',
        get_line_color=[0, 0, 0],
        line_width_min_pixels=2,
    )
]

# hazards
if show_hazards:
    for hazard in hole_data['hazards']:
        layers.append(
            pdk.Layer(
                'PolygonLayer',
                data=[{
                    'coordinates': hazard['coordinates'],
                    'color': [255, 255, 0, 120] if hazard['type'] == 'bunker' else [0, 0, 255, 100]
                }],
                get_polygon='coordinates',
                get_fill_color='color',
                get_line_color=[0, 0, 0],
                line_width_min_pixels=2,
            )
        )

shot_colors = {
    "Drive": [255, 0, 0, 160],      #red
    "Iron Shot": [255, 165, 0, 160],  # orange
    "Approach": [255, 255, 0, 160],   # yellow
    "Chip": [0, 255, 0, 160],        # green
    "Putt": [0, 0, 255, 160]         # blue
}

for i, simulation in enumerate(all_simulations):
    simulation['color'] = simulation['shot_type'].map(shot_colors)
    
    layers.extend([
        pdk.Layer(
            'ArcLayer',
            data=simulation,
            get_source_position=['lon', 'lat'],
            get_target_position=['lon2', 'lat2'],
            get_source_color='color',
            get_target_color='color',
            auto_highlight=True,
            width_scale=0.0001,
            get_width='distance / 50',
            width_min_pixels=2,
            width_max_pixels=5,
        ),
        pdk.Layer(
            'ScatterplotLayer',
            data=simulation,
            get_position=['lon2', 'lat2'],
            get_color='color',
            get_radius=5,
        ),
        pdk.Layer(
            'TextLayer',
            data=simulation,
            get_position=['lon2', 'lat2'],
            get_text='shot_type',
            get_color=[255, 255, 255, 200],  
            get_size=12,
            get_alignment_baseline='bottom',
        )
    ])

st.pydeck_chart(
    pdk.Deck(
        map_style='mapbox://styles/mapbox/satellite-streets-v11',
        initial_view_state={
            'latitude': hole_data["tee"][0],
            'longitude': hole_data["tee"][1],
            'zoom': 18,
            'pitch': 60,
        },
    )
)

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
    plt.legend()
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
