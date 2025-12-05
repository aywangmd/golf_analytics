import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import sqlite3
from auth import get_user_shots, init_db
import geopandas as gpd
from shapely.geometry import Point
from shapely.affinity import translate, rotate, scale
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Golf Simulation", page_icon="⛳", layout="wide")

# Initialize database
init_db()

# Initialize session state
if 'user_id' not in st.session_state:
    st.error("Please log in to access this page.")
    st.stop()

st.title("🏌️ Golf Round Simulation")
st.markdown("Simulate rounds using your personal shot distributions and actual course geometry")

# Load hole geometry data
@st.cache_data
def load_hole_geometry():
    """Load hole geometry data (greens, fairways, bunkers, tees)"""
    try:
        greens = gpd.read_file("StreamlitApp/pages/data/greens.csv")
        greens['hole'] = [1, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 2, 3, 5, 4, 7, 6]
        
        fairways = gpd.read_file("StreamlitApp/pages/data/fairways.csv")
        fairways['hole'] = [1, 3, 2, 5, 4, 6, 8, 9, 11, 12, 13, 16, 17, 18, 18, 2]
        
        bunkers = gpd.read_file("StreamlitApp/pages/data/bunkers.csv")
        bunkers['hole'] = [1, 1, 1, 2, 8, 8, 9, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 13, 14, 14, 14, 15, 15, 16, 16, 16, 16, 17, 17, 17, 17, 17, 18, 18, 18, 2, 3, 3, 5, 5, 5, 5, 5, 4, 7, 6, 6, 6, 6]
        
        tees = gpd.read_file("StreamlitApp/pages/data/tees.csv")
        tees['hole'] = [1, 3, 3, 2, 5, 5, 4, 4, 7, 7, 6, 8, 9, 10, 11, 11, 11, 12, 13, 13, 14, 14, 15, 15, 16, 17, 18, 18]
        
        return greens, fairways, bunkers, tees
    except Exception as e:
        st.warning(f"Could not load hole geometry data: {e}")
        return None, None, None, None

# Load geometry data
greens, fairways, bunkers, tees = load_hole_geometry()

# Load user shot data
def load_user_shots(user_id):
    """Load and process user shot data"""
    shots = get_user_shots(user_id)
    if not shots:
        return pd.DataFrame()
    
    # Load shots into DataFrame
    if len(shots) > 0:
        df = pd.DataFrame(shots, columns=[
            'id', 'user_id', 'shot_type', 'carry', 'club_speed',
            'ball_speed', 'launch_angle', 'spin_rate', 'face_angle',
            'face_to_path', 'club_path', 'attack_angle', 'launch_direction', 'timestamp'
        ])
    else:
        df = pd.DataFrame()
    
    # Rename columns to match expected format
    df = df.rename(columns={
        'shot_type': 'Shot Type',
        'carry': 'Carry (yards)',
        'club_speed': 'Club Speed (MPH)',
        'ball_speed': 'Ball Speed (MPH)',
        'launch_angle': 'Launch Angle (Deg)',
        'spin_rate': 'Spin Rate (RPM)',
        'face_angle': 'Face Angle (Deg)',
        'face_to_path': 'Face to Path (Deg)',
        'club_path': 'Club Path (Deg)',
        'attack_angle': 'Attack Angle (Deg)',
        'launch_direction': 'Launch Direction (Deg)'
    })
    
    # Convert numeric columns
    numeric_columns = [
        'Carry (yards)', 'Club Speed (MPH)', 'Ball Speed (MPH)',
        'Launch Angle (Deg)', 'Spin Rate (RPM)', 'Face Angle (Deg)',
        'Face to Path (Deg)', 'Club Path (Deg)', 'Attack Angle (Deg)',
        'Launch Direction (Deg)'
    ]
    
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df.dropna(subset=numeric_columns)

# Load shot data
shots_df = load_user_shots(st.session_state.user_id)

# Add refresh button
if st.button("🔄 Refresh Shot Data"):
    st.rerun()

if shots_df.empty:
    st.error("No shot data found! Please add some shots first.")
    # st.write("**Debug Info:**")
    # st.write(f"- User ID: {st.session_state.user_id}")
    # st.write(f"- Raw shots from database: {len(get_user_shots(st.session_state.user_id))}")
    # st.write("- Make sure you have added shots through the Player Data page")
    
    # Show raw data for debugging
    raw_shots = get_user_shots(st.session_state.user_id)
    if raw_shots:
        st.write("**Raw shots from database:**")
        st.write(raw_shots)
    else:
        st.write("**No raw shots found in database**")
    
    st.stop()

st.success(f"Loaded {len(shots_df)} shots from your history!")

# Show shot type breakdown
st.write("**Shot Type Breakdown:**")
shot_type_counts = shots_df['Shot Type'].value_counts()
st.write(shot_type_counts)

# # Show sample data
# st.write("**Sample of your shots:**")
# st.write(shots_df[['Shot Type', 'Carry (yards)', 'Face to Path (Deg)', 'Launch Direction (Deg)']].head())

# Course setup sidebar
st.sidebar.header("🏌️ Course Setup")

# Always use actual course geometry when available
use_geometry = greens is not None

if greens is not None:
    # Get available holes from geometry data
    available_holes = sorted(set(greens['hole'].dropna().unique()) & 
                            set(fairways['hole'].dropna().unique()) &
                            set(tees['hole'].dropna().unique()))
    
    # Calculate hole distances from geometry
    def get_hole_distance(hole_num):
        """Calculate hole distance from tee to green centroid"""
        try:
            hole_tees = tees[tees['hole'] == hole_num]
            hole_greens = greens[greens['hole'] == hole_num]
            
            if not hole_tees.empty and not hole_greens.empty:
                tee_centroid = hole_tees.geometry.unary_union.centroid
                green_centroid = hole_greens.geometry.unary_union.centroid
                
                # Calculate distance in degrees
                lat_diff = green_centroid.y - tee_centroid.y
                lon_diff = green_centroid.x - tee_centroid.x
                
                # Convert to yards (approximate)
                lat_yards = lat_diff * 69 * 1760  # degrees to yards
                lon_yards = lon_diff * 69 * 1760 * np.cos(np.radians(tee_centroid.y))
                
                distance = np.sqrt(lat_yards**2 + lon_yards**2)
                return int(distance)
        except:
            pass
        return 400  # Default distance
    
    # Get distances for all 18 holes
    hole_distances = [get_hole_distance(i+1) for i in range(18)]
else:
    st.error("❌ Course geometry data not available. Please ensure geometry files are loaded.")
    st.stop()

# Course difficulty settings
st.sidebar.subheader("Course Difficulty")
rough_penalty = st.sidebar.slider("Rough Penalty (%)", 0, 50, 15)
bunker_penalty = st.sidebar.slider("Bunker Penalty (%)", 0, 50, 25)
wind_factor = st.sidebar.slider("Wind Factor", 0.0, 2.0, 1.0, 0.1)

# Simulation parameters
st.sidebar.subheader("Simulation Parameters")
n_simulations = st.sidebar.slider("Number of Rounds", min_value=100, max_value=10000, value=500, step=100)
show_details = st.sidebar.checkbox("Show Detailed Analysis", value=False)

class GolfSimulator:
    def __init__(self, shots_df, hole_distances, rough_penalty=15, bunker_penalty=25, wind_factor=1.0,
                 greens=None, fairways=None, bunkers=None, tees=None, use_geometry=False):
        self.shots_df = shots_df
        self.hole_distances = hole_distances
        self.rough_penalty = rough_penalty / 100
        self.bunker_penalty = bunker_penalty / 100
        self.wind_factor = wind_factor
        self.use_geometry = use_geometry
        
        # Store geometry data
        self.greens = greens
        self.fairways = fairways
        self.bunkers = bunkers
        self.tees = tees
        
        # Create shot type distributions
        self.shot_distributions = self._create_shot_distributions()
        
        # Define course states
        self.states = ["Tee", "Fairway", "Rough", "Bunker", "Green", "Hole"]
        
        # State distance thresholds (as percentages of hole distance) - fallback if no geometry
        self.state_thresholds = {
            "Tee": 1.0,      # Start at 100% of hole distance
            "Fairway": 0.7,   # Fairway ends at 70% of hole distance
            "Rough": 0.3,    # Rough ends at 30% of hole distance  
            "Bunker": 0.2,   # Bunker ends at 20% of hole distance
            "Green": 0.05,   # Green starts at 5% of hole distance
            "Hole": 0.0      # Hole at 0% of hole distance
        }
    
    def _get_hole_geometry(self, hole_num):
        """Get geometry for a specific hole"""
        if not self.use_geometry or self.greens is None:
            return None, None, None, None
        
        try:
            hole_greens = self.greens[self.greens['hole'] == hole_num]
            hole_fairways = self.fairways[self.fairways['hole'] == hole_num]
            hole_bunkers = self.bunkers[self.bunkers['hole'] == hole_num]
            hole_tees = self.tees[self.tees['hole'] == hole_num]
            
            return hole_greens, hole_fairways, hole_bunkers, hole_tees
        except:
            return None, None, None, None
    
    def _determine_state_from_geometry(self, lat, lon, hole_num):
        """Determine state based on actual geometry (point-in-polygon check)"""
        if not self.use_geometry:
            return None
        
        point = Point(lon, lat)  # Note: Point takes (x, y) = (lon, lat)
        
        hole_greens, hole_fairways, hole_bunkers, hole_tees = self._get_hole_geometry(hole_num)
        
        if hole_greens is None:
            return None
        
        # Check in order: Green, Bunker, Fairway, Tee (most specific first)
        if not hole_greens.empty:
            for _, green in hole_greens.iterrows():
                if green.geometry.contains(point) or green.geometry.buffer(0.00001).contains(point):
                    return "Green"
        
        if hole_bunkers is not None and not hole_bunkers.empty:
            for _, bunker in hole_bunkers.iterrows():
                if bunker.geometry.contains(point) or bunker.geometry.buffer(0.00001).contains(point):
                    return "Bunker"
        
        if not hole_fairways.empty:
            for _, fairway in hole_fairways.iterrows():
                if fairway.geometry.contains(point) or fairway.geometry.buffer(0.00001).contains(point):
                    return "Fairway"
        
        if not hole_tees.empty:
            for _, tee in hole_tees.iterrows():
                if tee.geometry.contains(point) or tee.geometry.buffer(0.00001).contains(point):
                    return "Tee"
        
        # If not in any defined area, assume rough
        return "Rough"
        
    def _create_shot_distributions(self):
        """Create probability distributions for each shot type based on player history"""
        distributions = {}
        
        for shot_type in self.shots_df['Shot Type'].unique():
            shot_data = self.shots_df[self.shots_df['Shot Type'] == shot_type]
            
            if len(shot_data) < 1:  # Lowered requirement to 1 shot
                continue
                
            # Create distributions for key metrics
            distributions[shot_type] = {
                'carry': {
                    'mean': shot_data['Carry (yards)'].mean(),
                    'std': shot_data['Carry (yards)'].std() if len(shot_data) > 1 else shot_data['Carry (yards)'].mean() * 0.1,  # Use 10% of mean as std if only 1 shot
                    'min': shot_data['Carry (yards)'].min(),
                    'max': shot_data['Carry (yards)'].max()
                },
                'accuracy': {
                    'face_to_path_mean': shot_data['Face to Path (Deg)'].mean(),
                    'face_to_path_std': shot_data['Face to Path (Deg)'].std() if len(shot_data) > 1 else 3.0,  # Default std if only 1 shot
                    'launch_direction_mean': shot_data['Launch Direction (Deg)'].mean(),
                    'launch_direction_std': shot_data['Launch Direction (Deg)'].std() if len(shot_data) > 1 else 2.0  # Default std if only 1 shot
                },
                'count': len(shot_data)
            }
        
        return distributions
    
    def _sample_shot_distance(self, shot_type, hole_distance):
        """Sample shot distance based on player's distribution"""
        if shot_type not in self.shot_distributions:
            # Fallback to generic distances
            generic_distances = {
                'Drive': (200, 300),
                'Iron Shot': (150, 200),
                'Approach': (100, 150),
                'Chip': (20, 50),
                'Putt': (5, 20)
            }
            mean_dist = np.mean(generic_distances.get(shot_type, (100, 150)))
            std_dist = (generic_distances.get(shot_type, (100, 150))[1] - generic_distances.get(shot_type, (100, 150))[0]) / 4
        else:
            dist_info = self.shot_distributions[shot_type]['carry']
            mean_dist = dist_info['mean']
            std_dist = dist_info['std']
        
        # Apply wind factor
        mean_dist *= self.wind_factor
        
        # Sample from normal distribution
        sampled_dist = np.random.normal(mean_dist, std_dist)
        
        # Ensure reasonable bounds
        min_dist = max(5, mean_dist * 0.3)  # At least 30% of mean
        max_dist = min(hole_distance * 1.2, mean_dist * 1.7)  # At most 120% of hole distance
        
        sampled_dist = np.clip(sampled_dist, min_dist, max_dist)
        
        return sampled_dist
    
    def _determine_shot_accuracy(self, shot_type, current_state):
        """Determine shot accuracy based on player's face-to-path and launch direction"""
        if shot_type not in self.shot_distributions:
            # Generic accuracy
            face_to_path_error = np.random.normal(0, 3)
            launch_direction_error = np.random.normal(0, 2)
        else:
            acc_info = self.shot_distributions[shot_type]['accuracy']
            face_to_path_error = np.random.normal(
                acc_info['face_to_path_mean'], 
                acc_info['face_to_path_std']
            )
            launch_direction_error = np.random.normal(
                acc_info['launch_direction_mean'], 
                acc_info['launch_direction_std']
            )
        
        # Adjust accuracy based on current state
        if current_state == "Rough":
            face_to_path_error *= (1 + self.rough_penalty)
            launch_direction_error *= (1 + self.rough_penalty)
        elif current_state == "Bunker":
            face_to_path_error *= (1 + self.bunker_penalty)
            launch_direction_error *= (1 + self.bunker_penalty)
        
        # Determine landing state based on accuracy
        total_error = abs(face_to_path_error) + abs(launch_direction_error)
        
        if total_error < 2:
            return "Fairway"
        elif total_error < 5:
            return "Rough"
        else:
            return "Bunker"
    
    def _determine_next_shot_type(self, distance_to_hole, current_state, shot_number):
        """Determine next shot type based on distance and current state"""
        if current_state == "Green":
            return "Putt"
        elif distance_to_hole < 20:
            return "Chip"
        elif distance_to_hole < 100:
            return "Approach"
        elif distance_to_hole < 200:
            return "Iron Shot"
        elif shot_number == 1:
            return "Drive"
        else:
            return "Iron Shot"
    
    def _sample_shot_features(self, shot_type):
        """Sample all relevant shot features from user's distributions"""
        if shot_type not in self.shot_distributions:
            # Fallback to generic values
            return {
                'face_to_path': np.random.normal(0, 3),
                'launch_direction': np.random.normal(0, 2),
                'launch_angle': np.random.normal(12, 3),
                'spin_rate': np.random.normal(3000, 500)
            }
        
        acc_info = self.shot_distributions[shot_type]['accuracy']
        
        # Sample from user's actual distributions
        face_to_path = np.random.normal(
            acc_info['face_to_path_mean'],
            acc_info['face_to_path_std']
        )
        launch_direction = np.random.normal(
            acc_info['launch_direction_mean'],
            acc_info['launch_direction_std']
        )
        
        # Get additional features if available
        shot_data = self.shots_df[self.shots_df['Shot Type'] == shot_type]
        
        launch_angle = shot_data['Launch Angle (Deg)'].mean() if len(shot_data) > 0 else 12
        launch_angle_std = shot_data['Launch Angle (Deg)'].std() if len(shot_data) > 1 else 3
        launch_angle = np.random.normal(launch_angle, launch_angle_std)
        
        spin_rate = shot_data['Spin Rate (RPM)'].mean() if len(shot_data) > 0 else 3000
        spin_rate_std = shot_data['Spin Rate (RPM)'].std() if len(shot_data) > 1 else 500
        spin_rate = np.random.normal(spin_rate, spin_rate_std)
        
        return {
            'face_to_path': face_to_path,
            'launch_direction': launch_direction,
            'launch_angle': launch_angle,
            'spin_rate': spin_rate
        }
    
    def _calculate_landing_position(self, start_lat, start_lon, target_lat, target_lon, shot_distance, 
                                   face_to_path, launch_direction, current_state):
        """Calculate landing position based on shot distance, face-to-path, and launch direction"""
        # Calculate direction to target
        lat_diff = target_lat - start_lat
        lon_diff = target_lon - start_lon
        
        # Calculate bearing (angle from north)
        target_bearing = np.arctan2(lon_diff, lat_diff)
        
        # Apply face-to-path error (affects left/right)
        # Positive face-to-path = ball goes right
        lateral_error = np.radians(face_to_path * 2)  # Scale factor for effect
        
        # Apply launch direction error (affects overall direction)
        directional_error = np.radians(launch_direction)
        
        # Combine errors
        bearing = target_bearing + lateral_error + directional_error
        
        # Adjust for current state (rough/bunker reduce accuracy)
        if current_state == "Rough":
            bearing += np.radians(np.random.normal(0, self.rough_penalty * 2))
        elif current_state == "Bunker":
            bearing += np.radians(np.random.normal(0, self.bunker_penalty * 2))
        
        # Convert shot distance from yards to degrees (approximate)
        distance_deg = shot_distance / (69 * 1760)  # yards to degrees
        
        # Calculate landing position
        end_lat = start_lat + (distance_deg * np.cos(bearing))
        end_lon = start_lon + (distance_deg * np.sin(bearing))
        
        return end_lat, end_lon, face_to_path, launch_direction
    
    def _calculate_transition_probabilities(self, current_state, distance_to_hole, hole_distance, 
                                          current_lat=None, current_lon=None, target_lat=None, target_lon=None, hole_num=None):
        """Calculate transition probabilities based on shot distributions and distance"""
        # Initialize landing position variables
        landing_lat = None
        landing_lon = None
        face_to_path = None
        launch_direction = None
        
        shot_type = self._determine_next_shot_type(distance_to_hole, current_state, 1)
        
        # Sample shot distance
        shot_distance = self._sample_shot_distance(shot_type, hole_distance)
        
        # Sample all shot features from user's distributions
        shot_features = self._sample_shot_features(shot_type)
        face_to_path = shot_features['face_to_path']
        launch_direction = shot_features['launch_direction']
        
        # If using geometry, calculate actual landing position
        if self.use_geometry and current_lat is not None and current_lon is not None and target_lat is not None and target_lon is not None:
            landing_lat, landing_lon, face_to_path, launch_direction = self._calculate_landing_position(
                current_lat, current_lon, target_lat, target_lon, shot_distance, 
                face_to_path, launch_direction, current_state
            )
            
            # Determine state from geometry
            geometry_state = self._determine_state_from_geometry(landing_lat, landing_lon, hole_num)
            
            if geometry_state is not None:
                final_state = geometry_state
            else:
                # Fallback to distance-based
                new_distance = max(0, distance_to_hole - shot_distance)
                final_state = self._determine_state_from_distance(new_distance, hole_distance)
        else:
            # Use distance-based state determination (fallback)
            # Still use face-to-path and launch direction to determine accuracy
            total_error = abs(face_to_path) + abs(launch_direction)
            
            landing_state = "Fairway"
            if total_error > 5:
                landing_state = "Bunker"
            elif total_error > 2:
                landing_state = "Rough"
            
            new_distance = max(0, distance_to_hole - shot_distance)
            final_state = self._determine_state_from_distance(new_distance, hole_distance)
            
            # Override with accuracy-based state if it's worse
            state_hierarchy = {"Tee": 0, "Fairway": 1, "Rough": 2, "Bunker": 3, "Green": 4, "Hole": 5}
            if state_hierarchy[landing_state] > state_hierarchy[final_state]:
                final_state = landing_state
        
        # Don't set to "Hole" here - let simulate_hole handle completion
        # This ensures the final position is recorded as "Green" for visualization
        return final_state, shot_distance, landing_lat, landing_lon, face_to_path, launch_direction
    
    def _get_accuracy_error(self, shot_type, current_state):
        """Get accuracy error in degrees"""
        if shot_type not in self.shot_distributions:
            face_to_path_error = np.random.normal(0, 3)
            launch_direction_error = np.random.normal(0, 2)
        else:
            acc_info = self.shot_distributions[shot_type]['accuracy']
            face_to_path_error = np.random.normal(acc_info['face_to_path_mean'], acc_info['face_to_path_std'])
            launch_direction_error = np.random.normal(acc_info['launch_direction_mean'], acc_info['launch_direction_std'])
        
        # Adjust accuracy based on current state
        if current_state == "Rough":
            face_to_path_error *= (1 + self.rough_penalty)
            launch_direction_error *= (1 + self.rough_penalty)
        elif current_state == "Bunker":
            face_to_path_error *= (1 + self.bunker_penalty)
            launch_direction_error *= (1 + self.bunker_penalty)
        
        # Combine errors (approximate)
        total_error = np.sqrt(face_to_path_error**2 + launch_direction_error**2)
        return total_error
    
    def _determine_state_from_distance(self, distance_to_hole, hole_distance):
        """Determine state based on distance thresholds (fallback method)"""
        distance_ratio = distance_to_hole / hole_distance if hole_distance > 0 else 0
        
        if distance_ratio <= self.state_thresholds["Hole"]:
            return "Hole"
        elif distance_ratio <= self.state_thresholds["Green"]:
            return "Green"
        elif distance_ratio <= self.state_thresholds["Bunker"]:
            return "Bunker"
        elif distance_ratio <= self.state_thresholds["Rough"]:
            return "Rough"
        elif distance_ratio <= self.state_thresholds["Fairway"]:
            return "Fairway"
        else:
            return "Tee"
    
    def simulate_hole(self, hole_distance, hole_number):
        """Simulate a single hole using Markov chain transitions"""
        strokes = 0
        current_state = "Tee"
        distance_to_hole = hole_distance
        
        # Get starting and target positions from geometry if available
        current_lat, current_lon = None, None
        target_lat, target_lon = None, None
        
        if self.use_geometry:
            hole_greens, hole_fairways, hole_bunkers, hole_tees = self._get_hole_geometry(hole_number)
            
            if not hole_tees.empty and not hole_greens.empty:
                # Start at tee centroid
                tee_centroid = hole_tees.geometry.unary_union.centroid
                current_lat, current_lon = tee_centroid.y, tee_centroid.x
                
                # Target is green centroid
                green_centroid = hole_greens.geometry.unary_union.centroid
                target_lat, target_lon = green_centroid.y, green_centroid.x
        
        # Track shot sequence for analysis
        shot_sequence = []
        
        final_state = "Green"

        while current_state != final_state and strokes < 15:  # Max 10 strokes per hole
            # Calculate transition probabilities and sample next state
            next_state, shot_distance, landing_lat, landing_lon, face_to_path, launch_direction = self._calculate_transition_probabilities(
                current_state, distance_to_hole, hole_distance,
                current_lat, current_lon, target_lat, target_lon, hole_number
            )
            
            # Update position if using geometry
            if self.use_geometry and landing_lat is not None and landing_lon is not None:
                current_lat, current_lon = landing_lat, landing_lon
                
                # Recalculate distance to hole based on actual position
                if target_lat is not None and target_lon is not None:
                    lat_diff = target_lat - current_lat
                    lon_diff = target_lon - current_lon
                    lat_yards = lat_diff * 69 * 1760
                    lon_yards = lon_diff * 69 * 1760 * np.cos(np.radians(current_lat))
                    distance_to_hole = np.sqrt(lat_yards**2 + lon_yards**2)
                else:
                    distance_to_hole = max(0, distance_to_hole - shot_distance)
            else:
                # Update distance based on shot distance
                distance_to_hole = max(0, distance_to_hole - shot_distance)
            
            # Update state
            current_state = next_state
            strokes += 1
            
            # Check if ball is on green and close enough to hole
            if current_state == "Green" and distance_to_hole <= 2:
                # Record final shot on green (ending state should be Green)
                shot_sequence.append({
                    'stroke': strokes,
                    'from_state': "Green",  # Keep as Green for visualization
                    'shot_distance': shot_distance,
                    'distance_to_hole': 0,
                    'distance_ratio': 0,
                    'lat': current_lat,
                    'lon': current_lon,
                    'face_to_path': face_to_path,
                    'launch_direction': launch_direction
                })
                # Stop simulation (ball is holed)
                break
            else:
                # Record shot details
                shot_sequence.append({
                    'stroke': strokes,
                    'from_state': current_state,
                    'shot_distance': shot_distance,
                    'distance_to_hole': distance_to_hole,
                    'distance_ratio': distance_to_hole / hole_distance if hole_distance > 0 else 0,
                    'lat': current_lat,
                    'lon': current_lon,
                    'face_to_path': face_to_path,
                    'launch_direction': launch_direction
                })
        
        return strokes, shot_sequence
    
    def simulate_round(self):
        """Simulate a complete 18-hole round"""
        round_scores = []
        hole_details = []
        
        for hole_num, hole_distance in enumerate(self.hole_distances, 1):
            strokes, shot_sequence = self.simulate_hole(hole_distance, hole_num)
            round_scores.append(strokes)
            
            # Calculate par for this hole
            par = 3 if hole_distance < 250 else 4 if hole_distance < 450 else 5
            
            hole_details.append({
                'hole': hole_num,
                'distance': hole_distance,
                'strokes': strokes,
                'par': par,
                'score_vs_par': strokes - par,
                'shot_sequence': shot_sequence
            })
        
        return round_scores, hole_details

# Initialize simulator
simulator = GolfSimulator(
    shots_df, 
    hole_distances, 
    rough_penalty, 
    bunker_penalty, 
    wind_factor,
    greens=greens,
    fairways=fairways,
    bunkers=bunkers,
    tees=tees,
    use_geometry=True
)

# Display shot distributions
st.subheader("📊 Your Shot Distributions")
col1, col2 = st.columns(2)

with col1:
    st.write("**Shot Type Statistics:**")
    shot_stats = []
    for shot_type, dist_info in simulator.shot_distributions.items():
        shot_stats.append({
            'Shot Type': shot_type,
            'Count': dist_info['count'],
            'Avg Carry': f"{dist_info['carry']['mean']:.1f} yds",
            'Std Dev': f"{dist_info['carry']['std']:.1f} yds"
        })
    
    st.dataframe(pd.DataFrame(shot_stats))

with col2:
    # Create visualization of shot distributions
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Define colors for different shot types
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8', '#F7DC6F']
    
    for i, (shot_type, dist_info) in enumerate(simulator.shot_distributions.items()):
        if dist_info['count'] >= 1:  # Changed from >= 3 to >= 1
            carry_data = shots_df[shots_df['Shot Type'] == shot_type]['Carry (yards)']
            
            if len(carry_data) > 0:
                color = colors[i % len(colors)]  # Cycle through colors
                # For single shots, create a bar instead of histogram
                if len(carry_data) == 1:
                    ax.bar(carry_data.iloc[0], 1, alpha=0.7, label=f"{shot_type} (n={dist_info['count']})", 
                          width=10, color=color, edgecolor='black')
                else:
                    ax.hist(carry_data, alpha=0.7, label=f"{shot_type} (n={dist_info['count']})", 
                           bins=min(10, len(carry_data)), color=color, edgecolor='black')
    
    ax.set_xlabel('Carry Distance (yards)')
    ax.set_ylabel('Frequency')
    ax.set_title('Your Shot Distance Distributions')
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

# Display Markov Chain State Thresholds
# st.subheader("🎯 Course State Thresholds")
# st.write("**Distance thresholds for each course state (as % of hole distance):**")
# threshold_df = pd.DataFrame([
#     {"State": "Tee", "Threshold": "100%", "Description": "Starting position"},
#     {"State": "Fairway", "Threshold": "70%", "Description": "Good lie, full shots"},
#     {"State": "Rough", "Threshold": "30%", "Description": "Difficult lie, reduced distance"},
#     {"State": "Bunker", "Threshold": "20%", "Description": "Sand penalty, accuracy issues"},
#     {"State": "Green", "Threshold": "5%", "Description": "Putting surface"},
#     {"State": "Hole", "Threshold": "0%", "Description": "Ball holed"}
# ])
# st.dataframe(threshold_df)

# Visualization function
def visualize_single_simulation(simulator, hole_num, hole_distance):
    """Visualize a single hole simulation on a plot similar to geoplot_hole"""
    if not simulator.use_geometry or greens is None:
        st.warning("Visualization requires geometry data. Please ensure geometry files are loaded.")
        return None
    
    # Get hole geometry
    hole_greens = greens[greens['hole'] == hole_num].copy()
    hole_fairways = fairways[fairways['hole'] == hole_num].copy()
    hole_bunkers = bunkers[bunkers['hole'] == hole_num].copy()
    hole_tees = tees[tees['hole'] == hole_num].copy()
    
    if hole_tees.empty or hole_greens.empty:
        st.warning(f"Geometry data not available for hole {hole_num}")
        return None
    
    # Run a single simulation
    strokes, shot_sequence = simulator.simulate_hole(hole_distance, hole_num)
    
    # Get tee and green centroids
    tee_centroid = hole_tees.geometry.unary_union.centroid
    green_centroid = hole_greens.geometry.unary_union.centroid
    
    # Transform geometries (same as geoplot_hole)
    hole_tees['geometry'] = hole_tees.translate(xoff=-tee_centroid.x, yoff=-tee_centroid.y)
    hole_fairways['geometry'] = hole_fairways.translate(xoff=-tee_centroid.x, yoff=-tee_centroid.y)
    hole_greens['geometry'] = hole_greens.translate(xoff=-tee_centroid.x, yoff=-tee_centroid.y)
    
    # Rotate so green is "up"
    green_centroid_translated = hole_greens.geometry.unary_union.centroid
    dx = green_centroid_translated.x
    dy = green_centroid_translated.y
    angle = np.degrees(np.arctan2(dx, dy))
    
    hole_tees['geometry'] = hole_tees.rotate(angle, origin=(0,0))
    hole_fairways['geometry'] = hole_fairways.rotate(angle, origin=(0,0))
    hole_greens['geometry'] = hole_greens.rotate(angle, origin=(0,0))
    
    # Convert to yards
    latitude_to_yards = 69 * 1760
    longitude_to_yards = 69 * 1760 * np.cos(np.radians(tee_centroid.y))
    hole_tees['geometry'] = hole_tees.scale(xfact=longitude_to_yards, yfact=latitude_to_yards, origin=(0,0))
    hole_fairways['geometry'] = hole_fairways.scale(xfact=longitude_to_yards, yfact=latitude_to_yards, origin=(0,0))
    hole_greens['geometry'] = hole_greens.scale(xfact=longitude_to_yards, yfact=latitude_to_yards, origin=(0,0))
    
    # Transform shot positions
    shot_positions = []
    shot_positions.append({
        'x': 0,
        'y': 0,
        'state': "Tee",
        'stroke': 0,
        'distance': 0
    })
    for shot in shot_sequence:
        if shot['lat'] is not None and shot['lon'] is not None:
            # Transform shot position
            shot_point = Point(shot['lon'], shot['lat'])
            shot_point = translate(shot_point, xoff=-tee_centroid.x, yoff=-tee_centroid.y)
            shot_point = rotate(shot_point, angle, origin=(0,0))
            shot_point = scale(shot_point, xfact=longitude_to_yards, yfact=latitude_to_yards, origin=(0,0))
            shot_positions.append({
                'x': shot_point.x,
                'y': shot_point.y,
                'state': shot['from_state'],
                'stroke': shot['stroke'],
                'distance': shot['shot_distance']
            })
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Plot course features
    if not hole_fairways.empty:
        hole_fairways.plot(ax=ax, color='palegreen', edgecolor='black', alpha=0.5, label='Fairway')
    if not hole_greens.empty:
        hole_greens.plot(ax=ax, color='darkgreen', edgecolor='black', alpha=0.7, label='Green')
    if not hole_tees.empty:
        hole_tees.plot(ax=ax, color='blue', edgecolor='black', alpha=0.5, label='Tee')
    if hole_bunkers is not None and not hole_bunkers.empty:
        # Transform bunkers too
        hole_bunkers_plot = hole_bunkers.copy()
        hole_bunkers_plot['geometry'] = hole_bunkers_plot.translate(xoff=-tee_centroid.x, yoff=-tee_centroid.y)
        hole_bunkers_plot['geometry'] = hole_bunkers_plot.rotate(angle, origin=(0,0))
        hole_bunkers_plot['geometry'] = hole_bunkers_plot.scale(xfact=longitude_to_yards, yfact=latitude_to_yards, origin=(0,0))
        hole_bunkers_plot.plot(ax=ax, color='yellow', edgecolor='black', alpha=0.5, label='Bunker')
    
    # Plot shot path
    if shot_positions:
        state_colors = {
            'Tee': 'blue',
            'Fairway': 'green',
            'Rough': 'brown',
            'Bunker': 'yellow',
            'Green': 'darkgreen',
            'Hole': 'red'
        }
        
        # Plot shot positions
        for i, pos in enumerate(shot_positions):
            color = state_colors.get(pos['state'], 'gray')
            ax.scatter(pos['x'], pos['y'], c=color, s=100, edgecolor='black', linewidth=2, zorder=5)
            ax.annotate(f"{pos['stroke']}", (pos['x'], pos['y']), fontsize=10, ha='center', va='center', 
                       color='white', weight='bold', zorder=6)
        
        # Plot shot path (lines connecting shots)
        for i in range(len(shot_positions) - 1):
            x1, y1 = shot_positions[i]['x'], shot_positions[i]['y']
            x2, y2 = shot_positions[i+1]['x'], shot_positions[i+1]['y']
            ax.plot([x1, x2], [y1, y2], 'k--', alpha=0.5, linewidth=2, zorder=4)
    
    ax.set_title(f'Hole {hole_num} Simulation - {strokes} Strokes')
    ax.set_xlabel("Yards Left/Right of Line of Play")
    ax.set_ylabel("Yards from Tee")
    ax.legend()
    ax.axis('equal')
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    
    # Show shot details
    st.write(f"**Shot Sequence:**")
    shot_df = pd.DataFrame(shot_sequence)
    if not shot_df.empty:
        display_df = shot_df[['stroke', 'from_state', 'shot_distance', 'distance_to_hole', 
                             'face_to_path', 'launch_direction']].copy()
        display_df.columns = ['Stroke', 'State', 'Shot Distance (yds)', 'Distance to Hole (yds)', 
                              'Face to Path (°)', 'Launch Direction (°)']
        st.dataframe(display_df.round(2))
    
    return strokes

def visualize_full_round(simulator, hole_distances):
    """Visualize a full 18-hole round simulation using visualize_single_simulation for each hole"""
    if not simulator.use_geometry or greens is None:
        st.warning("Visualization requires geometry data. Please ensure geometry files are loaded.")
        return
    
    total_strokes = 0
    hole_results = []
    
    # First, simulate all holes to get strokes for summary
    for hole_num in range(1, 19):
        hole_distance = hole_distances[hole_num - 1] if hole_num <= len(hole_distances) else 400
        
        # Check if geometry is available
        hole_greens = greens[greens['hole'] == hole_num].copy()
        hole_tees = tees[tees['hole'] == hole_num].copy()
        
        if not hole_tees.empty and not hole_greens.empty:
            strokes, _ = simulator.simulate_hole(hole_distance, hole_num)
            total_strokes += strokes
            hole_results.append({
                'Hole': hole_num,
                'Strokes': strokes,
                'Distance (yds)': hole_distance
            })
    
    # Display summary first
    st.markdown("---")
    st.subheader("📊 Round Summary")
    
    # Total strokes
    st.metric("Total Strokes", f"{total_strokes}")
    
    # Hole-by-hole breakdown
    if hole_results:
        st.write("**Hole-by-Hole Breakdown:**")
        summary_df = pd.DataFrame(hole_results)
        st.dataframe(summary_df)
    
    # Display each hole's visualization in an expander
    st.markdown("---")
    st.subheader("🏌️ Hole Visualizations")
    
    for hole_num in range(1, 19):
        hole_distance = hole_distances[hole_num - 1] if hole_num <= len(hole_distances) else 400
        
        # Check if geometry is available
        hole_greens = greens[greens['hole'] == hole_num].copy()
        hole_tees = tees[tees['hole'] == hole_num].copy()
        
        if not hole_tees.empty and not hole_greens.empty:
            # Get strokes for this hole from summary
            hole_strokes = next((r['Strokes'] for r in hole_results if r['Hole'] == hole_num), None)
            
            with st.expander(f"🏌️ Hole {hole_num} - {hole_strokes} Strokes" if hole_strokes else f"🏌️ Hole {hole_num}"):
                # Use visualize_single_simulation for each hole
                visualize_single_simulation(simulator, hole_num, hole_distance)

def run_statistical_simulation(simulator, hole_distances, n_simulations=10000):
    """Run simulations without visualizations and show detailed statistics"""
    all_round_scores = []
    all_hole_details = []
    
    # Create progress bar and status text
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Run simulations
    for i in range(n_simulations):
        round_scores, hole_details = simulator.simulate_round()
        all_round_scores.append(sum(round_scores))
        all_hole_details.extend(hole_details)
        
        # Update progress bar
        progress = (i + 1) / n_simulations
        progress_bar.progress(progress)
        status_text.text(f"Simulated {i + 1:,} / {n_simulations:,} rounds ({progress*100:.1f}%)")
    
    # Clear progress bar and status text when done
    progress_bar.empty()
    status_text.empty()
    
    # Convert to arrays for analysis
    round_totals = np.array(all_round_scores)
    
    # Calculate statistics
    mean_score = np.mean(round_totals)
    std_score = np.std(round_totals)
    median_score = np.median(round_totals)
    min_score = np.min(round_totals)
    max_score = np.max(round_totals)
    
    # Percentiles
    p25 = np.percentile(round_totals, 25)
    p75 = np.percentile(round_totals, 75)
    p90 = np.percentile(round_totals, 90)
    p95 = np.percentile(round_totals, 95)
    
    # Display results
    st.subheader("📈 Overall Round Statistics")
    
    # Key statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Average Score", f"{mean_score:.1f}")
    with col2:
        st.metric("Best Round", f"{min_score}")
    with col3:
        st.metric("Worst Round", f"{max_score}")
    with col4:
        st.metric("Standard Deviation", f"{std_score:.1f}")
    
    # Detailed statistics
    st.subheader("📊 Detailed Score Statistics")
    
    stats_data = {
        'Metric': ['Mean', 'Median', 'Standard Deviation', 'Min', 'Max', 
                  '25th Percentile', '75th Percentile', '90th Percentile', '95th Percentile'],
        'Value': [f"{mean_score:.2f}", f"{median_score:.2f}", f"{std_score:.2f}", 
                 f"{min_score}", f"{max_score}", f"{p25:.2f}", f"{p75:.2f}", 
                 f"{p90:.2f}", f"{p95:.2f}"]
    }
    
    st.dataframe(pd.DataFrame(stats_data))
    
    # Histogram visualization
    st.subheader("📊 Score Distribution")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hist(round_totals, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    ax.axvline(mean_score, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_score:.1f}')
    ax.axvline(median_score, color='green', linestyle='--', linewidth=2, label=f'Median: {median_score:.1f}')
    ax.set_xlabel('Total Score')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Distribution of {n_simulations:,} Simulated Rounds')
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    
    # Hole-by-hole analysis
    st.subheader("🏌️ Hole-by-Hole Stroke Statistics")
    
    hole_df = pd.DataFrame(all_hole_details)
    hole_stats = hole_df.groupby('hole').agg({
        'strokes': ['mean', 'std', 'min', 'max', 'median'],
        'distance': 'first',
        'par': 'first'
    }).round(2)
    
    hole_stats.columns = ['Avg Strokes', 'Std Dev', 'Min Strokes', 'Max Strokes', 'Median Strokes', 'Distance', 'Par']
    hole_stats['Avg vs Par'] = hole_stats['Avg Strokes'] - hole_stats['Par']
    hole_stats = hole_stats[['Distance', 'Par', 'Avg Strokes', 'Median Strokes', 'Std Dev', 'Min Strokes', 'Max Strokes', 'Avg vs Par']]
    
    st.write("**Statistics for each hole across all simulations:**")
    st.dataframe(hole_stats)
    
    # Hole difficulty visualization
    fig, ax = plt.subplots(figsize=(14, 6))
    
    holes = hole_stats.index
    avg_strokes = hole_stats['Avg Strokes']
    par = hole_stats['Par']
    
    x = np.arange(len(holes))
    width = 0.35
    
    ax.bar(x - width/2, avg_strokes, width, alpha=0.7, label='Average Strokes', color='skyblue')
    ax.bar(x + width/2, par, width, alpha=0.7, label='Par', color='lightcoral')
    ax.set_xlabel('Hole Number')
    ax.set_ylabel('Strokes')
    ax.set_title('Average Strokes vs Par by Hole')
    ax.set_xticks(x)
    ax.set_xticklabels(holes)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    st.pyplot(fig)
    
    # Contribution of each hole to total score
    st.subheader("📊 Hole Contribution to Total Score")
    
    # Calculate contribution percentage
    total_avg_strokes = hole_stats['Avg Strokes'].sum()
    hole_stats['Contribution %'] = (hole_stats['Avg Strokes'] / total_avg_strokes * 100).round(2)
    
    contribution_df = hole_stats[['Avg Strokes', 'Contribution %']].copy()
    contribution_df.columns = ['Average Strokes', 'Contribution to Total Score (%)']
    
    st.write("**How much each hole contributes to the average total score:**")
    st.dataframe(contribution_df)
    
    # Visualization of hole contributions
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(holes, hole_stats['Contribution %'], alpha=0.7, color='steelblue', edgecolor='black')
    ax.set_xlabel('Hole Number')
    ax.set_ylabel('Contribution (%)')
    ax.set_title('Percentage Contribution of Each Hole to Total Score')
    ax.set_xticks(holes)
    ax.grid(True, alpha=0.3, axis='y')
    
    st.pyplot(fig)
    
    # Standard deviation by hole
    st.subheader("📊 Stroke Variability by Hole")
    
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(holes, hole_stats['Std Dev'], alpha=0.7, color='coral', edgecolor='black')
    ax.set_xlabel('Hole Number')
    ax.set_ylabel('Standard Deviation (Strokes)')
    ax.set_title('Stroke Variability (Standard Deviation) by Hole')
    ax.set_xticks(holes)
    ax.grid(True, alpha=0.3, axis='y')
    
    st.pyplot(fig)
    
    # Store results in session state
    st.session_state.simulation_results = {
        'round_totals': round_totals,
        'mean_score': mean_score,
        'std_score': std_score,
        'hole_details': all_hole_details,
        'hole_stats': hole_stats
    }

# Single simulation visualization button
st.markdown("---")
st.subheader("Simulations on Course")

st.text("Hole Selection:")
selected_hole = st.number_input("Select Hole Number", min_value=1, max_value=18, value=1)
hole_num = selected_hole
if st.button("🎯 Visualize Single Simulation", type="secondary"):
    viz_hole_distance = hole_distances[int(hole_num) - 1] if hole_num <= len(hole_distances) else 400
    visualize_single_simulation(simulator, int(hole_num), viz_hole_distance)

if st.button("🏌️ Simulate Full Round (18 Holes)", type="secondary"):
    visualize_full_round(simulator, hole_distances)

if st.button(f"📊 Run {n_simulations} Simulations", type="primary"):
    run_statistical_simulation(simulator, hole_distances, n_simulations)

