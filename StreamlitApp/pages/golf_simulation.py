import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.mixture import GaussianMixture
import sqlite3
from auth import get_user_shots, init_db
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
st.markdown("Simulate 10,000 rounds using your personal shot distributions")

# Load user shot data
def load_user_shots(user_id):
    """Load and process user shot data"""
    shots = get_user_shots(user_id)
    if not shots:
        return pd.DataFrame()
    
    # Handle variable number of columns (old vs new format)
    if len(shots) > 0:
        first_shot_columns = len(shots[0])
        
        if first_shot_columns == 14:  # Old format without location data
            df = pd.DataFrame(shots, columns=[
                'id', 'user_id', 'shot_type', 'carry', 'club_speed',
                'ball_speed', 'launch_angle', 'spin_rate', 'face_angle',
                'face_to_path', 'club_path', 'attack_angle', 'launch_direction', 'timestamp'
            ])
            # Add empty location columns
            df['origin_lat'] = None
            df['origin_lon'] = None
            df['destination_lat'] = None
            df['destination_lon'] = None
        else:  # New format with location data
            df = pd.DataFrame(shots, columns=[
                'id', 'user_id', 'shot_type', 'carry', 'club_speed',
                'ball_speed', 'launch_angle', 'spin_rate', 'face_angle',
                'face_to_path', 'club_path', 'attack_angle', 'launch_direction',
                'origin_lat', 'origin_lon', 'destination_lat', 'destination_lon', 'timestamp'
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
        'launch_direction': 'Launch Direction (Deg)',
        'origin_lat': 'Origin Latitude',
        'origin_lon': 'Origin Longitude',
        'destination_lat': 'Destination Latitude',
        'destination_lon': 'Destination Longitude'
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
    st.write("**Debug Info:**")
    st.write(f"- User ID: {st.session_state.user_id}")
    st.write(f"- Raw shots from database: {len(get_user_shots(st.session_state.user_id))}")
    st.write("- Make sure you have added shots through the Player Data page")
    
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

# Course distances input
st.sidebar.subheader("Course Distances (yards)")
hole_distances = []
for i in range(18):
    distance = st.sidebar.number_input(
        f"Hole {i+1} Distance", 
        min_value=100, 
        max_value=600, 
        value=400 if i < 4 else 350 if i < 8 else 300 if i < 12 else 250 if i < 16 else 200,
        key=f"hole_{i+1}"
    )
    hole_distances.append(distance)

# Course difficulty settings
st.sidebar.subheader("Course Difficulty")
rough_penalty = st.sidebar.slider("Rough Penalty (%)", 0, 50, 15)
bunker_penalty = st.sidebar.slider("Bunker Penalty (%)", 0, 50, 25)
wind_factor = st.sidebar.slider("Wind Factor", 0.0, 2.0, 1.0, 0.1)

# Simulation parameters
st.sidebar.subheader("Simulation Parameters")
n_simulations = st.sidebar.slider("Number of Rounds", 1000, 50000, 10000, 1000)
show_details = st.sidebar.checkbox("Show Detailed Analysis", value=False)

class GolfSimulator:
    def __init__(self, shots_df, hole_distances, rough_penalty=15, bunker_penalty=25, wind_factor=1.0):
        self.shots_df = shots_df
        self.hole_distances = hole_distances
        self.rough_penalty = rough_penalty / 100
        self.bunker_penalty = bunker_penalty / 100
        self.wind_factor = wind_factor
        
        # Create shot type distributions
        self.shot_distributions = self._create_shot_distributions()
        
        # Define course states
        self.states = ["Tee", "Fairway", "Rough", "Bunker", "Green", "Hole"]
        
        # State distance thresholds (as percentages of hole distance)
        self.state_thresholds = {
            "Tee": 1.0,      # Start at 100% of hole distance
            "Fairway": 0.7,   # Fairway ends at 70% of hole distance
            "Rough": 0.3,    # Rough ends at 30% of hole distance  
            "Bunker": 0.2,   # Bunker ends at 20% of hole distance
            "Green": 0.05,   # Green starts at 5% of hole distance
            "Hole": 0.0      # Hole at 0% of hole distance
        }
        
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
    
    def _calculate_transition_probabilities(self, current_state, distance_to_hole, hole_distance):
        """Calculate transition probabilities based on shot distributions and distance"""
        shot_type = self._determine_next_shot_type(distance_to_hole, current_state, 1)
        
        # Sample shot distance
        shot_distance = self._sample_shot_distance(shot_type, hole_distance)
        
        # Determine accuracy-based landing state
        landing_state = self._determine_shot_accuracy(shot_type, current_state)
        
        # Calculate new distance after shot
        new_distance = max(0, distance_to_hole - shot_distance)
        
        # Determine final state based on distance thresholds
        distance_ratio = new_distance / hole_distance
        
        if distance_ratio <= self.state_thresholds["Hole"]:
            final_state = "Hole"
        elif distance_ratio <= self.state_thresholds["Green"]:
            final_state = "Green"
        elif distance_ratio <= self.state_thresholds["Bunker"]:
            final_state = "Bunker"
        elif distance_ratio <= self.state_thresholds["Rough"]:
            final_state = "Rough"
        elif distance_ratio <= self.state_thresholds["Fairway"]:
            final_state = "Fairway"
        else:
            final_state = "Tee"
        
        # Override with accuracy-based state if it's worse
        state_hierarchy = {"Tee": 0, "Fairway": 1, "Rough": 2, "Bunker": 3, "Green": 4, "Hole": 5}
        if state_hierarchy[landing_state] > state_hierarchy[final_state]:
            final_state = landing_state
        
        return final_state, shot_distance
    
    def simulate_hole(self, hole_distance, hole_number):
        """Simulate a single hole using Markov chain transitions"""
        strokes = 0
        current_state = "Tee"
        distance_to_hole = hole_distance
        
        # Track shot sequence for analysis
        shot_sequence = []
        
        while current_state != "Hole" and strokes < 10:  # Max 10 strokes per hole
            # Calculate transition probabilities and sample next state
            next_state, shot_distance = self._calculate_transition_probabilities(
                current_state, distance_to_hole, hole_distance
            )
            
            # Update state and distance
            current_state = next_state
            distance_to_hole = max(0, distance_to_hole - shot_distance)
            strokes += 1
            
            # Record shot details
            shot_sequence.append({
                'stroke': strokes,
                'from_state': current_state,
                'shot_distance': shot_distance,
                'distance_to_hole': distance_to_hole,
                'distance_ratio': distance_to_hole / hole_distance
            })
            
            # Special case: if we're on the green and close enough, go to hole
            if current_state == "Green" and distance_to_hole <= 2:
                current_state = "Hole"
                distance_to_hole = 0
        
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
simulator = GolfSimulator(shots_df, hole_distances, rough_penalty, bunker_penalty, wind_factor)

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

# Run simulation
if st.button("🎯 Run Simulation", type="primary"):
    with st.spinner("Running 10,000 round simulations..."):
        all_round_scores = []
        all_hole_details = []
        
        # Run simulations
        for i in range(n_simulations):
            round_scores, hole_details = simulator.simulate_round()
            all_round_scores.append(sum(round_scores))
            all_hole_details.extend(hole_details)
        
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
        st.subheader("📈 Simulation Results")
        
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
        st.subheader("📊 Detailed Statistics")
        
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
        
        # Histogram
        plt.xlim(0, 150)
        plt.hist(round_totals, bins=10, alpha=0.7, color='skyblue', edgecolor='black')
        plt.axvline(mean_score, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_score:.1f}')
        plt.axvline(median_score, color='green', linestyle='--', linewidth=2, label=f'Median: {median_score:.1f}')
        plt.xlabel('Total Score')
        plt.ylabel('Frequency')
        plt.title(f'Distribution of {n_simulations:,} Simulated Rounds')
        plt.legend()
        plt.grid(True, alpha=0.3)
        st.pyplot(plt)
        
        # Score ranges
        st.subheader("🎯 Score Range Analysis")
        
        score_ranges = [
            (60, 70, "Excellent"),
            (70, 80, "Good"),
            (80, 90, "Average"),
            (90, 100, "Below Average"),
            (100, 200, "Poor")
        ]
        
        range_data = []
        for min_score_range, max_score_range, label in score_ranges:
            count = np.sum((round_totals >= min_score_range) & (round_totals < max_score_range))
            percentage = (count / len(round_totals)) * 100
            range_data.append({
                'Score Range': f"{min_score_range}-{max_score_range}",
                'Category': label,
                'Count': count,
                'Percentage': f"{percentage:.1f}%"
            })
        
        st.dataframe(pd.DataFrame(range_data))
        
        # Hole-by-hole analysis
        if show_details:
            st.subheader("🏌️ Hole-by-Hole Analysis")
            
            hole_df = pd.DataFrame(all_hole_details)
            hole_stats = hole_df.groupby('hole').agg({
                'strokes': ['mean', 'std', 'min', 'max'],
                'distance': 'first',
                'par': 'first'
            }).round(2)
            
            hole_stats.columns = ['Avg Strokes', 'Std Dev', 'Min Strokes', 'Max Strokes', 'Distance', 'Par']
            hole_stats['Avg vs Par'] = hole_stats['Avg Strokes'] - hole_stats['Par']
            
            st.dataframe(hole_stats)
            
            # Hole difficulty visualization
            fig, ax = plt.subplots(figsize=(12, 6))
            
            holes = hole_stats.index
            avg_strokes = hole_stats['Avg Strokes']
            par = hole_stats['Par']
            
            ax.bar(holes, avg_strokes, alpha=0.7, label='Average Strokes')
            ax.plot(holes, par, 'r--', linewidth=2, label='Par')
            ax.set_xlabel('Hole Number')
            ax.set_ylabel('Strokes')
            ax.set_title('Hole Difficulty Analysis')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            st.pyplot(fig)
            
            # Shot sequence analysis
            st.subheader("📊 Shot Sequence Analysis")
            
            # Analyze state transitions
            state_transitions = {}
            for hole_detail in all_hole_details:
                for shot in hole_detail['shot_sequence']:
                    from_state = shot['from_state']
                    if from_state not in state_transitions:
                        state_transitions[from_state] = []
                    state_transitions[from_state].append(shot['shot_distance'])
            
            # Display state transition statistics
            transition_stats = []
            for state, distances in state_transitions.items():
                if distances:
                    transition_stats.append({
                        'From State': state,
                        'Avg Shot Distance': f"{np.mean(distances):.1f} yds",
                        'Std Dev': f"{np.std(distances):.1f} yds",
                        'Count': len(distances)
                    })
            
            if transition_stats:
                st.dataframe(pd.DataFrame(transition_stats))
        
        # Store results in session state
        st.session_state.simulation_results = {
            'round_totals': round_totals,
            'mean_score': mean_score,
            'std_score': std_score,
            'hole_details': all_hole_details
        }

# Display course setup summary
st.subheader("🏌️ Course Setup Summary")
course_df = pd.DataFrame({
    'Hole': range(1, 19),
    'Distance (yards)': hole_distances,
    'Par': [3 if d < 250 else 4 if d < 450 else 5 for d in hole_distances]
})

st.dataframe(course_df)

total_distance = sum(hole_distances)
course_par = sum([3 if d < 250 else 4 if d < 450 else 5 for d in hole_distances])

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Distance", f"{total_distance:,} yards")
with col2:
    st.metric("Course Par", f"{course_par}")
with col3:
    st.metric("Average Hole Distance", f"{total_distance/18:.0f} yards")
