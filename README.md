# Golf Analytics

## Instructions

1. Install the Streamlit package
```pip install streamlit ```

2. Install dependencies from requirements.txt
```pip install requirements.txt```

3. Run the Streamlit dashboard (locally)
```streamlit run ~/StreamlitApp/welcome.py```


## Dependencies
```
streamlit
pandas
numpy
matplotlib
scikit-learn
seaborn
langchain_deepseek
geopandas
pydeck
shapely
```

## Dashboard Pages
- welcome.py: basic overview of app
- auth.py: sign up, log in
- player data.py: players can input data in the form of csv or input boxes
- coach.py: textual NLP-based golf coaching, uses user inputs & our research
- geospatial.py: visualizes Clifton Park holes
- golf_simulation.py: simulation of rounds, using Monte Carlo methods and Markov Chains
- research.py: our research methods, excluding the simulation model
