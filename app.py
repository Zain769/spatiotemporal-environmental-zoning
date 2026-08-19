import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
os.environ.setdefault("OMP_NUM_THREADS", "4")
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import folium
import joblib

DATA_FILE = "C:/Users/USER/Downloads/R/Python/rural_landscape_dataset.csv"
os.makedirs("results", exist_ok=True)
df = pd.read_csv(DATA_FILE)
df = df.drop_duplicates()
df["timestamp"] = pd.to_datetime(
    df["timestamp"],
    errors="coerce"
)

df = df.dropna()
df["year"] = df["timestamp"].dt.year
df["month"] = df["timestamp"].dt.month
df["month_sin"] = np.sin(
    2 * np.pi * df["month"] / 12
)
df["month_cos"] = np.cos(
    2 * np.pi * df["month"] / 12
)
features = [

    "soil_moisture",
    "land_surface_temp",
    "ndvi",
    "lst",
    "month_sin",
    "month_cos"

]

X = df[features]

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)
silhouette_results = {}

for number_zones in range(2, 7):

    candidate_model = KMeans(

        n_clusters=number_zones,

        n_init=20,

        random_state=42

    )

    candidate_labels = candidate_model.fit_predict(X_scaled)

    silhouette_results[number_zones] = silhouette_score(

        X_scaled,

        candidate_labels

    )


number_zones = max(

    silhouette_results,

    key=silhouette_results.get

)

model = KMeans(

    n_clusters=number_zones,

    n_init=20,

    random_state=42

)

df["zone_id"] = model.fit_predict(X_scaled)

silhouette = silhouette_results[number_zones]

print("\nSilhouette scores by zone count:")

for count, score in silhouette_results.items():

    print(f"{count} zones: {score:.3f}")

print(f"\nSelected environmental zones: {number_zones}")
print(f"Silhouette score: {silhouette:.3f}")
profile_features = [

    "soil_moisture",
    "land_surface_temp",
    "ndvi",
    "lst"
]
zone_profiles = df.groupby("zone_id")[profile_features].mean()

zone_profiles["vegetation_rank"] = zone_profiles["ndvi"].rank(
    ascending=False,
    method="first"
)

zone_profiles["moisture_rank"] = zone_profiles["soil_moisture"].rank(
    ascending=False,
    method="first"
)

zone_profiles["temperature_rank"] = zone_profiles["lst"].rank(
    ascending=False,
    method="first"
)

zone_names = {}

for zone_id, profile in zone_profiles.iterrows():

    if profile["vegetation_rank"] == 1:

        zone_name = "Vegetation-rich zone"

    elif profile["moisture_rank"] == 1:

        zone_name = "Moisture-dominant zone"

    elif profile["temperature_rank"] == 1:

        zone_name = "Warm low-vegetation zone"

    else:

        zone_name = "Balanced environmental zone"

    zone_names[zone_id] = f"{zone_name} {zone_id + 1}"


df["Environmental_Zone"] = df["zone_id"].map(zone_names)

zone_profiles["Environmental_Zone"] = zone_profiles.index.map(
    zone_names
)
gis_columns = [

    "sensor_id",

    "latitude",

    "longitude",

    "soil_moisture",

    "land_surface_temp",

    "ndvi",

    "lst",

    "land_cover_label",

    "Environmental_Zone",

    "zone_id"

]
df[gis_columns].to_csv(

    "results/environmental_zones.csv",

    index=False

)

import folium
from folium import Element
center_lat = df["latitude"].mean()
center_lon = df["longitude"].mean()


m = folium.Map(

    location=[
        center_lat,
        center_lon
    ],

    zoom_start=5,

    tiles=None

)
folium.TileLayer(

    "CartoDB positron",

    name="Light Map"

).add_to(m)


folium.TileLayer(

    "OpenStreetMap",

    name="OpenStreetMap"

).add_to(m)

zone_palette = [

    "#167A58",
    "#D89038",
    "#2D7DD2",
    "#C95D63",
    "#7A6FA8",
    "#4E9F78"

]
zone_colors = {

    zone_names[zone_id]: zone_palette[zone_id % len(zone_palette)]

    for zone_id in sorted(zone_names)

}
for zone_name, color in zone_colors.items():

    layer = folium.FeatureGroup(

        name=zone_name,

        show=True

    )


    class_data = df[

        df["Environmental_Zone"] == zone_name

    ]

    for _, row in class_data.iterrows():

        popup_html = f"""

        <div style="font-family:Arial; width:230px;">

            <h4 style="margin-bottom:8px;">
                {zone_name}
            </h4>

            <b>Sensor:</b> {row['sensor_id']}<br>
            <b>Reference label:</b> {row['land_cover_label']}<br>
            <b>NDVI:</b> {row['ndvi']:.3f}<br>
            <b>LST:</b> {row['lst']:.2f} °C<br>
            <b>Soil Moisture:</b> {row['soil_moisture']:.3f}<br>
            <b>Latitude:</b> {row['latitude']:.4f}<br>
            <b>Longitude:</b> {row['longitude']:.4f}

        </div>

        """


        folium.CircleMarker(

            location=[

                row["latitude"],

                row["longitude"]

            ],

            radius=6,

            color=color,

            fill=True,

            fill_color=color,

            fill_opacity=0.75,

            weight=1,

            popup=folium.Popup(

                popup_html,

                max_width=300

            )

        ).add_to(layer)


    layer.add_to(m)
    
folium.LayerControl(

    collapsed=False

).add_to(m)


m.get_root().header.add_child(

    Element(
        """
        <style>
            .leaflet-control-layers {
                margin-top: 150px !important;
                margin-right: 18px !important;
                border: 1px solid rgba(20, 54, 45, 0.14) !important;
                border-radius: 10px !important;
                box-shadow: 0 8px 24px rgba(17, 38, 32, 0.16) !important;
            }
        </style>
        """
    )

)
total_points = len(df)

number_zones = df[
    "Environmental_Zone"
].nunique()

mean_ndvi = df[
    "ndvi"
].mean()

mean_lst = df[
    "lst"
].mean()


zone_counts = (

    df[
        "Environmental_Zone"
    ]

    .value_counts()

)

dashboard_html = f"""

<style>

@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

:root {{
    --ink: #17332b;
    --muted: #6b7d76;
    --line: #dfe9e4;
    --paper: #f7faf8;
    --green: #167a58;
    --green-dark: #0d4f3b;
    --mint: #e4f2ec;
    --amber: #d89038;
}}

body {{

    margin: 0;

    font-family: 'DM Sans', sans-serif;

    color: var(--ink);

    background: var(--paper);

}}

#dashboard-header {{

    position: fixed;

    top: 0;

    left: 0;

    right: 0;

    height: 85px;

    background: linear-gradient(120deg, #0b4031 0%, #167a58 62%, #43a879 100%);

    color: white;

    z-index: 9999;

    padding: 14px 32px;

    box-sizing: border-box;

    display: flex;

    align-items: center;

    justify-content: center;

}}

#dashboard-header h1 {{

    margin: 0;

    font-family: 'Space Grotesk', sans-serif;

    font-size: 25px;

    letter-spacing: 0;

}}

#dashboard-nav {{

    position: fixed;

    top: 85px;

    left: 0;

    right: 0;

    height: 50px;

    background: #ffffff;

    border-bottom: 1px solid var(--line);

    z-index: 9998;

    display: flex;

    align-items: center;

    padding: 0 22px;

    box-shadow: 0 4px 18px rgba(17, 38, 32, 0.06);

}}

.nav-button {{

    border: none;

    background: transparent;

    padding:
        15px 16px;

    cursor: pointer;

    font-size: 14px;

    color: var(--muted);

    font-weight: 600;

    transition: color 160ms ease, background 160ms ease;

}}

.nav-button i {{

    margin-right: 8px;

    color: var(--green);

}}

.nav-button:hover {{

    background: var(--mint);

    color: var(--green-dark);

}}

.dashboard-panel {{

    position: fixed;

    top: 135px;

    left: 0;

    right: 0;

    bottom: 0;

    background: var(--paper);

    z-index: 9997;

    padding: 34px clamp(22px, 5vw, 72px);

    overflow-y: auto;

    display: none;

}}

.dashboard-panel.active {{

    display: block;

    animation: panel-in 220ms ease-out;

}}

@keyframes panel-in {{
    from {{ opacity: 0; transform: translateY(6px); }}
    to {{ opacity: 1; transform: translateY(0); }}

}}

.card-container {{

    display: grid;

    grid-template-columns:
        repeat(
            auto-fit,
            minmax(
                200px,
                1fr
            )
        );

    gap: 20px;

    margin-bottom: 30px;

}}

.stat-card {{

    background: #ffffff;

    border-radius: 10px;

    padding: 22px 24px;

    border: 1px solid var(--line);

    border-top: 4px solid var(--green);

    box-shadow:
        0 2px 8px
        0 8px 24px rgba(17, 38, 32, 0.06);

    transition: transform 160ms ease, box-shadow 160ms ease;

}}

.stat-card:hover {{

    transform: translateY(-3px);

    box-shadow: 0 12px 28px rgba(17, 38, 32, 0.11);

}}

.stat-card h3 {{

    margin: 0;

    color: var(--muted);

    font-size: 13px;

}}

.stat-card .value {{

    font-size: 28px;

    font-weight: bold;

    color: var(--green-dark);

    margin-top: 8px;

}}

table {{

    width: 100%;

    border-collapse:
        collapse;

}}

th,
td {{

    padding: 12px;

    border-bottom: 1px solid var(--line);

    text-align: left;

}}

th {{

    background: var(--mint);

    color: var(--green-dark);

    font-size: 12px;

    text-transform: uppercase;

    letter-spacing: 0.06em;

}}

.section-title {{

    color: var(--green-dark);

    font-family: 'Space Grotesk', sans-serif;

    margin-top: 10px;

}}

.description {{

    max-width: 900px;

    line-height: 1.7;

    color: var(--muted);

}}

.close-dashboard {{

    position: absolute;

    right: 25px;

    top: 20px;

    border: none;

    background: #dc3545;

    color: white;

    padding: 8px 15px;

    border-radius: 6px;

    cursor: pointer;

}}

@media (max-width: 700px) {{
    #dashboard-header {{ height: 90px; padding: 14px 18px; }}
    #dashboard-header h1 {{ font-size: 19px; text-align: center; line-height: 1.15; }}
    #dashboard-nav {{ top: 90px; height: 48px; padding: 0 8px; overflow-x: auto; }}
    .nav-button {{ flex: 0 0 auto; padding: 14px 10px; font-size: 12px; }}
    .dashboard-panel {{ top: 138px; padding: 24px 16px; }}
    .leaflet-control-layers {{ margin-top: 165px !important; margin-right: 10px !important; }}
}}

</style>

<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.2/css/all.min.css">

<div id="dashboard-header">

    <h1>
        <i class="fa-solid fa-leaf" aria-hidden="true"></i>
        Spatiotemporal Environmental Zoning of Rural Landscapes Using GIS and Unsupervised Machine Learning
    </h1>

</div>
<div id="dashboard-nav">

    <button
        class="nav-button"
        onclick="showPanel('overview')">

        <i class="fa-solid fa-chart-line" aria-hidden="true"></i> Overview

    </button>
    <button
        class="nav-button"
        onclick="showPanel('environment')">

        <i class="fa-solid fa-seedling" aria-hidden="true"></i> Environment

    </button>
    <button
        class="nav-button"
        onclick="showPanel('ml')">

        <i class="fa-solid fa-object-group" aria-hidden="true"></i> Zoning Model

    </button>
    <button
        class="nav-button"
        onclick="showPanel('about')">

        <i class="fa-solid fa-circle-info" aria-hidden="true"></i> About

    </button>
    <button
        class="nav-button"
        onclick="closePanel()">

        <i class="fa-solid fa-map" aria-hidden="true"></i> Map

    </button>

</div>
<div
    id="overview"
    class="dashboard-panel">

    <h2 class="section-title">
        Landscape Overview
    </h2>

    <div class="card-container">

        <div class="stat-card">

            <h3>
                OBSERVATIONS
            </h3>

            <div class="value">
                {total_points:,}
            </div>

        </div>


        <div class="stat-card">

            <h3>
                ENVIRONMENTAL ZONES
            </h3>

            <div class="value">
                {number_zones}
            </div>

        </div>


        <div class="stat-card">

            <h3>
                MEAN NDVI
            </h3>

            <div class="value">
                {mean_ndvi:.3f}
            </div>

        </div>


        <div class="stat-card">

            <h3>
                MEAN LST
            </h3>

            <div class="value">
                {mean_lst:.1f} °C
            </div>

        </div>

    </div>


    <h3>
        Environmental Zone Distribution
    </h3>


    <table>

        <tr>

            <th>
                Environmental Zone
            </th>

            <th>
                Observations
            </th>

            <th>
                Percentage
            </th>

        </tr>


        {"".join(

            f'''

            <tr>

                <td>
                    {cls}
                </td>

                <td>
                    {count}
                </td>

                <td>
                    {(count / total_points * 100):.1f}%
                </td>

            </tr>

            '''

            for cls, count
            in zone_counts.items()

        )}

    </table>

</div>
<div
    id="environment"
    class="dashboard-panel">

    <h2 class="section-title">
        Environmental Indicators
    </h2>

    <p class="description">

        The dashboard combines environmental and
        spatial indicators including Normalized
        Difference Vegetation Index (NDVI),
        Land Surface Temperature (LST), and
        soil moisture, vegetation, temperature,
        and seasonal signals to identify
        environmental zones.

    </p>
    <div class="card-container">

        <div class="stat-card">

            <h3>
                NDVI
            </h3>

            <div class="value">
                {mean_ndvi:.3f}
            </div>

        </div>


        <div class="stat-card">

            <h3>
                LAND SURFACE TEMPERATURE
            </h3>

            <div class="value">
                {mean_lst:.1f} °C
            </div>

        </div>


        <div class="stat-card">

            <h3>
                SOIL MOISTURE
            </h3>

            <div class="value">
                {df['soil_moisture'].mean():.3f}
            </div>

        </div>

    </div>

</div>

<div
    id="ml"
    class="dashboard-panel">

    <h2 class="section-title">
        Environmental Zoning Model
    </h2>


    <div class="card-container">

        <div class="stat-card">

            <h3>
                MODEL
            </h3>

            <div class="value"
                 style="font-size:20px;">

                K-Means Clustering

            </div>

        </div>


        <div class="stat-card">

            <h3>
                SILHOUETTE SCORE
            </h3>

            <div class="value">

                {silhouette:.3f}

            </div>

        </div>


        <div class="stat-card">

            <h3>
                ZONES SELECTED
            </h3>

            <div class="value">

                {number_zones}

            </div>

        </div>


        <div class="stat-card">

            <h3>
                OBSERVATIONS
            </h3>

            <div class="value">

                {total_points:,}

            </div>

        </div>

    </div>


    <h3>
        Zoning Features
    </h3>

    <ul>

        <li>
            Soil Moisture
        </li>

        <li>
            Land Surface Temperature
        </li>

        <li>
            NDVI
        </li>

        <li>
            LST
        </li>

        <li>
            Seasonal Month Signals
        </li>

        <li>
            Standardized Environmental Variables
        </li>

    </ul>

</div>

<div
    id="about"
    class="dashboard-panel">

    <h2 class="section-title">
        About This Project
    </h2>


    <p class="description">

        This project identifies environmental zones
        across a rural landscape using spatial,
        seasonal, and remote-sensing observations.

        K-Means clustering groups observations with
        similar soil moisture, vegetation, temperature,
        and seasonal profiles. Existing land-cover
        labels are retained as reference information,
        not as the prediction target.

    </p>


    <h3>
        Technology Stack
    </h3>

    <p class="description">

        Python • Pandas • Scikit-learn • Folium
        • Matplotlib • GIS • Remote Sensing

    </p>


    <h3>
        Interactive Map
    </h3>

    <p class="description">

        Use the layer control on the map to switch
        environmental zones on and off. Click
        individual points to inspect their sensor,
        environmental, seasonal, and reference-label
        attributes.

    </p>

</div>
<script>
function showPanel(panelId) {{

    document
        .querySelectorAll(
            '.dashboard-panel'
        )
        .forEach(
            function(panel) {{

                panel.classList.remove(
                    'active'
                );

            }}
        );


    document
        .getElementById(panelId)
        .classList.add(
            'active'
        );

}}


function closePanel() {{

    document
        .querySelectorAll(
            '.dashboard-panel'
        )
        .forEach(
            function(panel) {{

                panel.classList.remove(
                    'active'
                );

            }}
        );

}}

</script>

"""
m.get_root().html.add_child(

    Element(
        dashboard_html
    )

)
m.save(

    "results/rural_landscape_dashboard.html"

)

joblib.dump(

    model,

    "results/kmeans_environmental_zoning_model.pkl"

)

joblib.dump(

    scaler,

    "results/environmental_feature_scaler.pkl"

)
