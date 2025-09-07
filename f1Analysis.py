import fastf1
import pandas as pd
import numpy as np
import mesures
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN  
from itertools import product
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

class f1Analysis:
    @staticmethod
    def get_data(year, event, session_name):
        session = fastf1.get_session(year, event, session_name)
        session.load()
        laps = session.laps
        laps['LapTime'] = laps['LapTime']
        laps['Sector1Time'] = laps['Sector1Time'].dt.total_seconds()
        laps['Sector2Time'] = laps['Sector2Time'].dt.total_seconds()
        laps['Sector3Time'] = laps['Sector3Time'].dt.total_seconds()
        laps['LapsInStint'] = laps.groupby(['DriverNumber', 'Stint'])['LapNumber'].transform('count')
        # Mark OutLap as the lap where PitOutTime is not null (driver just exited the pits)
        laps['OutLap'] = laps['PitOutTime'].notna()
        # Mark InLap as the lap where PitInTime is not null (driver is about to enter the pits)
        laps['InLap'] = laps['PitInTime'].notna()
        # We compute the fuel level of the number of laps left in the stint
        laps['FuelLevel'] = laps['LapsInStint'] - (laps['LapNumber'] - laps.groupby(['DriverNumber', 'Stint'])['LapNumber'].transform('min'))
        # Map compounds to numerical values, use -1 for any other/unexpected value
        return laps

    @staticmethod
    def get_free_practice(year, event):
        sessions = ['FP1', 'FP2', 'FP3']
        all_data = []
        for session_name in sessions:
            session_data = f1Analysis.get_data(year, event, session_name)
            session_data['Session'] = session_name  # Optional: tag the session
            all_data.append(session_data)
        # Combine all sessions into one DataFrame
        return pd.concat(all_data, ignore_index=True)
    
    # Tuning DBSCAN parameters based on how well we classify the biggest cluster
    @staticmethod
    def tune_DBSCAN(features, data):
        # Remove inlaps and outlaps, drop NaNs in selected features
        data_clean = data[~data['InLap'] & ~data['OutLap']].dropna(subset=features)
        X = data_clean[features].astype(float).values
        X_scaled = StandardScaler().fit_transform(X)

        eps_values = np.arange(0.1, 1.5, 0.1)
        min_samples_values = range(3, 10)
        results = []

        for eps, min_samples in product(eps_values, min_samples_values):
            db = DBSCAN(eps=eps, min_samples=min_samples, metric="chebyshev")
            labels = db.fit_predict(X_scaled)
            
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)     
            score = mesures.biggest_cluster_quality_score(X_scaled, labels)
            results.append({
                'eps': eps,
                'min_samples': min_samples,
                'clean_cluster_score': score,
                'n_clusters': n_clusters
            })

        results_df = pd.DataFrame(results)
        best = results_df.loc[results_df['clean_cluster_score'].idxmax()]
        print("Best Params according to biggest cluster quality score:")
        print(best)
        return best

    @staticmethod
    def perform_dbscan(params, data, features, column_name='DbscanCluster'):
        data_clean = data[~data['InLap'] & ~data['OutLap']].dropna(subset=features)
        best_eps = params['eps']
        best_min_samples = int(params['min_samples'])
        X = data_clean[features].astype(float).values
        X_scaled = StandardScaler().fit_transform(X)
        # Run DBSCAN
        dbscan_best = DBSCAN(eps=best_eps, min_samples=best_min_samples, metric="chebyshev")
        clusters = dbscan_best.fit_predict(X_scaled)
        # Assign cluster labels back to the clean subset in original data
        data.loc[data_clean.index, column_name] = clusters
        print(data[column_name].value_counts())
        return data
    
    @staticmethod
    def plot_sectors(data, cluster_column):
        # Create a new array: 0 for biggest cluster, others become 1
        unique_labels, counts = np.unique(data[cluster_column], return_counts=True)
        biggest_cluster_label = unique_labels[np.argmax(counts)]
        cluster_binary = np.where(data[cluster_column] == biggest_cluster_label, 0, 1)
        
        cmap = ListedColormap(['#1f77b4', '#ff7f0e'])  # blue for 0, orange for others
        norm = BoundaryNorm(boundaries=[-0.5, 0.5, 1.5], ncolors=2)

        # 3D scatter plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        sc = ax.scatter(
            data['Sector1Time'],
            data['Sector2Time'],
            data['Sector3Time'],
            c=cluster_binary,
            cmap=cmap,
            norm=norm,
            s=70,
            alpha=0.8,
            edgecolor='black',
            linewidth=0.5
        )

        # Axis labels
        ax.set_xlabel('Sector 1 Time (s)', fontsize=12, labelpad=5)
        ax.set_ylabel('Sector 2 Time (s)', fontsize=12, labelpad=5)
        ax.set_zlabel('Sector 3 Time (s)', fontsize=12, labelpad=5)

        # Set a better view angle
        ax.view_init(elev=30, azim=135)

        # Add a colorbar with binary cluster labels
        cbar = fig.colorbar(sc, ax=ax, pad=0.2, shrink=0.7, ticks=[0, 1])
        cbar.ax.set_yticklabels(['Clean Laps', 'Unclean Laps'])

        ax.grid(True)
        plt.show()

    @staticmethod
    def tune_KMeans(features, data, max_clusters=10):
        data_clean = data[~data['InLap'] & ~data['OutLap']].dropna(subset=features)
        X = data_clean[features].astype(float).values
        X_scaled = StandardScaler().fit_transform(X)

        results = []

        for k in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(X_scaled)

            # Use cluster0 quality score instead of silhouette
            score = mesures.biggest_cluster_quality_score(X_scaled, labels)

            results.append({
                'n_clusters': k,
                'cluster0_score': score,
                'inertia': kmeans.inertia_
            })

        results_df = pd.DataFrame(results)
        best = results_df.loc[results_df['cluster0_score'].idxmax()]
        print("Best Params according to biggest cluster quality score:")
        print(best)
        return best
    
    @staticmethod
    def perform_kmeans(params, data, features, column_name='KMeansCluster'):
        data_clean = data[~data['InLap'] & ~data['OutLap']].dropna(subset=features)
        n_clusters = int(params['n_clusters'])

        X = data_clean[features].astype(float).values
        X_scaled = StandardScaler().fit_transform(X)

        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(X_scaled)

        data.loc[data_clean.index, column_name] = clusters
        print(data[column_name].value_counts())

        return data
    
    @staticmethod
    def plot_clean_distributions(data, dbscan_cluster_column, kmeans_cluster_column):
        # Define the clean cluster as the biggest ones
        clean_DBSCAN = data[dbscan_cluster_column].value_counts().idxmax()
        # CLEAN for KMeans
        clean_KMeans = data[kmeans_cluster_column].value_counts().idxmax()
        # Extract LapTimes in seconds
        laptimes_all = data['LapTime'].dt.total_seconds().dropna()
        laptimes_dbscan = data[data[dbscan_cluster_column] == clean_DBSCAN]['LapTime'].dt.total_seconds()
        laptimes_kmeans = data[data[kmeans_cluster_column] == clean_KMeans]['LapTime'].dt.total_seconds()

        # Define common bin edges
        min_time = min(laptimes_all.min(), laptimes_dbscan.min(), laptimes_kmeans.min())
        max_time = max(laptimes_all.max(), laptimes_dbscan.max(), laptimes_kmeans.max())
        bins = np.linspace(min_time, max_time, 50)

        # Compute histograms (density=True for shape comparison)
        hist_all, _ = np.histogram(laptimes_all, bins=bins, density=False)
        hist_dbscan, _ = np.histogram(laptimes_dbscan, bins=bins, density=False)
        hist_kmeans, _ = np.histogram(laptimes_kmeans, bins=bins, density=False)

        # Use bin centers for plotting
        bin_centers = (bins[:-1] + bins[1:]) / 2

        # Plotting
        plt.figure(figsize=(10, 6))

        plt.plot(bin_centers, hist_all, linestyle=':', label='All Laps', color='gray', linewidth=2)
        plt.plot(bin_centers, hist_dbscan, linestyle='--', label='DBSCAN (Clean)', color='tab:blue', linewidth=2)
        plt.plot(bin_centers, hist_kmeans, linestyle='-.', label='KMeans (Clean)', color='tab:orange', linewidth=2)

        # Formatting
        plt.title('Lap Time Distributions (Line Comparison)', fontsize=14)
        plt.xlabel('Lap Time (seconds)', fontsize=12)
        plt.ylabel('Density', fontsize=12)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def compute_sector_variance(data):
        # Ensure required columns are present
        required_cols = ['Sector1Time', 'Sector2Time', 'Sector3Time']
        if not all(col in data.columns for col in required_cols):
            raise ValueError(f"Missing one of the required columns: {required_cols}")

        # Drop rows with NaNs in sector times
        data = data.dropna(subset=required_cols).copy()

        # Extract sector times
        sectors = data[required_cols].values

        # Standardize sector times (mean=0, std=1)
        scaler = StandardScaler()
        standardized_sectors = scaler.fit_transform(sectors)

        # Get standardized values for each sector
        s1_std = standardized_sectors[:, 0]
        s2_std = standardized_sectors[:, 1]
        s3_std = standardized_sectors[:, 2]

        # Compute average delta from theoretical best time (which now is also standardized)
        avg = (s1_std + s2_std + s3_std) / 3

        # Compute variance based on standardized times
        data['SectorVariance'] = (
            (s1_std - avg) ** 2 +
            (s2_std - avg) ** 2 +
            (s3_std - avg) ** 2
        )
        return data