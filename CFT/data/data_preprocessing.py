# -*- coding: utf-8 -*-
"""
Data Preprocessing Module for Groundwater Prediction.

Functionality:
1. Feature normalization (separate normalization for groundwater features, MinMax for others)
2. Hydrological clustering (optional)
3. Train/Val/Test dataset splitting
4. Multi-channel feature preparation
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import pickle
import os
from scipy import stats
import warnings

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    CARTOPY_AVAILABLE = True
except ImportError:
    print("Warning: Cartopy not available. Spatial visualization will be skipped.")
    CARTOPY_AVAILABLE = False


# ============== Feature Category Configuration ==============
FEATURE_CATEGORIES = {
    'groundwater': ['Groundwater_'],
    'temperature': ['tasmax', 'tasmin_', 'tas_'],
    'precipitation': ['precipitation_'],
    'population': ['population_'],
    'evapotranspiration': ['E_'],
    'drought_index': ['pdsi_'],
    'humidity': ['q_'],
    'soil_moisture': ['soil_']
}

# Reverse mapping: prefix -> category
PREFIX_TO_CATEGORY = {}
for category, prefixes in FEATURE_CATEGORIES.items():
    for prefix in prefixes:
        PREFIX_TO_CATEGORY[prefix] = category


class GroundwaterDataPreprocessor:
    """
    Groundwater Data Preprocessor.

    Provides comprehensive data preprocessing pipeline for groundwater level prediction,
    including feature normalization, hydrological clustering, and multi-channel feature preparation.
    """

    # Feature prefix list (for grid-based processing)
    FEATURE_PREFIXES = [
        'Groundwater_', 'population_', 'precipitation_', 'tasmax', 
        'tasmin_', 'tas_', 'E_', 'pdsi_', 'q_', 'soil_'
    ]

    def __init__(self, target_name='Groundwater_', n_clusters=5):
        self.target_name = target_name
        self.n_clusters = n_clusters

        # Normalizers
        self.scaler_dynamic = None
        self.scaler_spatial = None
        self.cluster_scaler = None
        self.feature_normalizers = {}  # Feature-specific normalization parameters
        self.normalization_metadata = {}

        # Clustering results
        self.kmeans_model = None
        self.cluster_labels = None
        self.cluster_site_indices = None
        self.cluster_bounds = None
        self.cluster_env_profiles = None

        # Anomaly detection
        self.anomalous_sites = []
        self.outlier_records = []

        # Dataset split
        self.train_split_year = None
        self.val_split_year = None

        # Clustering parameters
        self.clustering_random_state = 42
        self.clustering_n_init = 10

    # ============== Utility Methods ==============

    @staticmethod
    def build_feature_column_name(feature_prefix, year):
        """Unified feature column name builder."""
        if feature_prefix == 'tasmax':
            return f"{feature_prefix}{year}"
        elif feature_prefix.endswith('_'):
            return f"{feature_prefix}{year}"
        else:
            return f"{feature_prefix}_{year}"

    def get_dataset_type_for_year(self, year):
        """Determine dataset type based on year."""
        if self.train_split_year is None:
            return 'train'
        if year <= self.train_split_year:
            return 'train'
        elif self.val_split_year is not None and year <= self.val_split_year:
            return 'val'
        else:
            return 'test'

    def get_train_validation_test_splits(self, years):
        """Get year splits for train/validation/test sets."""
        train_years = [y for y in years if self.get_dataset_type_for_year(y) == 'train']
        val_years = [y for y in years if self.get_dataset_type_for_year(y) == 'val']
        test_years = [y for y in years if self.get_dataset_type_for_year(y) == 'test']
        return train_years, val_years, test_years

    def _get_feature_category(self, feature_name):
        """Get feature category."""
        for prefix, category in PREFIX_TO_CATEGORY.items():
            if feature_name.startswith(prefix):
                return category
        return 'other'

    def _extract_feature_prefix(self, feature_name):
        """Extract prefix from feature name."""
        for prefix in sorted(PREFIX_TO_CATEGORY.keys(), key=len, reverse=True):
            if feature_name.startswith(prefix):
                return prefix
        return feature_name[:-4] if feature_name[-4:].isdigit() else feature_name

    # ============== Clustering Methods ==============

    def set_clustering_params(self, random_state=None, n_init=None):
        """Set clustering parameters."""
        if random_state is not None:
            self.clustering_random_state = random_state
        if n_init is not None:
            self.clustering_n_init = n_init
        print(f"Clustering params: random_state={self.clustering_random_state}, n_init={self.clustering_n_init}")

    def perform_clustering(self, data, years, train_split_year):
        """
        Perform hydrological fingerprint clustering.

        Args:
            data: Raw DataFrame
            years: List of years
            train_split_year: Training set split year

        Returns:
            clustering_features: Clustering feature matrix
        """
        if self.n_clusters <= 0:
            print("Clustering disabled (n_clusters=0)")
            self.cluster_labels = None
            self.cluster_site_indices = None
            self.cluster_bounds = {}
            return None

        print(f"[Cluster] Performing hydrological fingerprint clustering K={self.n_clusters}...")

        # Build clustering features
        clustering_features, site_indices = self._build_clustering_features(
            data, years, train_split_year
        )

        if len(clustering_features) == 0:
            print("[Cluster] Warning: Insufficient valid sites, using geographic clustering")
            return self._fallback_geographic_clustering(data, train_split_year)

        # Standardize and cluster
        self.cluster_scaler = StandardScaler()
        features_scaled = self.cluster_scaler.fit_transform(clustering_features)

        self.kmeans_model = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.clustering_random_state,
            n_init=self.clustering_n_init
        )
        self.cluster_labels = self.kmeans_model.fit_predict(features_scaled)

        # Evaluate clustering quality
        silhouette_avg = silhouette_score(features_scaled, self.cluster_labels)
        print(f"[Cluster] Clustering complete. Silhouette score: {silhouette_avg:.3f}")

        # Analyze cluster characteristics
        self._analyze_clusters(clustering_features, site_indices, data)
        self.cluster_site_indices = site_indices

        # Save cluster environmental profiles
        self._save_cluster_profiles(data, years, train_split_year)

        return clustering_features

    def _build_clustering_features(self, data, years, train_split_year):
        """Build clustering feature matrix."""
        # Extract year list
        year_cols = {col[-4:]: col for col in data.columns if col[-4:].isdigit()}
        years_sorted = sorted(year_cols.keys())

        # Feature prefix mapping
        prefix_mapping = {
            'tasmax': 'temp', 'tasmin_': 'temp', 'tas_': 'temp',
            'precipitation_': 'precip', 'population_': 'pop',
            'E_': 'evap', 'pdsi_': 'pdsi', 'q_': 'humid', 'soil_': 'soil'
        }

        feature_names = ['lon', 'lat', 'temp', 'precip', 'pop', 'evap',
                        'pdsi', 'humid', 'soil', 'gw_mean', 'gw_cv', 'gw_range']

        clustering_features = []
        site_indices = []

        for idx in range(len(data)):
            features = []

            # 1. Spatial information
            lat = data.at[idx, 'Latitude'] if 'Latitude' in data.columns else 0
            lon = data.at[idx, 'Longitude'] if 'Longitude' in data.columns else 0
            features.extend([lon, lat])

            # 2. Environmental features
            env_values = {k: [] for k in prefix_mapping.values()}
            for yr in years_sorted:
                yr_int = int(yr)
                if yr_int <= train_split_year:
                    for prefix, key in prefix_mapping.items():
                        col_name = self.build_feature_column_name(prefix, yr)
                        if col_name in data.columns:
                            val = data.at[idx, col_name]
                            if not pd.isna(val):
                                env_values[key].append(val)

            for key in prefix_mapping.values():
                features.append(np.mean(env_values[key]) if env_values[key] else 0.0)

            # 3. Hydrological background
            gw_values = []
            for yr in years_sorted:
                yr_int = int(yr)
                if yr_int <= train_split_year:
                    gw_col = f"Groundwater_{yr}"
                    if gw_col in data.columns:
                        val = data.at[idx, gw_col]
                        if not pd.isna(val):
                            gw_values.append(val)

            if gw_values:
                gw_mean = np.mean(gw_values)
                gw_std = np.std(gw_values) if len(gw_values) > 1 else 0.0
                gw_cv = gw_std / (abs(gw_mean) + 0.1)
                gw_range = np.max(gw_values) - np.min(gw_values) if len(gw_values) > 1 else 0.0
            else:
                gw_mean, gw_cv, gw_range = 0.0, 0.0, 0.0

            features.extend([gw_mean, gw_cv, gw_range])

            # Validity check
            min_required = max(1, len(gw_values) // 2)
            if len(gw_values) >= min_required and len(features) >= 5:
                clustering_features.append(features)
                site_indices.append(idx)

        print(f"[Cluster] Clustering features: {len(clustering_features)} sites x {len(feature_names)} features")
        return np.array(clustering_features), site_indices

    def _analyze_clusters(self, features, site_indices, data):
        """Analyze cluster characteristics."""
        self.cluster_bounds = {}
        feature_names = ['lon', 'lat', 'temp', 'precip', 'pop', 'evap',
                        'pdsi', 'humid', 'soil', 'gw_mean', 'gw_cv', 'gw_range']

        for cluster_id in range(self.n_clusters):
            mask = self.cluster_labels == cluster_id
            cluster_feats = features[mask]

            if len(cluster_feats) == 0:
                continue

            center_lat = np.mean(cluster_feats[:, 1])
            center_lon = np.mean(cluster_feats[:, 0])
            gw_mean = np.mean(cluster_feats[:, 9]) if cluster_feats.shape[1] > 9 else 0

            name = self._infer_region_name(center_lat, center_lon, gw_mean)

            stats = {}
            for i, fname in enumerate(feature_names):
                if i < cluster_feats.shape[1]:
                    stats[fname] = {'mean': float(np.mean(cluster_feats[:, i])),
                                   'std': float(np.std(cluster_feats[:, i]))}

            self.cluster_bounds[f'Cluster_{cluster_id}'] = {
                'name': name, 'count': len(cluster_feats),
                'center_lat': float(center_lat), 'center_lon': float(center_lon),
                'gw_mean': float(gw_mean), 'stats': stats
            }
            print(f"[Cluster] {name}: {len(cluster_feats)} sites")

    def _infer_region_name(self, lat, lon, gw_mean):
        """Infer region name based on geographic coordinates."""
        if lat < -25:
            return 'Tropical North' if lon < 135 else 'Northern Arid Zone'
        elif lat < -30:
            return 'Central Arid Zone' if lon > 145 or lon < 120 else 'Murray-Darling Basin'
        elif lat < -35:
            return 'Southeast' if lon > 145 else ('Southwest' if lon < 120 else 'Eastern Coast')
        else:
            return 'Southern Temperate'

    def _fallback_geographic_clustering(self, data, train_split_year):
        """Geographic clustering fallback method."""
        print("[Cluster] Using geographic clustering fallback")
        coords = data[['Latitude', 'Longitude']].values

        self.cluster_scaler = MinMaxScaler()
        coords_scaled = self.cluster_scaler.fit_transform(coords)

        self.kmeans_model = KMeans(n_clusters=self.n_clusters, random_state=self.clustering_random_state)
        self.cluster_labels = self.kmeans_model.fit_predict(coords_scaled)

        self.cluster_site_indices = list(range(len(data)))
        self._analyze_clusters(np.hstack([coords, np.zeros((len(coords), 10))]),
                               self.cluster_site_indices, data)
        return coords

    def _save_cluster_profiles(self, data, years, train_split_year):
        """Save cluster environmental profiles."""
        self.cluster_env_profiles = {}

        for cluster_id in range(self.n_clusters):
            mask = self.cluster_labels == cluster_id
            cluster_sites = np.array(self.cluster_site_indices)[mask]

            profile = {
                'n_sites': len(cluster_sites),
                'lat_range': (data.iloc[cluster_sites]['Latitude'].min(),
                             data.iloc[cluster_sites]['Latitude'].max()),
                'lon_range': (data.iloc[cluster_sites]['Longitude'].min(),
                             data.iloc[cluster_sites]['Longitude'].max())
            }
            self.cluster_env_profiles[f'Cluster_{cluster_id}'] = profile

        print(f"[Cluster] Saved {len(self.cluster_env_profiles)} cluster profiles")

    # ============== Normalization Methods ==============

    def normalize_data(self, data, years):
        """
        Feature normalization processing.

        Returns:
            dynamic_data_normalized: Normalized dynamic features
            spatial_data_normalized: Normalized spatial coordinates
            spatial_data_original: Original spatial coordinates
            dynamic_features: Dynamic feature list
        """
        print("Starting feature normalization...")

        # Extract dynamic features
        dynamic_features = [col for col in data.columns if col[-4:].isdigit()]
        print(f"  Dynamic features count: {len(dynamic_features)}")

        # Categorize features
        categories = {cat: [] for cat in FEATURE_CATEGORIES.keys()}
        categories['other'] = []

        for feat in dynamic_features:
            cat = self._get_feature_category(feat)
            categories[cat].append(feat)

        for cat, feats in categories.items():
            if feats:
                print(f"  {cat}: {len(feats)} features")

        # Compute training year indices
        train_year_indices = [i for i, yr in enumerate(years) if yr <= self.train_split_year]

        # Initialize
        self.feature_normalizers = {}
        self.normalization_metadata = {'feature_strategies': {}}

        # Normalize dynamic features
        dynamic_data = data[dynamic_features].values
        dynamic_data_normalized = np.zeros_like(dynamic_data)

        for i, feat in enumerate(dynamic_features):
            feat_data = dynamic_data[:, i]
            category = self._get_feature_category(feat)

            # Collect training data
            train_data = self._collect_train_data(feat, data, years, train_year_indices)

            # Normalize
            normalized, params = self._normalize_feature(feat_data, train_data, category, feat)
            dynamic_data_normalized[:, i] = normalized
            self.feature_normalizers[feat] = params
            self.normalization_metadata['feature_strategies'][feat] = params

        # Normalize spatial features
        spatial_features = ['Latitude', 'Longitude']
        self.scaler_spatial = MinMaxScaler()
        spatial_data = data[spatial_features].values
        spatial_data_original = spatial_data.copy()
        spatial_data_normalized = self.scaler_spatial.fit_transform(spatial_data)

        print(f"Normalization complete: {len(dynamic_features)} features")
        return dynamic_data_normalized, spatial_data_normalized, spatial_data_original, dynamic_features

    def _collect_train_data(self, feature_name, data, years, train_year_indices):
        """Collect training data for computing normalization parameters."""
        prefix = self._extract_feature_prefix(feature_name)
        train_data_list = []

        for train_idx in train_year_indices:
            train_year = years[train_idx]
            col_name = self.build_feature_column_name(prefix, train_year)
            if col_name in data.columns:
                values = data[col_name].values
                train_data_list.extend(values[~np.isnan(values)])

        return np.array(train_data_list) if train_data_list else np.array([])

    def _normalize_feature(self, feature_data, train_data, category, feature_name):
        """
        Unified normalization method.

        Args:
            feature_data: Current feature data
            train_data: Training data
            category: Feature category
            feature_name: Feature name

        Returns:
            normalized: Normalized data
            params: Normalization parameters
        """
        # Handle empty data
        if len(train_data) == 0 or np.all(np.isnan(train_data)):
            return np.zeros_like(feature_data), {'method': 'zero_fill'}

        valid_train = train_data[~np.isnan(train_data)]
        if len(valid_train) == 0:
            return np.zeros_like(feature_data), {'method': 'zero_fill'}

        # Category-specific normalization
        if category == 'groundwater':
            return self._normalize_groundwater(feature_data, valid_train)
        elif category in ['temperature', 'drought_index']:
            return self._normalize_with_negatives(feature_data, valid_train)
        else:
            return self._minmax_normalize(feature_data, valid_train)

    def _normalize_groundwater(self, feature_data, train_data):
        """
        Groundwater normalization: positive values [0,1], negative values [-1,0].
        """
        positive = train_data[train_data >= 0]
        negative = train_data[train_data < 0]

        pos_min = float(np.min(positive)) if len(positive) > 0 else 0.0
        pos_max = float(np.max(positive)) if len(positive) > 0 else 100.0

        neg_min = float(np.min(negative)) if len(negative) > 0 else -20.0
        neg_max = float(np.max(negative)) if len(negative) > 0 else -0.001
        if neg_max >= -0.001:
            neg_max = -0.001

        normalized = np.zeros_like(feature_data)

        # Positive values -> [0, 1]
        pos_mask = feature_data >= 0
        if np.any(pos_mask) and pos_max > pos_min:
            normalized[pos_mask] = (feature_data[pos_mask] - pos_min) / (pos_max - pos_min)

        # Negative values -> [-1, 0)
        neg_mask = feature_data < 0
        if np.any(neg_mask) and neg_max > neg_min:
            neg_values = feature_data[neg_mask]
            neg_norm = (neg_values - neg_min) / (neg_max - neg_min) - 1
            normalized[neg_mask] = np.clip(neg_norm, -1.0, -1e-10)

        return normalized, {
            'method': 'separate_groundwater',
            'positive': {'min': pos_min, 'max': pos_max},
            'negative': {'min': neg_min, 'max': neg_max}
        }

    def _normalize_with_negatives(self, feature_data, train_data):
        """Separate normalization supporting negative values."""
        neg_ratio = np.sum(train_data < 0) / len(train_data)

        if neg_ratio > 0.05:  # Use separate normalization when negatives > 5%
            return self._separate_normalize(feature_data, train_data)
        return self._minmax_normalize(feature_data, train_data)

    def _separate_normalize(self, feature_data, train_data):
        """Separate normalization (positive and negative values handled independently)."""
        positive = train_data[train_data >= 0]
        negative = train_data[train_data < 0]

        normalized = np.zeros_like(feature_data)
        params = {'method': 'separate', 'positive': {}, 'negative': {}}

        # Positive values
        if len(positive) > 0:
            p_min, p_max = float(np.min(positive)), float(np.max(positive))
            if p_max > p_min:
                mask = feature_data >= 0
                normalized[mask] = (feature_data[mask] - p_min) / (p_max - p_min)
            params['positive'] = {'min': p_min, 'max': p_max}

        # Negative values
        if len(negative) > 0:
            n_min, n_max = float(np.min(negative)), float(np.max(negative))
            if n_max > n_min:
                mask = feature_data < 0
                normalized[mask] = (feature_data[mask] - n_min) / (n_max - n_min) - 1
                normalized[mask] = np.clip(normalized[mask], -1.0, -1e-10)
            params['negative'] = {'min': n_min, 'max': n_max}

        return normalized, params

    def _minmax_normalize(self, feature_data, train_data):
        """Standard MinMax normalization to [0,1]."""
        d_min, d_max = float(np.min(train_data)), float(np.max(train_data))

        if d_max > d_min:
            normalized = (feature_data - d_min) / (d_max - d_min)
        else:
            normalized = np.ones_like(feature_data) * 0.5

        return normalized, {'method': 'minmax', 'min': d_min, 'max': d_max}

    def inverse_normalize(self, normalized_data, feature_name):
        """
        Inverse normalization.

        Args:
            normalized_data: Normalized data
            feature_name: Feature name

        Returns:
            original_scale: Data in original scale
        """
        if feature_name not in self.feature_normalizers:
            return normalized_data  # Cannot inverse normalize, return original

        params = self.feature_normalizers[feature_name]
        method = params.get('method', 'minmax')

        if method == 'minmax':
            d_min, d_max = params['min'], params['max']
            return normalized_data * (d_max - d_min) + d_min

        elif method == 'separate_groundwater':
            pos = params['positive']
            neg = params['negative']

            original = np.zeros_like(normalized_data)

            # [0, 1] -> Positive range
            pos_mask = normalized_data >= 0
            if np.any(pos_mask) and pos['max'] > pos['min']:
                original[pos_mask] = normalized_data[pos_mask] * (pos['max'] - pos['min']) + pos['min']

            # [-1, 0) -> Negative range
            neg_mask = (normalized_data < 0) & (normalized_data >= -1)
            if np.any(neg_mask) and neg['max'] > neg['min']:
                original[neg_mask] = (normalized_data[neg_mask] + 1) * (neg['max'] - neg['min']) + neg['min']

            return original

        return normalized_data

    # ============== Clustering Feature Preparation ==============

    def prepare_cluster_features(self, data_length):
        """Prepare cluster one-hot features."""
        if self.cluster_labels is None:
            return None

        actual_n = max(self.n_clusters, int(np.max(self.cluster_labels)) + 1)
        features = np.zeros((data_length, actual_n))

        for i, site_idx in enumerate(self.cluster_site_indices):
            cid = self.cluster_labels[i]
            if cid < actual_n:
                features[site_idx, cid] = 1.0
            else:
                features[site_idx] = 1.0 / actual_n

        # Unclustered sites
        clustered = set(self.cluster_site_indices) if self.cluster_site_indices else set()
        for idx in range(data_length):
            if idx not in clustered:
                features[idx] = 1.0 / actual_n

        return features

    # ============== Multi-Channel Features ==============

    def prepare_multichannel_features(self, data, years):
        """Prepare multi-channel features."""
        print("Preparing multi-channel features...")

        available = {}
        for prefix in self.FEATURE_PREFIXES:
            cols = []
            for yr in years:
                col = self.build_feature_column_name(prefix, yr)
                if col in data.columns:
                    cols.append(col)

            if cols:
                available[prefix] = {
                    'columns': cols,
                    'n_years': len(cols),
                    'data': data[cols].values
                }

        print(f"  Available feature types: {list(available.keys())}")

        return {
            'multichannel_available': available,
            'multichannel_order': list(available.keys()),
            'multichannel_data': {k: v['data'] for k, v in available.items()}
        }

    # ============== Anomaly Detection ==============

    def detect_anomalous_sites(self, data):
        """Detect anomalous sites."""
        if self.cluster_labels is None:
            print("No clustering results, skipping anomaly detection")
            return

        print("Detecting anomalous sites...")
        self.anomalous_sites = []

        for i, site_idx in enumerate(self.cluster_site_indices):
            cluster_id = self.cluster_labels[i]
            cluster_name = f'Cluster_{cluster_id}'

            if cluster_name not in self.cluster_bounds:
                continue

            bounds = self.cluster_bounds[cluster_name]

            # Geographic distance
            lat = data.at[site_idx, 'Latitude']
            lon = data.at[site_idx, 'Longitude']
            geo_dist = np.sqrt((lat - bounds['center_lat'])**2 + (lon - bounds['center_lon'])**2)

            # Numerical statistics
            gw_values = []
            for col in data.columns:
                if self.target_name in col and col[-4:].isdigit():
                    val = data.at[site_idx, col]
                    if not pd.isna(val):
                        gw_values.append(val)

            if not gw_values:
                continue

            site_mean = np.mean(gw_values)
            site_std = np.std(gw_values) if len(gw_values) > 1 else 0.0

            # Anomaly detection
            reasons = []
            if geo_dist > 2.0:
                reasons.append(f"Geographic anomaly (distance: {geo_dist:.2f})")
            if site_std > site_mean * 0.8:
                reasons.append(f"Excessive variation (std: {site_std:.2f})")

            if reasons:
                self.anomalous_sites.append({
                    'site_idx': site_idx,
                    'site_id': data.at[site_idx, 'Site ID'] if 'Site ID' in data.columns else site_idx,
                    'cluster': cluster_name,
                    'lat': lat, 'lon': lon,
                    'reasons': reasons,
                    'geo_distance': geo_dist,
                    'site_mean': site_mean, 'site_std': site_std
                })

        print(f"  Detected {len(self.anomalous_sites)} anomalous sites")

    # ============== Main Processing Flow ==============
    
    def process_data(self, data, train_split_ratio=0.8, val_split_ratio=0.1, 
                    enable_multichannel=False):
        """
        Complete data preprocessing pipeline.

        Args:
            data: Raw DataFrame
            train_split_ratio: Training set ratio
            val_split_ratio: Validation set ratio
            enable_multichannel features

        Returns: Enable multi-channel:
            dict: Processing results
        """
        print("=" * 50)
        print("Starting data preprocessing...")
        print("=" * 50)

        # 1. Extract year information
        dynamic_features = [col for col in data.columns if col[-4:].isdigit()]
        years = sorted(set(int(col[-4:]) for col in dynamic_features))
        n_years = len(years)

        # 2. Dataset splitting
        train_end = int(n_years * train_split_ratio) - 1
        val_end = int(n_years * (train_split_ratio + val_split_ratio)) - 1

        self.train_split_year = years[train_end]
        self.val_split_year = years[val_end] if val_end < n_years else None

        print(f"Dataset split: Train {years[0]}-{self.train_split_year}, "
              f"Val {years[train_end+1]}-{self.val_split_year}, "
              f"Test {years[val_end+1]}-{years[-1]}")

        # 3. Save original data
        data_original = data.copy()
        site_ids = data['Site ID'].values if 'Site ID' in data.columns else np.arange(len(data))

        # 4. Clustering (if enabled)
        if self.n_clusters > 0:
            self.perform_clustering(data, years, self.train_split_year)
            self.detect_anomalous_sites(data)
        else:
            print("Clustering disabled")
            self.cluster_labels = None
            self.cluster_site_indices = None
            self.cluster_bounds = {}
        
        # 5. Normalization
        (dynamic_norm, spatial_norm, spatial_orig,
         dyn_features) = self.normalize_data(data, years)

        # 6. Cluster features
        cluster_features = self.prepare_cluster_features(len(data)) if self.n_clusters > 0 else None

        # 7. Multi-channel features
        multichannel_data = None
        if enable_multichannel:
            multichannel_data = self.prepare_multichannel_features(data, years)

        # 8. Dataset splitting
        train_years, val_years, test_years = self.get_train_validation_test_splits(years)

        # 9. Site statistics
        site_means, site_stds = {}, {}
        for idx, site_id in enumerate(site_ids):
            vals = []
            for yr in years:
                if yr <= self.train_split_year:
                    col = self.build_feature_column_name(self.target_name, yr)
                    if col in dyn_features:
                        val = data[col].iloc[idx]
                        if not pd.isna(val):
                            vals.append(val)

            if vals:
                site_means[site_id] = np.mean(vals)
                site_stds[site_id] = np.std(vals) if np.std(vals) > 0 else 1.0
            else:
                site_means[site_id] = 0.0
                site_stds[site_id] = 1.0

        # 10. Build results
        processed_data_norm = data.copy()
        for i, feat in enumerate(dyn_features):
            processed_data_norm[feat] = dynamic_norm[:, i]
        
        result = {
            'processed_data': processed_data_norm,
            'processed_data_original_scale': data,
            'original_data': data_original,
            'dynamic_data_normalized': dynamic_norm,
            'spatial_data_normalized': spatial_norm,
            'spatial_data_original': spatial_orig,
            'cluster_features': cluster_features,
            'site_ids': site_ids,
            'site_means': site_means,
            'site_stds': site_stds,
            'years': years,
            'dynamic_features': dyn_features,
            'train_split_year': self.train_split_year,
            'val_split_year': self.val_split_year,
            'train_years': train_years,
            'val_years': val_years,
            'test_years': test_years,
            'feature_normalizers': self.feature_normalizers,
            'normalization_metadata': self.normalization_metadata,
            'scaler_dynamic': self.scaler_dynamic,
            'scaler_spatial': self.scaler_spatial
        }
        
        if multichannel_data:
            result.update(multichannel_data)

        # Save anomaly records
        if self.outlier_records:
            pd.DataFrame(self.outlier_records).to_csv('outlier_corrections.csv', index=False)
        if self.anomalous_sites:
            pd.DataFrame(self.anomalous_sites).to_csv('anomalous_sites.csv', index=False)

        # Visualization
        if self.n_clusters > 0 and CARTOPY_AVAILABLE:
            self._generate_clustering_viz(data_original)

        print(f"Preprocessing complete: {len(site_ids)} sites, {len(years)} years")
        return result

    def _generate_clustering_viz(self, data_original):
        """Generate clustering visualization."""
        if not CARTOPY_AVAILABLE:
            return

        try:
            fig = plt.figure(figsize=(15, 10))
            ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
            ax.coastlines()
            ax.add_feature(cfeature.BORDERS, linestyle=':')
            ax.set_extent([112, 156, -44, -10])

            colors = ['red', 'blue', 'green', 'orange', 'purple']

            for i, site_idx in enumerate(self.cluster_site_indices):
                cid = self.cluster_labels[i]
                lat = data_original.at[site_idx, 'Latitude']
                lon = data_original.at[site_idx, 'Longitude']
                ax.scatter(lon, lat, color=colors[cid % len(colors)],
                          s=30, alpha=0.7, transform=ccrs.PlateCarree())

            for name, bounds in self.cluster_bounds.items():
                cid = int(name.split('_')[1])
                ax.scatter(bounds['center_lon'], bounds['center_lat'],
                          color=colors[cid % len(colors)], marker='*', s=200,
                          edgecolor='black', transform=ccrs.PlateCarree())

            plt.savefig('clustering_result.png', dpi=150, bbox_inches='tight')
            plt.close()
            print("Clustering visualization saved: clustering_result.png")
        except Exception as e:
            print(f"Visualization generation failed: {e}")

    # ============== Save/Load ==============

    def save_preprocessor(self, filepath):
        """Save preprocessor."""
        state = {
            'target_name': self.target_name,
            'n_clusters': self.n_clusters,
            'scaler_dynamic': self.scaler_dynamic,
            'scaler_spatial': self.scaler_spatial,
            'cluster_scaler': self.cluster_scaler,
            'feature_normalizers': self.feature_normalizers,
            'normalization_metadata': self.normalization_metadata,
            'kmeans_model': self.kmeans_model,
            'cluster_labels': self.cluster_labels,
            'cluster_site_indices': self.cluster_site_indices,
            'cluster_bounds': self.cluster_bounds,
            'cluster_env_profiles': self.cluster_env_profiles,
            'anomalous_sites': self.anomalous_sites,
            'outlier_records': self.outlier_records,
            'train_split_year': self.train_split_year,
            'val_split_year': self.val_split_year,
            'clustering_random_state': self.clustering_random_state,
            'clustering_n_init': self.clustering_n_init
        }

        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        print(f"Preprocessor saved: {filepath}")

    def load_preprocessor(self, filepath):
        """Load preprocessor."""
        with open(filepath, 'rb') as f:
            state = pickle.load(f)

        for key, value in state.items():
            setattr(self, key, value)
        print(f"Preprocessor loaded: {filepath}")
