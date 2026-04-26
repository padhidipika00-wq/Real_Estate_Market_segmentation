#Python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
from datetime import datetime

# 1. LOAD & CLEAN
clients = pd.read_csv('clients.csv')
properties = pd.read_csv('properties.csv')
properties['sale_price'] = properties['sale_price'].str.replace('$', '', regex=False).str.replace(',', '', regex=False).astype(float)

# 2. FEATURE ENGINEERING
client_inv = properties.groupby('client_ref').agg(
    total_investment=('sale_price', 'sum'),
    num_properties=('listing_id', 'count'),
    avg_floor_area=('floor_area_sqft', 'mean')
).reset_index()

df = pd.merge(clients, client_inv, left_on='client_id', right_on='client_ref', how='left').fillna(0)

def get_age(dob):
    fmt = '%m-%d-%Y' if '-' in dob else '%m/%d/%Y'
    return datetime.now().year - datetime.strptime(dob, fmt).year

df['age'] = df['date_of_birth'].apply(get_age)

# 3. PREPROCESSING
cat_cols = ['client_type', 'region', 'acquisition_purpose', 'referral_channel', 'loan_applied']
num_cols = ['age', 'satisfaction_score', 'total_investment', 'num_properties']

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_cols),
    ('cat', OneHotEncoder(drop='first', sparse_output=False), cat_cols)
])

X = preprocessor.fit_transform(df)

# 4. CLUSTERING
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X)

# 5. MAPPING SEGMENTS
# (Logic based on data analysis in Step 1)
mapping = {1: 'Global Investors', 0: 'First-Time Buyers', 3: 'Corporate Buyers', 2: 'Luxury Investors'}
df['Buyer_Segment'] = df['Cluster'].map(mapping)

# 6. SAVE ASSETS
df.to_csv('final_buyer_segmentation.csv', index=False)
print("Pipeline Complete: final_buyer_segmentation.csv generated.")