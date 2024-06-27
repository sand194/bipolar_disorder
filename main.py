import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

df = pd.read_csv("C:/Users/Lenovo/source/repos/bipolar_disorder/data/dataset.csv")
#pd.set_option('display.max_columns', None)

print(df.columns)
#(df.iloc[:, 26])

#Preparing dataset

#Removing timestamp column
df = df.drop(df.columns[0], axis=1)
#print(df)

#Filling names column
def fill_empty_names_with_ordinals(df, column_name="Name"):
    """
    Fills empty values in "Name"

    Parameters:
    df (pandas.DataFrame): dataframe
    column_name (str): name of column (default "Name")

    Returns:
    pandas.DataFrame: DataFrame with filled values
    """
    # Counting missing values
    num_missing = df[column_name].isna().sum()

    # generating numbers in the order
    ordinal_numbers = [f"Name_{i}" for i in range(1, num_missing + 1)]

    # indexes where values are empty
    missing_indices = df[df[column_name].isna()].index

    # filling empty spaces
    for idx, ordinal in zip(missing_indices, ordinal_numbers):
        df.at[idx, column_name] = ordinal

    return df

df = fill_empty_names_with_ordinals(df)

names = df['Name']


#print(df)

df.columns.values[4] = "Are you familiar with the concept of bipolar disorder?"
df.columns = df.columns.str.strip()
#print(df.columns[4])
#print(df["Are you familiar with the concept of bipolar disorder?"].to_string())

print(df.to_string())

replace_dict = {
    'NaN': 1,
    'Threapy': 2,
    'Medication': 3
}

pd.set_option('future.no_silent_downcasting', True)
df['Age Group'] = df['Age Group'].replace({'under 20': 1, '20-25': 2, '25-35': 3, 'above 35': 4}).infer_objects(copy=False)
df['Gender'] = df['Gender'].replace({'Male': 1, 'Female': 2}).infer_objects(copy=False)
df['Employment Status'] = df['Employment Status'].replace({'Student': 1, 'Unemployed': 2, 'Employed': 3}).infer_objects(copy=False)
df['Are you familiar with the concept of bipolar disorder?'] = df['Are you familiar with the concept of bipolar disorder?'].replace({'Not familiar at all': 1, 'Somewhat familiar': 2, 'Very familiar': 3}).infer_objects(copy=False)
df['Have you or someone you know been diagnosed with bipolar disorder?'] = df['Have you or someone you know been diagnosed with bipolar disorder?'].replace({'No': 1, 'Prefer not to say': 2, 'Yes': 3}).infer_objects(copy=False)
df['If yes ,then how long ago?'] = df['If yes ,then how long ago?'].replace({'Never': 1, 'Few years ago': 2, 'Recently': 3}).infer_objects(copy=False)
df['How often do you experience periods of intense Happiness?'] = df['How often do you experience periods of intense Happiness?'].replace({'Rarely': 1, 'Occasionally': 2, 'Frequently': 3, 'Almost constantly': 4}).infer_objects(copy=False)
df['During high-energy periods, do you find yourself excessively active or talkative?'] = df['During high-energy periods, do you find yourself excessively active or talkative?'].replace({'Rarely or never': 1, 'Sometimes': 2, 'Often': 3, 'Always': 4}).infer_objects(copy=False)
df['How would you describe your sleep patterns during mood changes?'] = df['How would you describe your sleep patterns during mood changes?'].replace({'Insomnia': 1, 'Decreased need for sleep': 2, 'No significant change': 3, 'Increased need for sleep': 4}).infer_objects(copy=False)
df['Have you engaged in impulsive or risky activities during elevated mood periods?'] = df['Have you engaged in impulsive or risky activities during elevated mood periods?'].replace({'Rarely or never': 1, 'Occasionally': 2, 'Frequently': 3, 'Always': 4}).infer_objects(copy=False)
df['Do you experience persistent sadness, low energy, or loss of interest in activities?'] = df['Do you experience persistent sadness, low energy, or loss of interest in activities?'].replace({'Rarely or never': 1, 'Sometimes': 2, 'Often': 3, 'Always': 4}).infer_objects(copy=False)
df['How would you rate your ability to concentrate during mood episodes?'] = df['How would you rate your ability to concentrate during mood episodes?'].replace({'Poor': 1, 'Fair': 2, 'Good': 3, 'Excellent': 4}).infer_objects(copy=False)
df['Have your relationships been significantly affected by your mood swings?'] = df['Have your relationships been significantly affected by your mood swings?'].replace({'Not at all': 1, 'Somewhat': 2, 'Moderately': 3, 'Severely': 4}).infer_objects(copy=False)
df['Have you ever had thoughts of self-harm or suicide?'] = df['Have you ever had thoughts of self-harm or suicide?'].replace({'Never': 1, 'Rarely': 2, 'Sometimes': 3, 'Often': 4}).infer_objects(copy=False)
df['Have you sought professional help or treatment for your mood symptoms?'] = df['Have you sought professional help or treatment for your mood symptoms?'].replace({'No': 1, 'Unsure': 2, 'Considering it': 3, 'Yes': 4}).infer_objects(copy=False)
df['How would you describe the overall impact of mood swings on your daily life?'] = df['How would you describe the overall impact of mood swings on your daily life?'].replace({'Minimal': 1, 'Moderate': 2, 'Significant': 3, 'Severe': 4}).infer_objects(copy=False)
df['Have you experienced any change in appetite and weight  during mood episodes?'] = df['Have you experienced any change in appetite and weight  during mood episodes?'].replace({'Minimal': 1, 'Moderate': 2, 'Significant': 3, 'Severe': 4}).infer_objects(copy=False)
df['Have you faced any difficulties in maintaing focused and concentration during extreme emotional state?'] = df['Have you faced any difficulties in maintaing focused and concentration during extreme emotional state?'].replace({'Minimal': 1, 'Moderate': 2, 'Significant': 3, 'Severe': 4}).infer_objects(copy=False)
df['Do you tend to avoid or reduce participation in social activities when experiencing manic episodes?'] = df['Do you tend to avoid or reduce participation in social activities when experiencing manic episodes?'].replace({'Never': 1, 'Occasionally': 2, 'Frequently': 3, 'Always': 4}).infer_objects(copy=False)
df['Are there specific triggers that tend to precede your mood episodes?'] = df['Are there specific triggers that tend to precede your mood episodes?'].replace({'Never': 1, 'Occasionally': 2, 'Frequently': 3, 'Always': 4}).infer_objects(copy=False)
df['Are you currently receiving treatment for bipolar disorder?'] = df['Are you currently receiving treatment for bipolar disorder?'].replace({'No': 1, 'Yes': 2}).infer_objects(copy=False)
df['What type of treatment have you received for bipolar disorder?'] = df['What type of treatment have you received for bipolar disorder?'].fillna('nan')
df['What type of treatment have you received for bipolar disorder?'] = df['What type of treatment have you received for bipolar disorder?'].replace({'nan': 1, 'Threapy': 2, 'Medication':3})
df['How do  your family and friends support  you in managing bipolar disorder?'] = df['How do  your family and friends support  you in managing bipolar disorder?'].replace({'Rarely or never': 1, 'Occasionally': 2, 'Frequently':3, 'Always': 4}).infer_objects(copy=False)
df['Do you feel optimistic(confident) about your future managing bipolar disorder?'] = df['Do you feel optimistic(confident) about your future managing bipolar disorder?'].replace({'No': 1, 'Slightly no': 2, 'Slightly yes':3, 'Yes': 4}).infer_objects(copy=False)
df['How do  your family and friends support  you in managing bipolar disorder?'] = df['How do  your family and friends support  you in managing bipolar disorder?'].replace({'No': 1, 'Slightly no': 2, 'Slightly yes':3, 'Yes':4}).infer_objects(copy=False)
df['Are there any specific goals or improvement you hope to achieve in your journey with bipolar disorder?'] = df['Are there any specific goals or improvement you hope to achieve in your journey with bipolar disorder?'].replace({'No': 1, 'Yes': 2}).infer_objects(copy=False)

print(df.to_string())

df_numeric = df.drop(df.columns[0], axis = 1)

inertia = []
silhouette_scores = []

for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(df_numeric)
    inertia.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(df_numeric, kmeans.labels_))

# Elbow's method
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.plot(range(2, 11), inertia, marker='o')
plt.xlabel('Clusters amount')
plt.ylabel('Innertion')
plt.title('Elbow method')

# Silhouette's method
plt.subplot(1, 2, 2)
plt.plot(range(2, 11), silhouette_scores, marker='o')
plt.xlabel('Clasters amount')
plt.ylabel('Silhouettes index')
plt.title('Silhouette index for different number of clusters')

plt.show()

# KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(df_numeric)

df['Cluster'] = clusters

pca = PCA(n_components=2)
pca_components = pca.fit_transform(df_numeric)

# PCA with names
plt.figure(figsize=(10, 8))
scatter = plt.scatter(pca_components[:, 0], pca_components[:, 1], c=clusters, cmap='viridis')

# Adding names
for i, name in enumerate(names):
    plt.text(pca_components[i, 0], pca_components[i, 1], name, fontsize=8)

plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.title('PCA Clustering Visualization with Names')
plt.colorbar(scatter)
plt.show()