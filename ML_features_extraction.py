import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN



def plot_corr_matrix(df):
    corr_matrix = df.corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.xticks([])
    plt.yticks(fontsize=7)
    plt.title("Features correlation matrix")
    plt.show()


def plot_predict_labels(df):
    x = df['mask_name']
    y = df['Cluster']

    plt.plot(x, y)
    plt.xticks(fontsize=7, rotation=90)
    plt.title('Prediction of the labels')
    plt.show()


def plot_PCA(df, numeric_columns):
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(df[numeric_columns])
    df["PC1"] = pca_result[:, 0]
    df["PC2"] = pca_result[:, 1]
    df["Month"] = [i.split("_")[1] for i in df["mask_name"]]

    plt.figure(figsize=(10, 6))
    palette = sns.color_palette("husl", 12)
    sns.scatterplot(
        x="PC1", y="PC2", hue="Month", data=df, palette=palette, s=100, legend="full"
    )

    plt.title("PCA")
    plt.legend(title="Month")
    plt.show()

    pca_df = pd.DataFrame(pca_result, columns=['PC1', 'PC2'])
    return pca_result, pca_df



file = "extraction_results_Beauvais.csv"
data = pd.read_csv(file)

# Keeping only numerical columns (features)
columns_to_drop = [column for column in data.columns if "diagnostics" in column]
data = data.drop(columns=columns_to_drop)
numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns

# Standardisation
scaler = StandardScaler()
data_scaled = data.copy()
data_scaled[numeric_columns] = scaler.fit_transform(data[numeric_columns])

# Correlated features elimination
#plot_corr_matrix(data_scaled[numeric_columns])
correlated_features_to_drop = [' glszm_ZoneVariance', ' glrlm_RunPercentage']
data_scaled = data_scaled.drop(columns=correlated_features_to_drop)
numeric_columns = data_scaled.select_dtypes(include=['float64', 'int64']).columns
#plot_corr_matrix(data_scaled[numeric_columns])


#================================================
#              K-MEAN
#================================================
X = data_scaled.values 
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(X)
data_scaled['Cluster'] = kmeans.labels_
plot_predict_labels(data_scaled)

"""
#================================================
#                DBSCAN
#================================================
pca_result, pca_df = plot_PCA(data_scaled, numeric_columns)

dbscan = DBSCAN(eps=1, min_samples=5) 
labels = dbscan.fit_predict(pca_result)
pca_df['Cluster'] = labels

plt.figure(figsize=(12, 6))
sns.scatterplot(
    x=pca_df['PC1'], 
    y=pca_df['PC2'], 
    hue=pca_df['Cluster'], 
    palette='tab10', 
    legend="full"
)
plt.title("Clustering DBSCAN")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()
"""


"""
#================================================
#         TESTING THE STANDARDISATION
#               m = 0, std = 1
#================================================

means = data_scaled[numeric_columns].mean()  
stds = data_scaled[numeric_columns].std()    

print("Moyennes des features standardisées :\n", means)
print("\nÉcarts-types des features standardisées :\n", stds)
"""