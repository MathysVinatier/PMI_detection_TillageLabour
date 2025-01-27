import os
import numpy as np
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

    plt.scatter(x, y)
    plt.xticks(fontsize=7, rotation=90)
    plt.title('Prediction of the labels')
    plt.show()


def plot_month_repartition(df):
    month_str = {1:"January", 2:"February", 3:"March", 4:"April", 5:"May", 6:"June", 7:"July", 8:"August", 9:"September", 10:"October", 11:"November", 12:"December"}
    df["Month"] = [int(i.split("_")[1]) for i in df["mask_name"]]
    df['Month Name'] = df['Month'].map(month_str)

    sns.histplot(data=df.sort_values(by='Month', ascending=True), x='Month Name', hue='Cluster', multiple='stack', shrink=0.8, palette='tab10')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_PCA(df, numeric_columns, plot):
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(df[numeric_columns])
    df["PC1"] = pca_result[:, 0]
    df["PC2"] = pca_result[:, 1]
    df["Month"] = [i.split("_")[1] for i in df["mask_name"]]

    if plot == True:
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


def plot_DBSCAN(data_df, pca_df):
    plt.figure(figsize=(12, 6))
    sns.scatterplot(
        x=pca_df['PC1'], 
        y=pca_df['PC2'], 
        hue=data_df['Cluster'], 
        palette='tab10', 
        legend="full"
    )
    plt.title("Clustering DBSCAN")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.show()


def plot_inertia(df):
    inertia = []
    K_list = [i for i in range(1, 11)]
    for k in K_list:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(df)
        inertia.append(kmeans.inertia_)

    plt.figure(figsize=(8, 5))
    plt.plot(K_list, inertia, marker='o')
    plt.xlabel('Clusters')
    plt.ylabel('Inertia')
    plt.show()


def plot_images_per_month(folder, df, month):    
    df["Month"] = [int(i.split("_")[1]) for i in df["mask_name"]]
    df = df[df["Month"] == month]
    num_images = len(df)
    num_row = num_images//5 + 1
    fig, axes = plt.subplots(num_row, 5, figsize=(15, num_row*3)) 
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        if i < len(df):
            current_date_result = df["mask_name"].iloc[i]
            fname = os.path.join(folder, current_date_result + "_gray.png")

            try:
                img = plt.imread(fname)
                ax.imshow(img)
                ax.set_title(f"Date: {current_date_result}")
                ax.axis("off") 
            except FileNotFoundError:
                ax.set_title(f"Image not found for {current_date_result}")
                ax.axis("off")

    plt.tight_layout()
    plt.show()


#================================================
#             FEATURES PREPARATION
#================================================
file = "Features_extraction_analysis/extraction_results_Catillon.csv"
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
correlated_features_to_drop = [' glszm_LargeAreaEmphasis', ' glrlm_LongRunEmphasis'] # To change depending on the dataset
data_scaled = data_scaled.drop(columns=correlated_features_to_drop)
numeric_columns = data_scaled.select_dtypes(include=['float64', 'int64']).columns
#plot_corr_matrix(data_scaled[numeric_columns])


#================================================
#             IMAGES CHECKING
#================================================
#plot_images_per_month("Features_extraction_analysis/grays", data_scaled, 12)


"""
#================================================
#              K-MEAN
#================================================
X = data_scaled[numeric_columns]
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(X)
data_scaled['Cluster'] = kmeans.labels_

#plot_predict_labels(data_scaled)
#plot_month_repartition(data_scaled)
"""

"""
#================================================
#                DBSCAN
#================================================
pca_result, pca_df = plot_PCA(data_scaled, numeric_columns, plot=True)

dbscan = DBSCAN(eps=1.1, min_samples=5) 
labels = dbscan.fit_predict(pca_result)
data_scaled['Cluster'] = labels

#plot_DBSCAN(data_scaled, pca_df)
#plot_month_repartition(data_scaled)
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

"""
#================================================
#             DEPRECATED IMAGES  
#================================================
16_01_20,v3.1.0,1.24.3,2.3.1,1.4.1,3.8.0,"{'minimumROIDimensions': 2, 'minimumROISize': None, 'normalize': False, 'normalizeScale': 1, 'removeOutliers': None, 'resampledPixelSpacing': None, 'interpolator': 'sitkBSpline', 'preCrop': False, 'padDistance': 5, 'distances': [1], 'force2D': False, 'force2Ddimension': 0, 'resegmentRange': None, 'label': 255, 'additionalInfo': True, 'binWidth': 1, 'enableDiagnostics': False}",{'Original': {}},2836e3fe3b14ea9c478186c12d9f6bbd063e23a0,2D,"(1.0, 1.0)","(512, 254)",34.98957308070866,0.0,132.0,fc5b746d5a3eedf45ba85b6ddcf2cfbb513ddce4,"(1.0, 1.0)","(512, 254)","(0, 0, 512, 254)",70540,1,"(217.85083640487667, 142.3380209810037)","(217.85083640487667, 142.3380209810037)",133.3412539278468,0.96435187393676,0.1262407860393172,3.726964946478899,0.12140146528265108,271.8356597000868,938.5111249934494,1299.4097957842864,0.1173235043946697,0.016556040575265933,964121.56,31.368,884507.2944,0.0035440884604479726,5.4100994502506944e-05,32.441174530103346,1815.9558001744515,0.8366071033339388
18_02_20,v3.1.0,1.24.3,2.3.1,1.4.1,3.8.0,"{'minimumROIDimensions': 2, 'minimumROISize': None, 'normalize': False, 'normalizeScale': 1, 'removeOutliers': None, 'resampledPixelSpacing': None, 'interpolator': 'sitkBSpline', 'preCrop': False, 'padDistance': 5, 'distances': [1], 'force2D': False, 'force2Ddimension': 0, 'resegmentRange': None, 'label': 255, 'additionalInfo': True, 'binWidth': 1, 'enableDiagnostics': False}",{'Original': {}},4211693e75638444ad8aba788e3b659702fee4ad,2D,"(1.0, 1.0)","(512, 254)",48.72203340305118,38.0,132.0,ee7ac8279743276cb4e722a34850fde9ed539f55,"(1.0, 1.0)","(512, 254)","(0, 42, 266, 212)",38453,1,"(129.2918107819936, 151.44657113879282)","(129.2918107819936, 151.44657113879282)",122.60381678948383,0.9688018570688984,0.20334890760037896,2.8409289633295334,0.08742763188703578,557.3347090384966,461.75900370895835,264.15853342589764,0.07539723818687749,0.0010076976529301998,3571990.2321428573,12.678571428571429,3100487.2955994895,0.0014563233037734376,8.56615210068277e-05,132.70719002315934,620.0383321178581,0.8809129587360895
12_02_20,v3.1.0,1.24.3,2.3.1,1.4.1,3.8.0,"{'minimumROIDimensions': 2, 'minimumROISize': None, 'normalize': False, 'normalizeScale': 1, 'removeOutliers': None, 'resampledPixelSpacing': None, 'interpolator': 'sitkBSpline', 'preCrop': False, 'padDistance': 5, 'distances': [1], 'force2D': False, 'force2Ddimension': 0, 'resegmentRange': None, 'label': 255, 'additionalInfo': True, 'binWidth': 1, 'enableDiagnostics': False}",{'Original': {}},a626d08099c41ffa31ce2fe7805d14a50c2939b9,2D,"(1.0, 1.0)","(512, 254)",57.53695558562992,57.0,132.0,572005a4cb8412b659c7a6553db7149791d27894,"(1.0, 1.0)","(512, 254)","(410, 129, 102, 74)",3470,1,"(443.11469740634004, 175.0893371757925)","(443.11469740634004, 175.0893371757925)",19.5891328348724,0.961312529054531,0.04248052369545156,5.472775241343011,0.2009355358366559,26.195572007763015,57.45784475299694,263.6373117407216,0.26253602305475504,0.015104214151326388,2542.7916666666665,5.833333333333333,1236.2703993055557,0.0276657060518732,0.0030450340986779515,1.603503256758095,1026.9590375894743,6.569097194793948
10_02_18,v3.1.0,1.24.3,2.3.1,1.4.1,3.8.0,"{'minimumROIDimensions': 2, 'minimumROISize': None, 'normalize': False, 'normalizeScale': 1, 'removeOutliers': None, 'resampledPixelSpacing': None, 'interpolator': 'sitkBSpline', 'preCrop': False, 'padDistance': 5, 'distances': [1], 'force2D': False, 'force2Ddimension': 0, 'resegmentRange': None, 'label': 255, 'additionalInfo': True, 'binWidth': 1, 'enableDiagnostics': False}",{'Original': {}},83f2a21448755e585d430cfc6d7c2b34af5d5b00,2D,"(1.0, 1.0)","(512, 254)",71.59879429133858,0.0,132.0,fc5b746d5a3eedf45ba85b6ddcf2cfbb513ddce4,"(1.0, 1.0)","(512, 254)","(0, 0, 512, 254)",70540,1,"(217.85083640487667, 142.3380209810037)","(217.85083640487667, 142.3380209810037)",0.0,1.0,1.0,-3.203426503814917e-16,0.005528459213541385,39123.70957469084,509.5,6.541031696852523,0.0072228522823929685,2.009690082476877e-10,4975891600.0,1.0,0.0,1.417635384179189e-05,1000000.0,0.0,0.0,0.0
12_07_20,v3.1.0,1.24.3,2.3.1,1.4.1,3.8.0,"{'minimumROIDimensions': 2, 'minimumROISize': None, 'normalize': False, 'normalizeScale': 1, 'removeOutliers': None, 'resampledPixelSpacing': None, 'interpolator': 'sitkBSpline', 'preCrop': False, 'padDistance': 5, 'distances': [1], 'force2D': False, 'force2Ddimension': 0, 'resegmentRange': None, 'label': 255, 'additionalInfo': True, 'binWidth': 1, 'enableDiagnostics': False}",{'Original': {}},8bd43e867d1cc93e63924e6553216c01691c443d,2D,"(1.0, 1.0)","(512, 254)",49.35243140994095,0.0,132.0,fc5b746d5a3eedf45ba85b6ddcf2cfbb513ddce4,"(1.0, 1.0)","(512, 254)","(0, 0, 512, 254)",70540,1,"(217.85083640487667, 142.3380209810037)","(217.85083640487667, 142.3380209810037)",105.9021855057122,0.9768748398053657,0.3528607571809378,2.4078909244755273,0.10283477642865665,1510.712729273853,602.0878715655335,417.63387724373564,0.05932095265097817,0.01497838916728713,11820982.356435644,13.534653465346535,11333197.668659935,0.001431811738020981,3.9694239000255306e-05,30.29740333115967,1569.114213823499,2.0292068388951243
20_01_20,v3.1.0,1.24.3,2.3.1,1.4.1,3.8.0,"{'minimumROIDimensions': 2, 'minimumROISize': None, 'normalize': False, 'normalizeScale': 1, 'removeOutliers': None, 'resampledPixelSpacing': None, 'interpolator': 'sitkBSpline', 'preCrop': False, 'padDistance': 5, 'distances': [1], 'force2D': False, 'force2Ddimension': 0, 'resegmentRange': None, 'label': 255, 'additionalInfo': True, 'binWidth': 1, 'enableDiagnostics': False}",{'Original': {}},daf3f462a5b0062741cdafd094c34ec8cea39fe5,2D,"(1.0, 1.0)","(512, 254)",24.295044906496063,0.0,132.0,fc5b746d5a3eedf45ba85b6ddcf2cfbb513ddce4,"(1.0, 1.0)","(512, 254)","(0, 0, 512, 254)",70540,1,"(217.85083640487667, 142.3380209810037)","(217.85083640487667, 142.3380209810037)",39.78158031070496,0.9637127678264565,0.06097404490437333,4.70507819606287,0.12448039942046196,147.8469240048758,649.6294393267435,1654.8097929594353,0.13666005103487383,0.010110214170115314,321521.32235294115,26.83058823529412,293973.1335640138,0.006024950382761554,0.0005162360176134763,1.5689605881875461,690.0297010388624,0.9678575302511572
21_01_20,v3.1.0,1.24.3,2.3.1,1.4.1,3.8.0,"{'minimumROIDimensions': 2, 'minimumROISize': None, 'normalize': False, 'normalizeScale': 1, 'removeOutliers': None, 'resampledPixelSpacing': None, 'interpolator': 'sitkBSpline', 'preCrop': False, 'padDistance': 5, 'distances': [1], 'force2D': False, 'force2Ddimension': 0, 'resegmentRange': None, 'label': 255, 'additionalInfo': True, 'binWidth': 1, 'enableDiagnostics': False}",{'Original': {}},daf3f462a5b0062741cdafd094c34ec8cea39fe5,2D,"(1.0, 1.0)","(512, 254)",24.295044906496063,0.0,132.0,fc5b746d5a3eedf45ba85b6ddcf2cfbb513ddce4,"(1.0, 1.0)","(512, 254)","(0, 0, 512, 254)",70540,1,"(217.85083640487667, 142.3380209810037)","(217.85083640487667, 142.3380209810037)",39.78158031070496,0.9637127678264565,0.06097404490437333,4.70507819606287,0.12448039942046196,147.8469240048758,649.6294393267435,1654.8097929594353,0.13666005103487383,0.010110214170115314,321521.32235294115,26.83058823529412,293973.1335640138,0.006024950382761554,0.0005162360176134763,1.5689605881875461,690.0297010388624,0.9678575302511572
19_03_20,v3.1.0,1.24.3,2.3.1,1.4.1,3.8.0,"{'minimumROIDimensions': 2, 'minimumROISize': None, 'normalize': False, 'normalizeScale': 1, 'removeOutliers': None, 'resampledPixelSpacing': None, 'interpolator': 'sitkBSpline', 'preCrop': False, 'padDistance': 5, 'distances': [1], 'force2D': False, 'force2Ddimension': 0, 'resegmentRange': None, 'label': 255, 'additionalInfo': True, 'binWidth': 1, 'enableDiagnostics': False}",{'Original': {}},9fc3e64ee35a7235bd7288c2a97af963e9b74f38,2D,"(1.0, 1.0)","(512, 254)",25.54968934547244,0.0,132.0,fc5b746d5a3eedf45ba85b6ddcf2cfbb513ddce4,"(1.0, 1.0)","(512, 254)","(0, 0, 512, 254)",70540,1,"(217.85083640487667, 142.3380209810037)","(217.85083640487667, 142.3380209810037)",47.53374749163919,0.9681632434954687,0.05998329709105423,4.693514480548709,0.12718640717872728,112.13608436217825,849.513905329525,2348.6407394682597,0.15917918911256027,0.007756712203500272,173654.57209302325,30.57674418604651,146743.31411573826,0.006095832151970513,0.0003372650554677519,2.7365566831896766,883.5621034866308,1.0257444592315563
19_06_18,v3.1.0,1.24.3,2.3.1,1.4.1,3.8.0,"{'minimumROIDimensions': 2, 'minimumROISize': None, 'normalize': False, 'normalizeScale': 1, 'removeOutliers': None, 'resampledPixelSpacing': None, 'interpolator': 'sitkBSpline', 'preCrop': False, 'padDistance': 5, 'distances': [1], 'force2D': False, 'force2Ddimension': 0, 'resegmentRange': None, 'label': 255, 'additionalInfo': True, 'binWidth': 1, 'enableDiagnostics': False}",{'Original': {}},064769f6f8729125daf1f80fb8bb4a17583fe357,2D,"(1.0, 1.0)","(512, 254)",27.03439499261811,0.0,132.0,fc5b746d5a3eedf45ba85b6ddcf2cfbb513ddce4,"(1.0, 1.0)","(512, 254)","(0, 0, 512, 254)",70540,1,"(217.85083640487667, 142.3380209810037)","(217.85083640487667, 142.3380209810037)",42.16372444022604,0.9782579317907238,0.07414655391317747,4.254005399040887,0.11082939031575147,301.3896222084975,685.4580313148608,1152.7497733204273,0.11479302523390983,0.010985786394571987,932446.4710743802,25.388429752066116,847481.4481934294,0.0034306776297136376,0.0003239495399032793,5.83866470546183,490.91704957972496,1.1455134827389524
02_08_19,v3.1.0,1.24.3,2.3.1,1.4.1,3.8.0,"{'minimumROIDimensions': 2, 'minimumROISize': None, 'normalize': False, 'normalizeScale': 1, 'removeOutliers': None, 'resampledPixelSpacing': None, 'interpolator': 'sitkBSpline', 'preCrop': False, 'padDistance': 5, 'distances': [1], 'force2D': False, 'force2Ddimension': 0, 'resegmentRange': None, 'label': 255, 'additionalInfo': True, 'binWidth': 1, 'enableDiagnostics': False}",{'Original': {}},b6cebf0456c96009f0d73b485cfbbf20725e7286,2D,"(1.0, 1.0)","(512, 254)",38.656849778543304,35.0,132.0,8f2c26c87ab5008e88b797cea5604a78924b500e,"(1.0, 1.0)","(512, 254)","(0, 37, 512, 217)",63029,1,"(219.64319598914784, 152.6517158768186)","(219.64319598914784, 152.6517158768186)",8.706185136326493,0.9734802128653691,0.05213294395475811,5.323213277597495,0.15027074816395225,48.14442515498623,1069.1857054299426,4064.961910876258,0.21940693966269495,0.013259227913395752,17158.52794411178,63.271457085828345,13201.716177425587,0.015897444033698773,0.000538989990860635,0.1324155596575847,714.2245715724612,1.565151495138984
"""