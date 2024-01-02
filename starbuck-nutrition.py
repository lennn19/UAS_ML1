import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

df = pd.read_csv('starbucks.csv')

x = df.drop(['Unnamed: 0','item','type'], axis=1)

st.header("Starbucks Nutrition")
st.write('Data yang berisikan mengenai nutrition di starbucks')
st.subheader('Data Asli')
st.write(df)

#menampilkan elbow
clusters = []
for i in range(1, 10):
    km = KMeans(n_clusters=i).fit(x)
    clusters.append(km.inertia_)

fig, ax = plt.subplots(figsize=(12, 8))
sns.lineplot(x=list(range(1, 10)), y=clusters, ax=ax)
ax.set_title('Mencari Elbow')
ax.set_xlabel('Number of Clusters')
ax.set_ylabel('Inertia')
plt.show()

st.set_option('deprecation.showPyplotGlobalUse', False)
elbo_plot = st.pyplot()

st.sidebar.subheader("Nilai Jumlah K")
clust = st.sidebar.slider("Pilih jumlah cluster :",1,10)

def k_means(n_clust):
    kmean = KMeans(n_clusters=n_clust).fit(x)
    x['Labels'] = kmean.labels_

    plt.figure(figsize=(10, 8))
    sns.scatterplot(x='fat', y='carb', hue='Labels', size='Labels', markers=True, palette=sns.color_palette('hls', n_colors=n_clust), data=x)
    for label in x['Labels']:
        plt.annotate(label,
                 (x[x['Labels'] == label]['fat'].mean(),
                  x[x['Labels'] == label]['carb'].mean()),
                 horizontalalignment='center',
                 verticalalignment='center',
                 size=20, weight='bold',
                 color='black')
       
    st.header('Cluster Plot')
    st.pyplot()
    st.write(x)

k_means(clust)