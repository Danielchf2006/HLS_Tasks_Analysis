import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error
pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', None)  # None means no limit
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)

HLS_data = pd.read_csv('data_intel.csv')
print(HLS_data.head())
HLS_data.info()
print(HLS_data.describe())

vars = HLS_data.columns[3:]
figure, axes = plt.subplots(len(vars), 3, figsize=(15, 30))
sns.set(font_scale=0.8)
figure.subplots_adjust(hspace=0.8, wspace=0.6)
for var in vars:
    if var != 'alm':
        sns.scatterplot(x=var, y='alm', data=HLS_data, ax=axes[vars.tolist().index(var),0], alpha=0.4).set_title(f'{var} vs alm', fontsize=7, weight='bold')
    else:
        continue
    if var != 'clock_speed':
        sns.scatterplot(x=var, y='clock_speed', data=HLS_data, ax=axes[vars.tolist().index(var),1], alpha=0.4).set_title(f'{var} vs clock_speed', fontsize=7, weight='bold')
    else:
        continue
    sns.histplot(x=var, data=HLS_data, ax=axes[vars.tolist().index(var),2]).set_title('Distribution', fontsize=7, weight='bold')
plt.show()
zvs = HLS_data.select_dtypes(include=[np.number])

print(zvs.corr()['alm'])
print(zvs.corr()['clock_speed'])

HLS_data_np = HLS_data.to_numpy()
print(HLS_data.shape)

X = HLS_data_np[:, [HLS_data.columns.get_loc('clock_speed'),
                    HLS_data.columns.get_loc('reg'),
                    HLS_data.columns.get_loc('dsp'),
                    HLS_data.columns.get_loc('ram'),
                    HLS_data.columns.get_loc('mlab')]]

y = HLS_data_np[:, HLS_data.columns.get_loc('alm')]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
model_HLS_data = LinearRegression()
model_HLS_data.fit(X_train, y_train)
y_pred = model_HLS_data.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error (MSE): {mse}')
print(f'R-squared (R²): {r2}')

HLS_data_test = np.array([[1, 1, 1, 1, 1]])
HLS_data_pred = model_HLS_data.predict(HLS_data_test)
print(f"Prediction for Adaptive Logic Module (ALM): {HLS_data_pred[0]}")

coef = model_HLS_data.coef_
Variable_names = ['clock speed','reg', 'dsp', 'ram', 'mlab']
for var_names, var_coef in zip(Variable_names, coef):
    print(f"{var_names}:{var_coef}")

#------------------------------------------


X1 = HLS_data_np[:, [HLS_data.columns.get_loc('alm'),
                    HLS_data.columns.get_loc('reg'),
                    HLS_data.columns.get_loc('dsp'),
                    HLS_data.columns.get_loc('ram'),
                    HLS_data.columns.get_loc('mlab')]]

y1 = HLS_data_np[:, HLS_data.columns.get_loc('clock_speed')]

X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.2, random_state=0)
model_HLS_data_2 = LinearRegression()
model_HLS_data_2.fit(X1_train, y1_train)
y1_pred = model_HLS_data_2.predict(X1_test)

mse = mean_squared_error(y1_test, y1_pred)
r2 = r2_score(y1_test, y1_pred)
print(f'Mean Squared Error (MSE): {mse}')
print(f'R-squared (R²): {r2}')

HLS_data_test1 = np.array([[1, 1, 1, 1, 1]])
HLS_data_pred1 = model_HLS_data_2.predict(HLS_data_test1)
print(f"Prediction for clock Speed: {HLS_data_pred1[0]}")

coef1 = model_HLS_data_2.coef_
Variable_names1 = ['alm','reg', 'dsp', 'ram', 'mlab']
for var_names1, var_coef1 in zip(Variable_names1, coef1):
    print(f"{var_names1}:{var_coef1}")

#--------------------------------------------------------

features = HLS_data[['clock_speed', 'alm', 'reg', 'dsp', 'ram', 'mlab']]
task_names = HLS_data[['name', 'name_unique']]
scaler = StandardScaler() # Normalize the data/ make sure there are no bias and scale all the data.
scaled_HLS_features = scaler.fit_transform(features)

# Determine the optimal number of clusters using the Elbow method
inertia = [] #how internally coherent each data within the cluster is ot the centroid.
k_values = range(1, 10)

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_HLS_features)
    inertia.append(kmeans.inertia_) #float/ it's kind of similar to R^2 for regression model = served as evaluation of the model

# Plot the Elbow curve
plt.figure(figsize=(8, 5))
plt.plot(k_values, inertia, 'bo-') #formatting the graph/ elbow method to tell us which k value is the most optimal
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
plt.show()

# Choose the optimal k (based on the elbow curve, choose k = 3)
k_optimal = 3

# Applying K-means clustering
kmeans = KMeans(n_clusters=k_optimal, random_state=42)
clusters = kmeans.fit_predict(scaled_HLS_features)

HLS_data['cluster'] = clusters

# Analyze the clusters by looking at the mean values of the features within each cluster
#HLS_data_with_names = HLS_data[['name', 'name_unique', 'clock_speed', 'alm', 'reg', 'dsp', 'ram', 'mlab', 'cluster']]
numeric_columns = HLS_data.select_dtypes(include=['number']).columns
cluster_analysis = HLS_data.groupby('cluster')[numeric_columns].mean()
print(cluster_analysis)
#print task names
tasks_by_cluster = HLS_data.groupby('cluster')['name_unique'].apply(list)
for cluster_id, tasks in tasks_by_cluster.items():
    print(f"Cluster {int(cluster_id) + 1} contains the following tasks:")
    for task in tasks:
        print(f"  - {task}")

#Cluster 0 represents tasks with lower clock speeds and moderate resource usage across the board. These tasks might be less performance-intensive but still require a decent amount of hardware resources.
#Cluster 1 is the most resource-intensive. Tasks in this cluster demand high hardware resources and could be the most critical or computationally heavy tasks. They likely require more powerful hardware or optimization for efficiency.
#Cluster 2 consists of tasks with high clock speeds but minimal resource usage. These tasks might be optimized for speed, requiring less computational resources while maintaining high performance. They could be lightweight tasks, ideal for scenarios where speed is crucial but resources are limited.
