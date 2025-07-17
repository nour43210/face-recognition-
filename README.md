🧠 Face Recognition GUI with PCA, LDA, KNN, and t-SNE
This is a Tkinter-based GUI app (styled with ttkbootstrap) that allows users to explore face recognition using machine learning and dimensionality reduction techniques.

🔧 Main Features:
GUI Components:

Dropdown menu to select ML models: KNN, LDA, PCA, t-SNE

Buttons to train the model and visualize results

Label to display accuracy

Model Training:

Uses K-Nearest Neighbors (KNN) for classification

Reduces data dimensions using either:

PCA: Principal Component Analysis

LDA: Linear Discriminant Analysis

t-SNE: t-Distributed Stochastic Neighbor Embedding (for 2D visualization only)

Visualization:

Plots 2D scatter of face embeddings using seaborn and matplotlib

📊 Dataset:
Assumes a dataset of face images (olivetti_faces) loaded using sklearn.datasets.fetch_olivetti_faces.
