# Advanced-Movie-Recommendation-System

We had two approaches for a Movie Recommendation System:
1. Content Based Recommendation
2. State-of-art Nvidia Merlin Recommendation Models

Built a content based movie recommendation system using the TMDB 5000 Movie Dataset. The system will recommend movies that are similar to a given movie based on its genres, keywords, cast, crew and overview. Our implementation is present in the Content Based Movie System directory. We have also deployed the project with a web app to Heroku - https://content-based-movie-recommend-6104417e7e90.herokuapp.com/

Built and trained a recommender system using NVIDIA's Merlin models on the MovieLens 25 million dataset. The python notebook for the same is present in the Nvidia Merlin Implementation directory. The notebook is heavily inspired from Nvidia's official implementation present on their github - https://github.com/NVIDIA-Merlin


Merlin Models and NVTabular Integration

This notebook demonstrates how to use Merlin models and NVTabular together to build and train a recommender system. Merlin models are a collection of PyTorch-based neural network architectures for recommender systems, such as Deep Learning Recommendation Model (DLRM) and Neural Collaborative Filtering (NCF). NVTabular is a library for accelerating and scaling data preprocessing and feature engineering for tabular data. By using them together, you can achieve high performance and scalability for both data processing and model training.

Data Preparation

We use the MovieLens 25M dataset, which contains 25 million ratings and one million tag applications applied to 62,000 movies by 162,000 users. The dataset can be downloaded from (https://grouplens.org/datasets/movielens/25m/). We use NVTabular to load and preprocess the data, such as filtering out low-frequency items, splitting into train and validation sets, applying categorical encoding and normalization, and generating negative samples. We also use NVTabular to create TensorFlow and PyTorch dataloaders that can feed the preprocessed data to the models.

Model Training

We use two Merlin models in this notebook: DLRM and NCF. DLRM is a two-tower model that combines embeddings of categorical features with numerical features, and passes them through multiple fully connected layers. NCF is a matrix factorization model that learns embeddings of users and items, and computes their dot product as the prediction. Both models are implemented in PyTorch and can be easily customized and extended. We use the same hyperparameters for both models, such as learning rate, batch size, number of epochs, and embedding dimension. We use the binary cross entropy loss as the objective function, and the area under the ROC curve (AUC) as the evaluation metric. We train the models on a single GPU using the PyTorch dataloader from NVTabular.

Results

We compare the performance of DLRM and NCF on the validation set after each epoch of training. We plot the AUC scores for both models, and observe that DLRM achieves a higher AUC than NCF throughout the training process. This suggests that DLRM can better capture the interactions between categorical and numerical features, as well as the non-linearities in the data. NCF, on the other hand, relies on the dot product of user and item embeddings, which may not be sufficient to model the complex preferences of users and items.

Conclusion

This notebook shows how to use Merlin models and NVTabular together to build and train a recommender system. We demonstrate how to preprocess the MovieLens 25M dataset using NVTabular, and how to create PyTorch dataloaders that can feed the data to the models. We also show how to train two Merlin models: DLRM and NCF, using PyTorch on a single GPU. We compare the performance of both models on the validation set, and find that DLRM achieves a higher AUC than NCF. This indicates that DLRM can better model the complex interactions between features than NCF.

Highlights
Demonstrated proficiency in PyTorch, NVTabular, and recommender systems by creating a comprehensive documentation for the code
Evaluated and compared the performance of DLRM and NCF models on the validation set, and analyzed the results using AUC metric.

Explored and experimented with different hyperparameters, embeddings, and layers for the models, and optimized them for the best performance
Built and trained a recommender system using Merlin models and NVTabular on the MovieLens 25M dataset
Achieved an AUC of 0.81 on the validation set using DLRM, a state-of-the-art neural network architecture for recommender systems
Accelerated and scaled data preprocessing and feature engineering using NVTabular, a library for tabular data
Implemented and customized DLRM and NCF models in PyTorch, and trained them on a single GPU using PyTorch dataloaders from NVTabular

