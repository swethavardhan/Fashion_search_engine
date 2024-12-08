# Fashion_search_engine

This project implements a **Fashion Search** system that allows users to search for clothing items using either **image-based queries** or **text-based queries**. The system recommends the most similar items based on a set of embeddings generated from both the images and descriptions of clothing items.

## Project Overview
The system consists of two components:

1. **Image Search**: Allows users to upload an image, and the system recommends the most similar clothing items based on visual features extracted from the image using the ResNet50 model.
2. **Text Search**: Users can enter a description (e.g., "red floral dress"), and the system returns the most similar items based on the textual descriptions, processed using a sentence transformer model (`paraphrase-MiniLM-L6-v2`).

The system combines **image embeddings** and **text embeddings** to provide accurate recommendations for fashion items based on both visual and descriptive features.

## Features

- **Image Search**: Upload an image, and the system suggests similar items based on visual features using the ResNet50 model.
- **Text Search**: Input a clothing description (e.g., "blue denim jacket"), and the system returns similar items based on the description using text embeddings.
- **Enhanced Image Quality**: Images are enhanced for clarity before displaying in the results.
- **Clothing Matching**: The system provides a list of the most similar clothing items based on image or text input.

## Dataset

This project uses the **Myntra Fashion Dataset** from [Kaggle](https://www.kaggle.com/datasets). The dataset contains clothing items from Myntra, including their descriptions, images, and various attributes. The dataset is used to generate text embeddings from clothing descriptions and visual embeddings from clothing images, enabling the recommendation system for fashion search based on both image and text input.

The applicaion
![Screenshot 2024-12-08 201755 - Copy](https://github.com/user-attachments/assets/a47f379f-d090-4932-8aba-402f0381d7e7)
![Screenshot 2024-12-08 201755](https://github.com/user-attachments/assets/1274ef1e-9ea1-404f-8744-eb281e835935)
![Screenshot 2024-12-08 203908](https://github.com/user-attachments/assets/dec44f11-c68f-49fd-bfd8-305acdfa6ad9)







