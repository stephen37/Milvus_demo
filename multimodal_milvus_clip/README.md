# Multimodal Milvus CLIP Demo

This repository demonstrates how to use [Milvus](https://github.com/milvus-io/milvus), a powerful vector database, with CLIP (Contrastive Language–Image Pre-Training) to perform multimodal searches. With this setup, you can search through image and text data efficiently.

## Features
- **Milvus Integration**: Utilises Milvus for storing and querying vector data.
- **CLIP Model**: Employs OpenAI's CLIP model to encode images and text into vectors.
- **Multimodal Search**: Perform similarity searches across different data modalities (images and text).

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/stephen37/Milvus_demo.git
    ```
2. Navigate to the project directory:
    ```sh
    cd Milvus_demo/multimodal_milvus_clip
    ```
3. Install the required packages:
    ```sh
    poetry install 
    ```
4. Start Milvus standalone: 
    ```sh
    docker-compose up -d
    ```

## Usage

1. Launch the Jupyter notebook:
    ```sh
    jupyter-lab multimodal_demo.ipynb
    ```
2. Follow the instructions in the notebook to set up Milvus, load the CLIP model, and perform searches.

## Support
* Star Us on GitHub: If you find this project useful, please [give us a star on GitHub](https://github.com/milvus-io/milvus)!
* Join Our Community: Join our [Discord](https://discord.gg/FG6hMJStWu) to connect with other users and developers.

## Contributing

Feel free to open issues or submit pull requests for improvements.