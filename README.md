# Comparing Multimodal Machine Learning to Lung Cancer Classification

This repository contains a project that compares the performance of different machine learning models in classifying lung cancer. The models included in this comparison are:

- Convolutional Neural Network (CNN)
- Transfer Learning using VGG16
- Transfer Learning using ResNet50
- Random Forest

The entire project is implemented in a single Jupyter notebook for ease of use and reproducibility.

## Repository Structure

The repository is structured as follows:

- `multimodal-machine-learning-model-to-lung-cancer-classification.ipynb`: The main Jupyter notebook file containing the implementation of the models.
- `Dataset/`: A folder containing images of multiple categories used for training and testing the models.
- `Models/`: A folder for saving the trained models.

## Getting Started

To get started with this project, follow the steps below.

### Prerequisites

Ensure you have the following installed:

- Python 3.7 or higher
- Jupyter Notebook
- Git

You will also need the following Python packages, which can be installed using `pip`:

```sh
pip install numpy pandas matplotlib scikit-learn tensorflow keras
```

### Cloning the Repository

Clone this repository to your local machine using the following command:

```sh
git clone https://github.com/algo-tushar/comparing-multimodal-machine-learning-to-lung-cancer-classification.git
cd comparing-multimodal-machine-learning-to-lung-cancer-classification
```

### Dataset

The `Dataset` folder should contain images categorized into appropriate subfolders. Ensure that your dataset is organized correctly for the models to be trained effectively.

### Running the Notebook

Launch Jupyter Notebook and open the main notebook file:

```sh
jupyter notebook multimodal-machine-learning-model-to-lung-cancer-classification.ipynb
```

Follow the instructions in the notebook to run the cells sequentially. The notebook includes:

1. **Data Loading and Preprocessing**: Code to load and preprocess the dataset.
2. **Model Definitions**: Definitions and configurations for the CNN, VGG16, ResNet50, and Random Forest models.
3. **Training and Evaluation**: Code to train and evaluate each model.
4. **Results and Comparisons**: Visualizations and comparisons of the performance metrics for each model.

### Saving Models

Trained models will be saved in the `Models` folder. Ensure this folder exists in the root directory of the project before running the notebook.

## Results

The performance of each model is evaluated based on accuracy, precision, recall, and F1-score. Detailed results and comparisons can be found in the final sections of the notebook.

## Contributing

Contributions are welcome! Please fork this repository and submit pull requests with any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For any questions or inquiries, please contact [abubakar.tushar@gmail.com].

---

**Disclaimer**: This project is for educational and research purposes only. The models and results should not be used for clinical decision-making without further validation.
