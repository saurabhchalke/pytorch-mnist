# PyTorch MNIST Classifier

This project demonstrates a simple neural network to classify handwritten digits from the MNIST dataset using PyTorch. It includes scripts for training a model, saving it, exporting it to ONNX format, and running inferences using both the original PyTorch model and the ONNX model.

## Project Structure

- `model.py`: Contains the neural network model definition.
- `train.py`: Script to train the model on the MNIST dataset.
- `export_onnx.py`: Script to export the trained model to ONNX format.
- `run_onnx.py`: Script to load and run inferences using the ONNX model.
- `requirements.txt`: List of dependencies for the project.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/saurabhchalke/pytorch-mnist.git
   cd pytorch-mnist

2. Set up a virtual environment:

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Run the training script:

   ```bash
   python train.py
   ```

5. Run the export script:

   ```bash
    python export_onnx.py
    ```

6. Run the inference script:

    ```bash
     python run_onnx.py
     ```

Contributing
------------
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

License
-------
[MIT](https://choosealicense.com/licenses/mit/)

Project Status
--------------
This project is was a fun learning project. Users are welcome to suggest improvements and report bugs.
