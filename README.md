# Mango Leaf Disease Prediction

This project is designed to predict mango leaf diseases using machine learning. It includes a Flask web application and a Tkinter-based GUI application for user-friendly interaction.

## Table of Contents
- [Introduction](#introduction)
- [Web Application](#web-application)
  - [Usage](#web-application-usage)
- [GUI Application](#gui-application)
  - [Usage](#gui-application-usage)
- [Project Structure](#project-structure)
- [License](#license)
- [Contributing](#contributing)

## Introduction

Mango Leaf Disease Prediction is a project that uses a trained machine learning model to classify diseases in mango leaves based on input images. The project consists of two main components: a web application and a GUI application.

## Web Application

The web application allows users to upload an image of a mango leaf, and it provides real-time predictions for the disease type. It uses a Flask backend for image processing and prediction.

### Web Application Usage

1. Clone the repository to your local machine:

```bash
git clone https://github.com/pypi-ahmad/mango-leaf-disease-prediction.git
cd mango-leaf-disease-prediction
```

2. Install the required Python packages (you may want to use a virtual environment):

```bash
pip install -r requirements.txt
```

3. Run the web application:

```bash
python WEB_APP.py
```

4. Open a web browser and navigate to `http://localhost:5000` to use the application.

## GUI Application

The GUI application provides a user-friendly interface for uploading and classifying mango leaf images. It uses Tkinter for the graphical interface and the same trained model for predictions.

### GUI Application Usage

1. Clone the repository to your local machine (if not already done):

```bash
git clone https://github.com/pypi-ahmad/mango-leaf-disease-prediction.git
cd mango-leaf-disease-prediction
```

2. Install the required Python packages (you may want to use a virtual environment):

```bash
pip install -r requirements.txt
```

3. Run the GUI application:

```bash
python GUI_APP.py
```

4. A window will appear, allowing you to browse and classify images.

## Project Structure

The project is organized as follows:

```
Mango Leaf Disease Prediction/
├── WEB_APP.py
├── GUI_APP.py
├── mango_leaf_disease_model.h5
├── templates/
│   ├── index.html
│   └── result.html
├── uploads/
├── .gitignore
├── LICENSE (optional)
├── README.md
├── requirements.txt
```

## License

This project is open-source and available under the [MIT License](LICENSE) 

## Contributing

Contributions are welcome! If you'd like to contribute to this project or report issues, please visit the [GitHub repository](https://github.com/pypi-ahmad/mango-leaf-disease-prediction) for more information.

Feel free to contribute, report issues, or provide feedback. Happy coding!
