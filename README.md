# Text Emotion Analyzer

![Beta](https://img.shields.io/badge/status-beta-yellow)
![Python](https://img.shields.io/badge/python-3.x-blue)
![License](https://img.shields.io/badge/license-MIT-green)

Welcome to the Text Emotion Analyzer, a Django-based web application for analyzing the emotional content of text using machine learning models.

## Installation

### Prerequisites

- Python 3.x
- Virtualenv

### Steps

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/KCprsnlcc/DjangoAnalyzer.git
    cd DjangoAnalyzer
    ```

2. **Create and Activate Virtual Environment**:
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows use `.\.venv\Scripts\activate`
    ```

3. **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4. **Apply Migrations**:
    ```bash
    python manage.py migrate
    ```

5. **Run the Server**:
    ```bash
    python manage.py runserver
    ```

6. Open your web browser and go to `http://127.0.0.1:8000`.

## Notes 📓

- Ensure the pre-trained emotion detection model (`tf_model.h5`) is present in the `predictivemodel` directory.
- Adjust the `analyzer/views.py` file to suit your specific requirements for emotion analysis.

## Contributing

Contributions are welcome! Please create an issue or submit a pull request for any improvements or bug fixes.

## Contact

For any questions or inquiries, please contact [kcpersonalacc@gmail.com](mailto:kcpersonalacc@gmail.com).

---

Thank you for using our Text Emotion Analyzer! Your feedback is valuable to us. 🧡
