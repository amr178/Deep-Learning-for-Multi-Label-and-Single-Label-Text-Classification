# Text Classification and Model Comparison 

This repository contains the source code and report , which focuses on **single-label text classification** using various Recurrent Neural Network (RNN) architectures and a comparative analysis of their performance.

---

## Project Explanation

The main goal of this assignment was to **classify text documents into a single topic category**. This task involved a complete machine learning pipeline, from raw data to final model evaluation, emphasizing the comparison of different sequence models.

### 1. Data Preparation
The text data (likely containing **TITLE** and **ABSTRACT** fields) was rigorously cleaned and prepared to ensure the models learned from relevant features:
* **Text Cleaning:** Standard steps like lowercasing, removal of punctuation, numbers, and extra whitespace were performed to minimize noise.
* **Feature Engineering:** The `TITLE` and `ABSTRACT` were combined into a single, comprehensive text feature.
* **Normalization:** Both **Lemmatization** (reducing words to their dictionary form, e.g., 'running' to 'run') and **Stemming** (reducing words to their root form, e.g., 'finally' to 'final') were applied for robust feature representation.
* **Vectorization:** The prepared text was converted into numerical features using **TF-IDF (Term Frequency-Inverse Document Frequency)**, which weights words based on how frequently they appear in a document relative to the entire dataset.

### 2. Model Implementation
Four different Recurrent Neural Network (RNN) architectures were implemented and trained on the prepared data to compare their ability to capture dependencies in sequential text data:
1.  **RNN (Recurrent Neural Network):** The foundational sequence model, which processes data one step at a time, using an internal state/memory to process sequences.
2.  **BiRNN (Bidirectional RNN):** Extends the standard RNN by processing the sequence in both forward and backward directions, allowing it to capture context from future words as well as past words.
3.  **LSTM (Long Short-Term Memory):** An improvement on RNNs designed to overcome the vanishing gradient problem, enabling the model to learn and remember long-term dependencies using 'gates'.
4.  **BiLSTM (Bidirectional LSTM):** Combines the power of LSTM for long-term memory with the contextual awareness of bidirectional processing, making it highly effective for complex text tasks.

### 3. Evaluation
The models were trained on 50% of the data, monitored for generalization on a 20% validation set, and finally tested on an unseen 30% test set. The models were evaluated primarily on **Test Accuracy** and measures of **Generalization Ability** (comparing training vs. validation accuracy curves).

---

## Project Structure

| File Name | Description |
| :--- | :--- |
| `code.ipynb` | The main Jupyter Notebook containing the full code for preprocessing, model implementation, training, and a final **bar chart comparison** of test accuracies across all four models. |
| `report.pdf` | The formal project report detailing the full methodology, **deep analysis of results**, discussion on model stability, and final conclusions. |


---

## Key Findings

As concluded in the `report.pdf`, the **BiLSTM** model demonstrated **superior performance**:

* **Best Model:** **BiLSTM** achieved the **highest test accuracy** and proved to be the most robust and reliable model for this specific single-label classification task.
* **Generalization:** BiLSTM and LSTM showed the **best generalization**, with stable learning curves (validation accuracy closely following training accuracy).
* **Overfitting Risk:** Standard RNN and BiRNN showed a wider gap between training and validation accuracy, indicating a greater tendency toward **overfitting** to the training data.

---

## How to Run the Code

To execute the models and reproduce the results:

1.  **Dependencies:** Install the necessary libraries (e.g., `tensorflow`/`keras`, `scikit-learn`, `nltk`, `pandas`) using `pip`:
    ```bash
    pip install notebook pandas numpy scikit-learn tensorflow matplotlib nltk
    ```
2.  **Execute the Notebook:** Launch Jupyter Notebook or JupyterLab and open `code.ipynb`.
    ```bash
    jupyter notebook code.ipynb
    ```
3.  Run all cells sequentially to perform the complete pipeline.
