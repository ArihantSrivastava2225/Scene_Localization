# Scene Localization in Dense Images via Natural Language Queries

## Project Brief and Technical Summary

This project implements a deep learning system for **visual grounding**, a task that localizes specific sub-scenes or interactions within a dense image based on a natural language query. The system is designed to handle complex, multi-activity scenes by leveraging pre-trained vision and language models.

---

## Project Goal

The primary objective is to develop a deep learning model that can:
-   **Understand** and extract contextual visual features from dense images.
-   **Parse** a natural language query to create a semantically meaningful representation.
-   **Ground** the query in the image by outputting an accurate bounding box or a cropped image region corresponding to the described event.

---

## Model Overview

The core of the system is a two-stream deep learning architecture that fuses information from both the visual and linguistic domains.

-   **Image Encoder**: A pre-trained image encoder (likely a ResNet-based model) extracts visual features from the image.
-   **Text Encoder**: A pre-trained text encoder (likely a BERT-based model) transforms the natural language query into a sequence of contextual embeddings.
-   **Cross-Modal Grounding**: An attention mechanism then grounds the textual query in the image by aligning the text embeddings with the visual features to produce a set of confidence scores for potential bounding boxes.
-   **Output Heads**: A final set of layers uses these scores to predict the most accurate bounding box for the described scene.

---

## Usage Instructions

### 1. Setup and Prerequisites

1.  **Clone the Repository**: Begin by cloning the project repository to your local machine.
2.  **Dataset**: The training data is based on the **RefCOCO dataset**. You must download this dataset separately.
3.  **Data Preparation**: After downloading, run `prepare_refcoco.py` to set up the data folder structure. Note that you must temporarily move any existing `coco` folder from the `data` directory before running this script.

### 2. Inference

After the model is trained and you have a checkpoint file, you can run inference from your terminal.

-   **Command:**
    ```bash
    python main.py --mode infer --image <path_to_image> --query "<your_query>" --ckpt <path_to_checkpoint>
    ```
-   **Example (Local System):**
    ```bash
    python main.py --mode infer --image data\coco\val2014\COCO_val2014_000000000136.jpg --query "the giraffe nearer to the window and also nearer to the people watching" --ckpt checkpoints\model_epoch_50.weights.h5
    ```

### 3. Training

The training process is resource-intensive and is configured to run on a Kaggle notebook. The `main.py` script contains hardcoded paths to align with Kaggle's notebook environment.

-   **Kaggle Command (in a notebook cell):**
    ```bash
    ! python Scene_Localization/main.py --mode train --dataset_dir data --split train --year 2014
    ```
    > **NOTE:** The `--dataset_dir` is provided for consistency, but the `main.py` file is configured with static Kaggle paths for seamless execution.
-   **Local Command (if using a GPU):**
    ```bash
    python main.py --mode train --dataset_dir data --split train --year 2014
    ```