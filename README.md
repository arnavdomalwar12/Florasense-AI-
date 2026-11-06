# FloraSense AI

FloraSense AI is a deep-learning computer-vision project for identifying plant
species and recognizing plant diseases from leaf images. It combines trained
PyTorch classifiers with a Streamlit interface that displays predictions,
confidence scores, plant information, and basic care recommendations.

> **Important:** Predictions are limited to the classes used during training.
> The result is an assistive prediction and should not be treated as a
> confirmed botanical or plant-health diagnosis.

## Features

- Leaf-image classification through a simple web interface.
- Top-three predictions with confidence scores.
- Plant common name, scientific name, and description.
- Soil, sunlight, watering, temperature, and management guidance.
- Separate experiments for campus species, Kaggle species, and PlantVillage
  disease classification.
- MobileNetV3-Large and ResNet18 model checkpoints.
- Reproducible manifests, training histories, classification reports, and
  evaluation metrics.

## Project Workflow

```text
Leaf image
    |
    v
Resize to 224 x 224 and normalize
    |
    v
Trained CNN model (MobileNetV3-Large or ResNet18)
    |
    v
Species/disease prediction and confidence score
    |
    v
Plant profile and care recommendations
```

## Repository Structure

```text
.
├── FloraSense_Final_Input_Based_DL.ipynb  # Main training and evaluation notebook
├── florasense-ai-space/
│   ├── app.py                             # Streamlit application
│   ├── care_profiles.json                 # Plant information and care profiles
│   ├── models/                            # Deployment checkpoint
│   └── requirements.txt
├── models/                                # Trained model checkpoints
├── outputs/                               # Metrics, manifests, histories, reports
├── Campus_plants/                         # Campus plant image dataset
├── PlantVillage-Dataset-master/           # PlantVillage source dataset
├── archive/                               # Prepared train/validation/test data
├── Common Leaf images/                    # Example images for testing
├── Output Screenshots/                    # Application output screenshots
├── Documents/                             # Reports and presentation
└── build_florasense_report.py             # Report-generation utility
```

## Datasets

The repository includes the project datasets used for local training and
evaluation. The following links document the public sources and related
references:

| Dataset or task | Contents | Public source |
|---|---|---|
| PlantVillage disease dataset | Healthy and diseased crop leaves; used for the 38-class disease task | [PlantVillage on Kaggle](https://www.kaggle.com/datasets/emmarex/plantdisease) |
| PlantVillage original project | Original dataset repository and documentation | [PlantVillage GitHub](https://github.com/spMohanty/PlantVillage-Dataset) |
| PlantVillage research dataset | Dataset card and downloadable reference | [PlantVillage on Hugging Face](https://huggingface.co/datasets/mohanty/PlantVillage) |
| Kaggle species task | 12 Indian/common plant species used by the species classifier | [Kaggle datasets](https://www.kaggle.com/datasets?search=indian+plant+species) |
| Campus plants task | 10 campus plant classes collected for this project | Local `Campus_plants/` directory |

The local dataset folders are large. If GitHub displays a download warning,
clone the repository with Git LFS support enabled and ensure that all dataset
objects have finished downloading.

## Models and Results

The project evaluates two architectures:

- **MobileNetV3-Large:** lightweight and suitable for web or edge deployment.
- **ResNet18:** compact residual network with strong classification accuracy.

Representative test results recorded in
[`outputs/final_test_metrics.csv`](outputs/final_test_metrics.csv):

| Task | Model | Test images | Accuracy |
|---|---|---:|---:|
| Campus species | MobileNetV3-Large | 3,634 | 99.725% |
| Campus species | ResNet18 | 3,634 | 99.752% |
| Kaggle species | MobileNetV3-Large | 642 | 99.065% |
| Kaggle species | ResNet18 | 642 | 99.377% |
| PlantVillage disease | MobileNetV3-Large | 8,146 | 99.619% |
| PlantVillage disease | ResNet18 | 8,146 | 99.767% |

These are test-set results. Real-world performance can vary with lighting,
background, image quality, leaf orientation, and unsupported plant classes.

## Running the Streamlit Application

### 1. Create an environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.\.venv\Scripts\Activate.ps1
```

Activate it on macOS/Linux:

```bash
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r florasense-ai-space/requirements.txt
```

### 3. Start the application

```bash
streamlit run florasense-ai-space/app.py
```

Upload a JPG, JPEG, or PNG leaf image. The deployed demonstration uses the
MobileNetV3-Large Kaggle-species checkpoint in
`florasense-ai-space/models/`.

## Training and Evaluation

Open [`FloraSense_Final_Input_Based_DL.ipynb`](FloraSense_Final_Input_Based_DL.ipynb)
to inspect dataset preparation, preprocessing, training, validation, testing,
model comparison, and report generation. The `outputs/` directory contains
class mappings, split manifests, histories, latency measurements, and
classification reports produced during the experiments.

## Application Screenshots

### Application home and image upload

![FloraSense application home](Output%20Screenshots/Screenshot%202026-10-19%20194543.png)

### Prediction result

![Plant prediction result](Output%20Screenshots/Screenshot%202026-11-06%20101750.png)

### Plant information and care profile

![Plant care information](Output%20Screenshots/Screenshot%202026-11-06%20101816.png)

### Prediction details

![Prediction details](Output%20Screenshots/Screenshot%202026-11-06%20101834.png)

### Additional output

![Additional FloraSense output](Output%20Screenshots/Screenshot%202026-11-06%20101851.png)

![FloraSense result screen](Output%20Screenshots/Screenshot%202026-11-06%20101918.png)

## Technology Stack

- Python
- PyTorch and TorchVision
- Streamlit
- Pillow
- NumPy
- MobileNetV3-Large and ResNet18
- Transfer learning and image classification

## Limitations and Future Scope

The application only recognizes classes represented in its training data.
Blurred images, unusual backgrounds, poor lighting, and multiple overlapping
leaves can reduce reliability. Future work can add more species and diseases,
camera-based recognition, multilingual guidance, mobile deployment, and
location-based campus plant mapping.

## License and Dataset Attribution

Review the individual dataset licenses and attribution requirements before
redistributing the raw images. The PlantVillage dataset and its research
publication are linked above. This repository contains project code, trained
artifacts, reports, and locally assembled datasets for educational and
research use.
