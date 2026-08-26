# Weather Condition Classification with Transfer Learning

Eleven class image classifier for weather conditions (**dew, fogsmog, frost, glaze, hail, lightning,
rain, rainbow, rime, sandstorm, snow**), trained on the
[Weather Image Recognition](https://www.kaggle.com/datasets/jehanbhathena/weather-dataset) dataset with
MobileNetV2 transfer learning and deployed as a Streamlit app.

**87.2% accuracy**, **0.884 balanced accuracy** and **0.879 macro F1** on 1030 held out test photos.

**Author:** Eng. Noureldin Bassem Mohamed

## The problem

A weather monitoring company runs roadside and rooftop cameras that take a new photo every few minutes.
Nothing downstream can use a photo until somebody tags the condition by hand, and that tag is what
triggers the useful part, warning a maintenance crew about ice on a bridge camera or pausing an
irrigation system while it rains. Manual tagging does not scale as more cameras come online, so the model
tags the photo automatically and a person only opens the ones it is genuinely unsure about.

## What is in here

| file | what it is |
| --- | --- |
| `weather-classification.ipynb` | the full notebook: EDA, split, preprocessing, both training stages, evaluation, write up |
| `app.py` | the Streamlit app, loads the trained model and predicts on an uploaded photo |
| `weather_model.keras` | the trained model saved from the notebook |
| `class_names.json` | class order and input size, so the app matches the training setup |
| `requirements.txt` | what the app needs to run |

## The dataset

6862 photos in eleven folders, one per class, and the folder name is the label. There is no ready made
train/test split and the classes are not balanced, `rime` has 1160 photos and `rainbow` only 232, a 5x
gap. The photos come from different sources at 245 different sizes in a sample of 400, so everything gets
resized to the 224x224 that the backbone expects.

## Approach

Transfer learning on **MobileNetV2** in TensorFlow / Keras, trained in two stages:

1. **Model A** - backbone frozen, only the classifier head trained (`lr=1e-3`).
2. **Model B** - the last block of the backbone unfrozen and fine-tuned (`lr=1e-5`), BatchNorm kept frozen.

Since the dataset ships without a split, I split the index myself, stratified, **70% train / 15%
validation / 15% test**. Every decision (early stopping, how much to unfreeze, which model to keep) is
made on the validation slice and the test slice is only touched at the very end. The loss is class
weighted so the small classes are not sacrificed to the big ones.

| model | accuracy | balanced accuracy | macro F1 | weighted F1 |
| --- | --- | --- | --- | --- |
| A (frozen) | 0.845 | 0.855 | 0.855 | 0.844 |
| **B (fine-tuned)** | **0.872** | **0.884** | **0.879** | **0.872** |

## The interesting result

The errors are concentrated, not spread out. About a third of them are the ice classes arguing with each
other, **frost, glaze and rime**, in every direction. Frost has the lowest precision (0.674) and glaze the
lowest recall (0.729), which is the same problem seen from two sides: when the model sees ice on a
surface it leans toward frost. What actually separates those three is a fine texture, a crystal pattern
against a clear glazed layer against a frozen fog crust, and that is the first detail to soften in the
resize. **fogsmog against sandstorm** is second, 22 photos between them, both being a low contrast haze
where the real difference is the colour of the air.

Everything with its own colour signature is comfortably above 0.9 F1, and `rainbow` reaches 0.971 despite
being the smallest class, which is the class weighting doing its job.

Because of that, the app reports a confidence with every prediction. At a 0.70 confidence threshold the
model tags 84.5% of the photos by itself at 92.9% accuracy, so the review team only opens about one in six.

## Running the app locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Reproducing the notebook

The notebook downloads the dataset itself with `kagglehub`, so Restart & Run All is enough. It was run on
CPU (TensorFlow 2.21, Keras 3.15), where the two training stages take about 12 and 10 minutes.
