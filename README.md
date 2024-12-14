# ECS 174 Project: Deep Learning for Lumbar Classification
@authors Arjun Ashok (arjun3.ashok@gmail.com, arjashok@ucdavis.edu), Zhian Li
(zanli@ucdavis.edu, lionellee0126@gmail.com), Ayush
Tripathi(atripathi7783@gmail.com, atripathi@ucdavis.edu)

> As deep learning becomes increasingly entrenched in our lives, it's no
> surprise that research in how such models can assist medical professionals in
> diagnosis is a hot topic. In this project, we aim to explore how baseline 
> vision models (e.g. classic CNN), state-of-the-art vision models (U-Net, 
> Vision Transformer), and novel model architectures (Kolmogorov-Arnold 
> Networks) stack up against diagnosing lumbar conditions via radiology imagery.

## Usage
### Installation
```
pip install -r requirements.txt
```

### Data
Data is sourced from [this RSNA
competition](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification),
and is expected to be unzipped into a directory called `data/`. This directory
should contain `train_images/`, `test_images/`, and some CSVs from the download.

### Running the Pipeline
```
python3 -m src.pipeline.pipeline
```

### Included Modules
We have modules for the architecture `arch`, visualizations and metrics `utils`, 
and the pipeline `pipeline`. Weights and hyperparameters are automatically 
cached for safe-keeping.

## Report
All materials (slides, report raw files, visuals, etc.) are all included in the
`report/` directory. The final report pdf is available in the repo directory as
`report.pdf`.

The video presentation is viewable on google drive
[here](https://drive.google.com/file/d/1mL5y67l-aK-iSct2mpXplZprdInjz4Ll/view?usp=sharing).

