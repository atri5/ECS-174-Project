# ECS 174 Project: Deep Learning for Lumbar Classification
@authors Arjun Ashok (arjun3.ashok@gmail.com, arjashok@ucdavis.edu), Zhian Li (zanli@ucdavis.edu, lionellee0126@gmail.com) ,
Ayush Tripathi(atripathi7783@gmail.com, atripathi@ucdavis.edu)

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

