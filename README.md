# Material Semantic Segmentation on opensurfaces dataset (Pytorch)

- Dataset : OpenSurfaces (http://opensurfaces.cs.cornell.edu/)
- Task : Semantic Segmentation

![image](https://user-images.githubusercontent.com/96943196/152593728-eb26d12a-9457-4a5e-bccf-b6a4f85daf4d.png)


#### Training
```python
python train.py --gpus 0    # single gpu
```
```python
python train.py --gpus 0,1  # multi gpu : 0,1 or 0-3, ... 
```

#### Evaluation
```python
python eval_multipro.py
```
