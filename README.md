# Material Semantic Segmentation on opensurfaces dataset (Pytorch)


![image](https://user-images.githubusercontent.com/96943196/152592757-b28d6f8e-6e72-49c9-8027-581a78365f5c.png)
![image](https://user-images.githubusercontent.com/96943196/152593103-a5a01f16-dd1c-4585-8558-6828992530c5.png)


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
