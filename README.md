# ObfuNAS

This is the offcial implementation of ObfuNAS (ICCAD 2022) on NASbench-101 dataset.

## Description

This framework is proposed to defend against model architecture extraction by obfuscating the victim model architracture as a different one, while considering the accuracy the obfuscated model may achieve. ObfuNAS converts the DNN architecture obfuscation into a neural architecture search (NAS) problem. Using a combination of function-preserving obfuscation strategies, it ensures that the obfuscated DNN architecture can only achieve lower accuracy than the victim. 
## Getting Started

### Requirement
* Python 3.7
* torch 1.10
* nasbench library:    
```
git clone https://github.com/google-research/nasbench
```

### Download

* Dataset (provided by nasbench): https://storage.googleapis.com/nasbench/nasbench_only108.tfrecord
* More details are in the nasbench repo: https://github.com/google-research/nasbench/tree/master

### Executing program

* Change the victim spec in main.py/ evaluate.py and run

```
python main.py 
```
```
python evalute.py
```














