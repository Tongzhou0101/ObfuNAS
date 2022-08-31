This is the implementation of ObfuNAS on NASbench-101 dataset.

1. Requirments:
   1) Python 3.7
   2) torch 1.10
   3) nasbench library: 
      - git clone https://github.com/google-research/nasbench


2. Download dataset (provided by nasbench):
   https://storage.googleapis.com/nasbench/nasbench_only108.tfrecord


3. Data directory:
   1) flops.npy: consists of the flops of each unique spec in NASbench-101
   2) test.npy: consists of the test accuracy of each unique spec in NASbench-101
   3) population1.pkl: the masks specs of a selected victim model
   4) mask_flops1.npy: consists of the flops of each mask in population1.pkl
   5) mask_test1.npy: consists of the test accuracy of each mask in population1.pkl

4. Code:
   1) settings.py: includes the spec definition 
   2) utilis.py: includes some helper functions
   3) mutation.py: mutate the victim spec to a mask spec according to obfuscation strategies
   4) main.py: use an evolutionary search to find the best mask for the victim with different flops constraints; one can also modify the hyper-parameters of the evolutionary search like popution size to adjust the search process
   5) evaluate.py: exhuastive search for small mask space (can be used to evaluate the evolutionary search result)

5. Usage:
   change the victim spec in main.py/ evaluate.py and run
   - python main.py 
   - python evalute.py


