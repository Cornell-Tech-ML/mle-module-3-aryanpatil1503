[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/vYQ4W4rf)
# MiniTorch Module 3

<img src="https://minitorch.github.io/minitorch.svg" width="50%">

* Docs: https://minitorch.github.io/

* Overview: https://minitorch.github.io/module3.html


You will need to modify `tensor_functions.py` slightly in this assignment.

* Tests:

```
python run_tests.py
```

* Note:

Several of the tests for this assignment will only run if you are on a GPU machine and will not
run on github's test infrastructure. Please follow the instructions to setup up a colab machine
to run these tests.

This assignment requires the following files from the previous assignments. You can get these by running

```bash




python sync_previous_module.py previous-module-dir current-module-dir
```

The files that will be synced are:

        minitorch/tensor_data.py minitorch/tensor_functions.py minitorch/tensor_ops.py minitorch/operators.py minitorch/module.py minitorch/autodiff.py minitorch/module.py project/run_manual.py project/run_scalar.py project/run_tensor.py

**Task 3.1: Parallel Check**
![image](https://github.com/Cornell-Tech-ML/mle-module-3-aryanpatil1503/assets/70055204/6539cbf9-781b-4aaf-97d4-c7721620db26)
* [Parallel_Check](parallel_check.txt)

**Task 3.4: Graph**
![module5graph](https://github.com/Cornell-Tech-ML/mle-module-3-aryanpatil1503/assets/70055204/cdfe1a96-8a4d-4f13-a0fb-081e79c6f0d4)

**Task 3.5: CPU and GPU Training Logs**

* Split Dataset
* CPU
* ![image](https://github.com/Cornell-Tech-ML/mle-module-3-aryanpatil1503/assets/70055204/3850a56b-9814-4c06-8e45-ca663eb6b3c7)
* [CPU_Logs](split_cpu_100.txt)
* GPU
* ![image](https://github.com/Cornell-Tech-ML/mle-module-3-aryanpatil1503/assets/70055204/60cb76f7-29ba-4b64-abf1-412170d16571)
* [GPU_Logs](split_gpu_100.txt)

* Simple Dataset
* CPU
* ![image](https://github.com/Cornell-Tech-ML/mle-module-3-aryanpatil1503/assets/70055204/dd74bde0-6a83-4934-9343-e15046c52767)
* [CPU_Logs](simple_cpu_100.txt)
* GPU
* ![image](https://github.com/Cornell-Tech-ML/mle-module-3-aryanpatil1503/assets/70055204/7f2293b4-4a87-4adc-8d3e-4b65bd59c586)
* [GPU_Logs](simple_gpu_100.txt)

* XOR Dataset
* CPU
* ![image](https://github.com/Cornell-Tech-ML/mle-module-3-aryanpatil1503/assets/70055204/c4b64c6b-54d6-4202-bff8-9395d6d06cfc)
* [CPU_Logs](xor_cpu_100.txt)
* GPU
* ![image](https://github.com/Cornell-Tech-ML/mle-module-3-aryanpatil1503/assets/70055204/66f48b8c-72e6-4ef1-aa86-67515adeb501)
* [GPU_Logs](xor_gpu_100.txt)

* Diag Dataset
* CPU
* ![image](https://github.com/Cornell-Tech-ML/mle-module-3-aryanpatil1503/assets/70055204/9376f7a7-0cda-47df-96d7-35df6eda75e5)
* [CPU_Logs](diag_cpu_100.txt)
* GPU
* ![image](https://github.com/Cornell-Tech-ML/mle-module-3-aryanpatil1503/assets/70055204/22b6c274-7b61-40a7-b102-8951168c100f)
* [GPU_Logs](diag_gpu_100.txt)

* Split Dataset Large Model (250)
* CPU
* ![image](https://github.com/Cornell-Tech-ML/mle-module-3-aryanpatil1503/assets/70055204/73e3cc65-dcf1-4d8e-8bc7-050cb3556f55)
* [CPU_Logs](split_large_cpu_250.txt)
* GPU:
* ![image](https://github.com/Cornell-Tech-ML/mle-module-3-aryanpatil1503/assets/70055204/843f131c-31c8-47df-98b5-b439cb3a114f)
* [GPU_Logs](split_gpu_large_250.txt)









