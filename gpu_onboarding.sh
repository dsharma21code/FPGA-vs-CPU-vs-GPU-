cd ~/fpga-ai-comparison
source tf-env/bin/activate
pip uninstall tensorflow-cpu -y
pip install "tensorflow[and-cuda]" #install new tensorflow
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))" #test
nano benchmark_gpu.py #benchmark for gpu
python benchmark_gpu.py #run the code
