#use ubuntu system
cd ~/fpga-ai-comparison
source tf-env/bin/activate
pip uninstall tensorflow-cpu -y
#install tensorflow on Legion beforehand
nano benchmark_cpu.py
#open system -- paste .py code
python benchmark_cpu.py
#run code
