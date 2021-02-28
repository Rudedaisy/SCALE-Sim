# square vs rectangles (minor and extreme)
python scale.py --network='./topologies/conv_nets/ResNet50_ImageNet_baseline.csv' --run_name='RECTANGLES_ResNet50_ImageNet_64_64_baseline' --height=64 --width=64

python scale.py	--network='./topologies/conv_nets/ResNet50_ImageNet_baseline.csv' --run_name='RECTANGLES_ResNet50_ImageNet_128_32_baseline' --height=128 --width=32

python scale.py	--network='./topologies/conv_nets/ResNet50_ImageNet_baseline.csv' --run_name='RECTANGLES_ResNet50_ImageNet_32_128_baseline' --height=32 --width=128

python scale.py	--network='./topologies/conv_nets/ResNet50_ImageNet_baseline.csv' --run_name='RECTANGLES_ResNet50_ImageNet_256_16_baseline' --height=256 --width=16

python scale.py	--network='./topologies/conv_nets/ResNet50_ImageNet_baseline.csv' --run_name='RECTANGLES_ResNet50_ImageNet_16_256_baseline' --height=16 --width=256


python scale.py --network='./topologies/conv_nets/ResNet50_ImageNet_baseline.csv' --run_name='RECTANGLES_ResNet50_ImageNet_512_8_baseline' --height=512 --width=8

python scale.py --network='./topologies/conv_nets/ResNet50_ImageNet_baseline.csv' --run_name='RECTANGLES_ResNet50_ImageNet_8_512_baseline' --height=8 --width=512

python scale.py --network='./topologies/conv_nets/ResNet50_ImageNet_baseline.csv' --run_name='RECTANGLES_ResNet50_ImageNet_1024_4_baseline' --height=1024 --width=4

python scale.py --network='./topologies/conv_nets/ResNet50_ImageNet_baseline.csv' --run_name='RECTANGLES_ResNet50_ImageNet_4_1024_baseline' --height=4 --width=1024
