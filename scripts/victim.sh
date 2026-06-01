name_1="bonsai"
name_2="kitchen"


(mkdir -p "./output/victim/$name_2/"
python ./victim/gaussian-splatting/benchmark.py --gpu 6\
    -s ./example/MIP_Nerf_360_eps16/$name_2/ -m ./output/victim/$name_2/ 
)&

(mkdir -p "./output/victim/$name_1/"
python ./victim/gaussian-splatting/benchmark.py --gpu 7\
    -s ./example/MIP_Nerf_360_eps16/$name_1/ -m ./output/victim/$name_1/ 
)&