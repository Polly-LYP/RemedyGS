name_1="bonsai"
name_2="kitchen"


(mkdir -p "./output/victim/$name_2/"
python ./victim/gaussian-splatting/benchmark.py --gpu 6\
    -s ./example/output/$name_2/ -m ./output/victim/$name_2/ 
)&

(mkdir -p "./output/victim/$name_1/"
python ./victim/gaussian-splatting/benchmark.py --gpu 7\
    -s ./example/output/$name_1/ -m ./output/victim/$name_1/ 
)&