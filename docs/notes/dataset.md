## detection
DeepFashion: Fashion Landmark Detection
http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/LandmarkDetection.html

OSError: [Errno 107] Transport endpoint is not connected: '/unsullied/sharefs/liangjiajun/ceph-home'


rlaunch --cpu=40 --memory=10240 -- mdl dstool.py serve_subgroup --train 8 --validation 1 --num_group 16 --group_index 0
rlaunch --cpu=40 --memory=10240 -- mdl dstool.py serve_subgroup --train 8 --validation 1 --num_group 16 --group_index 1
rlaunch --cpu=40 --memory=10240 -- mdl dstool.py serve_subgroup --train 8 --validation 1 --num_group 16 --group_index 2
rlaunch --cpu=40 --memory=10240 -- mdl dstool.py serve_subgroup --train 8 --validation 1 --num_group 16 --group_index 3
rlaunch --cpu=40 --memory=10240 -- mdl dstool.py serve_subgroup --train 8 --validation 1 --num_group 16 --group_index 4
rlaunch --cpu=40 --memory=10240 -- mdl dstool.py serve_subgroup --train 8 --validation 1 --num_group 16 --group_index 5
rlaunch --cpu=40 --memory=10240 -- mdl dstool.py serve_subgroup --train 8 --validation 1 --num_group 16 --group_index 6
rlaunch --cpu=40 --memory=10240 -- mdl dstool.py serve_subgroup --train 8 --validation 1 --num_group 16 --group_index 7
rlaunch --cpu=40 --memory=10240 -- mdl dstool.py serve_subgroup --train 8 --validation 1 --num_group 16 --group_index 8
rlaunch --cpu=40 --memory=10240 -- mdl dstool.py serve_subgroup --train 8 --validation 1 --num_group 16 --group_index 9
rlaunch --cpu=40 --memory=10240 -- mdl dstool.py serve_subgroup --train 8 --validation 1 --num_group 16 --group_index 10
rlaunch --cpu=40 --memory=10240 -- mdl dstool.py serve_subgroup --train 8 --validation 1 --num_group 16 --group_index 11
rlaunch --cpu=40 --memory=10240 -- mdl dstool.py serve_subgroup --train 8 --validation 1 --num_group 16 --group_index 12
rlaunch --cpu=40 --memory=10240 -- mdl dstool.py serve_subgroup --train 8 --validation 1 --num_group 16 --group_index 13
rlaunch --cpu=40 --memory=10240 -- mdl dstool.py serve_subgroup --train 8 --validation 1 --num_group 16 --group_index 14
rlaunch --cpu=40 --memory=10240 -- mdl dstool.py serve_subgroup --train 8 --validation 1 --num_group 16 --group_index 15

rlaunch --charged-group=research_llcv --cpu=40 --memory=20240 -- mdl dstool.py serve_subgroup --train 8 --validation 1 --num_group 16 --group_index 0
rlaunch --charged-group=research_llcv --cpu=40 --memory=20240 -- mdl dstool.py serve_subgroup --train 8 --validation 1 --num_group 16 --group_index 1
rlaunch --charged-group=research_llcv --cpu=40 --memory=20240 -- mdl dstool.py serve_subgroup --train 8 --validation 1 --num_group 16 --group_index 2
rlaunch --charged-group=research_llcv --cpu=40 --memory=20240 -- mdl dstool.py serve_subgroup --train 8 --validation 1 --num_group 16 --group_index 3
rlaunch --charged-group=research_llcv --cpu=40 --memory=20240 -- mdl dstool.py serve_subgroup --train 8 --validation 1 --num_group 16 --group_index 4
rlaunch --charged-group=research_llcv --cpu=40 --memory=20240 -- mdl dstool.py serve_subgroup --train 8 --validation 1 --num_group 16 --group_index 5
rlaunch --charged-group=research_llcv --cpu=40 --memory=20240 -- mdl dstool.py serve_subgroup --train 8 --validation 1 --num_group 16 --group_index 6
rlaunch --charged-group=research_llcv --cpu=40 --memory=20240 -- mdl dstool.py serve_subgroup --train 8 --validation 1 --num_group 16 --group_index 7
rlaunch --charged-group=research_llcv --cpu=40 --memory=20240 -- mdl dstool.py serve_subgroup --train 8 --validation 1 --num_group 16 --group_index 8
rlaunch --charged-group=research_llcv --cpu=40 --memory=20240 -- mdl dstool.py serve_subgroup --train 8 --validation 1 --num_group 16 --group_index 9
rlaunch --charged-group=research_llcv --cpu=40 --memory=20240 -- mdl dstool.py serve_subgroup --train 8 --validation 1 --num_group 16 --group_index 10
rlaunch --charged-group=research_llcv --cpu=40 --memory=20240 -- mdl dstool.py serve_subgroup --train 8 --validation 1 --num_group 16 --group_index 11
rlaunch --charged-group=research_llcv --cpu=40 --memory=20240 -- mdl dstool.py serve_subgroup --train 8 --validation 1 --num_group 16 --group_index 12
rlaunch --charged-group=research_llcv --cpu=40 --memory=20240 -- mdl dstool.py serve_subgroup --train 8 --validation 1 --num_group 16 --group_index 13
rlaunch --charged-group=research_llcv --cpu=40 --memory=20240 -- mdl dstool.py serve_subgroup --train 8 --validation 1 --num_group 16 --group_index 14
rlaunch --charged-group=research_llcv --cpu=40 --memory=20240 -- mdl dstool.py serve_subgroup --train 8 --validation 1 --num_group 16 --group_index 15

感性认识benchmark做的不好的地方

活人死人比3:1

最开始的filelist版本和加了koala的版本对比

严重掉点， 重新train才有公平的结论
koala 数据 不能随便加