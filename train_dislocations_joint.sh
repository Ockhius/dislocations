python train_scmnet.py --config_file='delineation/configs/dislocation_matching_only_disp.yml'
python train_scmnet.py --config_file='delineation/configs/dislocation_matching_disp_and_warp_and var.yml'
python train_scmnet.py --config_file='delineation/configs/dislocation_matching_disp_and_warp.yml'
python train_scmnet.py --config_file='delineation/configs/dislocation_matching_disp_and_var.yml'
python train_scmnet.py --config_file='delineation/configs/dislocation_matching_only_warp.yml'
cd /cvlabsrc1/cvlab/datasets_anastasiia/descriptors/glam-log-polar
python modules/hardnet/hardnet.py