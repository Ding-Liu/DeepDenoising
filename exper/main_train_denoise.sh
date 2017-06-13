# One input parameter is needed, which is the job id.
#export PYTHONPATH='./caffe/python/':$PYTHONPATH
jobid=$1

if [ -z "${jobid}" ]; then
    echo "jobid is empty"
    exit
fi

gpu_id=0
save_path=model/${jobid}

base_lr=1e-4
snap_int=25000
image_list=/your/path/to/image/list/withdummylabels.txt
image_root=/your/path/to/image/folder/
batch_size=64
crop_size=40
sigma=30
crop_offset=3

solver_file=config/solver_denoise.prototxt
train_layer_file=config/train_denoise.prototxt

# if finetune model is needed
#finetune_model=model/denoise-s30.caffemodel

mkdir -p $save_path
rm -r $save_path/*

# prepare network/solver
train_layer_target=$save_path/$(basename $train_layer_file)
solver_target=$save_path/$(basename $solver_file)

cp $train_layer_file $train_layer_target
cp $solver_file $solver_target

sed -i 's:<TRAIN_NET_HOLDER>:./'$train_layer_target':g' $solver_target
sed -i 's:<SAVE_PATH_HOLDER>:./'$save_path':g' $solver_target
sed -i 's:<BASE_LR_HOLDER>:'${base_lr}':g' $solver_target
sed -i 's:<SNAP_INT_HOLDER>:'${snap_int}':g' $solver_target

sed -i 's:<IMAGE_LIST_HOLDER>:'$image_list':g' $train_layer_target
sed -i 's:<IMAGE_ROOT_HOLDER>:'$image_root':g' $train_layer_target
sed -i 's:<BATCH_SIZE_HOLDER>:'$batch_size':g' $train_layer_target
sed -i 's:<CROP_SIZE_HOLDER>:'$crop_size':g' $train_layer_target
sed -i 's:<SIGMA_HOLDER>:'$sigma':g' $train_layer_target
#sed -i 's:<CROP_OFFSET1_HOLDER>:'$(($crop_offset-2))':g' $train_layer_target
sed -i 's:<CROP_OFFSET2_HOLDER>:'$crop_offset':g' $train_layer_target

arg="tools/train_net.py --GPU=$gpu_id --solver=$solver_target"
#arg="tools/train_net.py --GPU=$gpu_id --solver=$solver_target --weights=$finetune_model"
echo $arg
python $arg
