# settings
nepoch=50
consistency="mse"
dataset="UCI"
classes="UCI_classes"
data_folder="../uci_data/" 

source /Users/lizliao/miniconda3/etc/profile.d/conda.sh && \
conda activate torch && \
# source /Users/lizliao/miniconda3/bin/activate base
# python --version
chmod +x single_run.bash
python3 main.py --training_mode "ft" \
        --data_percentage "1" \
        --dataset $dataset \
        --classes $classes \
        --data_folder $data_folder\
        --nepoch $nepoch \
        --consistency $consistency \
        --batchsize=32 \
python3 main.py --training_mode "supervised" \
        --data_percentage "1" \
        --dataset $dataset \
        --classes $classes \
        --data_folder $data_folder\
        --nepoch $nepoch \
        --consistency $consistency \
        --batchsize=32 \
# mv result/UCI result/UCI_3conv