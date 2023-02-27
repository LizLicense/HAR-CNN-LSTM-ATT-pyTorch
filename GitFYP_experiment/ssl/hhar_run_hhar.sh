
# settings

nepoch=50
# data_percentage="10"
consistency="mse"
dataset="HHAR"
classes="HHAR_classes"
data_folder="../hhar_data/" 
source /Users/lizliao/miniconda3/etc/profile.d/conda.sh && \
conda activate base && \

python3 main.py --training_mode "ssl" \
        --data_percentage "1" \
        --dataset $dataset \
        --classes $classes \
        --data_folder $data_folder\
        --nepoch $nepoch \
        --consistency $consistency\
        --batchsize=32 \



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



python3 main.py --training_mode "ssl" \
        --data_percentage "5" \
        --dataset $dataset \
        --classes $classes \
        --data_folder $data_folder\
        --nepoch $nepoch \
        --consistency $consistency\
        --batchsize=32 \




python3 main.py --training_mode "ft" \
        --data_percentage "5" \
        --dataset $dataset \
        --classes $classes \
        --data_folder $data_folder\
        --nepoch $nepoch \
        --consistency $consistency \
        --batchsize=32 \



python3 main.py --training_mode "supervised" \
        --data_percentage "5" \
        --dataset $dataset \
        --classes $classes \
        --data_folder $data_folder\
        --nepoch $nepoch \
        --consistency $consistency \
        --batchsize=32 \



python3 main.py --training_mode "ssl" \
        --data_percentage "10" \
        --dataset $dataset \
        --classes $classes \
        --data_folder $data_folder\
        --nepoch $nepoch \
        --consistency $consistency \
        --batchsize=32 \




python3 main.py --training_mode "ft" \
        --data_percentage "10" \
        --dataset $dataset \
        --classes $classes \
        --data_folder $data_folder\
        --nepoch $nepoch \
        --consistency $consistency \
        --batchsize=32 \



python3 main.py --training_mode "supervised" \
        --data_percentage "10" \
        --dataset $dataset \
        --classes $classes \
        --data_folder $data_folder\
        --nepoch $nepoch \
        --consistency $consistency \
        --batchsize=32 \



python3 main.py --training_mode "ssl" \
        --data_percentage "50" \
        --dataset $dataset \
        --classes $classes \
        --data_folder $data_folder\
        --nepoch $nepoch \
        --consistency $consistency \
        --batchsize=32 \



python3 main.py --training_mode "ft" \
    --data_percentage "50" \
    --dataset $dataset \
    --classes $classes \
    --data_folder $data_folder\
    --nepoch $nepoch \
    --consistency $consistency \
        --batchsize=32 \



python3 main.py --training_mode "supervised" \
        --data_percentage "50" \
        --dataset $dataset \
        --classes $classes \
        --data_folder $data_folder\
        --nepoch $nepoch \
        --consistency $consistency \
        --batchsize=32 \


mv result/HHAR result/HHAR_3CONV

