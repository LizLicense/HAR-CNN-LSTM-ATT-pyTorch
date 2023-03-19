
# settings

nepoch=50
# data_percentage="10"
# oversample=False
consistency="mse"
dataset="UCI"
classes="UCI_classes"
data_folder="../uci_data/" 
# source /Users/lizliao/miniconda3/etc/profile.d/conda.sh && \
# conda activate base && \

/Users/lizliao/miniconda3/bin/python3 main.py --training_mode "ssl" \
        --data_percentage "1" \
        --dataset $dataset \
        --classes $classes \
        --data_folder $data_folder\
        --nepoch $nepoch \
        --consistency $consistency\
        --batchsize=32 \


/Users/lizliao/miniconda3/bin/python3 main.py --training_mode "ft" \
        --data_percentage "1" \
        --dataset $dataset \
        --classes $classes \
        --data_folder $data_folder\
        --nepoch $nepoch \
        --consistency $consistency \
        --batchsize=32 \



/Users/lizliao/miniconda3/bin/python3 main.py --training_mode "supervised" \
        --data_percentage "1" \
        --dataset $dataset \
        --classes $classes \
        --data_folder $data_folder\
        --nepoch $nepoch \
        --consistency $consistency \
        --batchsize=32 \



/Users/lizliao/miniconda3/bin/python3 main.py --training_mode "ssl" \
        --data_percentage "5" \
        --dataset $dataset \
        --classes $classes \
        --data_folder $data_folder\
        --nepoch $nepoch \
        --consistency $consistency\
        --batchsize=32 \




/Users/lizliao/miniconda3/bin/python3 main.py --training_mode "ft" \
        --data_percentage "5" \
        --dataset $dataset \
        --classes $classes \
        --data_folder $data_folder\
        --nepoch $nepoch \
        --consistency $consistency \
        --batchsize=32 \



/Users/lizliao/miniconda3/bin/python3 main.py --training_mode "supervised" \
        --data_percentage "5" \
        --dataset $dataset \
        --classes $classes \
        --data_folder $data_folder\
        --nepoch $nepoch \
        --consistency $consistency \
        --batchsize=32 \



# /Users/lizliao/miniconda3/bin/python3 main.py --training_mode "ssl" \
#         --data_percentage "10" \
#         --dataset $dataset \
#         --classes $classes \
#         --data_folder $data_folder\
#         --nepoch $nepoch \
#         --consistency $consistency \
#         --batchsize=32 \




# /Users/lizliao/miniconda3/bin/python3 main.py --training_mode "ft" \
#         --data_percentage "10" \
#         --dataset $dataset \
#         --classes $classes \
#         --data_folder $data_folder\
#         --nepoch $nepoch \
#         --consistency $consistency \
#         --batchsize=32 \



# # /Users/lizliao/miniconda3/bin/python3 main.py --training_mode "supervised" \
# #         --data_percentage "10" \
# #         --dataset $dataset \
# #         --classes $classes \
# #         --data_folder $data_folder\
# #         --nepoch $nepoch \
# #         --consistency $consistency \
# #         --batchsize=32 \



/Users/lizliao/miniconda3/bin/python3 main.py --training_mode "ssl" \
        --data_percentage "50" \
        --dataset $dataset \
        --classes $classes \
        --data_folder $data_folder\
        --nepoch $nepoch \
        --consistency $consistency \
        --batchsize=32 \



/Users/lizliao/miniconda3/bin/python3 main.py --training_mode "ft" \
    --data_percentage "50" \
    --dataset $dataset \
    --classes $classes \
    --data_folder $data_folder\
    --nepoch $nepoch \
    --consistency $consistency \
        --batchsize=32 \



/Users/lizliao/miniconda3/bin/python3 main.py --training_mode "supervised" \
        --data_percentage "50" \
        --dataset $dataset \
        --classes $classes \
        --data_folder $data_folder\
        --nepoch $nepoch \
        --consistency $consistency \
        --batchsize=32 \


# mv result/UCI result/UCI_3conv

