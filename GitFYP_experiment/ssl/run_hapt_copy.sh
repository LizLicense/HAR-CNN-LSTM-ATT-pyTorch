
# settings
nepoch=50
consistency="mse"
dataset="HAPT"
classes="HAPT_classes"
data_folder="../hapt_data/" 

# env_script = '/Users/lizliao/miniconda3/bin/python'
#replace the conda environment path: /envs/env/bin/python /script_path/pythonfile.py
# python3 
# /Users/lizliao/miniconda3/bin/python main.py  --training_mode "ssl" \
#         --data_percentage "1" \
#         --dataset $dataset \
#         --classes $classes \
#         --data_folder $data_folder\
#         --nepoch $nepoch \
#         --consistency $consistency\
#         --batchsize=32 \
        

# /Users/lizliao/miniconda3/bin/python main.py  --training_mode "ft" \
#         --data_percentage "1" \
#         --dataset $dataset \
#         --classes $classes \
#         --data_folder $data_folder\
#         --nepoch $nepoch \
#         --consistency $consistency \
#         --batchsize=32 \

/Users/lizliao/miniconda3/bin/python main.py  --training_mode "ssl" \
        --data_percentage "5" \
        --dataset $dataset \
        --classes $classes \
        --data_folder $data_folder\
        --nepoch $nepoch \
        --consistency $consistency\
        --batchsize=32 \
        

/Users/lizliao/miniconda3/bin/python3 main.py  --training_mode "ft" \
        --data_percentage "5" \
        --dataset $dataset \
        --classes $classes \
        --data_folder $data_folder\
        --nepoch $nepoch \
        --consistency $consistency\
        --batchsize=32 \
        
mv /Users/lizliao/Downloads/HAR-CNN-LSTM/GitFYP_experiment/ssl/result/HAPT /Users/lizliao/Downloads/HAR-CNN-LSTM/GitFYP_experiment/ssl/result/HAPT_CONV3OVERSAMPLE
# /Users/lizliao/miniconda3/bin/python main.py  --training_mode "supervised" \
#         --data_percentage "5" \
#         --dataset $dataset \
#         --classes $classes \
#         --data_folder $data_folder\
#         --nepoch $nepoch \
#         --consistency $consistency \
#         --batchsize=32 \

# mv ssl/result/HAPT ssl/result/HAPT_conv3_nooversample
# /Users/lizliao/miniconda3/bin/python main.py  --training_mode "ssl" \
#         --data_percentage "10" \
#         --dataset $dataset \
#         --classes $classes \
#         --data_folder $data_folder\
#         --nepoch $nepoch \
#         --consistency $consistency\
#         --batchsize=32 \
        

# /Users/lizliao/miniconda3/bin/python main.py  --training_mode "ft" \
#         --data_percentage "10" \
#         --dataset $dataset \
#         --classes $classes \
#         --data_folder $data_folder\
#         --nepoch $nepoch \
#         --consistency $consistency \
#         --batchsize=32 \


# /Users/lizliao/miniconda3/bin/python main.py  --training_mode "ssl" \
#         --data_percentage "50" \
#         --dataset $dataset \
#         --classes $classes \
#         --data_folder $data_folder\
#         --nepoch $nepoch \
#         --consistency $consistency\
#         --batchsize=32 \
        

# /Users/lizliao/miniconda3/bin/python main.py  --training_mode "ft" \
#         --data_percentage "50" \
#         --dataset $dataset \
#         --classes $classes \
#         --data_folder $data_folder\
#         --nepoch $nepoch \
#         --consistency $consistency \
#         --batchsize=32 \


