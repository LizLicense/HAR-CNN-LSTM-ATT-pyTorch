
# ç
nepoch=50
# data_percentage="10"
consistency="kld"
dataset="HHAR"
classes="HHAR_classes"
data_folder="../hhar_data/" 

python3 main.py --training_mode "ssl" \
        --data_percentage "1" \
        --dataset $dataset \
        --classes $classes \
        --data_folder $data_folder\
        --nepoch $nepoch \
        --consistency $consistency \

python3 main.py --training_mode "ft" \
        --data_percentage "1" \
        --dataset $dataset \
        --classes $classes \
        --data_folder $data_folder\
        --nepoch $nepoch \
        --consistency $consistency \

python3 main.py --training_mode "ssl" \
        --data_percentage "5" \
        --dataset $dataset \
        --classes $classes \
        --data_folder $data_folder\
        --nepoch $nepoch \
        --consistency $consistency \

python3 main.py --training_mode "ft" \
        --data_percentage "5" \
        --dataset $dataset \
        --classes $classes \
        --data_folder $data_folder\
        --nepoch $nepoch \
        --consistency $consistency \

python3 main.py --training_mode "ssl" \
        --data_percentage "10" \
        --dataset $dataset \
        --classes $classes \
        --data_folder $data_folder\
        --nepoch $nepoch \
        --consistency $consistency \

python3 main.py --training_mode "ft" \
        --data_percentage "10" \
        --dataset $dataset \
        --classes $classes \
        --data_folder $data_folder\
        --nepoch $nepoch \
        --consistency $consistency \

python3 main.py --training_mode "ssl" \
        --data_percentage "50" \
        --dataset $dataset \
        --classes $classes \
        --data_folder $data_folder\
        --nepoch $nepoch \
        --consistency $consistency \

python3 main.py --training_mode "ft" \
    --data_percentage "50" \
    --dataset $dataset \
    --classes $classes \
    --data_folder $data_folder\
    --nepoch $nepoch \
    --consistency $consistency \

mv result/HHAR result/HHAR_kld

# consistency ="mse"

# #MSE
# python3 main.py --training_mode "ssl" \
#     --data_percentage "5" \
#     --dataset $dataset \
#     --classes $classes \
#     --data_folder $data_folder\
#     --nepoch $nepoch \
#     --consistency $consistency \

# python3 main.py --training_mode "ft" \
#     --data_percentage "5" \
#     --dataset $dataset \
#     --classes $classes \
#     --data_folder $data_folder\
#     --nepoch $nepoch \
#     --consistency $consistency \

# python3 main.py --training_mode "ssl" \
#     --data_percentage "10" \
#     --dataset $dataset \
#     --classes $classes \
#     --data_folder $data_folder\
#     --nepoch $nepoch \
#     --consistency $consistency \

# python3 main.py --training_mode "ft" \
#     --data_percentage "10" \
#     --dataset $dataset \
#     --classes $classes \
#     --data_folder $data_folder\
#     --nepoch $nepoch \
#     --consistency $consistency \

# python3 main.py --training_mode "ssl" \
#     --data_percentage "50" \
#     --dataset $dataset \
#     --classes $classes \
#     --data_folder $data_folder\
#     --nepoch $nepoch \
#     --consistency $consistency \

# python3 main.py --training_mode "ft" \
#     --data_percentage "50" \
#     --dataset $dataset \
#     --classes $classes \
#     --data_folder $data_folder\
#     --nepoch $nepoch \
#     --consistency $consistency \

# #change folder name
# mv result/HHAR result/HHAR_mse \

# consistency ="kld"
# #kld
# python3 main.py --training_mode "ssl" \
#     --data_percentage "5" \
#     --dataset $dataset \
#     --classes $classes \
#     --data_folder $data_folder\
#     --nepoch $nepoch \
#     --consistency $consistency \

# python3 main.py --training_mode "ft" \
#     --data_percentage "5" \
#     --dataset $dataset \
#     --classes $classes \
#     --data_folder $data_folder\
#     --nepoch $nepoch \
#     --consistency $consistency \

# python3 main.py --training_mode "ssl" \
#     --data_percentage "10" \
#     --dataset $dataset \
#     --classes $classes \
#     --data_folder $data_folder\
#     --nepoch $nepoch \
#     --consistency $consistency \

# python3 main.py --training_mode "ft" \
#     --data_percentage "10" \
#     --dataset $dataset \
#     --classes $classes \
#     --data_folder $data_folder\
#     --nepoch $nepoch \
#     --consistency $consistency \

# python3 main.py --training_mode "ssl" \
#     --data_percentage "50" \
#     --dataset $dataset \
#     --classes $classes \
#     --data_folder $data_folder\
#     --nepoch $nepoch \
#     --consistency $consistency \

# python3 main.py --training_mode "ft" \
#     --data_percentage "50" \
#     --dataset $dataset \
#     --classes $classes \
#     --data_folder $data_folder\
#     --nepoch $nepoch \
#     --consistency $consistency \

# mv result/HHAR result/HHAR_kld



