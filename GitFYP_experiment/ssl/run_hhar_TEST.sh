nepoch=50
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

mv result/HHAR result/HHAR_kld \



#MSE
python3 main.py --training_mode "ssl" \
    --data_percentage "1" \
    --dataset $dataset \
    --classes $classes \
    --data_folder $data_folder\
    --nepoch $nepoch \
    --consistency "mse" \

python3 main.py --training_mode "ft" \
    --data_percentage "1" \
    --dataset $dataset \
    --classes $classes \
    --data_folder $data_folder\
    --nepoch $nepoch \
    --consistency "mse"  \

python3 main.py --training_mode "ssl" \
    --data_percentage "5" \
    --dataset $dataset \
    --classes $classes \
    --data_folder $data_folder\
    --nepoch $nepoch \
    --consistency "mse" \

python3 main.py --training_mode "ft" \
    --data_percentage "5" \
    --dataset $dataset \
    --classes $classes \
    --data_folder $data_folder\
    --nepoch $nepoch \
    --consistency "mse"  \

python3 main.py --training_mode "ssl" \
    --data_percentage "10" \
    --dataset $dataset \
    --classes $classes \
    --data_folder $data_folder\
    --nepoch $nepoch \
    --consistency "mse" \

python3 main.py --training_mode "ft" \
    --data_percentage "10" \
    --dataset $dataset \
    --classes $classes \
    --data_folder $data_folder\
    --nepoch $nepoch \
    --consistency "mse"  \

python3 main.py --training_mode "ssl" \
    --data_percentage "50" \
    --dataset $dataset \
    --classes $classes \
    --data_folder $data_folder\
    --nepoch $nepoch \
    --consistency "mse" \

python3 main.py --training_mode "ft" \
    --data_percentage "50" \
    --dataset $dataset \
    --classes $classes \
    --data_folder $data_folder\
    --nepoch $nepoch \
    --consistency "mse" \

#change folder name
mv result/HHAR result/HHAR_mse \


# #criterion
python3 main.py --training_mode "ssl" \
    --data_percentage "1" \
    --dataset $dataset \
    --classes $classes \
    --data_folder $data_folder\
    --nepoch $nepoch \
    --consistency "criterion"\

python3 main.py --training_mode "ft" \
    --data_percentage "1" \
    --dataset $dataset \
    --classes $classes \
    --data_folder $data_folder\
    --nepoch $nepoch \
    --consistency "criterion" \

python3 main.py --training_mode "ssl" \
    --data_percentage "5" \
    --dataset $dataset \
    --classes $classes \
    --data_folder $data_folder\
    --nepoch $nepoch \
    --consistency "criterion" \

python3 main.py --training_mode "ft" \
    --data_percentage "5" \
    --dataset $dataset \
    --classes $classes \
    --data_folder $data_folder\
    --nepoch $nepoch \
    --consistency "criterion" \

python3 main.py --training_mode "ssl" \
    --data_percentage "10" \
    --dataset $dataset \
    --classes $classes \
    --data_folder $data_folder\
    --nepoch $nepoch \
    --consistency "criterion" \

python3 main.py --training_mode "ft" \
    --data_percentage "10" \
    --dataset $dataset \
    --classes $classes \
    --data_folder $data_folder\
    --nepoch $nepoch \
    --consistency "criterion" \

python3 main.py --training_mode "ssl" \
    --data_percentage "50" \
    --dataset $dataset \
    --classes $classes \
    --data_folder $data_folder\
    --nepoch $nepoch \
    --consistency "criterion" \

python3 main.py --training_mode "ft" \
    --data_percentage "50" \
    --dataset $dataset \
    --classes $classes \
    --data_folder $data_folder\
    --nepoch $nepoch \
    --consistency "criterion" \

mv result/HHAR result/HHAR_c \


#supervise

python3 main.py --training_mode "supervised" \
        --data_percentage "1" \
        --dataset $dataset \
        --classes $classes \
        --data_folder $data_folder\
        --nepoch $nepoch \
        --consistency "criterion" \

python3 main.py --training_mode "supervised" \
        --data_percentage "5" \
        --dataset $dataset \
        --classes $classes \
        --data_folder $data_folder\
        --nepoch $nepoch \
        --consistency "criterion" \

python3 main.py --training_mode "supervised" \
        --data_percentage "10" \
        --dataset $dataset \
        --classes $classes \
        --data_folder $data_folder\
        --nepoch $nepoch \
        --consistency "criterion" \

python3 main.py --training_mode "supervised" \
        --data_percentage "50" \
        --dataset $dataset \
        --classes $classes \
        --data_folder $data_folder\
        --nepoch $nepoch \
        --consistency "criterion" \


mv result/HHAR result/HHAR_supervised


