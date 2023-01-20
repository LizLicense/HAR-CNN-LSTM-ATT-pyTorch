nepoch=50
dataset="HHAR"
classes="HHAR_classes"
data_folder="../hhar_data/" 

#kld
# python3 main.py --training_mode "ssl" \
#         --data_percentage "1" \
#         --dataset $dataset \
#         --classes $classes \
#         --data_folder $data_folder\
#         --nepoch $nepoch \
#         --consistency "kld" \

# python3 main.py --training_mode "ft" \
#         --data_percentage "1" \
#         --dataset $dataset \
#         --classes $classes \
#         --data_folder $data_folder\
#         --nepoch $nepoch \
#         --consistency "kld" \

# python3 main.py --training_mode "ssl" \
#         --data_percentage "5" \
#         --dataset $dataset \
#         --classes $classes \
#         --data_folder $data_folder\
#         --nepoch $nepoch \
#         --consistency "kld" \

# python3 main.py --training_mode "ft" \
#         --data_percentage "5" \
#         --dataset $dataset \
#         --classes $classes \
#         --data_folder $data_folder\
#         --nepoch $nepoch \
#         --consistency "kld" \

# python3 main.py --training_mode "ssl" \
#         --data_percentage "10" \
#         --dataset $dataset \
#         --classes $classes \
#         --data_folder $data_folder\
#         --nepoch $nepoch \
#         --consistency "kld" \

# python3 main.py --training_mode "ft" \
#         --data_percentage "10" \
#         --dataset $dataset \
#         --classes $classes \
#         --data_folder $data_folder\
#         --nepoch $nepoch \
#         --consistency "kld" \

# python3 main.py --training_mode "ssl" \
#         --data_percentage "50" \
#         --dataset $dataset \
#         --classes $classes \
#         --data_folder $data_folder\
#         --nepoch $nepoch \
#         --consistency "kld" \

# python3 main.py --training_mode "ft" \
#     --data_percentage "50" \
#     --dataset $dataset \
#     --classes $classes \
#     --data_folder $data_folder\
#     --nepoch $nepoch \
#     --consistency "kld" \

# mv result/HHAR result/HHAR_kld_nor \



#MSE
# python3 main.py --training_mode "ssl" \
#     --data_percentage "1" \
#     --dataset $dataset \
#     --classes $classes \
#     --data_folder $data_folder\
#     --nepoch $nepoch \
#     --consistency "mse" \

# python3 main.py --training_mode "ft" \
#     --data_percentage "1" \
#     --dataset $dataset \
#     --classes $classes \
#     --data_folder $data_folder\
#     --nepoch $nepoch \
#     --consistency "mse"  \

# python3 main.py --training_mode "ssl" \
#     --data_percentage "5" \
#     --dataset $dataset \
#     --classes $classes \
#     --data_folder $data_folder\
#     --nepoch $nepoch \
#     --consistency "mse" \

# python3 main.py --training_mode "ft" \
#     --data_percentage "5" \
#     --dataset $dataset \
#     --classes $classes \
#     --data_folder $data_folder\
#     --nepoch $nepoch \
#     --consistency "mse"  \

# python3 main.py --training_mode "ssl" \
#     --data_percentage "10" \
#     --dataset $dataset \
#     --classes $classes \
#     --data_folder $data_folder\
#     --nepoch $nepoch \
#     --consistency "mse" \

# python3 main.py --training_mode "ft" \
#     --data_percentage "10" \
#     --dataset $dataset \
#     --classes $classes \
#     --data_folder $data_folder\
#     --nepoch $nepoch \
#     --consistency "mse"  \

# python3 main.py --training_mode "ssl" \
#     --data_percentage "50" \
#     --dataset $dataset \
#     --classes $classes \
#     --data_folder $data_folder\
#     --nepoch $nepoch \
#     --consistency "mse" \

# python3 main.py --training_mode "ft" \
#     --data_percentage "50" \
#     --dataset $dataset \
#     --classes $classes \
#     --data_folder $data_folder\
#     --nepoch $nepoch \
#     --consistency "mse" \

# #change folder name
# mv result/HHAR result/HHAR_mse \


#  #criterion
# python3 main.py --training_mode "ssl" \
#     --data_percentage "1" \
#     --dataset $dataset \
#     --classes $classes \
#     --data_folder $data_folder\
#     --nepoch $nepoch \
#     --consistency "criterion"\

# python3 main.py --training_mode "ft" \
#     --data_percentage "1" \
#     --dataset $dataset \
#     --classes $classes \
#     --data_folder $data_folder\
#     --nepoch $nepoch \
#     --consistency "criterion" \

# python3 main.py --training_mode "ssl" \
#     --data_percentage "5" \
#     --dataset $dataset \
#     --classes $classes \
#     --data_folder $data_folder\
#     --nepoch $nepoch \
#     --consistency "criterion" \

# python3 main.py --training_mode "ft" \
#     --data_percentage "5" \
#     --dataset $dataset \
#     --classes $classes \
#     --data_folder $data_folder\
#     --nepoch $nepoch \
#     --consistency "criterion" \

# python3 main.py --training_mode "ssl" \
#     --data_percentage "10" \
#     --dataset $dataset \
#     --classes $classes \
#     --data_folder $data_folder\
#     --nepoch $nepoch \
#     --consistency "criterion" \

# python3 main.py --training_mode "ft" \
#     --data_percentage "10" \
#     --dataset $dataset \
#     --classes $classes \
#     --data_folder $data_folder\
#     --nepoch $nepoch \
#     --consistency "criterion" \

# python3 main.py --training_mode "ssl" \
#     --data_percentage "50" \
#     --dataset $dataset \
#     --classes $classes \
#     --data_folder $data_folder\
#     --nepoch $nepoch \
#     --consistency "criterion" \

# python3 main.py --training_mode "ft" \
#     --data_percentage "50" \
#     --dataset $dataset \
#     --classes $classes \
#     --data_folder $data_folder\
#     --nepoch $nepoch \
#     --consistency "criterion" \

# mv result/HHAR result/HHAR_c \

# tri

# python3 main.py --training_mode "ssl" \
#     --data_percentage "1" \
#     --dataset $dataset \
#     --classes $classes \
#     --data_folder $data_folder\
#     --nepoch $nepoch \
#     --consistency "tri"\

# python3 main.py --training_mode "ft" \
#     --data_percentage "1" \
#     --dataset $dataset \
#     --classes $classes \
#     --data_folder $data_folder\
#     --nepoch $nepoch \
#     --consistency "tri" \

# python3 main.py --training_mode "ssl" \
#     --data_percentage "5" \
#     --dataset $dataset \
#     --classes $classes \
#     --data_folder $data_folder\
#     --nepoch $nepoch \
#     --consistency "tri" \

# python3 main.py --training_mode "ft" \
#     --data_percentage "5" \
#     --dataset $dataset \
#     --classes $classes \
#     --data_folder $data_folder\
#     --nepoch $nepoch \
#     --consistency "tri" \

# python3 main.py --training_mode "ssl" \
#     --data_percentage "10" \
#     --dataset $dataset \
#     --classes $classes \
#     --data_folder $data_folder\
#     --nepoch $nepoch \
#     --consistency "tri" \

# python3 main.py --training_mode "ft" \
#     --data_percentage "10" \
#     --dataset $dataset \
#     --classes $classes \
#     --data_folder $data_folder\
#     --nepoch $nepoch \
#     --consistency "tri" \

# python3 main.py --training_mode "ssl" \
#     --data_percentage "50" \
#     --dataset $dataset \
#     --classes $classes \
#     --data_folder $data_folder\
#     --nepoch $nepoch \
#     --consistency "tri" \

# python3 main.py --training_mode "ft" \
#     --data_percentage "50" \
#     --dataset $dataset \
#     --classes $classes \
#     --data_folder $data_folder\
#     --nepoch $nepoch \
#     --consistency "tri" \

# mv result/HHAR result/HHAR_tri \

# coe

python3 main.py --training_mode "ssl" \
    --data_percentage "1" \
    --dataset $dataset \
    --classes $classes \
    --data_folder $data_folder\
    --nepoch $nepoch \
    --consistency "coe"\

python3 main.py --training_mode "ft" \
    --data_percentage "1" \
    --dataset $dataset \
    --classes $classes \
    --data_folder $data_folder\
    --nepoch $nepoch \
    --consistency "coe" \

python3 main.py --training_mode "ssl" \
    --data_percentage "5" \
    --dataset $dataset \
    --classes $classes \
    --data_folder $data_folder\
    --nepoch $nepoch \
    --consistency "coe" \

python3 main.py --training_mode "ft" \
    --data_percentage "5" \
    --dataset $dataset \
    --classes $classes \
    --data_folder $data_folder\
    --nepoch $nepoch \
    --consistency "coe" \

python3 main.py --training_mode "ssl" \
    --data_percentage "10" \
    --dataset $dataset \
    --classes $classes \
    --data_folder $data_folder\
    --nepoch $nepoch \
    --consistency "coe" \

python3 main.py --training_mode "ft" \
    --data_percentage "10" \
    --dataset $dataset \
    --classes $classes \
    --data_folder $data_folder\
    --nepoch $nepoch \
    --consistency "coe" \

python3 main.py --training_mode "ssl" \
    --data_percentage "50" \
    --dataset $dataset \
    --classes $classes \
    --data_folder $data_folder\
    --nepoch $nepoch \
    --consistency "coe" \

python3 main.py --training_mode "ft" \
    --data_percentage "50" \
    --dataset $dataset \
    --classes $classes \
    --data_folder $data_folder\
    --nepoch $nepoch \
    --consistency "coe" \

mv result/HHAR result/HHAR_coe \


# #supervise

# # python3 main.py --training_mode "supervised" \
# #         --data_percentage "1" \
# #         --dataset $dataset \
# #         --classes $classes \
# #         --data_folder $data_folder\
# #         --nepoch $nepoch \
# #         --consistency "criterion" \

# # python3 main.py --training_mode "supervised" \
# #         --data_percentage "5" \
# #         --dataset $dataset \
# #         --classes $classes \
# #         --data_folder $data_folder\
# #         --nepoch $nepoch \
# #         --consistency "criterion" \

# # python3 main.py --training_mode "supervised" \
# #         --data_percentage "10" \
# #         --dataset $dataset \
# #         --classes $classes \
# #         --data_folder $data_folder\
# #         --nepoch $nepoch \
# #         --consistency "criterion" \

# # python3 main.py --training_mode "supervised" \
# #         --data_percentage "50" \
# #         --dataset $dataset \
# #         --classes $classes \
# #         --data_folder $data_folder\
# #         --nepoch $nepoch \
# #         --consistency "criterion" \


# # mv result/HHAR result/HHAR_supervised


