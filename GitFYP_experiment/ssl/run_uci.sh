
# settings
# training_mode="ssl"
nepoch=50
# data_percentage="10"
consistency="criterion"
dataset="UCI"
classes="UCI_classes"
data_folder="../uci_data/" 

    # python3 main.py --training_mode "ssl" \
    #     --data_percentage "10" \
    #     --dataset $dataset \
    #     --classes $classes \
    #     --data_folder $data_folder\
    #     --nepoch $nepoch \
    #     --consistency $consistency

    # python3 main.py --training_mode "ft" \
    #     --data_percentage "10" \
    #     --dataset $dataset \
    #     --classes $classes \
    #     --data_folder $data_folder\
    #     --nepoch $nepoch \
    #     --consistency $consistency

    # python3 main.py --training_mode "ssl" \
    #     --data_percentage "50" \
    #     --dataset $dataset \
    #     --classes $classes \
    #     --data_folder $data_folder\
    #     --nepoch $nepoch \
    #     --consistency $consistency

python3 main.py --training_mode "ft" \
    --data_percentage "50" \
    --dataset $dataset \
    --classes $classes \
    --data_folder $data_folder\
    --nepoch $nepoch \
    --consistency $consistency 

mv result/UCI result/UCI_C

