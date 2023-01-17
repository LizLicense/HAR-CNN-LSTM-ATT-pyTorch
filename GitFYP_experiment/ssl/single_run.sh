
# settings
training_mode="ft"
nepoch=50
data_percentage="1"
consistency="criterion"
dataset="UCI"
classes="UCI_classes"
data_folder="../uci_data/" 

python3 main.py --training_mode $training_mode \
    --data_percentage $data_percentage \
    --dataset $dataset \
    --classes $classes \
    --data_folder $data_folder\
    --nepoch $nepoch \
    --consistency $consistency