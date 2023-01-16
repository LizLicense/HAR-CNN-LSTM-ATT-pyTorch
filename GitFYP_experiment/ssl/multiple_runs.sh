
nepoch = 1
start=0
end=1

sleep_model="dsn"
for i in $(eval echo {$start..$end})
# criterion
do
   python3 main.py \
   --training_mode "ssl" \
   --data_percentage "50"\
   --dataset "UCI"\
   --consistency "criterion"\
   --classes "UCI_classes"\
   --data_folder "../uci_data/"\
   --nepoch $nepoch 
done

for i in $(eval echo {$start..$end})
do
   python3 main.py \
   --training_mode "ft" \
   --data_percentage "50"\
   --dataset "UCI"\
   --consistency "criterion"\
   --classes "UCI_classes"\
   --data_folder "../uci_data/"\
   --nepoch $nepoch 

done

python3 main.py \
   --training_mode "ssl" \
   --data_percentage "10"\
   --dataset "UCI"\
   --consistency "criterion"\
   --classes "UCI_classes" \
   --data_folder "../uci_data/"\
   --nepoch $nepoch \


python3 main.py \
   --training_mode "ft" \
   --data_percentage "10"\
   --dataset "UCI"\
   --consistency "criterion"\
   --classes "UCI_classes"\
   --data_folder "../uci_data/"\
   --nepoch $nepoch \

python3 main.py \
   --training_mode "ssl" \
   --data_percentage "5"\
   --dataset "UCI"\
   --consistency "criterion"\
   --classes "UCI_classes"\
   --data_folder "../uci_data/"\
   --nepoch $nepoch \



python3 main.py \
   --training_mode "ft" \
   --data_percentage "5"\
   --dataset "UCI"\
   --consistency "criterion"\
   --classes "UCI_classes"\
   --data_folder "../uci_data/"\
   --nepoch $nepoch \



# python3 main.py \
#    --training_mode "ssl" \
#    --data_percentage "1"\
#    --dataset "UCI"\
#    --consistency "criterion"\
#    --batchsize 32 \
#    --classes "UCI_classes"
#    --data_folder "../uci_data/"\
   # --nepoch $nepoch \




# python3 main.py \
#    # --device $device \
#    --training_mode "ft" \
#    --data_percentage "1"\
#    --dataset "UCI"\
#    --consistency "criterion"\
#    --batchsize 32\
#    --classes "UCI_classes"\
#    --data_folder "../uci_data/"\
# --nepoch $nepoch \

# kld
python3 main.py \
   --training_mode "ssl" \
   --data_percentage "50"\
   --dataset "UCI"\
   --consistency "kld"\
   --classes "UCI_classes"\
   --data_folder "../uci_data/"\
   --nepoch $nepoch \


python3 main.py \
   --training_mode "ft" \
   --data_percentage "50"\
   --dataset "UCI"\
   --consistency "kld"\
   --classes "UCI_classes"\
   --data_folder "../uci_data/"\
   --nepoch $nepoch \



python3 main.py \
   --training_mode "ssl" \
   --data_percentage "10"\
   --dataset "UCI"\
   --consistency "kld"\
   --classes "UCI_classes" \
   --data_folder "../uci_data/"\
   --nepoch $nepoch \


python3 main.py \
   --training_mode "ft" \
   --data_percentage "10"\
   --dataset "UCI"\
   --consistency "kld"\
   --classes "UCI_classes"\
   --data_folder "../uci_data/"\
   --nepoch $nepoch \

python3 main.py \
   --training_mode "ssl" \
   --data_percentage "5"\
   --dataset "UCI"\
   --consistency "kld"\
   --classes "UCI_classes"\
   --data_folder "../uci_data/"\
   --nepoch $nepoch \


python3 main.py \
   --training_mode "ft" \
   --data_percentage "5"\
   --dataset "UCI"\
   --consistency "kld"\
   --classes "UCI_classes"\
   --data_folder "../uci_data/"\
   --nepoch $nepoch \



# python3 main.py \
#    --training_mode "ssl" \
#    --data_percentage "1"\
#    --dataset "UCI"\
#    --consistency "kld"\
#    --batchsize 32 \
#    --classes "UCI_classes"\
#    --data_folder "../uci_data/"\
# --nepoch $nepoch \




# python3 main.py \
#    # --device $device \
#    --training_mode "ft" \
#    --data_percentage "1"\
#    --dataset "UCI"\
#    --consistency "kld"\
#    --batchsize 32\
#    --classes "UCI_classes"\
#    --data_folder "../uci_data/"\
# --nepoch $nepoch \


# mse
python3 main.py \
   --training_mode "ssl" \
   --data_percentage "50"\
   --dataset "UCI"\
   --consistency "mse"\
   --classes "UCI_classes"\
   --data_folder "../uci_data/"\
   --nepoch $nepoch \
 


python3 main.py \
   --training_mode "ft" \
   --data_percentage "50"\
   --dataset "UCI"\
   --consistency "mse"\
   --classes "UCI_classes"\
   --data_folder "../uci_data/"\
   --nepoch $nepoch \



python3 main.py \
   --training_mode "ssl" \
   --data_percentage "10"\
   --dataset "UCI"\
   --consistency "mse"\
   --classes "UCI_classes"\
   --data_folder "../uci_data/"\
   --nepoch $nepoch \


python3 main.py \
   --training_mode "ft" \
   --data_percentage "10"\
   --dataset "UCI"\
   --consistency "mse"\
   --classes "UCI_classes"\
   --data_folder "../uci_data/"\
   --nepoch $nepoch \

python3 main.py \
   --training_mode "ssl" \
   --data_percentage "5"\
   --dataset "UCI"\
   --consistency "mse"\
   --classes "UCI_classes"\
   --data_folder "../uci_data/"\
   --nepoch $nepoch \


python3 main.py \
   --training_mode "ft" \
   --data_percentage "5"\
   --dataset "UCI"\
   --consistency "mse"\
   --classes "UCI_classes"\
   --data_folder "../uci_data/"\
   --nepoch $nepoch \



# python3 main.py \
#    --training_mode "ssl" \
#    --data_percentage "1"\
#    --dataset "UCI"\
#    --consistency "mse"\
#    --batchsize 32 \
#    --classes "UCI_classes"\
#    --data_folder "../uci_data/"\
# --nepoch $nepoch \




# python3 main.py \
#    # --device $device \
#    --training_mode "ft" \
#    --data_percentage "1"\
#    --dataset "UCI"\
#    --consistency "kld"\
#    --batchsize 32\
#    --classes "UCI_classes"\
#    --data_folder "../uci_data/"\
# --nepoch $nepoch \
