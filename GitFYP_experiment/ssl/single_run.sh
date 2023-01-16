
# criterion
python3 main.py --training_mode "ft" \
--data_percentage "50" \
--dataset "UCI" \
--consistency "criterion" \
--classes "UCI_classes" \
--data_folder "../uci_data/" \
--nepoch 50 \

# python3 main.py --training_mode "ssl" --data_percentage "10" --dataset "UCI" --consistency "criterion" --classes "UCI_classes" --data_folder "../uci_data/" --nepoch 50
# python3 main.py --training_mode "ft" --data_percentage "10" --dataset "UCI" --consistency "criterion" --classes "UCI_classes" --data_folder "../uci_data/" --nepoch 50
