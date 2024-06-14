DATASET='fb237'
sf='k3'


python load_data.py -d ${DATASET} -st test --part full --hop 3 --ind_suffix "_ind" --suffix ${sf}
cp ./bertrl_data/fb237_hop3_full${sf}/test.tsv ./bertrl_data/fb237_hop3_full${sf}/dev.tsv 
cp ./bertrl_data/fb237_hop3_full${sf}/testcontext.tsv ./bertrl_data/fb237_hop3_full${sf}/devcontext.tsv 
python load_data.py -d ${DATASET} -st train --part full --hop 3 --ind_suffix "_ind" --suffix ${sf}
./train.sh

# python eval_bertrl.py -d fb237_hop3_full_neg10_max_inductive_test