# python3 trainer.py --name=bullet

python3 trainer.py --m alphazero_value\
    --name="bulletfuck" --lr=0.2 --reg_const=0.2\
    --batch_size=32 --gram_size=3\
    --glob_expression='KB_small.pgn' --max_saves=3\
    --num_steps=150 --log_every=1 --save_every=15