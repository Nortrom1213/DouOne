import multiprocessing as mp
import pickle

from douzero.env.game import GameEnv

def load_card_play_models(card_play_model_path_dict, model_type='lstm'):
    """
    Load deep agents or other agents based on the provided model paths.
    The deep agents will be instantiated with the specified model type.
    """
    players = {}
    for position in ['landlord', 'landlord_up', 'landlord_down']:
        if card_play_model_path_dict[position] == 'rlcard':
            from .rlcard_agent import RLCardAgent
            players[position] = RLCardAgent(position)
        elif card_play_model_path_dict[position] == 'random':
            from .random_agent import RandomAgent
            players[position] = RandomAgent()
        else:
            from .deep_agent import DeepAgent
            # Pass model_type parameter to DeepAgent
            players[position] = DeepAgent(position, card_play_model_path_dict[position],
                                            model_type=model_type)
    return players

def mp_simulate(card_play_data_list, card_play_model_path_dict, q, model_type='lstm'):
    """
    Run a simulation on a subset of evaluation data using deep agents
    with the specified model type.
    """
    players = load_card_play_models(card_play_model_path_dict, model_type=model_type)
    from douzero.env.game import GameEnv
    env = GameEnv(players)
    for idx, card_play_data in enumerate(card_play_data_list):
        env.card_play_init(card_play_data)
        while not env.game_over:
            env.step()
        env.reset()
    q.put((env.num_wins['landlord'],
           env.num_wins['farmer'],
           env.num_scores['landlord'],
           env.num_scores['farmer']))

def data_allocation_per_worker(card_play_data_list, num_workers):
    card_play_data_list_each_worker = [[] for k in range(num_workers)]
    for idx, data in enumerate(card_play_data_list):
        card_play_data_list_each_worker[idx % num_workers].append(data)

    return card_play_data_list_each_worker

def evaluate(landlord, landlord_up, landlord_down, eval_data, num_workers, model_type='lstm'):
    """
    Evaluate the performance of the deep agents using the given model type.
    """
    import pickle
    with open(eval_data, 'rb') as f:
        card_play_data_list = pickle.load(f)

    card_play_data_list_each_worker = data_allocation_per_worker(card_play_data_list, num_workers)
    del card_play_data_list

    card_play_model_path_dict = {
        'landlord': landlord,
        'landlord_up': landlord_up,
        'landlord_down': landlord_down
    }

    num_landlord_wins = 0
    num_farmer_wins = 0
    num_landlord_scores = 0
    num_farmer_scores = 0

    import multiprocessing as mp
    ctx = mp.get_context('spawn')
    q = ctx.SimpleQueue()
    processes = []
    for card_play_data in card_play_data_list_each_worker:
        p = ctx.Process(
                target=mp_simulate,
                args=(card_play_data, card_play_model_path_dict, q, model_type))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    for i in range(num_workers):
        result = q.get()
        num_landlord_wins += result[0]
        num_farmer_wins += result[1]
        num_landlord_scores += result[2]
        num_farmer_scores += result[3]

    num_total_wins = num_landlord_wins + num_farmer_wins
    print('WP results:')
    print('landlord : Farmers - {} : {}'.format(num_landlord_wins / num_total_wins, num_farmer_wins / num_total_wins))
    print('ADP results:')
    print('landlord : Farmers - {} : {}'.format(num_landlord_scores / num_total_wins, 2 * num_farmer_scores / num_total_wins))
