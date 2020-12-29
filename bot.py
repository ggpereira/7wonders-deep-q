from network import Agent 
from network import DeepQNetwork
# from utils import plot_learning_curve 
import numpy as np 
import sys 
import json
import transform 
import utils
import torch 
import strategies 
from sklearn import preprocessing


# CARD ACTIONS
BUILD = 1 
BUILD_WONDER = 2 
DISCARD = 3


def match(value, array):
    for item in array:
        if item == value:
            return True 
    return False 


def get_neighbors(players_state, player_id):
    neighbors = []
    id_player_left = player_id - 1
    id_player_right = player_id + 1

    if id_player_left < 0:
        id_player_left = len(players_state) - 1

    if id_player_right > (len(players_state) - 1):
        id_player_right = 0

    neighbors.append(players_state[str(id_player_left)])
    neighbors.append(players_state[str(id_player_right)])

    return neighbors


def get_card_reward(gamestatus, player_id, card_id):
    player_state = gamestatus['players'][str(player_id)]
    neighbors = get_neighbors(gamestatus['players'], player_id)

    print('card {}'.format(card_id))

    return strategies.card_weight_map[str(card_id)](player_state, gamestatus['game'], neighbors)


def get_valid_actions(playerstatus):
    arr_valid_actions = []
    hand = playerstatus['cards_hand_id']
    playable_cards = playerstatus['cards_playable_id']        
    
    for card in playable_cards:
        arr_valid_actions.append((BUILD * 100) + card)
    
    for card in hand:
        arr_valid_actions.append((DISCARD * 100) + card)
        if(playerstatus['can_build_wonder']):    
            arr_valid_actions.append((BUILD_WONDER * 100) + card)
    
    print('VALID LABEL ACTIONS: {}'.format(arr_valid_actions))

    return arr_valid_actions


def encode_actions(actions, era):
    encoder = preprocessing.LabelEncoder()
    
    if era == 1:
        print('ENCODING FOR: ERA {}'.format(era))
        encoder = encoder.fit(utils.AGE1_LABELS)    
    if era == 2:
        print('ENCODING FOR: ERA {}'.format(era))
        encoder = encoder.fit(utils.AGE2_LABELS)
    if era == 3: 
        print('ENCODING FOR: ERA {}'.format(era))
        encoder = encoder.fit(utils.AGE3_LABELS)
    
    transformed = encoder.transform(actions)
    
    return transformed


def encode_all_actions(actions):
    print('VALID_ACTIONS: {}'.format(actions))
    encoder = preprocessing.LabelEncoder()
    encoder.fit(utils.ALL_LABELS)
    transformed = encoder.transform(actions)
    return transformed     
    

def is_valid_label(label, playerstatus):
    action = int(label/100)
    card_id = label % 100
    is_playable = False

    if action == BUILD:
        playable_cards = playerstatus['cards_playable_id']
        is_playable = match(card_id, playable_cards) 
        return is_playable 
    
    if action == BUILD_WONDER:
        hand = playerstatus['cards_hand_id']
        is_playable = match(card_id, hand)
        is_playable = playerstatus['can_build_wonder']
        return is_playable 
    if action == DISCARD:
        hand = playerstatus['cards_hand_id']
        is_playable = match(card_id, hand)
        return is_playable 
    
    return is_playable 


def get_command(v):
    if v == BUILD:
        return 'build_structure'
    
    if v == BUILD_WONDER:
        return 'build_wonder'
    
    if v == DISCARD:
        return 'discard'


def get_card_name(card_id, playerstatus):
    idx = 0 
    hand = playerstatus['cards_hand_id']

    #find index 
    for i in range(len(hand)):
        if card_id == hand[i]:
            idx = i
            break 
    return playerstatus['cards_hand'][idx]


def write_json(label, bot_id, playerstatus):
    path = sys.argv[1] + '/player_' + str(bot_id + 1) + '.json'
    action = int(label/100)
    card_id = label % 100 

    command  = get_command(action)
    cardname = get_card_name(card_id, playerstatus)
    
    command_dict = {'command': {'subcommand': command, 'argument': cardname, 'extra': ''}}
    print('player_{}: plays - {}'.format(bot_id, command_dict))

    with open(path, 'w') as write_file:
        json.dump(command_dict, write_file) 

    file_ready = open(sys.argv[1] + '/ready.txt', 'a')
    file_ready.write('ready\n')
    file_ready.close()    


def check_age_finish(turn, end_turn):
    print('TURN: {}'.format(turn))
    print('END TURN: {}'.format(end_turn))
    if(turn == end_turn):
        print('ENTRA AQUI')
        return True 
    print('ENTRA NA CONDIÇÃO FALSA')
    return False 


def decode_action(enc_action, game_era):
    labelencoder = preprocessing.LabelEncoder()
    
    if(game_era == 1):
        labelencoder.fit(utils.AGE1_LABELS)
    if(game_era == 2):
        labelencoder.fit(utils.AGE2_LABELS)
    if(game_era == 3):
        labelencoder.fit(utils.AGE3_LABELS)

    return labelencoder.inverse_transform([enc_action])


def decode_all_actions(enc_action):
    labelencoder = preprocessing.LabelEncoder()
    labelencoder.fit(utils.ALL_LABELS)
    return labelencoder.inverse_transform([enc_action])


def play(player_id, agent, gamestatus):
    current_era = gamestatus['game']['era']
    current_turn = gamestatus['game']['turn']
    total_turns = 21
    total_ages =3 
    discard_turn =((total_turns/total_ages) * current_era) - 1
    end_turn = (total_turns/total_ages) * current_era
    done = False 

    # pega as informações do jogador 
    playerstatus = gamestatus['players'][str(player_id)]

    if gamestatus['game']['finished']:
        # gambiarra para subtrair 1 da contagem de eras quando obtiver o estado obtido após o último turno
        gamestatus['game']['era'] =  gamestatus['game']['era'] - 1
        done = True 
    

    if(utils.PREVIOUS_AGE != gamestatus['game']['era']):
        utils.DISCARD_COUNT = 0

    # transforma as informações do estado do jogador no vetor de entrada para a rede neural
    observation = transform.transform_features(playerstatus, gamestatus['game']['era'])

    # a partir daqui começa a guardar os estados e calcular as recompensas
    if current_turn > 0:
        points_obtained = playerstatus['points']['total'] - utils.PREVIOUS_SCORE
        reward = points_obtained + utils.PREVIOUS_REWARD

        print('vp obtained: {}'.format(points_obtained))
        print('reward received: {}'.format(reward))
        print('total score: {}'.format(gamestatus['players'][str(player_id)]['points']['total']))

        agent.store_transition(utils.PREVIOUS_OBS, utils.PREVIOUS_ACTION, reward, observation, done)
        agent.learn()
        print("<=================================================================================>")
    # se o jogo acabou não há ações para fazer
    if done:
        # salva os parâmetros aprendidos ao fim do jogo 
        print("##################################################################")
        torch.save(agent.q_eval.state_dict(), './checkpoint/checkpoint.pth')
        print('Model checkpoint.pth saved')
        print("##################################################################")
        return 

    # se não  estiver no turno de descarte joga normalmente
    if current_turn != discard_turn:
        # pega todas as ações válidas para o estado atual do jogador
        valid_actions = get_valid_actions(playerstatus)
        
        # codifica ações para as saídas da rede neural
        enc_valid_actions = encode_all_actions(np.asarray(valid_actions))
        print("<=====================================================================================>")
        # mostra o valor de epsilon(quanto maior, mais ações aleatórias)
        print('EPSILON: {}'.format(agent.epsilon))

        # escolhe a ação a ser enviada para o jogo
        action = agent.choose_action(observation, enc_valid_actions)

        # decodifica a ação para obter a carta e a ação
        d_action = decode_all_actions(action)[0]
    # caso seja o turno de descarte apenas descarta uma carta e guarda a ação
    else: 
        d_action = label=(3 * 100) + playerstatus['cards_hand_id'][0]
        action = encode_all_actions([d_action])[0]

    # escreve ação realizada no arquivo
    write_json(d_action, player_id, gamestatus['players'][str(player_id)])
    
    # guarda o score atual para calcular a diferença no próximo turno
    utils.PREVIOUS_SCORE = gamestatus['players'][str(player_id)]['points']['total'] 
    
    # guarda ação atual
    utils.PREVIOUS_ACTION = action 

    # guarda a observação atual 
    utils.PREVIOUS_OBS = observation 

    utils.PREVIOUS_AGE = gamestatus['game']['era']

    if int(d_action/100) == BUILD:
        utils.PREVIOUS_REWARD = get_card_reward(gamestatus, player_id, d_action % 100)
    elif int(d_action/100) == BUILD_WONDER:
        utils.PREVIOUS_REWARD = 4
    elif int(d_action/100) == DISCARD:
        utils.DISCARD_COUNT += 1
        if utils.DISCARD_COUNT > 2:
            utils.PREVIOUS_REWARD = -5
        else:
            utils.PREVIOUS_REWARD = 1