import utils
from sklearn import preprocessing

# AS FUNÇÕES ABAIXO TRANSFORMAM O ESTADO DO JOGADOR EM ENTRADAS PARA A REDE NEURAL
# used in networks per age
def transform_input(data, age):
    inputs = []
    if age == 1:
        inputs = transform_input_age1(data)
    elif age == 2:
        inputs = transform_input_age2(data) 
    elif age == 3:
        inputs = transform_input_age3(data)

    return inputs


# used in network for entire game
def transform_features(data, era):
    encoder = preprocessing.LabelEncoder()
    encoder = encoder.fit(utils.ALL_CARDS)
    t_row = []
    resource_neurons = []

    # initialize card neurons with zeros
    cards_neurons = [0 for i in range(len(utils.ALL_CARDS))]     
    
    for id in data['cards_hand_id']:
        index = encoder.transform([id])[0]
        cards_neurons[index] = 1

    resource_neurons.append(era)
    resource_neurons.append(data["amount"]["civilian"])
    # resource_neurons.append(data["amount"]["military"])
    resource_neurons.append(data["amount"]["commercial"])
    # resource_neurons.append(data["amount"]["raw_material"])
    # resource_neurons.append(data["amount"]["manufactured_goods"])
    # resource_neurons.append(data["amount"]["scientific"])
    resource_neurons.append(data["amount"]["guild"])
    resource_neurons.append(data["wonder_id"])
    resource_neurons.append(data["wonder_stage"])

    coins = data['resources']['coins']
    compass = data['resources']['compass']
    gear = data['resources']['gear']
    glass = data['resources']['glass']
    loom = data['resources']['loom']
    ore = data['resources']['ore']
    papyrus = data['resources']['papyrus']
    shields = data['resources']['shields']
    stone = data['resources']['stone']
    tablet = data['resources']['tablet']
    wood = data['resources']['wood']

    resource_neurons.append(coins)
    resource_neurons.append(compass)
    resource_neurons.append(gear)
    resource_neurons.append(glass)
    resource_neurons.append(loom)
    resource_neurons.append(ore)
    resource_neurons.append(papyrus)
    resource_neurons.append(shields)
    resource_neurons.append(stone)
    resource_neurons.append(tablet)
    resource_neurons.append(wood)
    
    t_row.extend(cards_neurons)
    t_row.extend(resource_neurons)

    print('SIZE OF INPUT: {}'.format(t_row))

    return t_row   



def transform_input_age1(data):
    encoder = preprocessing.LabelEncoder()
    encoder = encoder.fit(utils.CARDS_AGE1)
    t_row = []
    resource_neurons = []

    # initialize card neurons with zeros
    cards_neurons = [0 for i in range(len(utils.CARDS_AGE1))]
    
    for id in data["cards_hand_id"]: 
        index = encoder.transform([id])[0]
        cards_neurons[index] = 1

    resource_neurons.append(data["amount"]["civilian"])
    # resource_neurons.append(data["amount"]["military"])
    resource_neurons.append(data["amount"]["commercial"])
    # resource_neurons.append(data["amount"]["raw_material"])
    # resource_neurons.append(data["amount"]["manufactured_goods"])
    # resource_neurons.append(data["amount"]["scientific"])
    resource_neurons.append(data["amount"]["guild"])
    resource_neurons.append(data["wonder_id"])
    resource_neurons.append(data["wonder_stage"])

    coins = data['resources']['coins']
    compass = data['resources']['compass']
    gear = data['resources']['gear']
    glass = data['resources']['glass']
    loom = data['resources']['loom']
    ore = data['resources']['ore']
    papyrus = data['resources']['papyrus']
    shields = data['resources']['shields']
    stone = data['resources']['stone']
    tablet = data['resources']['tablet']
    wood = data['resources']['wood']

    resource_neurons.append(coins)
    resource_neurons.append(compass)
    resource_neurons.append(gear)
    resource_neurons.append(glass)
    resource_neurons.append(loom)
    resource_neurons.append(ore)
    resource_neurons.append(papyrus)
    resource_neurons.append(shields)
    resource_neurons.append(stone)
    resource_neurons.append(tablet)
    resource_neurons.append(wood)
    
    t_row.extend(cards_neurons)
    t_row.extend(resource_neurons)

    return t_row   


def transform_input_age2(data):
    encoder = preprocessing.LabelEncoder()
    encoder = encoder.fit(utils.CARDS_AGE2)
    t_row = []
    resource_neurons = []

    # initialize card neurons with zeros
    cards_neurons = [0 for i in range(len(utils.CARDS_AGE2))]
    
    for id in data["cards_hand_id"]: 
        index = encoder.transform([id])[0]
        cards_neurons[index] = 1

    resource_neurons.append(data["amount"]["civilian"])
    # resource_neurons.append(data["amount"]["military"])
    resource_neurons.append(data["amount"]["commercial"])
    # resource_neurons.append(data["amount"]["raw_material"])
    # resource_neurons.append(data["amount"]["manufactured_goods"])
    # resource_neurons.append(data["amount"]["scientific"])
    resource_neurons.append(data["amount"]["guild"])
    resource_neurons.append(data["wonder_id"])
    resource_neurons.append(data["wonder_stage"])

    coins = data['resources']['coins']
    compass = data['resources']['compass']
    gear = data['resources']['gear']
    glass = data['resources']['glass']
    loom = data['resources']['loom']
    ore = data['resources']['ore']
    papyrus = data['resources']['papyrus']
    shields = data['resources']['shields']
    stone = data['resources']['stone']
    tablet = data['resources']['tablet']
    wood = data['resources']['wood']

    resource_neurons.append(coins)
    resource_neurons.append(compass)
    resource_neurons.append(gear)
    resource_neurons.append(glass)
    resource_neurons.append(loom)
    resource_neurons.append(ore)
    resource_neurons.append(papyrus)
    resource_neurons.append(shields)
    resource_neurons.append(stone)
    resource_neurons.append(tablet)
    resource_neurons.append(wood)
    
    t_row.extend(cards_neurons)
    t_row.extend(resource_neurons)

    return t_row   


def transform_input_age3(data):
    encoder = preprocessing.LabelEncoder()
    encoder = encoder.fit(utils.CARDS_AGE3)
    t_row = []
    resource_neurons = []

    # initialize card neurons with zeros
    cards_neurons = [0 for i in range(len(utils.CARDS_AGE3))]
    
    for id in data["cards_hand_id"]: 
        index = encoder.transform([id])[0]
        cards_neurons[index] = 1

    resource_neurons.append(data["amount"]["civilian"])
    # resource_neurons.append(data["amount"]["military"])
    resource_neurons.append(data["amount"]["commercial"])
    # resource_neurons.append(data["amount"]["raw_material"])
    # resource_neurons.append(data["amount"]["manufactured_goods"])
    # resource_neurons.append(data["amount"]["scientific"])
    resource_neurons.append(data["amount"]["guild"])
    resource_neurons.append(data["wonder_id"])
    resource_neurons.append(data["wonder_stage"])

    coins = data['resources']['coins']
    compass = data['resources']['compass']
    gear = data['resources']['gear']
    glass = data['resources']['glass']
    loom = data['resources']['loom']
    ore = data['resources']['ore']
    papyrus = data['resources']['papyrus']
    shields = data['resources']['shields']
    stone = data['resources']['stone']
    tablet = data['resources']['tablet']
    wood = data['resources']['wood']

    resource_neurons.append(coins)
    resource_neurons.append(compass)
    resource_neurons.append(gear)
    resource_neurons.append(glass)
    resource_neurons.append(loom)
    resource_neurons.append(ore)
    resource_neurons.append(papyrus)
    resource_neurons.append(shields)
    resource_neurons.append(stone)
    resource_neurons.append(tablet)
    resource_neurons.append(wood)
    
    t_row.extend(cards_neurons)
    t_row.extend(resource_neurons)

    return t_row   
