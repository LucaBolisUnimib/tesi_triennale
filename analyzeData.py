import matplotlib.pyplot as plt
import numpy as np
import pickle
import numpy as np
import holoviews as hv
import statistics
hv.extension('bokeh')
# [state, action, agentReward, agent_q_values, selected_arm, q_valuesRecommender, rewards_recommender]

ENVIRONMENT = "raffinato"
FILENAME =  "envRaffinato_beta0.5_mbus50_osservazione.pkl" 
SIMULATION = 0

if ENVIRONMENT.lower() == "avanzato":
    ###### ENV_Avanzato ######
    STATE_NAME = ['','Healthy', 'Neutral1', 'Neutral2', 'Neutral3', 'Neutral4', 'Neutral5', 'Neutral6', 'Drug', 'Aftereffects1', 
                'Aftereffects2', 'Aftereffects3', 'Aftereffects4', 'Aftereffects5', 'Aftereffects6', 'Aftereffects7', 
                'Aftereffects8', 'Aftereffects9', 'Aftereffects10', 'Aftereffects11', 'Aftereffects12', 'Aftereffects13', 
                'Aftereffects14', '']
    ACTION_NAME = ["as2", "as3", "as4", "as5", "as6", "as7", "aG", "aW", "aD"]

    S0_STATE = 4
    HEALTHY_STATE = 1
    NEUTRAL_STATE = 2
    ADDICTION_STATE = 3
    AFTEREFFECT_STATE = 12
    aD_ACTION = 2
elif ENVIRONMENT.lower() == 'semplificato':
    ###### ENV_Semplificato ######
    FILENAME =  "env1_caso.pkl" 
    STATE_NAME = ['', 'Healthy', 'Neutral', 'Rec_Sys', 'Aftereffects', '']
    ACTION_NAME = ['aS2', 'aS3', 'aG', 'aW', 'aD']

    S0_STATE = 2 
    HEALTHY_STATE = 1
    NEUTRAL_STATE = 2
    AFTEREFFECT_STATE = 4
    aD_ACTION = 4
else:
    ###### ENV_Raffinato ######
    STATE_NAME = ['Healthy', 'Neutral1', 'Neutral2', 'RS1', 'RS2', 'RS3', 'RS4', 'RL1', 'RL2', 'RL3', 'RL4', 'Aftereffects', '']
    ACTION_NAME = ['aG', 'aW', 'aD']

    S0_STATE = 2
    HEALTHY_STATE = 1
    NEUTRAL_STATE = 2
    AFTEREFFECT_STATE = 12
    aD_ACTION = 2

# Agent indexes
AGENT_STATE_INDEX = 0
AGENT_ACTION_INDEX = 1
AGENT_REWARD_INDEX = 2
AGENT_QVALUES_INDEX = 3
SELECTED_ARM_INDEX = 4
RECOMMENDER_QVALUES_INDEX = 5

SAMPLE_STEP = 50

def getReward(data):
    rewardStepByStep = []
    reward = 0
    for ele in data:
        reward += ele[AGENT_REWARD_INDEX]
        rewardStepByStep.append(reward)
    return rewardStepByStep

def getQValues(data):
    qValues = []
    for ele in data:
        for arm in range(len(ele[RECOMMENDER_QVALUES_INDEX])):
            if len(qValues) == arm:
                qValues.append([])
            qValues[arm].append(ele[RECOMMENDER_QVALUES_INDEX][arm])
    return qValues

def getArm(data):
    arm = []
    for ele in data:
        arm.append(ele[SELECTED_ARM_INDEX])
    return arm

def getAgentState(data):
    states = []
    for ele in data:
        states.append(ele[AGENT_STATE_INDEX])
    return states

def data_plot(data):
    ### get the data to plot
    rewardStepByStep = getReward(data)
    qValues = getQValues(data)
    selectedArm = getArm(data)
    agentState = getAgentState(data)

    ### plot the data
    plt.figure(figsize=(16, 12))
    
    # Plot agent total_reward
    plt.subplot(2, 2, 1)
    plt.plot(rewardStepByStep, label='Total Reward')
    plt.title('Total Reward')
    plt.xlabel('Steps')
    plt.ylabel('Total reward')
    plt.legend()
    # plt.axvline(x=50, color='pink')  # Add red line at x=50 (Può servici per capire meglio)
    plt.grid(True, linestyle=':', color='gray', alpha=0.7)

    # Plot qvalues for each arm
    plt.subplot(2, 2, 2)
    n_arms = len(qValues)
    for arm in range(n_arms):
        plt.plot(qValues[arm], label='Qvalue arm ' + chr(ord('A') + arm))
    plt.title('Q-Value arms')
    plt.xlabel('Steps')
    plt.ylabel('Total Reward')
    plt.legend()
    # plt.axvline(x=50, color='pink')  # Add red line at x=50 (Può servici per capire meglio)
    plt.grid(True, linestyle=':', color='gray', alpha=0.7)  
   
    # Arm selected at each step
    plt.subplot(2, 2, 3)
    plt.scatter(np.arange(len(data)), selectedArm, np.full(shape=len(data),fill_value=0.2), marker='o', linestyle='-')
    arms_name = []
    for arm in np.arange(n_arms):
        arms_name.append('Arm ' + chr(ord('A') + arm))
    plt.yticks(np.arange(n_arms), arms_name)
    plt.title('Addicted Arm: 0 = Addictive, 1 = Healthy (C)')
    plt.xlabel('Steps')
    plt.ylabel('Choice')
    #plt.axvline(x=50, color='pink')  # Add red line at x=50 (Può servici per capire meglio)
    plt.grid(True, linestyle=':', color='gray', alpha=0.7)
    
    # Plot agent state
    plt.subplot(2, 2, 4)
    plt.scatter(np.arange(len(data)), agentState, np.full(shape=len(data),fill_value=0.2), marker='o', linestyle='-')
    plt.yticks(list(range(STATE_NAME)), STATE_NAME)
    plt.title('Addicted Arm: 0 = Addictive, 1 = Healthy (C)')
    plt.xlabel('Steps')
    plt.ylabel('Choice')
    #plt.axvline(x=50, color='pink')  # Add red line at x=50 (Può servici per capire meglio)
    plt.grid(True, linestyle=':', color='gray', alpha=0.7)

    plt.tight_layout()
    plt.show()

def read_data(filename):
    data = []
    with open(filename, 'rb') as f:
        try:
            while True and len(data):
                data.append(pickle.load(f))
        except EOFError:
            pass
    return data

def tmp(data):
    sum = 0
    count = np.zeros([5])
    for agent in data:
        for iter in agent:
            q_values = iter[AGENT_QVALUES_INDEX][AFTEREFFECT_STATE]
            for index in np.nonzero(q_values):
                    count[index] += 1
            # print(q_values) 
            sum += q_values
            #print(sum)
    print(count)
    print(sum / count)
### TEST ###

def getStateAgent(agent, locally):
    result = dict()
    interval = dict()
    count = 0
    for iter in agent:
        if count % SAMPLE_STEP == 0:
            result[count] = interval.copy()
            if locally:
                interval = dict()
        if iter[AGENT_STATE_INDEX] not in interval:
            interval[iter[0]] = 0
        interval[iter[0]] += 1
        count += 1
    return result

def getStateStats(data, locally):
    result = dict()
    for simulation in data:
        agentData = getStateAgent(simulation, locally)
        for key in agentData: # key is the step value such as 50, 500, 5000
            if key not in result:
                result[key] = dict()
            for state in agentData[key]:
                if state not in result[key]:
                    result[key][state] = []
                step_size = key
                if locally:
                    step_size = SAMPLE_STEP
                result[key][state].append(agentData[key][state] / step_size * 100)
    return result

def meanStandardDevation(stateStats):
    result = []
    for iter in stateStats:
        for state in stateStats[iter]:
            mean = statistics.mean(stateStats[iter][state] + ([0] * (SIMULATION - len(stateStats[iter][state]))))
            if len(stateStats[iter][state]) >= 2:
                stdev = statistics.stdev(stateStats[iter][state] + ([0] * (SIMULATION - len(stateStats[iter][state]))))
            else:
                stdev = 0
            result.append([iter, mean, stdev, state])
    return result

def draw_general_stats(data, locally):
    STATE_INDEX = 3
    stateStats = getStateStats(data, locally)
    graphData = meanStandardDevation(stateStats)

    dataToDraw = dict()
    for value in graphData:
        if value[STATE_INDEX] not in dataToDraw:
            dataToDraw[value[STATE_INDEX]] = []
        dataToDraw[value[STATE_INDEX]].append(value[:-1])
    
    result = None
    for state in dataToDraw:
        if result == None:
            result = hv.Curve(dataToDraw[state], label=STATE_NAME[state]).opts(axiswise=True) * hv.ErrorBars(dataToDraw[state]).opts(width = 1000, height = 500)
        else:
            result *= hv.Curve(dataToDraw[state], label=STATE_NAME[state]).opts(axiswise=True) * hv.ErrorBars(dataToDraw[state]).opts(width = 1000, height = 500)

    if locally:
        result.opts(title = "States stats locally")
    else:
        result.opts(title = "States stats")
    renderer = hv.renderer('bokeh')
    renderer.save(result, 'general_stats', 'html')
    return result

############

def getQvaluesAtIter(data):
    print("Iterazione: ")
    iter = int(input())
    print("Stato: ")
    state = input()
    print()
    stateIndex = STATE_NAME.index(state)
    
    pypy = []
    for i in range(len(data[iter][3][stateIndex])):
        print(ACTION_NAME[i], data[iter][3][stateIndex][i])
        pypy.append((ACTION_NAME[i], data[iter][3][stateIndex][i]))
        #Draw Bar plot
        # create some example data[statoAgente, azioneAgente, rewardAgenteComplessivo (Non serve, si può ricostruire eventualmente), rewardAgenteIterazione
        # [QvaluesAgente(matrice state x action)], raccomandazioneRecomender(A o B), [Qa_Recomender], rewardRecomender]
        #
        #
    pypy   = hv.Bars(pypy, 'Selected Action '+ ACTION_NAME[data[iter][1]],'Q Values')
    renderer = hv.renderer('bokeh')
    renderer.save(pypy, 'PLOT', 'html')

def reportQValuesRecomender(data):
    #percentages = (values/values.sum(axis=0)).T*100
    percentages = []
    number_of_arms = len(data[0][RECOMMENDER_QVALUES_INDEX])
    for i in range(number_of_arms):   
        percentages.append([])

    for i in range(len(data)): #ogni step
        sumQa = np.sum(data[i][RECOMMENDER_QVALUES_INDEX]) 
        if sumQa != 0:
            for j in range(number_of_arms):   
                percentages[j].append((data[i][RECOMMENDER_QVALUES_INDEX][j]/sumQa) * 100)
        

    overlay = hv.Overlay([hv.Area(percentages[i], vdims=[hv.Dimension('value', unit='%')]) for i in range(number_of_arms)])
    renderer = hv.renderer('bokeh')
    renderer.save(hv.Area.stack(overlay), 'Recomender', 'html')

def get_addicted_agents_info_locally(data, step):
    addicted_agents = 0
    for agent in range(len(data)):
        addicted_action = 0
        real_step_size = min(step + SAMPLE_STEP, len(data[agent]))
        # An errore value is returned
        if real_step_size - step <= 1:
            return [-1, -1]
        for iter in range(step, real_step_size):
            if  data[agent][iter][AGENT_STATE_INDEX] == AFTEREFFECT_STATE or \
                (data[agent][iter][AGENT_STATE_INDEX] == ADDICTION_STATE and data[agent][iter][AGENT_ACTION_INDEX] == aD_ACTION):
                    addicted_action += 1
        # if more than half of the actions done by the agent are addicted action the agent is labeled as an 
        # addicted agent
        if addicted_action >= SAMPLE_STEP/2:
            addicted_agents += 1
    return [SIMULATION - addicted_agents, addicted_agents]

def get_addicted_agents_info(agent, addictive_dict):
    addicted_action = 0
    non_adiction_action = 0
    neutral_action = 0
    optimal_state = S0_STATE
    for iter in range(len(agent)):
        if ENVIRONMENT.lower() == 'semplificato' or ENVIRONMENT.lower() == 'avanzato':
            if agent[iter][AGENT_STATE_INDEX] == optimal_state:
                non_adiction_action += 1
                if optimal_state != HEALTHY_STATE:
                    optimal_state = optimal_state - 1
                else:
                    optimal_state = S0_STATE
            else:
                optimal_state = min(max(agent[iter][AGENT_STATE_INDEX] - 1, HEALTHY_STATE), S0_STATE)
                addicted_action += 1
        else:
            if agent[iter][AGENT_STATE_INDEX] == optimal_state:
                non_adiction_action += 1
                if optimal_state == HEALTHY_STATE:
                    optimal_state = 0
                elif optimal_state == 0:
                    optimal_state = S0_STATE
                else:
                    optimal_state = HEALTHY_STATE
            else:
                optimal_state = S0_STATE
                addicted_action += 1

        if iter % SAMPLE_STEP == 0 and iter != 0:
            # print(addicted_action, non_adiction_action)
            if addicted_action >= SAMPLE_STEP/2:
                addictive_dict['addicted'][iter] += 1
            #elif neutral_action >= iter/3:
            #    addictive_dict['neutral'][iter] += 1
            else:
                addictive_dict['non_addicted'][iter] += 1
            addicted_action = 0
            non_adiction_action = 0
        elif iter == 0:
            addictive_dict['addicted'][iter] = 0
            addictive_dict['non_addicted'][iter] += 1
            #addictive_dict['neutral'][iter] += 1

# sample_by_sample = True means that agents_info are calculated only considering the last SAMPLE_STEP steps
# sample_by_sample = False means that agents_info are calculated considering all the step
def draw_agents_addiction_info(data, sample_by_sample, graph, marker):
    result = dict()
    result['addicted'] = {0 : 0}
    result['neutral'] = {0 : 0}
    result['non_addicted'] = {0 : 900}
    if sample_by_sample:
        for step in range(0, len(data[0]), SAMPLE_STEP):
            agent_state = get_addicted_agents_info_locally(data, step)
            if agent_state != [-1, -1]:
                if step + SAMPLE_STEP not in result['addicted']:
                    result['addicted'][step + SAMPLE_STEP] = 0
                result['addicted'][step + SAMPLE_STEP] = agent_state[1]
                if step + SAMPLE_STEP not in result['non_addicted']:
                    result['non_addicted'][step + SAMPLE_STEP] = 0
                result['non_addicted'][step + SAMPLE_STEP] = agent_state[0]
                if step + SAMPLE_STEP not in result['neutral']:
                    result['neutral'][step + SAMPLE_STEP] = 0
                result['neutral'][step + SAMPLE_STEP] = agent_state[0]
    else:
        for step in range(0, len(data[0]), SAMPLE_STEP):
            result['addicted'][step] = 0
            result['non_addicted'][step] = 0
            result['neutral'][step] = 0
        for agent in data:
            get_addicted_agents_info(agent, result)

    if graph == None:
        graph = hv.Curve(sorted(result['non_addicted'].items())).opts(line_dash = marker, color = 'blue', axiswise=True, width = 1000, height = 500)
    else:
        graph *= hv.Curve(sorted(result['non_addicted'].items())).opts(line_dash = marker, color = 'blue', axiswise=True, width = 1000, height = 500)
    graph *= hv.Curve(sorted(result['addicted'].items())).opts(line_dash = marker, color = 'red', axiswise=True, width = 1000, height = 500)
    # graph *= hv.Curve(sorted(result['neutral'].items()), label = 'Neutral agent').opts(axiswise=True, width = 1000, height = 500)
    graph.opts(title = "Agents stats")
    renderer = hv.renderer('bokeh')
    renderer.save(graph, 'Agents_stats', 'html')
    return graph

def get_agents_state_info(agent, result):
    for step in range(0, len(agent), SAMPLE_STEP):
        if agent[step][AGENT_STATE_INDEX] not in result:
            result[agent[step][AGENT_STATE_INDEX]] = {0 : 0}
        if step not in result[agent[step][AGENT_STATE_INDEX]]:
            result[agent[step][AGENT_STATE_INDEX]][step] = 0
        result[agent[step][AGENT_STATE_INDEX]][step] += 1

def draw_agents_state_info(data):
    result = dict()
    for agent in data:
        get_agents_state_info(agent, result)

    graph = None
    for state in result:
        if graph == None:
            graph = hv.Curve(sorted(result[state].items()), label = STATE_NAME[state]).opts(axiswise=True, width = 1000, height = 500)
        else:
            graph *= hv.Curve(sorted(result[state].items()), label = STATE_NAME[state]).opts(axiswise=True, width = 1000, height = 500)
    graph.opts(title = "Agents state stats")
    renderer = hv.renderer('bokeh')
    renderer.save(graph, 'Agents_state_stats', 'html')
    return graph

def get_arms_choice(agent, result, accpeted):
    count = dict()
    for iter in range(0, len(agent)):
        if (agent[iter][AGENT_STATE_INDEX] >= 2 and agent[iter][AGENT_STATE_INDEX] <= 11 and \
                ((accpeted and agent[iter][AGENT_ACTION_INDEX] == aD_ACTION) or (not accpeted))):
            arm = agent[iter][SELECTED_ARM_INDEX]
            if arm not in count:
                count[arm] = 0
            count[arm] += 1

        for recomm in count:
            if recomm not in result:
                result[recomm] = dict()
            if iter not in result[recomm]:
                result[recomm][iter] = 0
            result[recomm][iter] += count[recomm]

def draw_arms_choice(data, accpeted):
    result = dict()
    for agent in data:
        get_arms_choice(agent, result, accpeted)

    ARMS_NAME = []
    for arm in range(len(result)):
        ARMS_NAME.append('Arm ' + chr(ord('A') + arm))

    graph = None
    # print("###", result)
    for arm in result:
        if graph == None:
            graph = hv.Curve(sorted(result[arm].items()), label = ARMS_NAME[arm]).opts(axiswise=True, width = 1000, height = 500)
        else:
            graph *= hv.Curve(sorted(result[arm].items()), label = ARMS_NAME[arm]).opts(axiswise=True, width = 1000, height = 500)
    if accpeted:
        graph.opts(title = "Recommender arms accepted stats")
    else:
        graph.opts(title = "Recommender arms proposed stats")
    renderer = hv.renderer('bokeh')
    renderer.save(graph, 'Recommender_arm_stats', 'html')
    return graph

def make_report(graphs):
    result = None
    for graph in graphs:
        if result == None:
            result = graph.opts(axiswise=True)
        else:
            result += graph.opts(axiswise=True)
    name = 'aaac' # FILENAME # FILENAME.split("/")[1][:-4]
    result.opts(title = name)
    renderer = hv.renderer('bokeh')
    renderer.save(result, 'Report_' + name, 'html')

def internal_model(data):
    last_transition = data[-1][-1]
    #print(last_transition)
    for index_ss, state_start in enumerate(STATE_NAME[:-1]):
        if index_ss != 0:
            for index_a, action in enumerate(ACTION_NAME):
                for index_se, state_end in enumerate(STATE_NAME[:-1]):
                    print(state_start + " | " + action + " | " + state_end + " -> " + 
                          "Probability: " + str(last_transition[index_ss][index_a][index_se]['probability']) + 
                          " | Reward: " + str(last_transition[index_ss][index_a][index_se]['reward']))
                input()

def draw_rec_qvalues(data, graph, marker):
    qvalues = dict()
    for agent in data:
        tmp = [0] * len(data[0][0][RECOMMENDER_QVALUES_INDEX])
        for index, iter in enumerate(agent):
            if index % SAMPLE_STEP == 0 and index != 0:
                if index not in qvalues:
                    qvalues[index] = [0] * len(data[0][0][RECOMMENDER_QVALUES_INDEX])
                qvalues[index] += tmp# per evitare la divisione per 0
                tmp = [0] * len(data[0][0][RECOMMENDER_QVALUES_INDEX])
            #print(iter[RECOMMENDER_QVALUES_INDEX])
            #print(tmp)
            #input()
            tmp += iter[RECOMMENDER_QVALUES_INDEX]
            
            #input()

    for ele in qvalues:
        qvalues[ele] = [x / (SIMULATION * SAMPLE_STEP) for x in qvalues[ele]]
        #input()

    ARMS_NAME = []
    for arm in range(len(data[0][0][RECOMMENDER_QVALUES_INDEX])):
        ARMS_NAME.append('Arm ' + chr(ord('A') + arm))
    

    #graph = None
    color = ['red', 'blue', 'green', 'orange']
    count = 0
    for arm in range(len(data[0][0][RECOMMENDER_QVALUES_INDEX])):
        if graph == None:
            graph = hv.Curve(sorted([(iter, qvalues[iter][arm]) for iter in qvalues])).opts(line_dash = marker, color = color[count], axiswise=True, width = 1000, height = 500)
        else:
            graph *= hv.Curve(sorted([(iter, qvalues[iter][arm]) for iter in qvalues])).opts(line_dash = marker, color = color[count], axiswise=True, width = 1000, height = 500)
        count += 1
    graph.opts(title = "Recommender arms Q-Values")
    renderer = hv.renderer('bokeh')
    renderer.save(graph, 'Recommender_arms_Q-Values', 'html')
    return graph

def draw_multiple_simulation_graph(filenames):
    global SIMULATION
    graph_addicted = None
    q_values = None
    count = 0
    marker = [(8, 8), (2, 2), (20, 3, 2, 3)]
    for filename in filenames:
        data = read_data(filename)
        SIMULATION = len(data)
        print("Simulation: ", len(data))
        graph_addicted = draw_agents_addiction_info(data, False, graph_addicted, marker[count])
        q_values = draw_rec_qvalues(data, q_values, marker[count])
        count += 1

    return graph_addicted, q_values

def main():
    global SIMULATION
    #plt.ion()
    data = read_data(FILENAME)
    SIMULATION = len(data)
    print("Simulation: ", len(data))
    
    #for i in data:
    #    data_plot(i)
    #    internal_model(i)
    
    graph_to_draw = []
    # agent_info, q_values = draw_multiple_simulation_graph(['envSemplificato_beta0.0.pkl', 'envSemplificato_beta0.5_mbus1.pkl', 'envSemplificato_beta1.0_mbus1.pkl'])
    # graph_to_draw.append(agent_info)
    # graph_to_draw.append(q_values)
    graph_to_draw.append(draw_general_stats(data, False))
    graph_to_draw.append(draw_general_stats(data, True))
    graph_to_draw.append(draw_agents_addiction_info(data, False, None, 'solid'))
    graph_to_draw.append(draw_agents_state_info(data))
    graph_to_draw.append(draw_arms_choice(data, False))
    graph_to_draw.append(draw_arms_choice(data, True))
    graph_to_draw.append(draw_rec_qvalues(data, None, 'solid'))
    make_report(graph_to_draw)

if __name__ == "__main__":
    main()