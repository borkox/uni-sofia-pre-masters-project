import os, sys, getopt, glob
import gym
from collections import deque
import matplotlib.pyplot as plt
import numpy as np
from gym.envs.toy_text import FrozenLakeEnv
from pathlib import Path

environment_type = '3x3'
slippery = False
clean_output = True
output_base = "./outputs"
max_number_episodes = 60
opts, args = getopt.getopt(sys.argv[1:],"he:s:o:c:n:")
print (f"Script {sys.argv[0]} started with options: ", opts)
for opt, arg in opts:
    if opt in ("-h", "--help"):
        print ('USAGE:> actor-critic-frozen-lake-nest.py OPTIONS')
        print ('OPTIONS: ')
        print ('   -e <environment>')
        print ('        where <environment> is one of: 3x3,4x4, default 3x3')
        print ('   -s <slippery>')
        print ('       where <slippery> is one of: true,false, default false')
        print ('   -o <output>')
        print ('       where <output> is output folder, default "./output"')
        print ('   -c <clean_output>')
        print ('       where <clean_output> is true or false, default true.')
        print ('   -n <num_episodes>')
        print ('       where <num_episodes> is max number of episodes, default 60.')
        sys.exit()
    elif opt in ("-e"):
        if arg not in ("3x3","4x4"):
            print("Unknown environment:", arg)
            sys.exit()
        environment_type = arg
    elif opt in ("-s"):
        slippery = 'true' == arg.lower()
    elif opt in ("-o"):
        output_base = arg
    elif opt in ("-n"):
        max_number_episodes = int(arg)
    elif opt in ("-c"):
        clean_output = 'true' == arg.lower()
    else:
        print("Unknown option:", opt)
        sys.exit()


print ('Environment: ', environment_type)
print ('slippery: ', slippery)
print ('output_folder: ', output_base)
print ('clean_output: ', clean_output)

import nest.voltage_trace

output_folder = output_base + "/" + environment_type
# Ensure folder with resources exists
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
# Clean output folder
if clean_output:
    [f.unlink() for f in Path(output_folder).glob("*") if f.is_file()]

# number of episodes to run
NUM_EPISODES = max_number_episodes
# max steps per episode
MAX_STEPS = 100
# Saves scores to file evey SAVE_SCORES_STEPS steps
SAVE_SCORES_STEPS = 5
DRAW_ARROWS_STEPS = 5
# score agent needs for environment to be solved
SOLVED_HISTORY_SCORES_LEN = 10
SOLVED_MEAN_SCORE = 0.5
SOLVED_MIN_EPISODES = 3
# current time while it runs
current_time = 0
# STEP is milliseconds to wait WTA to become functional
STEP = 150
# Learn time is when WTA is still active and dopamine is activated
LEARN_TIME = 20
# REST_TIME is milliseconds to run rest for WTA and perform dopamine STDP
REST_TIME = 50
# Noise constants
NOISE_DA_NEURONS_WEIGHT = 0.01
NOISE_ALL_STATES_WEIGHT = 0.01
NOISE_RATE = 65000.
CRITIC_NOISE_RATE = 65500.
REWARD_STIMULUS_RATE = 65000.
STIMULUS_RATE = 65000.
WTA_NOISE_RATE = 500.

# ================================================
nest.set_verbosity("M_WARNING")
nest.ResetKernel()

# ================= Environment ==================
# Make environment
if environment_type == '3x3':
    env = FrozenLakeEnv(desc=["SFF",
                              "FFH",
                              "FFG"], is_slippery=slippery)
else:
    # Makes 4x4 environment
    env = FrozenLakeEnv(is_slippery=slippery)

WORLD_ROWS = env.nrow
WORLD_COLS = env.ncol
print("World dimensions: ", WORLD_COLS, WORLD_ROWS)
world_dim = {'x': WORLD_COLS, 'y': WORLD_ROWS}
num_actions = 4
possible_actions = [0, 1, 2, 3]
possible_actions_str = ["LEFT", "DOWN", "RIGHT", "UP"]
# ================================================

def plot_values(draw_image=False, image_name=None):
    if draw_image:
        fig, ax = plt.subplots()
        plt.cla()

    values_plot = []

    for i in range(world_dim['y']):
        values_plot.append([])
        for j in range(world_dim['x']):
            values_plot[i].append(np.mean([np.mean(nest.GetStatus(nest.GetConnections(states[j][i], actions[a]), 'weight')) for a in range(len(actions))]))
            if len(actions) == 4:
                q_left = np.mean(nest.GetStatus(nest.GetConnections(states[j][i], actions[0]), 'weight'))
                q_down = np.mean(nest.GetStatus(nest.GetConnections(states[j][i], actions[1]), 'weight'))
                q_right = np.mean(nest.GetStatus(nest.GetConnections(states[j][i], actions[2]), 'weight'))
                q_up = np.mean(nest.GetStatus(nest.GetConnections(states[j][i], actions[3]), 'weight'))
                if draw_image:
                    ax.arrow(j, i, (q_right-q_left)/10., (q_down-q_up)/10., head_width=0.05, head_length=0.1, fc='k', ec='k')


    values_plot = np.array(values_plot)
    with np.printoptions(precision=3, suppress=True):
        print(values_plot)

    if draw_image:
        # plt.imshow(values_plot, interpolation='none', vmax=1 * WEIGHT_SCALING, vmin=-1 * WEIGHT_SCALING)
        plt.imshow(values_plot, interpolation='none', vmax=1 * WEIGHT_SCALING, vmin=-1 * WEIGHT_SCALING)

        xlabels = np.arange(0, len(states))
        ylabels = np.arange(0, len(states[0]))

        # Set the major ticks at the centers and minor tick at the edges
        xlocs = np.arange(len(xlabels))
        ylocs = np.arange(len(ylabels))
        ax.xaxis.set_ticks(xlocs + 0.5, minor=True)
        ax.xaxis.set(ticks=xlocs, ticklabels=xlabels)
        ax.yaxis.set_ticks(ylocs + 0.5, minor=True)
        ax.yaxis.set(ticks=ylocs, ticklabels=ylabels)

        # Turn on the grid for the minor ticks
        ax.grid(True, which='minor', linestyle='-', linewidth=2)

        for txt in ax.texts:
            txt.set_visible(False)
        plt.savefig(image_name, format='png')

    # ax.annotate(".", ((position['x'] + 0.5)/len(states), (1-(position['y'] + 0.5)/len(states[0]))), size=160, textcoords='axes fraction', color='white')
    # plt.draw()
# ================================================


NUM_STATE_NEURONS = 20
NUM_WTA_NEURONS = 50
# WEIGHT_SCALING = 100 / NUM_STATE_NEURONS
WEIGHT_SCALING = 100 * 20 / NUM_STATE_NEURONS

# nest.ResetKernel()
# nest.set_verbosity("M_DEBUG")

rank = nest.Rank()
size = nest.NumProcesses()
seed = np.random.randint(0, 1000000)
num_threads = 1
nest.SetKernelStatus({"local_num_threads": num_threads})
nest.SetKernelStatus({"rng_seed": seed})
tau_pre = 20.
nest.SetDefaults("iaf_psc_alpha", {"tau_minus": tau_pre})

# Create states
states = []
all_states = None
for i in range(world_dim['x']):
    states.append([])
    for j in range(world_dim['y']):
        state_group = nest.Create('iaf_psc_alpha', NUM_STATE_NEURONS)
        states[i].append(state_group)
        if all_states is None:
            all_states = state_group
        else:
            all_states = all_states + state_group

# Create actions
actions = []
all_actions = None
for i in range(num_actions):
    action_group = nest.Create('iaf_psc_alpha', NUM_WTA_NEURONS)
    actions.append(action_group)
    if all_actions is None:
        all_actions = action_group
    else:
        all_actions = all_actions + action_group

# Create WTA circuit
wta_ex_weights = 10.5
wta_inh_weights = -2.6
wta_ex_inh_weights = 2.8
wta_noise_weights = 2.1

wta_inh_neurons = nest.Create('iaf_psc_alpha', NUM_WTA_NEURONS)

for i in range(len(actions)):
    nest.Connect(actions[i], actions[i], 'all_to_all', {'weight': wta_ex_weights})
    nest.Connect(actions[i], wta_inh_neurons, 'all_to_all', {'weight': wta_ex_inh_weights})

nest.Connect(wta_inh_neurons, all_actions, 'all_to_all', {'weight': wta_inh_weights})

wta_noise = nest.Create('poisson_generator', 10, {'rate': WTA_NOISE_RATE})
nest.Connect(wta_noise, all_actions, 'all_to_all', {'weight': wta_noise_weights})
nest.Connect(wta_noise, wta_inh_neurons, 'all_to_all', {'weight': wta_noise_weights * 0.9})

# Create stimulus
stimulus = nest.Create('poisson_generator', 1, {'rate': STIMULUS_RATE})
nest.Connect(stimulus, all_states, 'all_to_all', {'weight': 0.})

# Here, we are implementing the dopaminergic nueron pool, volume transmitter and dopamin-modulated synapse between states and actions

# Create DA pool
DA_neurons = nest.Create('iaf_psc_alpha', 100)
vol_trans = nest.Create('volume_transmitter', 1, {'deliver_interval': 10})
nest.Connect(DA_neurons, vol_trans, 'all_to_all')

# Create reward stimulus
reward_stimulus = nest.Create('poisson_generator', 1, {'rate': REWARD_STIMULUS_RATE})
nest.Connect(reward_stimulus, DA_neurons, 'all_to_all', {'weight': 0.})

tau_c = 50.0
tau_n = 20.0
tau_plus = 20.

# Connect states to actions
nest.CopyModel('stdp_dopamine_synapse', 'dopa_synapse', {
    'vt': vol_trans.get('global_id'), 'A_plus': 1, 'A_minus': .5, "tau_plus": tau_plus,
    'Wmin': -10., 'Wmax': 10., 'b': 0., 'tau_n': tau_n, 'tau_c': tau_c})

nest.Connect(all_states, all_actions, 'all_to_all', {'synapse_model': 'dopa_synapse', 'weight': 0.0})

nest.CopyModel('stdp_dopamine_synapse', 'dopa_synapse_critic', {
    'vt': vol_trans.get('global_id'), 'A_plus': 1, 'A_minus': .5, "tau_plus": tau_plus,
    'Wmin': -10., 'Wmax': 10., 'b': 0., 'tau_n': tau_n, 'tau_c': tau_c})

critic = nest.Create('iaf_psc_alpha', 50)
nest.Connect(all_states, critic, 'all_to_all', {'synapse_model': 'dopa_synapse_critic', 'weight': 0.0})
nest.Connect(critic, DA_neurons, 'all_to_all', {'weight': -5., 'delay': 50})
nest.Connect(critic, DA_neurons, 'all_to_all', {'weight': 5., 'delay': 1.})

critic_noise = nest.Create('poisson_generator', 1, {'rate': CRITIC_NOISE_RATE})
nest.Connect(critic_noise, critic)

# Create spike detector
sd_wta = nest.Create('spike_recorder')
nest.Connect(all_actions, sd_wta)
nest.Connect(wta_inh_neurons, sd_wta)
sd_actions = nest.Create('spike_recorder', num_actions)
for i in range(len(actions)):
    nest.Connect(actions[i], sd_actions[i])
sd_states = nest.Create('spike_recorder')
nest.Connect(all_states, sd_states)
sd_DA = nest.Create('spike_recorder', 1)
nest.Connect(DA_neurons, sd_DA, 'all_to_all')
sd_critic = nest.Create('spike_recorder', 1)
nest.Connect(critic, sd_critic, 'all_to_all')

# Create noise
noise = nest.Create('poisson_generator', 1, {'rate': NOISE_RATE})
nest.Connect(noise, all_states, 'all_to_all', {'weight': NOISE_ALL_STATES_WEIGHT})
nest.Connect(noise, DA_neurons, 'all_to_all', {'weight': NOISE_DA_NEURONS_WEIGHT})


# Init network
print(f"Observation space: {env.observation_space}")
print(f"Action space: {env.action_space.n}")

# track scores
scores = []

# track recent scores
recent_scores = deque(maxlen=SOLVED_HISTORY_SCORES_LEN)
prev_spikes = 0
# run episodes
for episode in range(NUM_EPISODES):
    nest.SetStatus(sd_actions, {"n_events": 0})
    nest.SetStatus(sd_wta, {"n_events": 0})
    nest.SetStatus(sd_states, {"n_events": 0})
    nest.SetStatus(sd_DA, {"n_events": 0})
    nest.SetStatus(sd_critic, {"n_events": 0})

    # init variables
    state = env.reset()
    print("STATE:+>>>>", state)
    done = False
    score = 0
    reward = 0
    step = 0
    # run episode, update online
    for _ in range(MAX_STEPS):

        # ENVIRONMENT
        state_x = int(state % WORLD_COLS)
        state_y = int(state / WORLD_COLS)


        nest.SetStatus(nest.GetConnections(stimulus, states[state_x][state_y]), {'weight': 1.})
        nest.SetStatus(wta_noise, {'rate': 3000.})

        print("State position: ", state_x, ", ", state_y)
        # for si in range(len(states)):
        nest.SetStatus(nest.GetConnections(stimulus, states[state_x][state_y]), {'weight': 1.})

        env.render()
        nest.Simulate(STEP)

        max_rate = -1
        chosen_action = -1
        for i in range(len(sd_actions)):
            rate = len([e for e in nest.GetStatus(sd_actions[i], keys='events')[0]['times'] if
                        e > current_time])  # calc the \"firerate\" of each actor population
            if rate > max_rate:
                max_rate = rate  # the population with the hightes rate wins
                chosen_action = i

        current_time += STEP
        print("chose action:", possible_actions[chosen_action], " ", possible_actions_str[chosen_action], " at step ",
              step)

        action = possible_actions[chosen_action]
        new_state, reward, done, _ = env.step(action)

        # stimulate new state
        nest.SetStatus(nest.GetConnections(stimulus, states[state_x][state_y]), {'weight': 0.})

        # apply reward
        print("Scaled reward:", float(reward) * WEIGHT_SCALING)
        nest.SetStatus(nest.GetConnections(reward_stimulus, DA_neurons), {'weight': float(reward) * WEIGHT_SCALING})

        # learn time
        if reward > 0:
            print("Learn time")
            nest.Simulate(LEARN_TIME)
            current_time += LEARN_TIME

        nest.SetStatus(nest.GetConnections(reward_stimulus, DA_neurons), {'weight': 0.0})

        nest.SetStatus(wta_noise, {'rate': 0.})
        # refactory time
        nest.Simulate(REST_TIME)
        current_time += REST_TIME

        # plotting
        plot_values()


        # if done:
        #     for i in range(0, 1):
        #         step = step + 1
        #         nest.SetStatus(dc_generator_reward, {"amplitude": -10.})
        #         nest.Simulate(STEP + REST_TIME)
        #         time += STEP + REST_TIME

        #     print("reward:", reward)

        # update episode score
        score += reward

        # if terminal state, next state val is 0
        if done:
            print(f"Episode {episode} finished after {step} timesteps")
            break

        # move into new state, discount I
        state = new_state
        step = step + 1

    # append episode score
    scores.append(score)
    recent_scores.append(score)

    # early stopping if we meet solved score goal
    if len(recent_scores) > SOLVED_MIN_EPISODES \
            and np.array(recent_scores).mean() >= SOLVED_MEAN_SCORE \
            and reward > SOLVED_MEAN_SCORE: # We want spikes to be snapshot when actually episode is solved
        print("SOLVED")
        break
    else:
        print('Mean score: ', np.array(recent_scores).mean())
    if len(scores) % SAVE_SCORES_STEPS == 0:
        print("Save scores")
        np.savetxt(output_folder + '/scores.txt', scores, delimiter=',')
    if len(scores) % DRAW_ARROWS_STEPS == 0:
        plot_values(draw_image=True, image_name=f'{output_folder}/arrows_{len(scores)}_{WORLD_COLS}x{WORLD_ROWS}.png')


# if reward > 0:
    #     break
np.savetxt(output_folder + '/scores.txt', scores, delimiter=',')
plot_values(draw_image=True, image_name=f'{output_folder}/arrows_{WORLD_COLS}_{WORLD_ROWS}.png')

print("====== all_states === all_actions ===")
print(nest.GetConnections(all_states, all_actions))

nest.raster_plot.from_device(sd_wta, hist=True, title="sd_wta")
plt.savefig(f'{output_folder}/sd_wta.png', format='png')
nest.raster_plot.from_device(sd_states, hist=True, title="sd_states")
plt.savefig(f'{output_folder}/sd_states.png', format='png')
nest.raster_plot.from_device(sd_DA, hist=True, title="sd_DA")
plt.savefig(f'{output_folder}/sd_DA.png', format='png')
nest.raster_plot.from_device(sd_critic, hist=True, title="sd_critic")
plt.savefig(f'{output_folder}/sd_critic.png', format='png')

# Print scores
os.system(f"python ./plot_scores.py -i {output_folder}/scores.txt -o {output_folder}/final_scores.png")