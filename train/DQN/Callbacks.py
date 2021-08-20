
def DQN_train(dqn, state, action, next_state, reward, done, info, writer, title, config):
    dqn.push([
        state,
        action,
        next_state,
        reward
    ])
    return dqn.learn()

def DQN_loss_parse(losses, writer, title):
    for loss, step in losses:
        writer.add_scalar(title+'/Loss', loss, step)
    return map(lambda x: x[0], losses)

def DQN_model(env, map_size, config):
    import DQN
    import Net
    eps_sche = DQN.EpsScheduler(1., 'Linear', lower_bound=0.1, target_steps=200000)
    if env.name == "TDDefense":
        net = Net.UNet(
            env.observation_space.shape[2], 64,
            map_size, map_size, None, 4, value_type='dependent'
        ).to(config.device)
    elif env.name == "TDAttack":
        net = Net.FCN(
            env.observation_space.shape[0], map_size, map_size,
            None, [4, *env.action_space.shape]
        )
    dqn = DQN.Model(eps_sche, env.action_space.n, net, config)
    return dqn