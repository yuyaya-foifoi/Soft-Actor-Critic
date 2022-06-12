def polyak_average(main_net, target_net, rho=0.995):
    for mp, tp in zip(main_net.parameters(), target_net.parameters()):
        tp.data.copy_(rho * tp.data + (1 - rho) * mp.data)
