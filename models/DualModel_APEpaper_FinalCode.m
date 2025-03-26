clear all
trail_number = 100;
T = 1000;           % Simulation time
Ns = 2;             % Number of sensory inputs
Na = 2;             % Number of actions
perf_save = zeros(trail_number,T);
wt_save=zeros(Ns, Na, trail_number,T);
wd_actor_save= zeros(Ns, Na, trail_number,T);
wd_critic_save= zeros(Ns,trail_number,T);
RPE_save = zeros(trail_number,T);
APE_save=zeros(trail_number,T);

for trails = 1:trail_number

    %params
    alpha_RPE  = 0.04;  % learning rate for RPE
    alpha_APE  = 0.02;  % learning rate for APE
    Anoise = 1;         % noise amplitude
    tau_APE = 100;      % time constant to compute APE
    tau_S = 10;         % filtering
    dt = 1;             % Time bin
    tau_wd = 100;       % weight decay

    % ini
    s = zeros(Ns,1);
    ad = zeros(Na,1);
    wd_actor = ones(Ns, Na);
    wd_critic = zeros(Ns, 1);
    at = zeros(Na,1);
    wt = ones(Ns, Na);
    arand = zeros(Na,1);
    awarded =  zeros(Na,1); % right rewarded (1 is right, 2 is left).
    pred_at  = zeros(Na,Ns);
    reward = zeros(1,T);
    reward_critic = zeros(1,T);
    V_critic = zeros(Ns,1);
    %  learning time
    for t = 1:T-1
        % sensory input, action rewarded
        s = 0*s;
        s(mod(t,2)+1) = 1;
        awarded = s;
        % s(1) = 1;
        % action
        ad = wd_actor'*s;
        at = wt'*s;
        arand = Anoise*rand(Na,1);          % noise on the action
        atot = ad + at+arand;               % sum of all actions
        [atotmax, atotmax_ind]= max(atot);  % max of actions is the action taken
        adone = zeros(Na,1);
        adone(atotmax_ind)= 1;              % action taken
        atail = zeros(Na,1);
        [atotmax, atotmax_ind]= max(at);
        atail(atotmax_ind)= 1;              
        pred_at(:,mod(t,2)+1) = (1-dt/tau_APE) * pred_at(:,mod(t,2)+1) + dt/tau_APE* atail;
        % efferent copy
        % learning pass
        V_critic = wd_critic.*s;
        reward(t) = sum(adone.*awarded);
        RPE = reward(t) -V_critic(mod(t,2)+1);
        wd_actor = wd_actor + alpha_RPE * RPE * s*adone';
        wd_actor = wd_actor.*(wd_actor>0);
        wd_actor = (1-dt/tau_wd) * wd_actor + dt/tau_wd; % go to steady state wd = 1
        wd_critic = wd_critic + alpha_RPE * RPE * s;
        wd_critic = wd_critic.*(wd_critic>0);
        [atotmax, atotmax_ind]= max(adone);
        APE =  1- pred_at(atotmax_ind,mod(t,2)+1);
        wt = wt + alpha_APE*APE.*(s*adone');
        wt = wt.*(wt>0);
        wt_save(:,:,trails, t)= wt;
        wd_actor_save(:,:,trails, t)= wd_actor;
        wd_critic_save(:,trails, t)= wd_critic;
        RPE_save(trails,t) = RPE;
        APE_save(trails,t)= APE;
    end

    reward_S = 0.5*ones(1,T);
    for t = 1:T-1
        reward_S(t+1)= (1-dt/tau_S) * reward_S(t) + dt/tau_S* reward(t);
    end

    perf_save(trails,:) = reward_S;

end