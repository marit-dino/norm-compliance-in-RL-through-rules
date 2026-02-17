%%% Windowing %%%

#external gcol(G,C,0) : G=0..ghosts-1, C=-radius..radius.
#external grow(G,R,0) : G=0..ghosts-1, R=-radius..radius.
#external goutside(0,G) : G=0..ghosts-1.
#external wall(C,R) : C=-radius..radius, R=-radius..radius.
#external eaten_ghost(N) : N=0..horizon.

ctd :- ctd_flag = 1.
permissive :- permissive_flag = 1.


% rewards for the actions of the agent
#external action(A,R) : A=0..3, R=0..3.

% player always starts at the 0 position
pcol(0,0).
prow(0,0).


%%% P_gen %%%

% movement of the agent
pmove(0, T) | pmove(1, T) :- T=1..horizon - 1.
prow(R + 1,T) | prow(R - 1,T) :- prow(R,T-1), pmove(0, T).
pcol(C,T) :- pcol(C,T-1), pmove(0, T).
pcol(C + 1,T) | pcol(C - 1,T) :- pcol(C,T-1), pmove(1, T).
prow(R,T) :- prow(R,T-1), pmove(1, T).

% hard restrictions (agent cannot leave window or move into a wall)
:- prow(R,T), pcol(C,T), wall(C,R).
sat :- grow(G,R,T), gcol(G,C,T), wall(C,R).

%%% P_check %%%

% movement of the ghosts
gmove(G,0, T) | gmove(G,1, T) :- G=0..ghosts-1, T=1..horizon - 1.
grow(G,R + 1,T) | grow(G,R - 1,T) :- grow(G,R,T-1), gmove(G,0, T).
gcol(G,C,T) :- gcol(G,C,T-1), gmove(G,0, T).
gcol(G,C + 1,T) | gcol(G,C - 1,T) :- gcol(G,C,T-1), gmove(G,1, T).
grow(G,R,T) :- grow(G,R,T-1), gmove(G,1, T).

% check saturation criteria
ok(0).
ok(0,G) :- G=0..ghosts-1.
sat :- ok(horizon-1).
:- not sat.

% saturation criteria for ctd
ok_ctd(0).
ok_ctd(0,G) :- G=0..ghosts-1.
sat_ctd :- ok_ctd(horizon-1).
:- not sat_ctd.

% saturation technique
gcol(G,C,T) :- C = -radius..radius, sat, T=1..horizon-1, G=0..ghosts-1.
grow(G,R,T) :- R = -radius..radius, sat, T=1..horizon-1, G=0..ghosts-1.
gmove(G,0,T) :- G=0..ghosts-1, T=1..horizon-1, sat.
gmove(G,1,T) :- G=0..ghosts-1, T=1..horizon-1, sat.

% each ghost either needs to be safely away from player or give penalty
% also consider ghosts previous position here, as pacman cannot "swap places"
% with ghosts like the agent can in gardener
ok(T,G,0) :- pcol(C,T), gcol(G,C',T-1), C != C', ok(T-1).
ok(T,G,1) :- pcol(C,T), gcol(G,C',T), C != C', ok(T-1).
ok(T,G,0) :- prow(R,T), grow(G,R',T-1), R != R', ok(T-1).
ok(T,G,1) :- prow(R,T), grow(G,R',T), R != R', ok(T-1).
ok(T,G) :- ok(T,G,0), ok(T,G,1).
ok(T,G) :- goutside(T,G), T=1..horizon-1.
good(T,G) | bad(T,G) :- G=0..ghosts-1, T=1..horizon-1.
ok(T,G) :- bad(T,G).
ok(T,G) :- eaten_ghost(T), permissive, G=0..ghosts-1, T=1..horizon-1.
ok(T) :- {ok(T,G) : G=0..ghosts-1} = ghosts, ok(T-1).

% if ghost eaten -> it cannot be eaten again (in this timeframe)
goutside(T+1,G) :- bad(T,G).
goutside(T+1,G) :- goutside(T,G), horizon(T).


% permissive
eaten_ghost(T+1) :- eaten_ghost(T), T=0..horizon-1.
eaten_ghost(T+1) :- bad(T,G), T=1..horizon-1, G=0..ghosts-1.

% ctd
ok_ctd(T,G) :- goutside(T,G), T=1..horizon-1.
ok_ctd(T) :- not ctd, T=0..horizon-1.
ok_ctd(T,G) :- ok(T,G), not bad(T,G).

ok_ctd(T,G) :- prow(R,T), pcol(C,T), grow(G,R,T), gcol(G,C,T), prow(R-1, T-1), ok_ctd(T-1), G=0..ghosts-1, T=1..horizon-1.
ok_ctd(T,G) :- bad_ctd(T,G).
good_ctd(T,G) | bad_ctd(T,G) :- G=0..ghosts-1, T=1..horizon-1.
ok_ctd(T) :- {ok_ctd(T,G) : G=0..ghosts-1} = ghosts, ok_ctd(T-1).

% penalty for violating the norm of eating a ghost
penalty(P,T) :- P = horizon - T, bad(T,G).
penalty_ctd(P, T) :- P = horizon - T, bad_ctd(T,G).
penalty_sum(P1+P2,T) :- penalty(P1,T), penalty_ctd(P2,T), ctd, T=1..horizon-1. 
penalty_sum(P1,T) :- penalty(P1,T), not penalty_ctd(P2,T), ctd, T=1..horizon-1, P2=0..horizon. 
penalty_sum(P2,T) :- not penalty(P1,T), penalty_ctd(P2,T), ctd, T=1..horizon-1, P1=0..horizon. 
penalty_sum(P,T) :- penalty(P,T), not ctd, T=1..horizon-1.

% encode the reward of the taken action
reward(X) :- action(0,X), pcol(0,0), pcol(1,1).
reward(X) :- action(1,X), pcol(0,0), pcol(-1,1).
reward(X) :- action(2,X), prow(0,0), prow(1,1).
reward(X) :- action(3,X), prow(0,0), prow(-1,1).


%%% Optimization %%%
% Here (in addition with weak constraints above) we implement the utilitarian
% policy fix described in section "Policy Fixing" in the main document.

% optimize for minimum reward (best action has a reward of 0, higher is worse)
% and minimal penalty
#minimize {S@1 : reward(S)}.
#minimize {S@2,T : penalty_sum(S, T)}.

#show pcol/2.
#show prow/2.