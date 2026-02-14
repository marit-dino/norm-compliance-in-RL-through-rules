% adapted from oftendeeprl

%%% Windowing %%%

#external gcol(G,C,0) : G=0..ghosts-1, C=-radius..radius.
#external grow(G,R,0) : G=0..ghosts-1, R=-radius..radius.
#external goutside(G) : G=0..ghosts-1.
#external wall(C,R) : C=-radius..radius, R=-radius..radius.
#external num_violations(N) : N=1..ghosts*max_horizon.
#external dynamic_horizon(N) : N=1..max_horizon.
#external eaten_ghost(N) : N=0..max_horizon.

horizon(H) :- H=1..N-1, dynamic_horizon(N).

ctd :- ctd_flag = 1.
permissive :- permissive_flag = 1.

% player always starts at the 0 position
pcol(0,0).
prow(0,0).


%%% P_gen %%%

% movement of the agent
pmove(0, T) | pmove(1, T) :- horizon(T).
prow(R + 1,T) | prow(R - 1,T) :- prow(R,T-1), pmove(0, T).
pcol(C,T) :- pcol(C,T-1), pmove(0, T).
pcol(C + 1,T) | pcol(C - 1,T) :- pcol(C,T-1), pmove(1, T).
prow(R,T) :- prow(R,T-1), pmove(1, T).

% hard restrictions (agent cannot leave window or move into a wall)
:- prow(R,T), pcol(C,T), wall(C,R).
sat :- grow(G,R,T), gcol(G,C,T), wall(C,R).

%%% P_check %%%

% movement of the ghosts
gmove(G,0, T) | gmove(G,1, T) :- G=0..ghosts-1, horizon(T).
grow(G,R + 1,T) | grow(G,R - 1,T) :- grow(G,R,T-1), gmove(G,0, T).
gcol(G,C,T) :- gcol(G,C,T-1), gmove(G,0, T).
gcol(G,C + 1,T) | gcol(G,C - 1,T) :- gcol(G,C,T-1), gmove(G,1, T).
grow(G,R,T) :- grow(G,R,T-1), gmove(G,1, T).

% check saturation criteria
ok(0).
ok(0,G) :- G=0..ghosts-1.
sat :- ok(H-1), dynamic_horizon(H).
:- not sat.

% saturation criteria for ctd
ok_ctd(0).
ok_ctd(0,G) :- G=0..ghosts-1.
sat_ctd :- ok_ctd(H-1), dynamic_horizon(H).
:- not sat_ctd.

% saturation technique
gcol(G,C,T) :- C = -radius..radius, sat, G=0..ghosts-1, horizon(T).
grow(G,R,T) :- R = -radius..radius, sat, G=0..ghosts-1, horizon(T).
gmove(G,0,T) :- G=0..ghosts-1, sat, horizon(T).
gmove(G,1,T) :- G=0..ghosts-1, sat, horizon(T).

% each ghost either needs to be safely away from player or give penalty
% also consider ghosts previous position here, as pacman cannot "swap places"
% with ghosts like the agent can in gardener
ok(T,G,0) :- pcol(C,T), gcol(G,C',T-1), C != C', ok(T-1).
ok(T,G,1) :- pcol(C,T), gcol(G,C',T), C != C', ok(T-1).
ok(T,G,0) :- prow(R,T), grow(G,R',T-1), R != R', ok(T-1).
ok(T,G,1) :- prow(R,T), grow(G,R',T), R != R', ok(T-1).
ok(T,G) :- ok(T,G,0), ok(T,G,1).
ok(T,G) :- goutside(G), horizon(T).
good(T,G) | bad(T,G) :- G=0..ghosts-1, horizon(T).
ok(T,G) :- bad(T,G).
ok(T,G) :- eaten_ghost(T), permissive, G=0..ghosts-1, horizon(T).
ok(T) :- {ok(T,G) : G=0..ghosts-1} = ghosts, ok(T-1).


%permissive 
eaten_ghost(T+1) :- eaten_ghost(T), T=0..N-1, dynamic_horizon(N).
eaten_ghost(T+1) :- bad(T,G), G=0..ghosts-1, horizon(T).


% ctd
ok_ctd(T,G) :- goutside(G), horizon(T).
ok_ctd(T) :- not ctd, horizon(T).
ok_ctd(T,G) :- ok(T,G), not bad(T,G).

ok_ctd(T,G) :- prow(R,T), pcol(C,T), grow(G,R,T), gcol(G,C,T), prow(R-1, T-1), ok_ctd(T-1), G=0..ghosts-1, horizon(T).
ok_ctd(T,G) :- bad_ctd(T,G).
good_ctd(T,G) | bad_ctd(T,G) :- G=0..ghosts-1, horizon(T).
ok_ctd(T) :- {ok_ctd(T,G) : G=0..ghosts-1} = ghosts, ok_ctd(T-1).

% penalty for violating the norm of eating a ghost
penalty(P,T) :- P = N - T, bad(T,G), dynamic_horizon(N).
penalty_ctd(P, T) :- P = N - T, bad_ctd(T,G), dynamic_horizon(N).
penalty_sum(P1+P2,T) :- penalty(P1,T), penalty_ctd(P2,T), ctd, horizon(T). 
penalty_sum(P1,T) :- penalty(P1,T), not penalty_ctd(P2,T), ctd, horizon(T), P2=0..max_horizon. 
penalty_sum(P2,T) :- not penalty(P1,T), penalty_ctd(P2,T), ctd, horizon(T), P1=0..max_horizon. 
penalty_sum(P,T) :- penalty(P,T), not ctd, horizon(T).

:- V1 = #sum{1, T, G : bad(T,G), T=0..H-1, dynamic_horizon(H), G=0..ghosts-1}, V2 = #sum{1, T, G : bad_ctd(T,G), T=0..H-1, dynamic_horizon(H), G=0..ghosts-1}, V = V1 + V2, V >= N, num_violations(N).

#show pcol/2.
#show prow/2.