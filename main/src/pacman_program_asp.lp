% adapted from oftendeeprl

%%% Windowing %%%

#external gcol(G,C,0) : G=0..ghosts-1, C=-radius..radius.
#external grow(G,R,0) : G=0..ghosts-1, R=-radius..radius.
#external goutside(G) : G=0..ghosts-1.
#external wall(C,R) : C=-radius..radius, R=-radius..radius.
#external num_violations(N) : N=1..num_norms*max_horizon.
#external horizon(N) : N=1..max_horizon.


% player always starts at the 0 position
pcol(0,0).
prow(0,0).


%%% P_gen %%%

% movement of the agent
pmove(0, T) | pmove(1, T) :- T=1..H - 1, horizon(H).
prow(R + 1,T) | prow(R - 1,T) :- prow(R,T-1), pmove(0, T).
pcol(C,T) :- pcol(C,T-1), pmove(0, T).
pcol(C + 1,T) | pcol(C - 1,T) :- pcol(C,T-1), pmove(1, T).
prow(R,T) :- prow(R,T-1), pmove(1, T).

% hard restrictions (agent cannot leave window or move into a wall)
:- prow(R,T), pcol(C,T), wall(C,R).
sat :- grow(G,R,T), gcol(G,C,T), wall(C,R).

%%% P_check %%%

% movement of the ghosts
gmove(G,0, T) | gmove(G,1, T) :- G=0..ghosts-1, T=1..H - 1, horizon(H).
grow(G,R + 1,T) | grow(G,R - 1,T) :- grow(G,R,T-1), gmove(G,0, T).
gcol(G,C,T) :- gcol(G,C,T-1), gmove(G,0, T).
gcol(G,C + 1,T) | gcol(G,C - 1,T) :- gcol(G,C,T-1), gmove(G,1, T).
grow(G,R,T) :- grow(G,R,T-1), gmove(G,1, T).

% check saturation criteria
ok(0).
ok(0,G) :- G=0..ghosts-1.
sat :- ok(H-1), horizon(H).
:- not sat.

% saturation technique
gcol(G,C,T) :- C = -radius..radius, sat, T=1..H-1, G=0..ghosts-1, horizon(H).
grow(G,R,T) :- R = -radius..radius, sat, T=1..H-1, G=0..ghosts-1, horizon(H).
gmove(G,0,T) :- G=0..ghosts-1, T=1..H-1, sat, horizon(H).
gmove(G,1,T) :- G=0..ghosts-1, T=1..H-1, sat, horizon(H).

% each ghost either needs to be safely away from player or give penalty
% also consider ghosts previous position here, as pacman cannot "swap places"
% with ghosts like the agent can in gardener
ok(T,G,0) :- pcol(C,T), gcol(G,C',T-1), C != C', ok(T-1).
ok(T,G,1) :- pcol(C,T), gcol(G,C',T), C != C', ok(T-1).
ok(T,G,0) :- prow(R,T), grow(G,R',T-1), R != R', ok(T-1).
ok(T,G,1) :- prow(R,T), grow(G,R',T), R != R', ok(T-1).
ok(T,G) :- ok(T,G,0), ok(T,G,1).
ok(T,G) :- goutside(G), T=1..H-1, horizon(H).
good(T,G) | bad(T,G) :- G=0..ghosts-1, T=1..H-1, horizon(H).
ok(T,G) :- bad(T,G).
ok(T) :- {ok(T,G) : G=0..ghosts-1} = ghosts, ok(T-1).

% penalty for violating the norm of eating a ghost
penalty(P, T) :- P = H - T, bad(T,F), horizon(H).

:- V = #sum{1, T, G : bad(T,G), T=0..H-1, G=0..ghosts-1, horizon(H)}, V >= N, num_violations(N).

#show pcol/2.
#show prow/2.