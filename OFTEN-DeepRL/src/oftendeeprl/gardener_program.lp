% external constants: frogs, horizon, radius, size


%%% Windowing %%%

% player always starts at the 0 position
pcol(0,0).
prow(0,0).

% frogs can be anywhere within the window
#external fcol(F,C,0) : F=0..frogs-1, C=-radius..radius.
#external frow(F,R,0) : F=0..frogs-1, R=-radius..radius.

% if a frog is outside the current window, it will not be considered for
% the computation
#external foutside(F) : F=0..frogs-1.

% depending on where the window is, the cells can contain walls or plants
#external wall(C,R) : C=-radius..radius, R=-radius..radius.
#external plant(C,R) : C=-radius..radius, R=-radius..radius.
#external target(C,R) : C=-radius..radius, R=-radius..radius.

% a multiplier for the reward
#external multi(M) : M=1..10.

% rewards for the actions of the agent
#external action(A, S, C, R) : A=0..3, S=0..3, C=-radius..radius, R=-radius..radius.


%%% P_gen %%%

% movement of the agent
pmove(0, T) | pmove(1, T) :- T=1..horizon - 1.
prow(R + 1,T) | prow(R - 1,T) :- prow(R,T-1), pmove(0, T).
pcol(C,T) :- pcol(C,T-1), pmove(0, T).
pcol(C + 1,T) | pcol(C - 1,T) :- pcol(C,T-1), pmove(1, T).
prow(R,T) :- prow(R,T-1), pmove(1, T).

% hard restrictions (agent cannot leave window or move into a wall)
:- pcol(C,_), C = -radius - 1.
:- pcol(C,_), C = radius + 1.
:- prow(R,_), R = -radius - 1.
:- prow(R,_), R = radius + 1.
:- prow(R,T), pcol(C,T), wall(C,R).

% penalty for violating norm when stepping onto a plant
:~ prow(R,T), pcol(C,T), plant(C,R), T > 0. [(horizon - T) * 5@1]

% penalty for using the same cell twice
:~ prow(R,T), pcol(C,T), prow(R,T1), pcol(C,T1), T != T1, multi(M). [M * 10@1]

% encode which action has been taken
action_taken(0, T, C, R) :- pcol(C,T), prow(R,T), prow(R+1,T+1).
action_taken(1, T, C, R) :- pcol(C,T), prow(R,T), prow(R-1,T+1).
action_taken(2, T, C, R) :- pcol(C,T), prow(R,T), pcol(C-1,T+1).
action_taken(3, T, C, R) :- pcol(C,T), prow(R,T), pcol(C+1,T+1).

% check the reward for each time step
reward(S, T) :- action_taken(A,T,C,R), action(A,B,C,R), multi(M), S = (horizon - T) * M * B.
reward(S, horizon+1):- prow(R,T), pcol(C,T), target(C,R), multi(M), S = (horizon - T) * M * 5.


%%% P_check %%%

% movement of the frogs
fmove(F,0, T) | fmove(F,1, T) :- F=0..frogs-1, T=1..horizon - 1.
frow(F,R + 1,T) | frow(F,R - 1,T) :- frow(F,R,T-1), fmove(F,0, T).
fcol(F,C,T) :- fcol(F,C,T-1), fmove(F,0, T).
fcol(F,C + 1,T) | fcol(F,C - 1,T) :- fcol(F,C,T-1), fmove(F,1, T).
frow(F,R,T) :- frow(F,R,T-1), fmove(F,1, T).

% saturate if frogs make illegal move
sat :- fcol(_,C,_), C = -radius - 1.
sat :- fcol(_,C,_), C = radius + 1.
sat :- frow(_,R,_), R = -radius - 1.
sat :- frow(_,R,_), R = radius + 1.
sat :- frow(F,R,T), fcol(F,C,T), wall(C,R).

% check saturation criteria
ok(0).
ok(0,F) :- F=0..frogs-1.
sat :- ok(horizon-1).
:- not sat.

% if deterministic instance
ok(T) :- T=1..horizon - 1, frogs = 0.

% saturate to M_sat as described in section "ASP Framework -> Saturation
% technique" in the main document
fcol(F,C,T) :- C = -radius..radius, sat, T=1..horizon-1, F=0..frogs-1.
frow(F,R,T) :- R = -radius..radius, sat, T=1..horizon-1, F=0..frogs-1.
fmove(F,0,T) :- F=0..frogs-1, T=1..horizon-1, sat.
fmove(F,1,T) :- F=0..frogs-1, T=1..horizon-1, sat.

% each frog either needs to be safely away from player or give penalty
ok(T,F) :- pcol(C,T), fcol(F,C',T), C != C', ok(T-1).
ok(T,F) :- prow(R,T), frow(F,R',T), R != R', ok(T-1).
good(T,G) | bad(T,G) :- G=0..frogs-1, T=1..horizon-1.
ok(T,F) :- bad(T,F).
ok(T,F) :- foutside(F), T=1..horizon-1.
ok(T) :- {ok(T,F) : F=0..frogs-1} = frogs, ok(T-1), frogs > 0.

% penalty for violating the norm of stepping onto a frog
penalty(P, T) :- P = (horizon - T) * 5, bad(T,F).


%%% Optimization %%%
% Here (in addition with weak constraints above) we implement the utilitarian
% policy fix described in section "Policy Fixing" in the main document.

% optimize for maximum reward and minimal penalty
% internally clingo negates #maximize statements and translates them into
% #minimize statements. Hence these optimization statements amounts to a 
% combined optimization that aims to maximize rewards - penalties
#minimize {S@1,T : penalty(S, T)}.
#maximize {R@1,T : reward(R,T)}.


#show action_taken/4.