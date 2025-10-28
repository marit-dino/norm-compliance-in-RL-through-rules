%%% P_gen %%%
% This program corresponds to the subprogram P_gen described in section
% "ASP Framework" in the main document. The telingo variant does not include a
% P_check subprogram, as saturation technique is not possible. See section
% "Conclusion" in the main document for more information.

#program always.

% penalty for violating norm when stepping onto a plant
:~ player(C,R), plant(C,R), step(N). [(horizon - N) * 5@1]

% penalty for using the same cell twice
:~ player(C,R), player_hist(N,C,R), multi(M). [M * 10@1]

#program initial.

% player always starts at the 0 position
player(0,0).
step(0).

#program dynamic.

% helper to keep track of visited cells
step(N+1) :- 'step(N).
player_hist(N,C,R) :- 'player(C,R), 'step(N).
player_hist(N,C,R) :- 'player_hist(N,C,R).

% movement of the agent
action(0) | action(1) | action(2) | action(3).
player(C, R + 1) :- 'player(C, R), action(0).
player(C, R - 1) :- 'player(C, R), action(1).
player(C - 1, R) :- 'player(C, R), action(2).
player(C + 1, R) :- 'player(C, R), action(3).

% check the reward for each time step
reward(S,N - 1) :- 'player(C,R), action(A), reward(A,B,C,R), step(N),multi(M), S = (horizon - N + 1) * B * M.
reward(S, horizon+1):- player(C,R), target(C,R), multi(M), step(N), S = (horizon - N) * M * 5.

% hard restrictions (agent cannot leave window or move into a wall)
:- player(C,R), wall(C,R).
:- player(C,R), C < -radius.
:- player(C,R), C > radius.
:- player(C,R), R < -radius.
:- player(C,R), R > radius.


%%% Optimization %%%
% Here (in addition with weak constraints above) we implement the utilitarian
% policy fix described in section "Policy Fixing" in the main document.

% optimize for maximum reward and minimal penalty
#maximize {S@1 : reward(S,N)}.


#show action/1.