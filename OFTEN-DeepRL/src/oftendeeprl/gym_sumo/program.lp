%%% external information provided by the python program

% wait time W for the ambulance with the worst wait time at edge E at time
% point 0
#external amb_edge(E,W,0) : E=0..edges-1, W=0..horizon.

% preference for changing state A of traffic light T
#external tl_pref(TL,A,0) : TL=0..tl-1, A=0..1.

% current phase of tl
#external tl_phase(TL,P,0) : TL=0..tl-1, P=0..1.

% decision to switch phase or not
tl_act(TL,0,T) | tl_act(TL,1,T) :- TL=0..tl-1, T=0..horizon-1, T \ 5 = 0.

% consequences of this decision
tl_phase(TL,P,T+5) :- tl_act(TL,0,T), tl_phase(TL,P,T).
tl_phase(TL,P',T+5) :- tl_act(TL,1,T), tl_phase(TL,P,T), P'=(P+1) \ 2.

% check ambulances
amb_edge(E,W+5,T+5) :- amb_edge(E,W,T), not green_edge(E,T), T < horizon - 1.

:~ amb_edge(E,W,T), W > 10. [W@1]
:~ tl_act(TL,A,T), tl_pref(TL,A',T), A != A'. [10@1]