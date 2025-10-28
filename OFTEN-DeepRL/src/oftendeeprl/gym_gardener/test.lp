action(1) | action(2)| action(3)| action(4).

bction(3) | bction(2).

reward(A) :- action(A).
penalty(A) :- bction(A).

#minimize {P@1: penalty(P)}.
#maximize {P@1: reward(P)}.