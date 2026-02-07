# thesis
Code for my master's thesis.

Open tasks:
- [x] start training from main
- [x] mine rules
- [ ] adapt feature extractor
- [ ] update process
    - [x] update method for rules to exlude specific states
    - [x] check whether less violations are possible in asp
    - [x] what to do if backtracking exceeds horizon? (should this even happen? No)
    - [x] fix "looping" in same violation for a few steps that happens from time to time
    - [x] store updated rule set
    - [x] everything can be enumerated because of the intervals -> adapt rule update process
    - [ ] save rules as string in pickle instead of object (-> add some tag to identify udpated rules)?
    - [x] prune rules only if there have been added enough
    - [x] check why "nothing to update" does not occur anymore
    - [x] deal with the case where pacman gets forbidden to move into a certain direction, but then repeatedly moves into a wall and afterwards eats a ghost
- [ ] documentation/comments
- [ ] add more complex norms
  - [x] add to norm check 
  - [x] often deep rl
  - [x] ghost_eaten over time in asp
  - [ ] add feature to features of rules
  - [ ] check whether it works for larger horizon
  - [x] move feature back to env (oftendeeprl)
- [x] evaluation: 
  - [x] count violations
  - [x] count number of used updated rules
  - [x] log rules which applied when violation occurs
- [ ] license(s)
- [ ] attributions for legible, oftendeeprl
- [x] (re)check (categorical) features in rule mining
- [x] clean up episode data collection
- [x] (don't) consider (absolute) features e.g. x, y in some rules?
- [x] rule pruning (check retention):
  - [x] combine rules
  - [x] move merging to rest of pruning
  - [x] instead of adding and then removing rules, check retention first
- [x] rule tag for creation
- [x] currently rules are updated at the wrong state
- [x] check how intervals affect mined rules
- [ ] fix bug with backtracking 2 states and then immediate violation (printing index is wrong / out of bounds)

TODO : 
mine rules and check whether f62 is contained (using a policy that has been trained longer)
update those rules and check whether that is done correctly